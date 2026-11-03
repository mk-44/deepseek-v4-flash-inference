import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from arch import Checkpoint, ModelArgs, Transformer, clock, load_module, upcast
from expert_store import Store
from full_store import load_full
from store_moe import store_block

CONFIG = "deepseek_official/0731/inference/config.json"
FIXED_HOME = "results/full_generation_niajtmvbhu_20261025T201703Z.json"
CORPUS = "corpus/texts.jsonl"


def small_model(args : ModelArgs, layers : list[int], n_gpu : int = 56, n_ram : int = 56):
    ckpt = Checkpoint("/v4local/ckpt")
    store = Store(ckpt, layers, n_gpu, n_ram)
    model = load_module(lambda : Transformer(args), ckpt, "", args, "cuda", skip = ("layers.", "mtp.")) # emb, norm, head, hc_head
    blocks = {layer : store_block(ckpt, args, store, layer) for layer in layers}
    model.layers = nn.ModuleList(blocks.values())
    return model, store, blocks

    
class RouteRecorder:
    def __init__(self, blocks : dict):
        self.layers = list(blocks)
        self.gates = {layer : block.ffn.gate for layer, block in blocks.items()}
        self.calls = {} # layer last call : selected_experts [uint8], wts [fp32], gap67 [fp32]
        self.mismatch = 0 # if top_6 differs in recomputation
        for layer, gate in self.gates.items():
            gate.register_forward_hook(self.recorder(layer))
    
    def recorder(self, layer : int):
        def record(gate, inputs, outputs):
            x, _ = inputs # [n, d] where n = b * s
            weights, indices = outputs
            
            if gate.hash:
                gap = torch.full((x.shape[0], ), float("inf"), device = x.device) # [n, ]
            else:
                scores = F.softplus(F.linear(upcast(x), upcast(gate.weight))).sqrt() + gate.bias # [n, E]
                top = scores.topk(gate.topk + 1, dim = -1) # [n, 7]
                gap = top.values[:, -2] - top.values[:, -1]
                self.mismatch += int((top.indices[:, : -1].sort(dim = -1).values != indices.sort(dim = -1).values).any(dim = -1).sum())
                
            self.calls[layer] = (indices.to(torch.uint8), weights, gap)
        return record
    
    def take(self, b : int, s : int): # put on cpu and clear it.
        L = len(self.layers)
        experts, weights, gap = (torch.stack([self.calls[l][i] for l in self.layers], dim = 1) for i in range(3)) # [n, L, 6], [n, L, 6], [n, L]
        self.calls = {}
        return {"experts" : experts.view(b, s, L, -1).cpu(), "weights" : weights.view(b, s, L, -1).cpu(), "gap67" : gap.view(b, s, L).cpu()}
    

MAX_TEXTS = 4
MAX_SLOTS = 3_000

def slots(texts : list[dict]) -> int:
    return len(texts) * max(len(t["ids"]) for t in texts)


def plan_passes(texts : list[dict]) -> list[list[dict]]:
    passes = []
    for t in sorted((t for t in texts if t["shape"] == "qa"), key = lambda t : len(t["ids"])):
        if passes and len(passes[-1]) < MAX_TEXTS and slots(passes[-1] + [t]) <= MAX_SLOTS:
            passes[-1].append(t)
        else:
            passes.append([t])
    passes += [[t] for t in texts if t["shape"] == "long"]
    return sorted(passes, key = slots, reverse = True)


EOS = 1 # <|end_of_sentence|>
def run_pass(model : Transformer, store : Store, recorder : RouteRecorder, texts : list[dict]):
    b, s = len(texts), max(len(t["ids"]) for t in texts)
    ids = torch.full((b, s), EOS, dtype = torch.long)
    for i, t in enumerate(texts):
        ids[i, : len(t["ids"])] = torch.tensor(t["ids"])
    ids = ids.to("cuda")
    
    n0, read0, wait0 = dict(store.n), store.read_s, store.wait_s
    torch.cuda.reset_peak_memory_stats()
    t0 = clock()
    model(ids) # all prefill
    seconds = clock() - t0
    routes = recorder.take(b, s) # [b, s, L, 6], [b, s, L, 6], [b, s, L]
    loads = {tier : store.n[tier] - n0[tier] for tier in n0}
    
    predicted = {"gpu" : 0, "ram" : 0, "disk" : 0}
    for l, layer in enumerate(recorder.layers):
        for e in routes["experts"][:, :, l].unique().tolist():
            predicted[store.homes[layer, e].tier] += 1
    assert predicted == loads, f"check 3: the routes predict loads {predicted}, the store counted {loads}"
    
    lines = []
    for i, t in enumerate(texts):
        T = len(t["ids"])
        line = {k : v[i, : T] for k, v in routes.items()}
        
        for l, layer in enumerate(recorder.layers):
            gate = recorder.gates[layer]
            if gate.hash:
                assert torch.equal(line["experts"][:, l].long(), gate.tid2eid[ids[i, : T]].long().cpu()), f"check 4: {t['id']}, layer {layer}"
        
        lines.append(line)

    stats = {
        "b" : b, 
        "s" : s, 
        "pads" : b * s - sum(len(t["ids"]) for t in texts), 
        "seconds" : round(seconds, 3), 
        "loads" : loads,
        "disk read s" : round(store.read_s - read0, 3), 
        "copy wait s" : round(store.wait_s - wait0, 3),
        "GPU peak GB" : round(torch.cuda.max_memory_allocated() / 1e9, 2)
    }
    
    return lines, stats

    
ROUTES = Path("routes")


def save_routes(text : dict, line : dict, stats : dict, folder : Path = ROUTES) -> Path: # one line's file, whole or not at all
    folder.mkdir(exist_ok = True)
    path = folder / f"{text['id']}.pt"
    tmp = path.with_suffix(".tmp")
    
    torch.save(
        {
            "id" : text["id"], 
            "ids" : torch.tensor(text["ids"], dtype = torch.int32), 
            "spans" : text["spans"],
            "experts" : line["experts"].clone(),
            "weights" : line["weights"].half(), 
            "gap67" : line["gap67"].half(), 
            "pass" : stats}, 
        tmp
    )
    
    os.replace(tmp, path)
    return path


def compare_routes(ours : dict, theirs : dict, n_hash : int, typical : torch.Tensor | None = None) -> dict: # the same positions from two runs
    # ours : experts [T, L, 6], weights [T, L, 6], gap67 [T, L]
    # theirs: experts, weights [T, L, 6]
    # columns 0 .. n_hash-1 are hash layers
    if typical is None:
        typical = ours["gap67"][:, n_hash :].float().median(dim = 0).values # [L - n_hash]
    e_o, i_o = ours["experts"].long().sort(dim = -1) # [T, L, 6]: each position's 6 experts by id, and where each sat in the pick
    e_t, i_t = theirs["experts"].long().sort(dim = -1) # [T, L, 6]
    same = (e_o == e_t).all(dim = -1) # [T, L]: the same 6 experts
    swapped = 6 - (e_o[..., :, None] == e_t[..., None, :]).any(dim = -1).sum(dim = -1) # [T, L]: how many of our 6 the other run didn't pick
    diff = (ours["weights"].gather(-1, i_o) - theirs["weights"].gather(-1, i_t)).abs().amax(dim = -1)[same] # where they agree
    r = {"positions" : same.shape[0],
         "hash same" : int(same[:, : n_hash].sum()), "hash total" : same[:, : n_hash].numel(),
         "learned same" : int(same[:, n_hash :].sum()), "learned total" : same[:, n_hash :].numel(),
         "largest weight difference" : diff.max().item() if diff.numel() else 0.0,
         "median learned gap67" : typical.median().item(),
         "misses" : [(p, l, round(ours["gap67"][p, l].item(), 5), int(swapped[p, l])) for p, l in (~same).nonzero().tolist()]} 
    # rounding can only flip a near-tie, closer than the layer's typical gap.
    
    r["ok"] = r["hash same"] == r["hash total"] and all(gap < typical[l - n_hash] for _, l, gap, _ in r["misses"])
    return r


def check_decode(model : Transformer, recorder : RouteRecorder, prompts : dict) -> list[dict]: # one prefill vs real decode
    n_hash = sum(recorder.gates[l].hash for l in recorder.layers)
    results = []
    for name, p in prompts.items():
        P, n = len(p["prompt"]), len(p["new_tokens"]) - 1 # n answer tokens went back in, one decode step each; the last never did
        model(torch.tensor([p["prompt"] + p["new_tokens"][: n]], device = "cuda")) # one prefill: [1, P + n]
        ours = {k : v[0] for k, v in recorder.take(1, P + n).items()} # [P + n, L, 6], [P + n, L, 6], [P + n, L]

        per_step = len({c["layer"] for c in p["calls"]}) # the run recorded one call per layer per step
        steps = [{c["layer"] : c for c in p["calls"][i : i + per_step]} for i in range(0, len(p["calls"]), per_step)] # prefill
        theirs = {k : torch.tensor([[steps[0][l][k][q] for l in recorder.layers] for q in range(P)] + # the prompt : rows of the prefill
                                   [[steps[j][l][k][0] for l in recorder.layers] for j in range(1, n + 1)]) # the answer : decode steps
                  for k in ("experts", "weights")} # [P + n, L, 6] each

        typical = ours["gap67"][:, n_hash :].float().median(dim = 0).values # [L - n_hash]: each learned layer's typical gap67, whole prefill
        for part, rows in (("prompt", slice(0, P)), ("answer", slice(P, P + n))):
            r = compare_routes({k : v[rows] for k, v in ours.items()}, {k : v[rows] for k, v in theirs.items()}, n_hash, typical)
            results.append({"prompt" : name, "part" : part, **r})
            m = r["misses"]
            print(f"  check 1, {name} {part}: {r['positions']} positions | hash layers {r['hash same']}/{r['hash total']} |"
                  f" learned {r['learned same']}/{r['learned total']} | {len(m)} misses, experts swapped per miss {sorted({x[3] for x in m})},"
                  f" largest gap67 {max((x[2] for x in m), default = 0):.5f} vs median {r['median learned gap67']:.4f} |"
                  f" largest weight difference {r['largest weight difference']:.3g} | {'ok' if r['ok'] else 'FAIL'}")
    return results


def check_alone(model : Transformer, store : Store, recorder : RouteRecorder, texts : list[dict]) -> list[dict]: # check 2: batch vs alone
    n_hash = sum(recorder.gates[l].hash for l in recorder.layers)
    together, _ = run_pass(model, store, recorder, texts)
    results = []
    for t, line in zip(texts, together):
        (alone,), _ = run_pass(model, store, recorder, [t])
        r = compare_routes(line, alone, n_hash)
        results.append({"line" : t["id"], **r})
        m = r["misses"]
        print(f"  check 2, {t['id']}: {r['positions']} positions, next to {slots(texts) // len(texts) - len(t['ids'])} pads in the batch |"
              f" hash layers {r['hash same']}/{r['hash total']} | learned {r['learned same']}/{r['learned total']} |"
              f" {len(m)} misses, experts swapped per miss {sorted({x[3] for x in m})},"
              f" largest gap67 {max((x[2] for x in m), default = 0):.5f} vs median {r['median learned gap67']:.4f} |"
              f" largest weight difference {r['largest weight difference']:.3g} | {'ok' if r['ok'] else 'FAIL'}")
    return results


if __name__ == "__main__":
    args = ModelArgs(**json.load(open(CONFIG)))
    dry = len(sys.argv) > 1
    texts = [json.loads(line) for line in open(CORPUS)]
    passes = plan_passes(texts)
    print(f"{len(texts)} lines, {sum(len(t['ids']) for t in texts):,} tokens -> {len(passes)} passes")

    t0 = time.perf_counter()
    if dry:
        model, store, blocks = small_model(args, list(range(int(sys.argv[1]))))
        folder = Path("/tmp/routes_dry")
    else:
        model, store = load_full(args, args.n_layers)
        blocks = dict(enumerate(model.layers))
        folder = ROUTES
    
    
    recorder = RouteRecorder(blocks)
    homes = {tier : sum(h.tier == tier for h in store.homes.values()) // len(blocks) for tier in ("gpu", "ram", "disk")}
    print(f"model ready in {(time.perf_counter() - t0) / 60:.1f} min | {len(blocks)} layers |"
          f" per layer {homes['gpu']} GPU + {homes['ram']} RAM + {homes['disk']} disk homes |"
          f" GPU free {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB + {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e9:.2f} GB cached")

    store.fetch(0, 0) # loads cuda kernels as well.

    check1 = check_decode(model, recorder, json.load(open(FIXED_HOME))["prompts"])
    check2 = check_alone(model, store, recorder, min((p for p in passes if len(p) > 1), key = slots))
    assert all(r["ok"] for r in check1 + check2), "check 1 or 2 failed: stopping before the corpus"
    
    t1 = time.perf_counter()
    for k, p in enumerate(passes):
        if all((folder / f"{t['id']}.pt").exists() for t in p):
            continue # saved by an earlier run
        lines, stats = run_pass(model, store, recorder, p)
        for t, line in zip(p, lines):
            save_routes(t, line, stats, folder)
        f = stats["loads"]
        print(f"pass {k + 1}/{len(passes)} [{stats['b']}, {stats['s']}]: {stats['seconds']:.1f} s | loads {f['gpu']}/{f['ram']}/{f['disk']} |"
              f" disk read {stats['disk read s']:.1f} s, copy {stats['copy wait s']:.1f} s | GPU peak {stats['GPU peak GB']:.2f} GB |"
              f" {(time.perf_counter() - t1) / 60:.1f} min so far | " + ", ".join(t["id"] for t in p))

    summary = {"layers" : len(blocks), "homes per layer" : homes, "check 1" : check1, "check 2" : check2,
               "recomputed top 6 differ" : recorder.mismatch,
               "passes" : [{"lines" : [t["id"] for t in p], **torch.load(folder / f"{p[0]['id']}.pt")["pass"]} for p in passes]}
    path = folder / "summary.json" if dry else Path(f"results/collect_routes_{os.uname().nodename}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    json.dump(summary, open(path, "w"), indent = 1)
    print(f"all {len(passes)} passes saved in {folder} | summary -> {path}")
    store.close()
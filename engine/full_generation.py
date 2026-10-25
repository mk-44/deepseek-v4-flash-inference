import json
import os
import time
import traceback
import torch
from arch import ChatTokenizer, ModelArgs, Transformer
from full_store import FullStore, load_full
from generation import RouteTrace, generate_greedy, teacher_forced_check
from store_moe import MoEWatch

N_NEW = 32 # max num new tokens per prompt.


class StepLog: # log after each generated token
    def __init__(self, watches : dict[int, MoEWatch], chat : ChatTokenizer):
        self.watches, self.chat = watches, chat
        self.rows = [] # prompt steps
        self.layer_s = {layer : [0.0, 0] for layer in watches} # decode steps : moe seconds and steps per layer.

    def __call__(self, step : int, seconds : float, token : torch.Tensor):
        w = list(self.watches.values())
        row = {
            "seconds" : seconds, 
            "moe" : sum(x.seconds for x in w), 
            "read" : sum(x.read for x in w), 
            "wait" : sum(x.wait for x in w),
            "fetched" : {tier : sum(x.fetched[tier] for x in w) for tier in ("gpu", "ram", "disk")},
            "peak" : max([x.mem0 + x.peak for x in w] + [torch.cuda.max_memory_allocated()]), 
            "token" : token.item()
        }
        
        self.rows.append(row)
        
        if step > 0: # a decode step
            for layer, x in self.watches.items():
                self.layer_s[layer][0] += x.seconds
                self.layer_s[layer][1] += 1
        f = row["fetched"]
        
        print(f"  step {step}: {seconds * 1e3:,.0f} ms |"
              f"  MoE {row['moe'] * 1e3:,.0f} ms (disk read {row['read'] * 1e3:,.0f}, copy {row['wait'] * 1e3:,.0f}) |"
              f"  fetched {f['gpu']}/{f['ram']}/{f['disk']} | -> {self.chat.tokenizer.decode([row['token']], skip_special_tokens = False)!r}")

        
def run_prompt(model : Transformer, args : ModelArgs, tokenizer : ChatTokenizer, trace : RouteTrace, log : StepLog, ids : list[int]) -> dict:
    prompt = torch.tensor([ids], device = "cuda") # [1, s]
    trace.calls, log.rows = [], []
    tokens, scores, seconds = generate_greedy(model, args, prompt, N_NEW, log, stop = tokenizer.eos_id)
    calls, trace.calls = trace.calls, []
    assert torch.isfinite(scores).all(), "some token logits are not finite"
    
    top5 = scores[0].log_softmax(dim = -1).topk(5) # 5 most likely first tokens.
    text = tokenizer.tokenizer.decode(tokens[0].tolist(), skip_special_tokens = False)
    print(f"TTFT {seconds[0]:.1f} s | first token's top-5: " +
          " | ".join(f"{tokenizer.tokenizer.decode([i])!r} {p:.2f}" for p, i in zip(top5.values.tolist(), top5.indices.tolist())))
    
    decode = log.rows[1:] # more than one token generated i.e. decode took place after prefill.
    
    if decode:
        n, s = len(decode), sorted(r["seconds"] for r in decode)
        mean = lambda key : sum(r[key] for r in decode) / n
        f = {tier : sum(r["fetched"][tier] for r in decode) / n for tier in ("gpu", "ram", "disk")}
        
        print(f"  {n} decode steps : per token p50 {s[n // 2]:.2f} s, p90 {s[int(0.9 * (n - 1))]:.2f} s, max {s[-1]:.2f} s ({n / sum(s):.2f} tokens/s)\n"
              f"  per token: disk read {mean('read'):.2f} s, copy waits {mean('wait'):.2f} s,"
              f" rest of the MoE {mean('moe') - mean('read') - mean('wait'):.2f} s,"
              f" outside the MoE {mean('seconds') - mean('moe'):.2f} s | fetched {f['gpu']:.0f}/{f['ram']:.0f}/{f['disk']:.0f}")
        
    
    print(f"  GPU peak {max(r['peak'] for r in log.rows) / 1e9:.2f} GB | text {text!r}")
    
    same, diff, margin = teacher_forced_check(model, prompt, tokens, scores)
    
    print(f"  check, one prefill over prompt + new tokens: same tokens {int(same.sum())}/{len(same)} |"
          f" largest score difference {diff.max():.3g} | smallest top-2 margin {margin.min():.3g}")
    
    for step in (~same).nonzero().flatten().tolist():
        print(f"  step {step} differs: score difference {diff[step]:.3g}, top-2 margin {margin[step]:.3g}")
    
    
    return {
        "prompt" : ids, 
        "new_tokens" : tokens[0].tolist(), 
        "text" : text, 
        "steps" : log.rows,
        "top5" : {"ids" : top5.indices.tolist(), "log_probs" : top5.values.tolist()},
        "check" : {"same" : same.tolist(), "diff" : diff.tolist(), "margin" : margin.tolist()}, 
        "calls" : calls
    }


def layer_table(args : ModelArgs, store : FullStore, log : StepLog, records : list[dict]) -> list[list[int]]: # each expert hits in decode.
    hits = [[0] * args.n_routed_experts for _ in log.layer_s] # [layer][expert]
    
    for r in records:
        for call in r["calls"]:
            if len(call["experts"]) == 1: # a decode step - one token, 6 experts
                for e in call["experts"][0]:
                    hits[call["layer"]][e] += 1
                    
    total = {"gpu" : 0, "ram" : 0, "disk" : 0}
    print("per layer, decode steps: MoE ms per token | hits gpu/ram/disk | experts used | the 5 most hit (expert x hits)")
    
    for layer, h in enumerate(hits):
        tiers = {"gpu" : 0, "ram" : 0, "disk" : 0}
        for e, k in enumerate(h):
            tiers[store.homes[layer, e].tier] += k
        for tier in total:
            total[tier] += tiers[tier]
        r = args.compress_ratios[layer]
        
        kind = ("SWA" if r == 0 else "CSA" if r == 4 else "HCA") + (" hash" if layer < args.n_hash_layers else "")
        seconds, steps = log.layer_s[layer]
        top = sorted(range(len(h)), key = lambda e : -h[e])[:5]
        
        print(f"  L{layer:02d} {kind:<8} | {seconds / max(steps, 1) * 1e3:4.0f} ms | {tiers['gpu']}/{tiers['ram']}/{tiers['disk']} |"
              f" {sum(k > 0 for k in h):3d} of {len(h)} | " + ", ".join(f"{e} x{h[e]}" for e in top))
    
    n = max(sum(total.values()), 1)
    print(f"decode hits: {sum(total.values()):,} | GPU {total['gpu'] / n:.0%}, RAM {total['ram'] / n:.0%}, disk {total['disk'] / n:.0%}")
    return hits

if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    t = time.perf_counter()
    model, store = load_full(args, args.n_layers)
    print(f"model ready in {(time.perf_counter() - t) / 60:.1f} min | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    blocks = dict(enumerate(model.layers))
    watches = {layer : MoEWatch(block.ffn, store) for layer, block in blocks.items()}
    trace = RouteTrace(blocks)
    tokenizer = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    log = StepLog(watches, tokenizer)
    ask = lambda text : tokenizer.encode_chat([{"role" : "user", "content" : text}])
    
    prompts = {
        "france" : [0] + tokenizer.tokenizer.encode("The capital of France is").ids, # 0 = <|begin_of_sentence|>
        "2+2" : ask("What is 2+2?"),
        "sky" : ask("Explain in two sentences why the sky is blue."),
        "license" : ask("Summarize this in one sentence:\n\n" + open("LICENSE").read()), # 266 tokens - nearly every expert in prefill
    }
    store.fetch(0, 0) # the first decode also loads cuda kernels.

    records = {}
    with torch.inference_mode():
        for name, ids in prompts.items():
            print(f"prompt {name!r}, {len(ids)} tokens:")
            try:
                records[name] = run_prompt(model, args, tokenizer, trace, log, ids)
            except Exception: # one prompt failure should not throw away the 15 min load
                traceback.print_exc()
                
    hits = layer_table(args, store, log, list(records.values()))
    path = f"results/full_generation_{os.uname().nodename}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    json.dump({"homes per layer" : {tier : n // len(blocks) for tier, n in store.count.items()}, "prompts" : records, "hits" : hits}, open(path, "w"))
    print(f"saved -> {path}")
    store.close()
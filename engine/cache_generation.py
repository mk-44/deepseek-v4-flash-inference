import json
import os
import sys
import time
import traceback
from pathlib import Path
import torch
from arch import ChatTokenizer, Checkpoint, ModelArgs, SequenceState, Transformer, clock
from cache_store import FIXED_HOME, FOLDER, CacheStore, box_sizes, load_cache
from full_store import saved_entries
from generation import LAST_STEP, generate_greedy, teacher_forced_check
from store_moe import MoEWatch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from expert_plan import E, INDEX, ROUTES, load_texts, make_plan

N_FIXED = 32 # new tokens for each of the fixed_home run 4 prompts - as we did same for that run.
N_TEST = 150 # new tokens for each test prompt
EVERY = 10 # a step line every 10 tokens

def steal_seconds() -> float: # cpu time stolen so far, summed over all cpus i.e. the VM's host ran someone else while ours had work.
    try:
        return int(open("/proc/stat").readline().split()[8]) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return float("nan")

DISK = {} # name of the disk under /v4local, found on the first call. / is an overlay, so os.stat gives no disk we can look up

def disk_counters() -> tuple[float, float]: # the disk under /v4local so far: seconds with a request in flight, and GB written
    try:
        rows = {f[2] : f for f in map(str.split, open("/proc/diskstats"))} # disk name to its counters
        if not DISK: # the first call comes after the load, so the disk that read the most since boot is ours
            DISK["name"] = max(rows, key = lambda name : int(rows[name][5])) # field 5 is sectors read
            print(f"time map: disk busy and written are {DISK['name']}'s, by /proc/diskstats")
        f = rows[DISK["name"]]
        return int(f[12]) / 1000, int(f[9]) * 512 / 1e9 # field 12 is ms with a request in flight, field 9 is sectors written of 512 bytes
    except (OSError, IndexError, ValueError, KeyError):
        return float("nan"), float("nan")

def map_line(rows : list[dict]) -> str: # the time map, averaged over the given steps.
    avg = lambda f : sum(f(r["map"]) for r in rows) / len(rows)
    wait = lambda k : avg(lambda m : m["gpu waits"][k])

    return (
        f"CPU working {avg(lambda m : m['cpu work']):.3f} s (our logging {avg(lambda m : m.get('log cpu', 0.0)) * 1e3:.2f} ms "
        f"| in the MoE {avg(lambda m : m['cpu work in moe']):.3f}, of it in disk read calls"
        f" {avg(lambda m : m['read cpu']):.3f}) | waiting for the GPU {avg(lambda m : sum(m['gpu waits'].values())):.3f} s (gate {wait('gate'):.3f},"
        f" swaps {wait('swaps'):.3f}, copies {wait('copies'):.3f}, step end {wait('step end'):.3f}, MoE timer {wait('moe clock'):.3f})"
        f" | waiting for the disk {avg(lambda m : m['disk wait']):.3f} s | not running {avg(lambda m : m['not running']):.3f} s"
        f" (steal, all CPUs {avg(lambda m : m['steal']):.3f} s) | disk busy {avg(lambda m : m['disk busy']):.3f} s, written {avg(lambda m : m['disk written']):.2f} GB"
    )


class CacheTrace: # one hook on every gate, picks go to the store first (the cache moves) and then into the record.
    def __init__(self, blocks : dict, store : CacheStore):
        self.store = store
        self.calls = [] # every gate call {layer, experts [tokens][6], weights [tokens][6]} - as RouteTrace recorded them.
        self.prepare_s = 0.0 # seconds inside store.prepare i.e. the moves list and their disk reads.
        self.gate_wait_s = 0.0 # seconds the cpu waited for the gpu to reach the gates.
        self.swap_wait_s = 0.0 # seconds the cpu waited for the swaps that prepare queued.
        self.log_cpu = 0.0 # cpu seconds spent keeping the record: the gate weights read back and the calls list.
        
        
        for layer, block in blocks.items():
            block.ffn.gate.register_forward_hook(self.hook(layer))

    def hook(self, layer : int):
        def after_gate(gate, inputs, outputs): # runs as the gate returns, before the MoE loop asks the store for any expert
            weights, indices = outputs # [tokens, 6], [tokens, 6]
            
            t = time.perf_counter()
            rows = indices.tolist() # cpu waits for the gate here before prepare queues any copy.
            self.gate_wait_s += time.perf_counter() - t
            c = time.thread_time()
            w = weights.tolist() # for the record only. the gate is done on the gpu, so this does not wait.
            self.log_cpu += time.thread_time() - c
            t = time.perf_counter()

            self.store.prepare(layer, rows) # the cache moves on the bytes: every expert of this call is then where the table says
            self.prepare_s += time.perf_counter() - t
            t = time.perf_counter()
            torch.cuda.synchronize() # wait for the swaps prepare queued. the MoE counts.tolist() would wait for them anyway, here the wait is timed
            self.swap_wait_s += time.perf_counter() - t
            
            c = time.thread_time()
            self.calls.append({"layer" : layer, "experts" : rows, "weights" : w})
            self.log_cpu += time.thread_time() - c
        return after_gate

class StepLog: # after each generated token - log its [time], [MoE time], [the store reads and moves in that step] and [the token].
    def __init__(self, watches : dict[int, MoEWatch], trace : CacheTrace, tokenizer : ChatTokenizer, every : int = 1):
        self.watches, self.trace, self.store, self.tokenizer = watches, trace, trace.store, tokenizer
        self.every = every
        self.rows = []
        self.own_s = 0.0 # cpu seconds inside this logger, after the steps.
        self.mark()
    
    def mark(self): # the store current values.
        self.n0, self.read0, self.wait0, self.prepare0 = dict(self.store.n), self.store.read_s, self.store.wait_s, self.trace.prepare_s
        self.gate0, self.moe_wait0, self.steal0, self.disk0 = self.trace.gate_wait_s, MoEWatch.waited, steal_seconds(), disk_counters()
        self.swap0, self.read_cpu0, self.moe_cpu0 = self.trace.swap_wait_s, self.store.read_cpu_s, MoEWatch.cpu
        self.log0 = MoEWatch.log_cpu + self.trace.log_cpu # our own measuring code, so far
        torch.cuda.reset_peak_memory_stats() # the step's gpu memory peak starts here. it is read once, when the step ends
        self.cpu0 = time.thread_time() # last, so the readings above don't count in the step's cpu time.
    
    def __call__(self, step : int, seconds : float, token : torch.Tensor):
        entry = time.thread_time()
        cpu = entry - self.cpu0 # cpu time this thread used in the step. its waits for the GPU spin, so they count here too.
        
        n = {k : v - self.n0[k] for k, v in self.store.n.items()} # current step counters
        read, read_cpu = self.store.read_s - self.read0, self.store.read_cpu_s - self.read_cpu0 # time inside the disk read calls, and the cpu part of it
        waits = {
            "gate" : self.trace.gate_wait_s - self.gate0, # for the gpu to reach each layer's gate
            "swaps" : self.trace.swap_wait_s - self.swap0, # for the swaps each layer's prepare queued
            "step end" : seconds - LAST_STEP["cpu"], # for the gpu to finish the token
            "copies" : self.store.wait_s - self.wait0, # for queued copies
            "moe clock" : MoEWatch.waited - self.moe_wait0 # the old MoE timer's own waits, 0 with GPU timestamps
        }
        in_moe = MoEWatch.cpu - self.moe_cpu0 - waits["gate"] - waits["swaps"] - waits["copies"] # cpu work inside the MoEs, without their waits for the gpu
        busy, written = (now - then for now, then in zip(disk_counters(), self.disk0))
        
        row = {
            "seconds" : seconds,
            "moe" : sum(w.gpu_seconds() for w in self.watches.values()), # current step MoE calls [prepare included.]
            "prepare" : self.trace.prepare_s - self.prepare0,
            "read" : read,
            "wait" : self.store.wait_s - self.wait0,
            "reads" : n["pre kept"] + n["miss"] + n["disk"], # experts read from experts.bin [keeps and misses in prepare, disk only fetches]
            "peak" : max([w.mem0 + w.peak for w in self.watches.values() if MoEWatch.memory] + [torch.cuda.max_memory_allocated()]), # MoE reads off: the step's own peak, since mark()
            "counts" : n,
            "token" : token.item(),
            "map" : {
                "cpu work" : cpu - sum(waits.values()),
                "cpu work in moe" : in_moe,
                "read cpu" : read_cpu,
                "log cpu" : MoEWatch.log_cpu + self.trace.log_cpu - self.log0, # our own measuring code inside the step
                "gpu waits" : waits,
                "disk wait" : read - read_cpu, # read time without its cpu part
                "not running" : seconds - cpu - (read - read_cpu), # the cpu part of the reads is already in cpu
                "steal" : steal_seconds() - self.steal0,
                "disk busy" : busy,
                "disk written" : written
            }
        }
        
        self.rows.append(row)
        if step % self.every == 0:
            gpu, ram, disk = (n["t1"] + n["pre t1"], n["t2"] + n["pre t2"]), (n["t3"] + n["pre t3"], n["t4"] + n["pre t4"]), n["miss"] + n["pre t5"]
            print(f"  step {step}: {seconds * 1e3:,.0f} ms | MoE {row['moe'] * 1e3:,.0f} ms (disk read {row['read'] * 1e3:,.0f} ms for {row['reads']:,} experts,"
                  f" copy wait {row['wait'] * 1e3:,.0f} ms) | GPU {gpu[0]:,} + {gpu[1]:,}, RAM {ram[0]:,} + {ram[1]:,}, disk {disk:,}"
                  f" | -> {self.tokenizer.tokenizer.decode([row['token']], skip_special_tokens = False)!r}")
            print(f"    time map: {map_line([row])}")
        self.mark()
        self.own_s += self.cpu0 - entry # this call's own cpu. it runs after the step's clock, so it is in no step's time.
        
def frozen_check(model : Transformer, trace : CacheTrace, prompt : torch.Tensor, tokens : torch.Tensor, scores : torch.Tensor) -> dict:
    # teacher forced check (one prefill over prompt + answer) with the cache frozen, so no move and none of the fetches counted.
    store = trace.store
    saved = dict(store.n), store.read_s, store.wait_s, trace.prepare_s
    store.frozen, trace.calls = True, []
    try:
        same, diff, margin = teacher_forced_check(model, prompt, tokens, scores) # [n_new] each
    finally:
        store.frozen, trace.calls = False, []
        store.n.update(saved[0])
        store.read_s, store.wait_s, trace.prepare_s = saved[1:]
    return {"same" : same.tolist(), "diff" : diff.tolist(), "margin" : margin.tolist()}


def run_prompt(
    model : Transformer, 
    args : ModelArgs, 
    tokenizer : ChatTokenizer, 
    trace : CacheTrace, 
    log : StepLog,
    ids : list[int], 
    n_new : int, 
    check : bool
) -> dict: # one prompt - greedy generation and its record.
    store = trace.store
    prompt = torch.tensor([ids], device = "cuda") # [1, s]
    trace.calls, log.rows, log.own_s = [], [], 0.0
    log.mark()
    
    n0, read0, wait0 = dict(store.n), store.read_s, store.wait_s
    tokens, scores, seconds = generate_greedy(model, args, prompt, n_new, log, stop = tokenizer.eos_id) # [1, n], [n, vocab], n
    top5 = scores[0].log_softmax(dim = -1).topk(5) # first token's 5 most likely options.
    
    record = {
        "prompt" : ids,
        "new_tokens" : tokens[0].tolist(),
        "text" : tokenizer.tokenizer.decode(tokens[0].tolist(), skip_special_tokens = False),
        "top5" : {"ids" : top5.indices.tolist(), "log_probs" : top5.values.tolist()},
        "steps" : log.rows,
        "calls" : trace.calls, # every gate call of the generation.
        "counts" : {k : v - n0[k] for k, v in store.n.items()}, # the prompt's counters
        "read" : store.read_s - read0,
        "wait" : store.wait_s - wait0,
        "check" : None
    }
    
    decode = log.rows[1:] # steps after the prefill
    
    if decode:
        print(f"  prefill {seconds[0]:.1f} s | {len(decode)} decode steps: {sum(r['seconds'] for r in decode) / len(decode):.2f} s and"
              f" {sum(r['reads'] for r in decode) / len(decode):.1f} disk reads per token | text {record['text']!r}")
        print(f"  per decode token: {map_line(decode)}")
        print(f"  the step logger itself: {log.own_s / len(log.rows) * 1e3:.2f} ms per step, after each step's clock, so in no step's time")


    if check:
        c = frozen_check(model, trace, prompt, tokens, scores)
        record["check"] = c
        print(f"  check, one prefill over prompt + new tokens, cache frozen: same tokens {sum(c['same'])}/{len(c['same'])} |"
              f" largest score difference {max(c['diff']):.3g} | smallest top-2 margin {min(c['margin']):.3g}")
    return record

def test_prompts(index : str = INDEX, folder : Path = ROUTES) -> dict[str, list[int]]: # test split of Q&A text of each kind [before the answer token starts.]
    prompts, kinds = {}, set()
    for t in json.load(open(index))["texts"]:
        if t["split"] == "test" and t["shape"] == "qa" and t["kind"] not in kinds:
            kinds.add(t["kind"])
            begin = next(b for role, b, _ in t["spans"] if role == "assistant") # answer's first token
            prompts[t["id"]] = torch.load(folder / f"{t['id']}.pt")["ids"][: begin].tolist() # upto user's turn, ending at '<｜Assistant｜></think>'
    return prompts

def first_difference(ours : list, theirs : list) -> int | None: # the first index where two lists differ, None if they are equal
    for k, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            return k
    return None if len(ours) == len(theirs) else min(len(ours), len(theirs))


def compare(record : dict, run : dict, n_layers : int) -> dict: # our record against the fixed_home run record.
    theirs = [c for c in run["calls"] if c["layer"] < n_layers] # gate calls at layers in run order.
    k = first_difference(record["calls"], theirs)
    
    return {
        "tokens" : record["new_tokens"] == run["new_tokens"],
        "calls" : k is None,
        "first differing call" : None if k is None else {"step" : k // n_layers, "layer" : k % n_layers},
        "top5" : record["top5"] == run["top5"],
        "check" : record["check"] == run["check"]
    }

def save(path : str, out : dict):
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)

def decode_stats(records : list[dict]) -> str: # prompt decode steps: time and disk reads per token and where the picks were.
    steps = [s for r in records for s in r["steps"][1:]] # as step 0 is the prefill
    
    if not steps:
        return "no decode steps"
    
    n, total = len(steps), lambda k : sum(s["counts"][k] for s in steps)
    picks = sum(total(k) for k in ("t1", "t2", "t3", "t4", "miss")) # 6 per layer per token
    share = lambda k : f"{total(k) / picks:.0%}"
    return (f"{n:,} tokens | {sum(s['seconds'] for s in steps) / n:.2f} s and {sum(s['reads'] for s in steps) / n:.1f} disk reads per token |"
            f" picks GPU {share('t1')} + {share('t2')}, RAM {share('t3')} + {share('t4')}, disk {share('miss')}")

def run_prompts(
    model : Transformer, 
    args : ModelArgs, 
    tokenizer : ChatTokenizer, 
    trace : CacheTrace, 
    log : StepLog, 
    plan : dict,
    n_layers : int, 
    path : str, 
    n_fixed : int = N_FIXED, 
    n_test : int = N_TEST
) -> dict:
    runs = json.load(open(FIXED_HOME))["prompts"]
    out = {"plan" : {k : v for k, v in plan.items() if k not in ("rate", "position uses")}, "prompts" : {}, "same as fixed_home" : {}}
    todo = [(name, run["prompt"], n_fixed, True) for name, run in runs.items()] + [(name, ids, n_test, False) for name, ids in test_prompts().items()]
    
    for name, ids, n_new, check in todo:
        print(f"prompt {name!r}, {len(ids)} tokens:")
        try:
            out["prompts"][name] = r = run_prompt(model, args, tokenizer, trace, log, ids, n_new, check)
        except Exception:
            traceback.print_exc()
            continue
        if name in runs:
            out["same as fixed_home"][name] = same = compare(r, runs[name], n_layers)
            k = same["first differing call"]
            print(f"  same as the fixed_home run: tokens {same['tokens']} | gate calls {same['calls']}"
                  + ("" if k is None else f" (first differing: step {k['step']}, layer {k['layer']})")
                  + f" | first-token top-5 {same['top5']} | check {same['check']}")
        save(path, out)
    
    print("decode steps only, the prefill left out:")
    for name, r in out["prompts"].items():
        print(f"  {name:26s} {decode_stats([r])}")
    print(f"  {'the test prompts':26s} {decode_stats([r for name, r in out['prompts'].items() if name not in runs])}")
    
    return out

def profile_tokens(model : Transformer, args : ModelArgs, trace : CacheTrace, ids : list[int], n_prof : int = 3, path : str | None = None) -> dict:
    # n_prof decode tokens under torch.profiler, after the prefill and one more step, where each token time goes on the cpu and the gpu.
    store, trace.calls = trace.store, []
    state = SequenceState(args, 1, model.embed.weight.dtype, device = "cuda")
    x = model(torch.tensor([ids], device = "cuda"), state).argmax(dim = -1, keepdim = True) # [1, 1] first prefill token.
    x = model(x, state).argmax(dim = -1, keepdim = True) # one decode step outside of profile, so no first-call cost gets in.
    
    n0, read0, seconds, tokens = dict(store.n), store.read_s, [], []
    
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], acc_events = True) as prof:
        # profiling events across multiple profiling cycles.
        for _ in range(n_prof):
            t = clock()
            x = model(x, state).argmax(dim = -1, keepdim = True) # [1, 1]
            seconds.append(clock() - t)
            tokens.append(x.item())
    
    if path:
        prof.export_chrome_trace(path) # every cpu op and gpu kernel on one timeline, for ui.perfetto.dev
    
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation] # kernels, copies and memsets on one stream, without the GPU spans of our labels
    kernels = {} # kernel name -> [gpu_seconds, count]
    
    for e in gpu:
        k = kernels.setdefault(e.name, [0.0, 0])
        k[0] += e.time_range.elapsed_us() / 1e6
        k[1] += 1
    
    ops = prof.key_averages() # one row per cpu op name, its self time leaves out the ops it called.
    waits = [a for a in ops if a.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")]
    n = n_prof
    t, read, wait = sum(seconds) / n, (store.read_s - read0) / n, sum(a.self_cpu_time_total for a in waits) / 1e6 / n
    
    result = {
        "tokens" : tokens, 
        "seconds" : seconds,
        "per token" : {
            "seconds" : t, 
            "disk read" : read, 
            "cpu waiting for the gpu" : wait, 
            "rest" : t - read - wait,
            "gpu busy" : sum(s for s, _ in kernels.values()) / n, 
            "gpu kernels" : len(gpu) / n,
            "waits" : sum(a.count for a in waits) / n,
            "disk reads" : sum(store.n[k] - n0[k] for k in ("pre kept", "miss", "disk")) / n
        },
              
        "top gpu" : [
            {"name" : name, "seconds" : s / n, "count" : c / n} 
            for name, (s, c) in sorted(kernels.items(), key = lambda kv : -kv[1][0])[:10]
        ],
              
        "top cpu" : [
            {"name" : a.key, "seconds" : a.self_cpu_time_total / 1e6 / n, "count" : a.count / n} 
            for a in sorted(ops, key = lambda a : -a.self_cpu_time_total)[:10]
        ]
    }
    
    p = result["per token"]
    
    print(f"profile, {n} decode tokens: per token {t:.3f} s = disk read {read:.3f} s ({p['disk reads']:.1f} experts)"
          f" + CPU waiting for the GPU {wait:.3f} s ({p['waits']:.0f} waits) + the rest {p['rest']:.3f} s (Python, and launching {p['gpu kernels']:,.0f} kernels)"
          f" | GPU busy {p['gpu busy']:.3f} s ({p['gpu busy'] / t:.0%})")
    
    for title, rows in (("the GPU's biggest kernels", result["top gpu"]), ("the CPU's biggest own times", result["top cpu"])):
        print(f"  {title}, per token: ms | count | name")
        for r in rows:
            print(f"    {r['seconds'] * 1e3:8.2f} | {r['count']:6.0f} | {r['name'][:100]}")
    
    return result
    
        
    

if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else args.n_layers
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else None # most cache places per layer: 0 = run S, fixed experts only; None = as many as the disk allows
    dry = n_layers < args.n_layers
    ckpt = Checkpoint(FOLDER)
    
    ckpt.entries.update(saved_entries(ckpt.folder))
    Path(FOLDER).mkdir(parents = True, exist_ok = True)
    box = box_sizes(ckpt, args.n_layers) # for all 43 layers, in the dry run too.
    
    print(f"box, sized for {args.n_layers} layers: GPU free {box['gpu_free'] / 1e9:.2f} GB, static weights {box['static'] / 1e9:.2f} GB -> {box['n_gpu']}"
          f" experts per layer | RAM available {box['ram_free'] / 2**30:.1f} GiB -> {box['n_ram']} | disk free {box['disk_free'] / 1e9:.0f} GB -> {box['n_disk']}"
          f" | the load's disk peak {box['load_peak'] / 1e9:.0f} GB")
    
    
    assert box["load_peak"] <= box["disk_free"], "the downloads and the disk-only experts won't fit on the local disk during the load"
    
    t = time.perf_counter()
    n_disk = box["n_disk"] if cap is None else min(box["n_disk"], E - box["n_gpu"] - box["n_ram"] + cap) # disk-only experts + at most cap cache copies
    plan = make_plan(load_texts(), box["n_gpu"], box["n_ram"], n_disk)
    
    print(f"plan in {time.perf_counter() - t:.0f} s: fixed experts by the {plan['order']!r} order | disk reads per token on the learn answers:"
          + "".join(f" {name} {reads:.1f}" for name, reads in plan["learn_reads"].items()))
    
    t = time.perf_counter()
    model, store, fp0 = load_cache(args, ckpt, plan, n_layers)
    load_s = time.perf_counter() - t
    blocks = dict(enumerate(model.layers))
    
    trace = CacheTrace(blocks, store)
    watches = {l : MoEWatch(b.ffn, store) for l, b in blocks.items()}
    tokenizer = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    
    log = StepLog(watches, trace, tokenizer, EVERY)
    sky = json.load(open(FIXED_HOME))["prompts"]["sky"]["prompt"]
    
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = "" if cap is None else f"_C{cap}" # S's files say so: ..._C0.json
    path = f"/tmp/cache_generation_dry{tag}.json" if dry else f"results/cache_generation_{os.uname().nodename}_{stamp}{tag}.json"
    with torch.inference_mode():
        out = run_prompts(model, args, tokenizer, trace, log, plan, n_layers, path, n_test = 10 if dry else N_TEST)
        out["box"], out["load seconds"] = box, load_s
        out["end check"] = store.check_end(fp0) 
        print(f"end check: {out['end check']}")
        save(path, out) # before the profile so if the profile fails, nothing is lost.
        try:
            out["profile"] = p = profile_tokens(model, args, trace, sky, 3, path.removesuffix(".json") + "_trace.json.gz")
            print(f"the profiled tokens are sky's tokens 2-4: {p['tokens'] == out['prompts']['sky']['new_tokens'][2:5]}")
        except Exception: # the profile must not cost the results.
            traceback.print_exc()
    
    save(path, out)
    same = [name for name, s in out["same as fixed_home"].items() if all(s[k] for k in ("tokens", "calls", "top5", "check"))]
    print(f"{len(out['prompts'])} prompts done | the same as the fixed_home run in tokens, gate calls, top-5 and check: {same} | saved -> {path}")
    store.close()
    
    
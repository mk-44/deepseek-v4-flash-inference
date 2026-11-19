import json
import os
import sys
import time
import traceback
from pathlib import Path
import torch
import expert_store
from arch import ChatTokenizer, Checkpoint, ModelArgs
from cache_store import FIXED_HOME, FOLDER, box_sizes, load_cache
from cache_generation import EVERY, N_FIXED, CacheTrace, StepLog, compare, decode_stats, map_line, profile_tokens, run_prompt, save
from full_store import saved_entries
from store_moe import MoEWatch
from expert_plan import load_texts, make_plan 

DEFAULTS = {
    "old_timer" : False, # True : the MoE timer that waits for the gpu at both ends of each MoE. False : gpu timestamps, nothing waits.
    "compiled_expand" : True, # True : fp4_decode through torch.compile. False : the plain fp4_decode.
}

ARM_SETS = {
    "timer_and_compiled_expand" : {
        "about" : "old timer, event timer, compiled expand, event timer again. the arms of results/arms_ng0pion3rd_20261118T205650Z.json, profiles now at the end",
        "arms" : [("old timer", {"old_timer" : True, "compiled_expand" : False}), ("event timer", {"compiled_expand" : False}),
                  ("event timer and compiled expand", {}), ("event timer, again", {"compiled_expand" : False})],
        "after" : [],
    },
    "profiler_slowdown" : {
        "about" : "the baseline twice, both profiled, then the baseline once more: does a profile leave every later CUDA call slower?",
        "arms" : [("baseline", {}), ("baseline, again", {})],
        "after" : [("baseline, after the profiles", {})],
    }
}

def apply(switches : dict) -> dict: # sets one arm's switches, returns all of them.
    s = {**DEFAULTS, **switches}
    MoEWatch.sync, expert_store.COMPILED_EXPAND = s["old_timer"], s["compiled_expand"]
    return s


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in ARM_SETS:
        sys.exit("usage: python engine/arms.py <layers> <arm set>. The arm sets:\n" + "\n".join(f"  {k}: {v['about']}" for k, v in ARM_SETS.items()))
    
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    
    n_layers, arm_set = int(sys.argv[1]), sys.argv[2]
    arms, after = ARM_SETS[arm_set]["arms"], ARM_SETS[arm_set]["after"]
    print(f"arm set {arm_set}: {ARM_SETS[arm_set]['about']}")

    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder))
    
    Path(FOLDER).mkdir(parents = True, exist_ok = True)
    box = box_sizes(ckpt, args.n_layers) # sized for all 43 layers even in the dry run too.
    assert box["load_peak"] <= box["disk_free"], "the downloads and the disk-only experts won't fit on the local disk during the load"
    plan = make_plan(load_texts(), box["n_gpu"], box["n_ram"], box["n_disk"])
    print(f"plan: {box['n_gpu']} GPU + {box['n_ram']} RAM places per layer -> C = {plan['n_cache']}, {plan['n_fixed']} fixed experts per layer")
    
    expert_store.COMPILED_EXPAND = False # the load runs the plain expand, so a failed compile can cost its arms only and never the load.
    model, store, fp0 = load_cache(args, ckpt, plan, n_layers)
    blocks = dict(enumerate(model.layers))
    trace = CacheTrace(blocks, store)
    watches = {l : MoEWatch(b.ffn, store) for l, b in blocks.items()}
    tokenizer = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    log = StepLog(watches, trace, tokenizer, EVERY)
    runs = json.load(open(FIXED_HOME))["prompts"] # the 4 prompts and their fixed_home records.
    
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = f"/tmp/arms_dry_{arm_set}.json" if n_layers < args.n_layers else f"results/arms_{arm_set}_{os.uname().nodename}_{stamp}.json"
    out = {
        "box" : box, 
        "plan" : {k : plan[k] for k in ("n_gpu", "n_ram", "n_disk", "n_cache", "n_fixed", "order")},
        "arm set" : arm_set, 
        "about" : ARM_SETS[arm_set]["about"], 
        "arms" : {}
    }
    
    with torch.inference_mode():
        apply(DEFAULTS)
        t = time.perf_counter()
        try: # compile both matrix shapes once, so no step pays for it.
            store.fetch(0, next(e for e in range(256) if store.homes[0, e].tier <= 2))
            compiled_ok = True
            print(f"the expand compiled in {time.perf_counter() - t:.1f} s")
        except Exception: # log it, the arms that need it are skipped, the others still run.
            traceback.print_exc()
            print("FAILED: the expand did not compile, so the arms that need it are skipped")
            compiled_ok, expert_store.COMPILED_EXPAND = False, False
    
        print("warm-up: '2+2' once, so first-call costs land in no arm")
        run_prompt(model, args, tokenizer, trace, log, runs["2+2"]["prompt"], 2, False)
        
        
        first = arms[0][0] # every arm is compared with this one
        jobs = [("timed", n, sw) for n, sw in arms] + [("profile", n, sw) for n, sw in arms] + [("timed", n, sw) for n, sw in after]
        for kind, name, switches in jobs:
            s = apply(switches)
            if s["compiled_expand"] and not compiled_ok:
                print(f"=== {kind} {name}: FAILED, skipped as the expand did not compile")
                out["arms"][name] = {"failed" : "the expand did not compile"}
                continue
            t = time.perf_counter()
            moved = store.restore(ckpt, plan, n_layers)
            same = store.fingerprints() == fp0
            print(f"=== {kind} {name} {s}: the cache is back to its state after the load, in {time.perf_counter() - t:.0f} s"
                  f" ({moved:,} experts re-read) | fast places hold the load's bytes: {same}")

            if kind == "profile":
                i = [n for n, _ in arms].index(name)
                try:
                    out["arms"][name]["profile"] = profile_tokens(model, args, trace, runs["sky"]["prompt"], 3, path.removesuffix(".json") + f"_arm{i}_trace.json.gz")
                except Exception: # the profile must not cost the results
                    traceback.print_exc()
                save(path, out)
                continue
            
            arm = out["arms"][name] = {
                "switches" : s, 
                "after the profiles" : name in dict(after), 
                "restored" : moved, 
                "load's bytes" : same,
                "prompts" : {}, 
                "same as fixed_home" : {}, 
                "same as the first arm" : {}
            }
            
            for p, run in runs.items():
                print(f"prompt {p!r}, {len(run['prompt'])} tokens:")
                try:
                    r = arm["prompts"][p] = run_prompt(model, args, tokenizer, trace, log, run["prompt"], N_FIXED, True)
                    c = arm["same as fixed_home"][p] = compare(r, run, n_layers)
                    
                    print(f"  same as the fixed_home run: tokens {c['tokens']} | gate calls {c['calls']} | first-token top-5 {c['top5']} | check {c['check']}")
                    f = out["arms"][first].get("prompts", {}).get(p)
                    
                    if name != first and f:
                        c = arm["same as the first arm"][p] = {**compare(r, f, n_layers), "reads" : [x["reads"] for x in r["steps"]] == [x["reads"] for x in f["steps"]]}
                        print(f"  same as {first}: tokens {c['tokens']} | gate calls {c['calls']} | top-5 {c['top5']} | check {c['check']} | reads per step {c['reads']}")
                
                except Exception: # one prompt failing must not cost the other arms.
                    traceback.print_exc()

            decode = [x for r in arm["prompts"].values() for x in r["steps"][1:]]
            print(f"  {name}: {decode_stats(list(arm['prompts'].values()))}")
            
            if decode:
                print(f"{name}, per decode token: {map_line(decode)}")
                t, r, n = (sum(x[k] for x in decode) / len(decode) for k in ("seconds", "read", "reads"))
                print(f"{name}, per decode token: {t:.3f} s = reading the disk {r:.3f} s ({n:.1f} reads, {r / n * 1e3 if n else 0:.1f} ms each) + the rest {t - r:.3f} s")
            
            save(path, out)
        
        tokens = {n : a["profile"]["tokens"] for n, a in out["arms"].items() if "profile" in a}
        print(f"profiled tokens {tokens} | the same in every arm: {len({str(v) for v in tokens.values()}) <= 1}")

        out["end check"] = store.check_end(fp0)
        print(f"end check: {out['end check']}")
    
    save(path, out)
    
    for name, arm in out["arms"].items():
        good = [p for p, s in arm.get("same as fixed_home", {}).items() if all(s[k] for k in ("tokens", "calls", "top5", "check"))]
        print(f"{name}: the same as the fixed_home run in tokens, gate calls, top-5 and check: {good}")
        
        if arm.get("same as the first arm"):
            good = [p for p, s in arm["same as the first arm"].items() if all(s[k] for k in ("tokens", "calls", "top5", "check", "reads"))]
            print(f"  the same as {first} in tokens, gate calls, top-5, check and reads per step: {good}")
    
    print("the rest per decode token, s: " + " | ".join(runs))
    
    for name, arm in out["arms"].items():
        rest = [[x["seconds"] - x["read"] for x in arm["prompts"][p]["steps"][1:]] if p in arm.get("prompts", {}) else [] for p in runs]
        print(f"  {name:32s} " + " | ".join(f"{sum(v) / len(v):.3f}" if v else "  -  " for v in rest))
    
    print(f"saved -> {path}")
    store.close()
            





if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else args.n_layers
    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder))
    Path(FOLDER).mkdir(parents = True, exist_ok = True)
    box = box_sizes(ckpt, args.n_layers) # sized for all 43 layers, in the dry run too
    assert box["load_peak"] <= box["disk_free"], "the downloads and the disk-only experts won't fit on the local disk during the load"
    plan = make_plan(load_texts(), box["n_gpu"], box["n_ram"], box["n_disk"])
    print(f"plan: {box['n_gpu']} GPU + {box['n_ram']} RAM places per layer -> C = {plan['n_cache']}, {plan['n_fixed']} fixed experts per layer")

    model, store, fp0 = load_cache(args, ckpt, plan, n_layers)
    blocks = dict(enumerate(model.layers))
    trace = CacheTrace(blocks, store)
    watches = {l : MoEWatch(b.ffn, store) for l, b in blocks.items()}
    tokenizer = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    log = StepLog(watches, trace, tokenizer, EVERY)
    runs = json.load(open(FIXED_HOME))["prompts"] # the 4 prompts and their fixed_home records

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = "/tmp/arms_dry.json" if n_layers < args.n_layers else f"results/arms_{os.uname().nodename}_{stamp}.json"
    out = {"box" : box, "plan" : {k : plan[k] for k in ("n_gpu", "n_ram", "n_disk", "n_cache", "n_fixed", "order")}, "arms" : {}}

    with torch.inference_mode():
        print("warm-up: '2+2' once, so first-call costs land in no arm")
        run_prompt(model, args, tokenizer, trace, log, runs["2+2"]["prompt"], 2, False)

        for i, (name, sync, compiled) in enumerate(ARMS):
            t = time.perf_counter()
            moved = store.restore(ckpt, plan, n_layers)
            same = store.fingerprints() == fp0
            print(f"=== arm {i}, {name}: the cache is back to its state after the load, in {time.perf_counter() - t:.0f} s"
                  f" ({moved:,} experts re-read) | fast places hold the load's bytes: {same}")
            MoEWatch.sync, expert_store.COMPILED_EXPAND = sync, compiled

            if compiled: # compile both matrix shapes now, so no step pays for it
                t = time.perf_counter()
                try:
                    store.fetch(0, next(e for e in range(256) if store.homes[0, e].tier <= 2))
                    print(f"  the expand compiled in {time.perf_counter() - t:.1f} s")
                except Exception: # log it and skip this arm: the earlier arms are already saved
                    traceback.print_exc()
                    print("  FAILED: the expand did not compile, so this arm is skipped")
                    out["arms"][name] = {"failed" : "the expand did not compile"}
                    expert_store.COMPILED_EXPAND = False
                    continue

            arm = out["arms"][name] = {"restored" : moved, "load's bytes" : same, "prompts" : {}, "same as fixed_home" : {}}
            for p, run in runs.items():
                print(f"prompt {p!r}, {len(run['prompt'])} tokens:")
                try:
                    r = arm["prompts"][p] = run_prompt(model, args, tokenizer, trace, log, run["prompt"], N_FIXED, True)
                    s = arm["same as fixed_home"][p] = compare(r, run, n_layers)
                    print(f"  same as the fixed_home run: tokens {s['tokens']} | gate calls {s['calls']} | first-token top-5 {s['top5']} | check {s['check']}")
                except Exception: # one prompt failing must not cost the other arms
                    traceback.print_exc()

            decode = [s for r in arm["prompts"].values() for s in r["steps"][1:]]
            print(f"  {name}: {decode_stats(list(arm['prompts'].values()))}")
            if decode:
                print(f"  {name}, per decode token: {map_line(decode)}")
                t, r, n = (sum(s[k] for s in decode) / len(decode) for k in ("seconds", "read", "reads"))
                print(f"  {name}, per decode token: {t:.3f} s = reading the disk {r:.3f} s ({n:.1f} reads, {r / n * 1e3 if n else 0:.1f} ms each) + the rest {t - r:.3f} s")
            try:
                arm["profile"] = profile_tokens(model, args, trace, runs["sky"]["prompt"], 3, path.removesuffix(".json") + f"_arm{i}_trace.json.gz")
            except Exception: # the profile must not cost the results
                traceback.print_exc()
            save(path, out)

        out["end check"] = store.check_end(fp0)
        print(f"end check: {out['end check']}")

    save(path, out)
    for name, arm in out["arms"].items():
        good = [p for p, s in arm.get("same as fixed_home", {}).items() if all(s[k] for k in ("tokens", "calls", "top5", "check"))]
        print(f"{name}: the same as the fixed_home run in tokens, gate calls, top-5 and check: {good}")
    print(f"saved -> {path}")
    store.close()

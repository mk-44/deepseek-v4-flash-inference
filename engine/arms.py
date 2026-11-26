import json
import os
import sys
import time
import traceback
from pathlib import Path
import torch
import expert_store
from arch import ChatTokenizer, Checkpoint, ModelArgs
from cache_store import FIXED_HOME, FOLDER, CacheStore, box_sizes, load_cache
from cache_generation import EVERY, N_FIXED, CacheTrace, StepLog, compare, decode_stats, map_line, profile_tokens, run_prompt, save
from full_store import saved_entries
from recordings import RECORDINGS, make_recordings, use_recordings
from moe_recordings import MOE_RECORDINGS, MoERecording, make_moe_recordings
from store_moe import MoEWatch
from expert_plan import load_texts, make_plan 

LATE_COPY_CYCLES = 20_000_000 
# gpu clock cycles, about 10 to 14 ms at 2 to 1.4 GHz. longer than our thread takes to reach any expert of a layer after its prepare - for test.

DEFAULTS = {
    # True : the MoE timer that waits for the gpu at both ends of each MoE. False : gpu timestamps, nothing waits.
    "old_timer" : False, 
    
    # True : fp4_decode through torch.compile. False : the plain fp4_decode.
    "compiled_expand" : True,
    
    # True : gpu memory read at both ends of each MoE, the old way. False : the step logger reads the step's peak once, outside the step's clock.
    "moe_memory_readings" : False,
    
    # True : every 1 token step replays one recording per hc_pre call site. False : the plain hc_pre, about 140 GPU jobs per call.
    "hc_pre_recordings" : False, 
    
    # True : the swap copies run on their own stream, only the expand of a moved expert waits for them on the gpu. 
    # False : on the default stream, our thread waits for them after each prepare.
    "swap_copy_stream" : False,
    
    # True : the copy stream sleeps LATE_COPY_CYCLES before each swap, so every copy lands late on purpose. False : no sleep.
    "late_swap_copies" : False,
    
    # True : every 1 token step replays one recording per layer for its 6 experts and shared expert. it holds the compiled expand. 
    # False : the plain MoE loop, about 20 GPU jobs per expert.
    "moe_recordings" : False
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
    },
    "logging_cost" : {
        "about" : "the baseline, then the old gpu memory reads in every MoE, then the baseline again: what those reads cost",
        "arms" : [("baseline", {}), ("GPU memory read in every MoE", {"moe_memory_readings" : True}), ("baseline, again", {})],
        "after" : [],
        "profile" : False, # no profiles: this set only compares timings
    },
    "hc_pre_recordings" : {
        "about" : "the baseline, then hc_pre replayed from its recordings in every 1 token step, then the baseline again: what the recordings save",
        "arms" : [("baseline", {}), ("hc_pre recordings", {"hc_pre_recordings" : True}), ("baseline, again", {})],
        "after" : [],
        "profile" : True, # profiles after all the timed arms: they count each arm's kernel launches
    },
    
    "swap_copy_stream" : {
        "about" : "hc_pre recordings on in every arm. the baseline, then the swap copies on their own stream, then the baseline again: what the copy stream saves. after the profiles, the copy stream with every copy late on purpose: its tokens must still match the baseline",
        "arms" : [("baseline", {"hc_pre_recordings" : True}), ("swap copy stream", {"hc_pre_recordings" : True, "swap_copy_stream" : True}),
                  ("baseline, again", {"hc_pre_recordings" : True})],
        "after" : [("swap copy stream, every copy late", {"hc_pre_recordings" : True, "swap_copy_stream" : True, "late_swap_copies" : True})],
        "profile" : True, # profiles after all the timed arms: they count each arm's copies and kernel launches
    },

    "expand_hc_pre_copy_stream" : {
        "about" : "the baseline, then the plain expand, then hc_pre recordings, then hc_pre recordings and the swap copy stream, then the baseline again: what the compiled expand, the recordings and the copy stream each save at full size",
        "arms" : [("baseline", {}), ("plain expand", {"compiled_expand" : False}), ("hc_pre recordings", {"hc_pre_recordings" : True}),
                  ("hc_pre recordings and swap copy stream", {"hc_pre_recordings" : True, "swap_copy_stream" : True}), ("baseline, again", {})],
        "after" : [],
        "profile" : True, # profiles after all the timed arms: they count each arm's kernel launches
    },
    
    "moe_recordings" : {
        "about" : "hc_pre recordings and the swap copy stream on in every arm. the baseline, then each layer's MoE replayed from its recording in every 1 token step, then the baseline again: what the MoE recordings save. after the profiles, the MoE recordings with every copy late on purpose: its tokens must still match the baseline",
        "arms" : [("baseline", {"hc_pre_recordings" : True, "swap_copy_stream" : True}),
                  ("MoE recordings", {"hc_pre_recordings" : True, "swap_copy_stream" : True, "moe_recordings" : True}),
                  ("baseline, again", {"hc_pre_recordings" : True, "swap_copy_stream" : True})],
        "after" : [("MoE recordings, every copy late", {"hc_pre_recordings" : True, "swap_copy_stream" : True, "moe_recordings" : True, "late_swap_copies" : True})],
        "profile" : True, # profiles after all the timed arms: they count each arm's kernel launches
    }
}

def apply(switches : dict) -> dict: # sets one arm's switches, returns all of them.
    s = {**DEFAULTS, **switches}
    MoEWatch.sync, expert_store.COMPILED_EXPAND, MoEWatch.memory = s["old_timer"], s["compiled_expand"], s["moe_memory_readings"]
    use_recordings(s["hc_pre_recordings"]) # Block.forward finds the recorded hc_pre or the plain one from its next call
    CacheStore.copy_stream, CacheStore.copy_sleep = s["swap_copy_stream"], LATE_COPY_CYCLES if s["late_swap_copies"] else 0 # the next swap reads them
    MoERecording.on = s["moe_recordings"] # the next MoE call replays its recording or runs the plain way
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

        torch.cuda.empty_cache() # so the reserved memory read below counts only what the recordings hold
        reserved, t = torch.cuda.memory_reserved(), time.perf_counter()
        try: # one recording per hc_pre call site, made once before any prompt runs. a failure costs only the arms that use them
            make_recordings(model.layers, args, model.embed.weight.dtype)
            recorded_ok = True
            print(f"hc_pre recordings: {len(RECORDINGS)} made in {time.perf_counter() - t:.1f} s | GPU memory they hold {(torch.cuda.memory_reserved() - reserved) / 2**20:.0f} MiB")
        except Exception: # log it, the arms that need them are skipped, the others still run.
            traceback.print_exc()
            print("FAILED: the hc_pre recordings were not made, so the arms that need them are skipped")
            recorded_ok = False
            
        moe_recorded_ok = False
        
        if compiled_ok: # the MoE recordings hold the compiled expand, so they need it
            torch.cuda.empty_cache() # so the reserved memory read below counts only what the MoE recordings hold
            reserved, t = torch.cuda.memory_reserved(), time.perf_counter()
            
            try: # one recording per layer's MoE, made once before any prompt runs. a failure costs only the arms that use them
                make_moe_recordings(model.layers, store, model.embed.weight.dtype)
                moe_recorded_ok = True
                print(f"MoE recordings: {len(MOE_RECORDINGS)} made in {time.perf_counter() - t:.1f} s "
                      f"| GPU memory they hold {(torch.cuda.memory_reserved() - reserved) / 2**20:.0f} MiB")
                
            except Exception: # log it, the arms that need them are skipped, the others still run.
                traceback.print_exc()
                print("FAILED: the MoE recordings were not made, so the arms that need them are skipped")

        print("warm-up: '2+2' once, so first-call costs land in no arm")
        run_prompt(model, args, tokenizer, trace, log, runs["2+2"]["prompt"], 2, False)
        
        
        first = arms[0][0] # every arm is compared with this one
        profiles = [("profile", n, sw) for n, sw in arms] if ARM_SETS[arm_set].get("profile", True) else [] # a set can skip its profiles with "profile" : False
        jobs = [("timed", n, sw) for n, sw in arms] + profiles + [("timed", n, sw) for n, sw in after]
        for kind, name, switches in jobs:
            s = apply(switches)
            if s["compiled_expand"] and not compiled_ok:
                print(f"=== {kind} {name}: FAILED, skipped as the expand did not compile")
                out["arms"][name] = {"failed" : "the expand did not compile"}
                continue
                
            if s["hc_pre_recordings"] and not recorded_ok: # without them the arm would run the plain hc_pre under the recordings label
                print(f"=== {kind} {name}: FAILED, skipped as the hc_pre recordings were not made")
                out["arms"][name] = {"failed" : "the hc_pre recordings were not made"}
                continue
            
            if s["moe_recordings"] and not moe_recorded_ok: # without them the arm would run the plain MoE under the recordings label
                print(f"=== {kind} {name}: FAILED, skipped as the MoE recordings were not made")
                out["arms"][name] = {"failed" : "the MoE recordings were not made"}
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
        if tokens: # a set without profiles has none to compare
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
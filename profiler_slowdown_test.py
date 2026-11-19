# Does one torch.profiler session leave every later CUDA call slower? It times launches, not math.
import gc, statistics, sys, time
sys.path[:0] = ["engine"]
import torch
from arch import hc_pre

def per_call(fn, n : int, rounds : int = 5) -> tuple[float, float]: # microseconds per call, wall and our thread's cpu, the middle of 5 rounds
    walls, cpus = [], []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t, c = time.perf_counter(), time.thread_time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t) / n * 1e6)
        cpus.append((time.thread_time() - c) / n * 1e6)
    return statistics.median(walls), statistics.median(cpus)

def label(): # an empty profiler label, the cost the moe and expand labels have when no profiler runs
    with torch.profiler.record_function("label"):
        pass

def profile_session(path : str): # set up like profile_tokens in engine/cache_generation.py: cpu and cuda, acc_events, a trace export, key_averages
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], acc_events = True) as prof:
        for _ in range(20):
            hc_pre(x, hc_fn, hc_scale, hc_base, 1e-6)
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    prof.key_averages()

if __name__ == "__main__":
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name()}")
    with torch.inference_mode():
        small_gpu, small_cpu = torch.zeros(16, device = "cuda"), torch.zeros(16)
        x = torch.randn(1, 1, 4, 4096, device = "cuda", dtype = torch.bfloat16) # one hc_pre input: 4 copies of the hidden state, random as only launches are timed
        hc_fn = torch.randn(24, 4 * 4096, device = "cuda") * 0.02 # [24, 16384], the real shapes
        hc_scale, hc_base = torch.rand(3, device = "cuda") + 0.5, torch.randn(24, device = "cuda") * 0.5
        tests = {
            "tiny GPU op" : (lambda : small_gpu.add_(1.0), 20_000), # one launch each
            "tiny CPU op" : (lambda : small_cpu.add_(1.0), 20_000), # no CUDA call: the control
            "hc_pre call" : (lambda : hc_pre(x, hc_fn, hc_scale, hc_base, 1e-6), 500), # 140 launches each
            "empty label" : (label, 20_000),
        }
        for fn, _ in tests.values(): # first calls load kernels, so they land in no round
            for _ in range(100):
                fn()
        results = {}
        for k, phase in enumerate(("before any profiler", "after 1 profiler session", "after 2 profiler sessions")):
            if k > 0:
                profile_session(f"/tmp/profiler_slowdown_test_{k}.json")
                gc.collect()
            results[phase] = {name : per_call(fn, n) for name, (fn, n) in tests.items()}
            print(f"{phase}: " + " | ".join(f"{name} {w:.2f} us ({c:.2f} us cpu)" for name, (w, c) in results[phase].items()))
        b = results["before any profiler"]
        for phase in list(results)[1:]:
            a = results[phase]
            print(f"change {phase}, per call: " + " | ".join(f"{name} {a[name][0] - b[name][0]:+.2f} us ({a[name][0] / b[name][0] - 1:+.0%})" for name in tests))
        g, c = (results["after 1 profiler session"][name][0] - b[name][0] for name in ("tiny GPU op", "tiny CPU op"))
        print(f"verdict: {'a slowdown is left behind' if g > 1.0 else 'no slowdown'} | a tiny GPU op is {g:+.2f} us after one profiler session, the CPU op (the control) {c:+.2f} us")

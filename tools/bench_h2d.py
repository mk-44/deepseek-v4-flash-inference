"""Phase 3: memory and PCIe. How much pinned RAM can we get, and how fast does data move between disk, RAM and the GPU?
    python tools/bench_h2d.py           # about 2 minutes
    python tools/bench_h2d.py --quick   # about 1 minute
Results go to results/bench_h2d_<host>_<time>.json/.txt, plus a printed summary:
1. pinned RAM: pinned memory (RAM locked in place, so the GPU can copy from it directly) taken 1 GiB at a time until
   8 GiB of RAM is left: how many experts the RAM tier can hold. It runs in a child process, which frees it all on exit;
2. copy speed RAM -> GPU and GPU -> RAM, pinned vs pageable (ordinary) memory, from 1 MiB to 256 MiB;
3. RAM -> GPU copies while BF16 matmuls run (the engine overlaps the two), and the matmuls' slowdown;
4. the first copy after 0.1 s and 1 s idle (the PCIe link and the GPU clocks slow down when idle);
5. disk -> pinned RAM -> GPU for 1, 3 and 6 experts (one decode step's misses), median and p99;
6. memory bandwidth: CPU RAM with 1-8 threads, alone and during GPU copies, and GPU memory (VRAM).
For 5 it fetches one model file from Hugging Face onto the local disk and deletes it afterwards (or use --files)."""
import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_storage import ALIGN, EXPERT, LOCAL, ROOT, GiB, MiB, Files, fetch, meta, model_files, ms_stats, read, save


def gpu_timed(fn, reps):
    """Median wall time of fn() over reps runs, each ended by a GPU synchronize (after one warm-up run)."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def pinned_ramp(leave_gib, cap_gib):
    """Take pinned memory 1 GiB at a time until leave_gib of RAM is left (or cap_gib, or an error)."""
    blocks, times, error = [], [], None
    try:
        while len(blocks) < cap_gib and psutil.virtual_memory().available > (leave_gib + 1) * GiB:
            t = time.perf_counter()
            blocks.append(torch.empty(GiB, dtype=torch.uint8, pin_memory=True))
            times.append(time.perf_counter() - t)
    except RuntimeError as e:
        error = str(e).strip().splitlines()[0][:200]
    return dict(gib=len(blocks), experts=len(blocks) * GiB // EXPERT, leave_gib=leave_gib, cap_gib=cap_gib,
                seconds_per_gib=statistics.median(times) if times else None,
                available_after_gib=psutil.virtual_memory().available / GiB, error=error)


def copy_table(sizes, reps):
    rows = []
    for size in sizes:
        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        pageable = torch.empty(size, dtype=torch.uint8)
        gpu = torch.empty(size, dtype=torch.uint8, device='cuda')
        row = dict(size=size)
        for name, dst, src in [('h2d_pinned', gpu, pinned), ('h2d_pageable', gpu, pageable),
                               ('d2h_pinned', pinned, gpu), ('d2h_pageable', pageable, gpu)]:
            row[name] = size / gpu_timed(lambda: dst.copy_(src, non_blocking=True), reps) / 1e9
        rows.append(row)
    return rows


def overlap(reps, n_mm=40, n_copy=6, size=256 * MiB):
    """RAM -> GPU copies on a side stream while BF16 matmuls run on the default stream, vs each alone."""
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    src = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(size, dtype=torch.uint8, device='cuda')
    mm_alone = gpu_timed(lambda: [torch.mm(a, b) for _ in range(n_mm)], 3) / n_mm
    copy_alone = gpu_timed(lambda: dst.copy_(src, non_blocking=True), reps)
    side = torch.cuda.Stream()
    mm0, mm1, cp0, cp1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    torch.cuda.synchronize()
    mm0.record()
    for _ in range(n_mm):
        torch.mm(a, b)
    mm1.record()
    with torch.cuda.stream(side):
        cp0.record()
        for _ in range(n_copy):
            dst.copy_(src, non_blocking=True)
        cp1.record()
    torch.cuda.synchronize()
    return dict(copy_alone_gbps=size / copy_alone / 1e9, copy_during_gbps=n_copy * size / (cp0.elapsed_time(cp1) / 1e3) / 1e9,
                mm_alone_ms=mm_alone * 1e3, mm_during_ms=mm0.elapsed_time(mm1) / n_mm)


def after_idle(gaps, reps):
    """Time one expert's RAM -> GPU copy right after the GPU has been idle for `gap` seconds."""
    src = torch.empty(EXPERT, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(EXPERT, dtype=torch.uint8, device='cuda')
    dst.copy_(src)
    out = {}
    for gap in gaps:
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            time.sleep(gap)
            t = time.perf_counter()
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out[str(gap)] = dict(p50_ms=statistics.median(times) * 1e3, max_ms=max(times) * 1e3)
    return out


def disk_to_gpu(files, counts, trials, chunks=8):
    """n experts read from the local disk with O_DIRECT into pinned RAM (n x chunks reads at once), then copied to the
    GPU: the time until they are all on the GPU, and the reading part alone."""
    n_max = max(counts)
    pinned = torch.empty(n_max * EXPERT + ALIGN, dtype=torch.uint8, pin_memory=True)
    shift = (-pinned.data_ptr()) % ALIGN
    host = pinned[shift:shift + n_max * EXPERT]
    view = memoryview(host.numpy())
    gpu = torch.empty(n_max * EXPERT, dtype=torch.uint8, device='cuda')
    piece, rng, out = EXPERT // chunks, random.Random(5), {}
    with ThreadPoolExecutor(64) as pool:
        for n in counts:
            total, reading = [], []
            for _ in range(trials):
                jobs = []
                for e in range(n):
                    fd, offset = files.place(rng, EXPERT)
                    jobs += [(fd, view[e * EXPERT + c * piece:e * EXPERT + (c + 1) * piece], offset + c * piece)
                             for c in range(chunks)]
                t = time.perf_counter()
                for f in [pool.submit(read, *job) for job in jobs]:
                    f.result()
                t_read = time.perf_counter()
                gpu[:n * EXPERT].copy_(host[:n * EXPERT], non_blocking=True)
                torch.cuda.synchronize()
                total.append(time.perf_counter() - t)
                reading.append(t_read - t)
            out[str(n)] = dict(**ms_stats(total), read_p50_ms=statistics.median(reading) * 1e3)
    return out


def cpu_bandwidth(threads, nbytes, reps):
    """Copy nbytes in RAM with torch on t threads: GB/s counted as bytes read + bytes written (as STREAM does)."""
    a, b = torch.ones(nbytes, dtype=torch.uint8), torch.zeros(nbytes, dtype=torch.uint8)
    old, out = torch.get_num_threads(), {}
    for t in threads:
        torch.set_num_threads(t)
        b.copy_(a)
        times = []
        for _ in range(reps):
            s = time.perf_counter()
            b.copy_(a)
            times.append(time.perf_counter() - s)
        out[str(t)] = 2 * nbytes / statistics.median(times) / 1e9
    torch.set_num_threads(old)
    return out


def cpu_bandwidth_during_copies(nbytes, reps, size=256 * MiB):
    """The 8-thread RAM copy while another thread keeps copying RAM -> GPU (both use the memory bus)."""
    src = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(size, dtype=torch.uint8, device='cuda')
    stop, moved = threading.Event(), []

    def pump():
        stream, n, t0 = torch.cuda.Stream(), 0, time.perf_counter()
        with torch.cuda.stream(stream):
            while not stop.is_set():
                dst.copy_(src, non_blocking=True)
                stream.synchronize()
                n += 1
        moved.append(n * size / (time.perf_counter() - t0) / 1e9)

    th = threading.Thread(target=pump)
    th.start()
    time.sleep(0.3)
    try:
        bw = cpu_bandwidth([8], nbytes, reps)['8']
    finally:
        stop.set()
        th.join()
    return dict(cpu_gbps=bw, h2d_gbps_meanwhile=moved[0])


def vram_bandwidth(nbytes, reps):
    x = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
    y = torch.empty_like(x)
    return 2 * nbytes / gpu_timed(lambda: y.copy_(x), reps) / 1e9


def summary(res):
    m, lines = res['meta'], []
    lines.append(f'bench_h2d  host {m["host"]}  {m["time"]}  commit {m["commit"]}  {"quick" if res["quick"] else "full"} run')
    p = res.get('pinned')
    if p:
        lines.append(f'pinned RAM       {p.get("gib")} GiB in 1 GiB blocks ({p.get("seconds_per_gib") or 0:.2f} s each),'
                     f' {p.get("available_after_gib", 0):.1f} GiB of RAM left -> {p.get("experts")} expert slots'
                     + (f'  [stopped by: {p["error"]}]' if p.get('error') else ''))
    if res.get('copies'):
        lines.append('copies GB/s      size          RAM->GPU pinned   pageable   GPU->RAM pinned   pageable')
        for r in res['copies']:
            lines.append(f'                 {r["size"] / MiB:7.2f} MiB   {r["h2d_pinned"]:13.1f} {r["h2d_pageable"]:10.1f}'
                         f' {r["d2h_pinned"]:17.1f} {r["d2h_pageable"]:10.1f}')
    o = res.get('overlap')
    if o:
        lines.append(f'during matmuls   RAM->GPU {o["copy_during_gbps"]:.1f} GB/s (alone {o["copy_alone_gbps"]:.1f});'
                     f' one 4096^3 BF16 matmul {o["mm_during_ms"]:.2f} ms (alone {o["mm_alone_ms"]:.2f})')
    i = res.get('after_idle')
    if i:
        lines.append('after idle       one expert RAM->GPU, median (max): ' + ', '.join(
            f'{"warm" if float(g) == 0 else f"after {g} s idle"} {v["p50_ms"]:.2f} ms ({v["max_ms"]:.2f})' for g, v in i.items()))
    d = res.get('disk_to_gpu')
    if d:
        lines.append('disk -> GPU      experts: p50 / p99 ms (reading alone, p50): ' + ', '.join(
            f'{n}: {v["p50_ms"]:.1f} / {v["p99_ms"]:.1f} ({v["read_p50_ms"]:.1f})' for n, v in d.items()))
    if res.get('cpu_gbps'):
        c = res['cpu_gbps']
        line = 'CPU RAM GB/s     ' + ', '.join(f'{t} thread{"s" if t != "1" else ""} {v:.1f}' for t, v in c.items())
        if res.get('cpu_during_copies'):
            dc = res['cpu_during_copies']
            line += f'; 8 threads during RAM->GPU copies {dc["cpu_gbps"]:.1f} (copies meanwhile {dc["h2d_gbps_meanwhile"]:.1f})'
        lines.append(line)
    if res.get('vram_gbps'):
        lines.append(f'VRAM             {res["vram_gbps"]:.0f} GB/s (the A6000 spec is 768)')
    if res.get('error'):
        lines.append(f'ERROR            {res["error"]}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true', help='about 1 minute: fewer repetitions, at most 4 GiB pinned')
    ap.add_argument('--files', nargs='+', help='read these existing files for test 5 instead of fetching one')
    ap.add_argument('--out', default=str(ROOT / 'results'), help='where the results go (default: results/ in the repo)')
    ap.add_argument('--ramp-child', nargs=2, type=float, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.ramp_child:
        print(json.dumps(pinned_ramp(*a.ramp_child)))
        return
    q, res, files, work = a.quick, dict(meta=meta(), quick=a.quick), None, LOCAL / 'h2d'
    reps = 5 if q else 20
    try:
        if not torch.cuda.is_available():
            print('CUDA is not available: only the CPU memory-bandwidth test runs', flush=True)
            res['cpu_gbps'] = cpu_bandwidth([1, 2, 4, 8], GiB, 3)
            return
        print('pinned RAM ramp (child process) ...', flush=True)
        r = subprocess.run([sys.executable, __file__, '--ramp-child', '8', '4' if q else '64'], capture_output=True, text=True)
        try:
            res['pinned'] = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            res['pinned'] = dict(error=f'child exit {r.returncode}: {r.stderr.strip()[-300:]}')
        print(f'  {res["pinned"]}', flush=True)
        print('copy speeds ...', flush=True)
        res['copies'] = copy_table([MiB, EXPERT, 64 * MiB, 256 * MiB], reps)
        res['overlap'] = overlap(reps)
        print('first copy after idle ...', flush=True)
        res['after_idle'] = after_idle([0.0, 0.1, 1.0], 3 if q else 10)
        if a.files:
            paths = [Path(p) for p in a.files]
        else:
            print('fetching one model file from Hugging Face for disk -> GPU ...', flush=True)
            paths = fetch(model_files(3, 1), work)[0]
        files = Files(paths)
        res['disk_to_gpu_odirect'] = files.direct
        res['disk_to_gpu'] = disk_to_gpu(files, [1, 3, 6], 10 if q else 100)
        print('memory bandwidth ...', flush=True)
        res['cpu_gbps'] = cpu_bandwidth([1, 2, 4, 8], GiB if q else 2 * GiB, 3 if q else 5)
        res['cpu_during_copies'] = cpu_bandwidth_during_copies(GiB if q else 2 * GiB, 3 if q else 5)
        res['vram_gbps'] = vram_bandwidth(GiB, reps)
    except BaseException as e:
        res['error'] = f'{type(e).__name__}: {e}'
        if not isinstance(e, Exception):
            raise
    finally:
        if files is not None:
            files.close()
        shutil.rmtree(work, ignore_errors=True)
        text = summary(res)
        stem = save(res, text, 'bench_h2d', a.out)
        print('\n' + text + f'\nsaved {stem}.json and .txt', flush=True)


if __name__ == '__main__':
    main()

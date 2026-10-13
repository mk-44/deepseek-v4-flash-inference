"""Phase 3, DP1: how fast does the local disk read model data? Results go to results/bench_storage_<host>_<time>.json/.txt.
    python tools/bench_storage.py            # full run, about 30-40 minutes: start it inside tmux
    python tools/bench_storage.py --quick    # about 2 minutes: run this first to check that everything works
The model's master copy is Hugging Face (D6): every session fetches the files onto the local disk /, which is wiped at
every stop. So only / is measured, on real model files:
1. fetch: model files from Hugging Face onto / (the per-session fill speed);
2. O_DIRECT: does / accept reads that skip Linux's file cache, as the engine's reads will?
3. write: sequential O_DIRECT writes (how fast / takes in a fill);
4. sweep: random reads from 128 KiB to one expert (12.75 MiB) with 1-64 reads in flight: GB/s, median and p99 latency;
5. expert groups: 1, 2, 3, 4, 6 or 12 experts at random places, each read in 1-32 chunks, timed until every chunk has
   arrived (one decode step's reads); this also gives t(6 experts) / t(3 experts);
6. sustained: 15 minutes of expert reads at 32 in flight, in 10-second windows (does the disk slow down?);
7. a fit per number in flight, latency = a + bytes / b: the simulator's input.
Everything it fetched or wrote is deleted at the end, also after an error or Ctrl+C (--keep keeps the model files).
probe.py and bench_h2d.py reuse its helpers (fetch, Files, read, sweep_cell)."""
import argparse
import itertools
import json
import mmap
import os
import platform
import random
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = 'deepseek-ai/DeepSeek-V4-Flash-0731', '7872f01b1d1fe23eabc4c98b48bffcef5a386062'
EXPERT = 13_369_344                  # bytes of one routed expert: FP4 w1, w3, w2 and their E8M0 scales
ALIGN = 4096                         # O_DIRECT needs the buffer address, file offset and size in multiples of 4 KiB
MiB, GiB = 2**20, 2**30
LOCAL = Path(os.environ.get('V4_LOCAL', '/v4local'))   # the local disk: free, wiped at every stop
O_DIRECT = getattr(os, 'O_DIRECT', 0)                  # Linux only; elsewhere reads go through the file cache
COLD_BYTES = 76.3e9                  # the experts that fit in neither VRAM nor pinned RAM (plan section 2.4)


def model_files(first, count):
    return [f'model-{i:05d}-of-00048.safetensors' for i in range(first, first + count)]


def fetch(names, dest):
    """Download these files of the pinned checkpoint from Hugging Face into dest; returns (paths, bytes, seconds).
    The Hugging Face cache is kept on the local disk: the model is never stored on /storage (D6)."""
    if os.environ.get('HF_HOME', '').startswith('/storage'):
        os.environ['HF_HOME'] = str(LOCAL / 'hf')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
    from huggingface_hub import snapshot_download
    t = time.perf_counter()
    snapshot_download(REPO, revision=REVISION, allow_patterns=names, local_dir=dest, max_workers=8)
    seconds = time.perf_counter() - t
    paths = [Path(dest) / n for n in names]
    return paths, sum(p.stat().st_size for p in paths), seconds


def aligned_buffer(nbytes):
    """Page-aligned memory (an anonymous mmap), as O_DIRECT requires; returns a writable memoryview."""
    return memoryview(mmap.mmap(-1, nbytes))


def open_direct(path, write=False):
    """Open with O_DIRECT; returns (fd, direct). direct is False when the filesystem refuses it: reads may then come
    from Linux's file cache and make the disk look faster than it is."""
    flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if write else os.O_RDONLY
    if O_DIRECT:
        fd = None
        try:
            fd = os.open(path, flags | O_DIRECT, 0o644)
            probe = aligned_buffer(ALIGN)
            os.pwritev(fd, [probe], 0) if write else os.preadv(fd, [probe], 0)   # some filesystems fail only here
            return fd, True
        except OSError:
            if fd is not None:
                os.close(fd)
    return os.open(path, flags, 0o644), False


def read(fd, view, offset):
    n = os.preadv(fd, [view], offset)
    if n != len(view):
        raise OSError(f'short read: {n} of {len(view)} bytes at offset {offset}')


class Files:
    """Model files opened for reading, with random 4 KiB-aligned places inside them."""

    def __init__(self, paths):
        opened = [open_direct(p) for p in paths]
        self.fds = [fd for fd, _ in opened]
        self.sizes = [os.fstat(fd).st_size for fd in self.fds]
        self.direct = all(d for _, d in opened)

    def place(self, rng, nbytes):
        i = rng.randrange(len(self.fds))
        return self.fds[i], rng.randrange((self.sizes[i] - nbytes) // ALIGN) * ALIGN

    def close(self):
        for fd in self.fds:
            os.close(fd)


def ms_stats(seconds):
    ms = np.asarray(seconds) * 1e3
    return dict(p50_ms=float(np.percentile(ms, 50)), p99_ms=float(np.percentile(ms, 99)))


def sweep_cell(files, size, depth, seconds, bufs):
    """depth threads each read `size` bytes at random places for `seconds`: GB/s and latency (one read's time)."""
    lats = [[] for _ in range(depth)]
    deadline = time.perf_counter() + seconds

    def worker(k):
        rng, view, out = random.Random(1000 * depth + k), bufs[k][:size], lats[k]
        while time.perf_counter() < deadline:
            fd, offset = files.place(rng, size)
            t = time.perf_counter()
            read(fd, view, offset)
            out.append(time.perf_counter() - t)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(depth) as pool:
        list(pool.map(worker, range(depth)))
    reads = sum(len(x) for x in lats)
    return dict(size=size, depth=depth, reads=reads, gbps=reads * size / (time.perf_counter() - t0) / 1e9,
                **ms_stats(list(itertools.chain(*lats))))


def group_trials(files, n, chunks, trials, pool, buf, seed):
    """n experts at random places, each read in `chunks` equal pieces, all at once; time until every piece arrives."""
    rng, piece, times = random.Random(seed), EXPERT // chunks, []
    for _ in range(trials):
        jobs = []
        for e in range(n):
            fd, offset = files.place(rng, EXPERT)
            jobs += [(fd, buf[e * EXPERT + c * piece:e * EXPERT + (c + 1) * piece], offset + c * piece) for c in range(chunks)]
        t = time.perf_counter()
        for f in [pool.submit(read, *job) for job in jobs]:
            f.result()
        times.append(time.perf_counter() - t)
    return dict(experts=n, chunks=chunks, trials=trials, gbps=n * EXPERT / float(np.median(times)) / 1e9, **ms_stats(times))


def sustained(files, seconds, depth, window, bufs):
    """depth threads read whole experts for `seconds`; returns GB/s in consecutive windows of `window` seconds."""
    ends = [[] for _ in range(depth)]
    t0 = time.perf_counter()
    deadline = t0 + seconds

    def worker(k):
        rng, view, out = random.Random(77 + k), bufs[k][:EXPERT], ends[k]
        while time.perf_counter() < deadline:
            fd, offset = files.place(rng, EXPERT)
            read(fd, view, offset)
            out.append(time.perf_counter() - t0)

    with ThreadPoolExecutor(depth) as pool:
        list(pool.map(worker, range(depth)))
    counts, _ = np.histogram(list(itertools.chain(*ends)), bins=np.arange(0, seconds + window / 2, window))
    return dict(depth=depth, window_s=window, gbps=[float(c * EXPERT / window / 1e9) for c in counts])


def write_test(path, nbytes, depth=8, block=8 * MiB):
    """Sequential O_DIRECT writes of nbytes, `depth` writes in flight, then fsync; the file is deleted afterwards."""
    fd, direct = open_direct(path, write=True)
    data = [aligned_buffer(block) for _ in range(depth)]
    for view in data:
        view[:] = os.urandom(block)
    blocks, counter = nbytes // block, itertools.count()

    def worker(k):
        while (i := next(counter)) < blocks:
            if os.pwritev(fd, [data[k]], i * block) != block:
                raise OSError('short write')

    t = time.perf_counter()
    try:
        with ThreadPoolExecutor(depth) as pool:
            list(pool.map(worker, range(depth)))
        os.fsync(fd)
        seconds = time.perf_counter() - t
    finally:
        os.close(fd)
        Path(path).unlink(missing_ok=True)
    return dict(bytes=blocks * block, seconds=seconds, gbps=blocks * block / seconds / 1e9, direct=direct)


def fits(cells):
    """Per number in flight: least-squares line, median latency (ms) = a + bytes / b, over the read sizes."""
    out = {}
    for depth in sorted({c['depth'] for c in cells}):
        cs = [c for c in cells if c['depth'] == depth]
        if len(cs) >= 2:
            slope, a = np.polyfit([c['size'] for c in cs], [c['p50_ms'] for c in cs], 1)
            out[str(depth)] = dict(a_ms=float(a), b_gbps=float(1e-6 / slope) if slope > 0 else None)
    return out


def meta():
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(host=socket.gethostname(), time=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), commit=commit,
                kernel=platform.release(), cpus=os.cpu_count())


def save(res, text, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'{name}_{res["meta"]["host"]}_{res["meta"]["time"].replace("-", "").replace(":", "")}'
    stem.with_suffix('.json').write_text(json.dumps(res, indent=1))
    stem.with_suffix('.txt').write_text(text + '\n')
    return stem


def summary(res):
    m, lines = res['meta'], []
    lines.append(f'bench_storage  host {m["host"]}  {m["time"]}  commit {m["commit"]}  {"quick" if res["quick"] else "full"} run')
    lines.append(f'free on /        {res["free_before_gb"]:.0f} GB before the fetch'
                 f' (the experts outside VRAM + pinned RAM need {COLD_BYTES / 1e9:.1f} GB)')
    if res.get('fetch'):
        f = res['fetch']
        lines.append(f'fetch from HF    {f["bytes"] / 1e9:.1f} GB in {f["seconds"]:.0f} s = {f["gbps"]:.2f} GB/s ({f["files"]} files)')
    if 'odirect' in res:
        lines.append(f'O_DIRECT         {"yes" if res["odirect"] else "NO: reads may come from the file cache"}')
    if res.get('write'):
        w = res['write']
        lines.append(f'write            {w["bytes"] / GiB:.0f} GiB at {w["gbps"]:.2f} GB/s (O_DIRECT {"yes" if w["direct"] else "no"})')
    if res.get('sweep'):
        lines.append('sweep GB/s       ' + 'size \\ in flight  ' + ''.join(f'{d:>7}' for d in res['plan']['depths']))
        for size in res['plan']['sizes']:
            row = {c['depth']: c for c in res['sweep'] if c['size'] == size}
            label = f'{size / MiB:.2f} MiB' if size >= MiB else f'{size // 1024} KiB'
            lines.append(f'                 {label:>17} ' + ''.join(f'{row[d]["gbps"]:7.2f}' if d in row else '      -'
                                                                  for d in res['plan']['depths']))
        best = max(res['sweep'], key=lambda c: c['gbps'])
        lines.append(f'sweep best       {best["size"] / MiB:.2f} MiB x {best["depth"]} in flight: {best["gbps"]:.2f} GB/s,'
                     f' p50 {best["p50_ms"]:.1f} ms, p99 {best["p99_ms"]:.1f} ms')
    if res.get('groups'):
        lines.append('expert groups    experts: p50 / p99 ms at the best chunk count (chunks)')
        best = {}
        for g in res['groups']:
            if g['experts'] not in best or g['p50_ms'] < best[g['experts']]['p50_ms']:
                best[g['experts']] = g
        lines.append('                 ' + '   '.join(f'{n}: {g["p50_ms"]:.1f} / {g["p99_ms"]:.1f} ({g["chunks"]})'
                                                     for n, g in sorted(best.items())))
        if 3 in best and 6 in best:
            r = best[6]['p50_ms'] / best[3]['p50_ms']
            lines.append(f't(6)/t(3)        {r:.2f}  (about 1: limited by response time, extra parallel reads are nearly free;'
                         f' about 2: limited by bandwidth)')
    if res.get('sustained'):
        g = res['sustained']['gbps']
        k = max(1, len(g) // 15)
        first, last = np.mean(g[:k]), np.mean(g[-k:])
        secs = len(g) * res['sustained']['window_s']
        lines.append(f'sustained        {f"{secs / 60:.0f} min" if secs >= 120 else f"{secs} s"} at {res["sustained"]["depth"]} in flight:'
                     f' {np.mean(g):.2f} GB/s average, first {first:.2f}, last {last:.2f}'
                     f' -> {"slows down" if last < 0.8 * first else "steady"}')
    if res.get('fits'):
        lines.append('fit              ' + '   '.join(f'{d} in flight: {f["a_ms"]:.2f} ms + bytes / {f["b_gbps"]:.2f} GB/s'
                                                     for d, f in res['fits'].items() if f['b_gbps']))
    if all(k in res for k in ('odirect', 'sustained', 'groups')) and res['sustained']['gbps']:
        g3 = min((g for g in res['groups'] if g['experts'] == 3), key=lambda g: g['p50_ms'], default=None)
        checks = [('sustained >= 1.5 GB/s', np.mean(res['sustained']['gbps']) >= 1.5),
                  ('3 experts p99 <= 20 ms', g3 is not None and g3['p99_ms'] <= 20),
                  (f'free >= {COLD_BYTES / 1e9:.0f} GB', res['free_before_gb'] >= COLD_BYTES / 1e9),
                  ('O_DIRECT', res['odirect'])]
        lines.append('DP1 first row    ' + ', '.join(f'{name}: {"yes" if ok else "NO"}' for name, ok in checks)
                     + (' -> local store qualifies' if all(ok for _, ok in checks) else ''))
    if res.get('error'):
        lines.append(f'ERROR            {res["error"]}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true', help='about 2 minutes: 2 files, fewer sizes, shorter runs')
    ap.add_argument('--files', nargs='+', help='use these existing files instead of fetching from Hugging Face')
    ap.add_argument('--keep', action='store_true', help='keep the fetched model files on /')
    ap.add_argument('--out', default=str(ROOT / 'results'), help='where the results go (default: results/ in the repo)')
    a = ap.parse_args()
    q = a.quick
    plan = dict(files=2 if q else 20, write_bytes=2 * GiB if q else 16 * GiB, cell_s=0.5 if q else 4.0,
                sizes=[128 * 1024, MiB, EXPERT] if q else [128 * 1024, 256 * 1024, 512 * 1024, MiB, 2 * MiB, 4 * MiB, 8 * MiB, EXPERT],
                depths=[1, 8, 32] if q else [1, 2, 4, 8, 16, 32, 64], experts=[1, 3, 6] if q else [1, 2, 3, 4, 6, 12],
                chunks=[1, 8] if q else [1, 4, 8, 16, 32], trials=10 if q else 200, sustain_s=30 if q else 900,
                window_s=5 if q else 10)
    bench = LOCAL / 'bench'
    res = dict(meta=meta(), quick=q, plan=plan)
    files = None
    print(f'bench_storage: {"quick" if q else "full"} run, working in {bench}', flush=True)
    try:
        bench.mkdir(parents=True, exist_ok=True)
        res['free_before_gb'] = shutil.disk_usage(bench).free / 1e9
        if a.files:
            paths = [Path(p) for p in a.files]
        else:
            fit = int((res['free_before_gb'] * 1e9 - plan['write_bytes'] - 10e9) / 3.6e9)   # files average ~3.5 GB
            plan['files'] = max(2, min(plan['files'], fit))
            print(f'fetching {plan["files"]} model files from Hugging Face ...', flush=True)
            paths, nbytes, seconds = fetch(model_files(1, plan['files']), bench / 'hf')
            res['fetch'] = dict(files=len(paths), bytes=nbytes, seconds=seconds, gbps=nbytes / seconds / 1e9)
            print(f'  {nbytes / 1e9:.1f} GB in {seconds:.0f} s = {nbytes / seconds / 1e9:.2f} GB/s', flush=True)
        files = Files(paths)
        res['odirect'] = files.direct
        print(f'O_DIRECT: {files.direct}', flush=True)
        res['write'] = write_test(bench / 'write_test.bin', plan['write_bytes'])
        print(f'write: {res["write"]["gbps"]:.2f} GB/s', flush=True)
        bufs = [aligned_buffer(EXPERT) for _ in range(max(plan['depths']))]
        res['sweep'] = []
        for size in plan['sizes']:
            for depth in plan['depths']:
                c = sweep_cell(files, size, depth, plan['cell_s'], bufs)
                res['sweep'].append(c)
                print(f'sweep {size / MiB:6.2f} MiB x {depth:2d}: {c["gbps"]:6.2f} GB/s  p50 {c["p50_ms"]:7.2f} ms'
                      f'  p99 {c["p99_ms"]:7.2f} ms', flush=True)
        res['fits'] = fits(res['sweep'])
        res['groups'] = []
        group_buf = aligned_buffer(max(plan['experts']) * EXPERT)
        with ThreadPoolExecutor(64) as pool:
            for n in plan['experts']:
                for chunks in plan['chunks']:
                    g = group_trials(files, n, chunks, plan['trials'], pool, group_buf, seed=100 * n + chunks)
                    res['groups'].append(g)
                    print(f'group {n:2d} experts x {chunks:2d} chunks: p50 {g["p50_ms"]:7.2f} ms  p99 {g["p99_ms"]:7.2f} ms',
                          flush=True)
        print(f'sustained read for {plan["sustain_s"]} s ...', flush=True)
        res['sustained'] = sustained(files, plan['sustain_s'], 32, plan['window_s'], bufs)
    except BaseException as e:
        res['error'] = f'{type(e).__name__}: {e}'
        if not isinstance(e, Exception):
            raise
    finally:
        if files is not None:
            files.close()
        cleaned = not a.keep and not a.files      # files given with --files are never deleted
        if cleaned:
            shutil.rmtree(bench, ignore_errors=True)
        text = summary(res)
        stem = save(res, text, 'bench_storage', a.out)
        print('\n' + text + f'\nsaved {stem}.json and .txt' + (f'; deleted {bench}' if cleaned else ''), flush=True)


if __name__ == '__main__':
    main()

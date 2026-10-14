"""Quick speed probe at every session start (about 30 s; the bootstrap runs it), because each session may land on
another machine. It prints one line and appends it to $V4_DATA/logs/probe.csv:
- Hugging Face: one model file (~3.5 GB) fetched onto the local disk, where each session's model comes from (D6);
- local disk: that file read back with O_DIRECT, whole experts (12.75 MiB) with 16 reads in flight, for 3 s;
- RAM -> GPU: one expert copied from pinned memory, the median of 20 copies.
The file is deleted afterwards.
    python i_o_check/probe.py                 # --file PATH reads an existing file instead of fetching one
"""
import argparse
import csv
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_storage import EXPERT, LOCAL, Files, aligned_buffer, fetch, meta, model_files, sweep_cell


def h2d_gbps():
    import torch
    if not torch.cuda.is_available():
        return None
    src = torch.empty(EXPERT, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(EXPERT, dtype=torch.uint8, device='cuda')
    times = []
    for i in range(23):
        t = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t)
    return EXPERT / statistics.median(times) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--file', help='read this existing file instead of fetching one from Hugging Face')
    a = ap.parse_args()
    m, work = meta(), LOCAL / 'probe'
    row = dict(time=m['time'], host=m['host'], hf_gbps=None, disk_gbps=None, disk_p99_ms=None, odirect=None, h2d_gbps=None)
    try:
        if a.file:
            paths = [Path(a.file)]
        else:
            paths, nbytes, seconds = fetch(model_files(2, 1), work)
            row['hf_gbps'] = nbytes / seconds / 1e9
        files = Files(paths)
        cell = sweep_cell(files, EXPERT, 16, 3.0, [aligned_buffer(EXPERT) for _ in range(16)])
        files.close()
        row.update(disk_gbps=cell['gbps'], disk_p99_ms=cell['p99_ms'], odirect=files.direct)
    except Exception as e:
        print(f'probe: the disk part failed: {type(e).__name__}: {e}')
    finally:
        shutil.rmtree(work, ignore_errors=True)
    row['h2d_gbps'] = h2d_gbps()

    def fmt(v, spec):
        return '-' if v is None else format(v, spec)

    print(f'Hugging Face {fmt(row["hf_gbps"], ".2f")} GB/s | local disk {fmt(row["disk_gbps"], ".2f")} GB/s'
          f' (p99 {fmt(row["disk_p99_ms"], ".1f")} ms, O_DIRECT {row["odirect"]}) | RAM -> GPU {fmt(row["h2d_gbps"], ".1f")} GB/s')
    log = Path(os.environ.get('V4_DATA', '/storage/v4flash')) / 'logs' / 'probe.csv'
    if log.parent.is_dir():
        new = not log.exists()
        with log.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == '__main__':
    main()

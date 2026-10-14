"""Phase 3 item 4, run with Phase 4: how fast the GPU does the engine's basic steps. Results go to
results/bench_gpu_<host>_<time>.json/.txt.
    python i_o_check/bench_gpu.py                        # about 2 minutes on the box
    python i_o_check/bench_gpu.py --device cpu --quick   # a dry run of the code paths
1. expanding one expert from FP4 + E8M0 scales (the checkpoint's format) to BF16 (what the GPU multiplies): plain
   PyTorch, with a 256-entry table that turns one byte into its two values, and the same function through
   torch.compile. Both are checked bit for bit against a slow decode, nibble by nibble, in float64;
2. BF16 matmuls at expert shapes: x [rows, 4096] times w1 [2048, 4096], rows 1 ... 16,384 (time, TFLOPS, and for few
   rows the speed of reading the weight); one whole expert (w1, w3, SiLU, w2) for 1 and 512 rows, and for 1 row
   starting from its FP4 bytes;
3. the cost of one small GPU operation (the launch overhead) and of one host read (a GPU -> CPU copy that waits),
   times the ~22,400 operations and ~387 host reads of one eager decode token (plan section 3.1)."""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_storage import ROOT, meta, save

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]   # the 16 FP4 codes
SHAPES = dict(w1=(2048, 4096), w3=(2048, 4096), w2=(4096, 2048))    # one expert's matrices, [out, in] values
OPS_PER_TOKEN, READS_PER_TOKEN = 22_400, 387                         # eager decode, plan section 3.1


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def per_call(fn, dev, min_s=0.2):
    """Seconds per call: one warm-up call, then n calls back to back with one wait at the end, n doubling until the
    batch takes min_s. Calls queue on the GPU without waiting for each other, as they do in the engine."""
    fn()
    sync(dev)
    n = 1
    while True:
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync(dev)
        dt = time.perf_counter() - t
        if dt >= min_s:
            return dt / n
        n *= 2


def pair_table(dev):
    """[256, 2] BF16: a byte -> (the value of its low 4 bits, the value of its high 4 bits)."""
    v, b = torch.tensor(E2M1), torch.arange(256)
    return torch.stack([v[b & 15], v[b >> 4]], dim=-1).to(dev, torch.bfloat16)


def expand(packed, scales, pairs):
    """One FP4 matrix to BF16. packed: uint8 [out, in/2], low 4 bits = the even element; scales: uint8 E8M0 codes
    [out, in/32], value 2^(code - 127). Exact: an FP4 value times a power of two always fits in BF16."""
    out = packed.shape[0]
    w = pairs[packed.int()].view(out, -1, 32)                        # [out, in/32, 32]
    s = torch.exp2(scales.float() - 127).to(torch.bfloat16)          # [out, in/32]
    return (w * s.unsqueeze(-1)).view(out, -1)                        # [out, in]


def expand_slow(packed, scales):
    """The same decode, nibble by nibble, in float64 on the CPU: the reference."""
    v, p = torch.tensor(E2M1, dtype=torch.float64), packed.cpu().long()
    w = torch.stack([v[p & 15], v[p >> 4]], dim=-1).flatten(-2)       # [out, in]
    return w * torch.exp2(scales.cpu().double() - 127).repeat_interleave(32, dim=-1)


def random_expert(dev):
    """FP4 bytes and scale codes of one expert. Codes 115-125 are scales 2^-12 ... 2^-2, as in real experts."""
    return {k: (torch.randint(0, 256, (o, i // 2), dtype=torch.uint8, device=dev),
                torch.randint(115, 126, (o, i // 32), dtype=torch.uint8, device=dev)) for k, (o, i) in SHAPES.items()}


def bench_expand(dev):
    pairs, fp4 = pair_table(dev), random_expert(dev)
    compiled = torch.compile(expand)
    t0 = time.perf_counter()
    exact = {}
    for name, fn in (('plain', expand), ('compiled', compiled)):
        exact[name] = all(torch.equal(fn(p, s, pairs).double().cpu(), expand_slow(p, s)) for p, s in fp4.values())
    compile_s = time.perf_counter() - t0
    bytes_in = sum(p.numel() + s.numel() for p, s in fp4.values())
    bytes_out = sum(o * i * 2 for o, i in SHAPES.values())
    out = dict(compile_s=compile_s, exact=exact, bytes_in=bytes_in, bytes_out=bytes_out)
    for name, fn in (('plain', expand), ('compiled', compiled)):
        t = per_call(lambda: [fn(p, s, pairs) for p, s in fp4.values()], dev)
        out[name] = dict(ms=t * 1e3, gbps=(bytes_in + bytes_out) / t / 1e9)
    return out


def bench_matmul(dev, rows):
    w = torch.randn(*SHAPES['w1'], device=dev, dtype=torch.bfloat16) * 0.02
    out = {}
    for r in rows:
        x = torch.randn(r, SHAPES['w1'][1], device=dev, dtype=torch.bfloat16)
        t = per_call(lambda: F.linear(x, w), dev)
        out[r] = dict(ms=t * 1e3, tflops=2 * r * w.numel() / t / 1e12, weight_gbps=w.numel() * 2 / t / 1e9)
    return out


def bench_expert(dev, rows):
    w = {k: torch.randn(o, i, device=dev, dtype=torch.bfloat16) * 0.02 for k, (o, i) in SHAPES.items()}
    pairs, fp4 = pair_table(dev), random_expert(dev)

    def expert(x, w1, w3, w2):
        return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)

    out = {}
    for r in rows:
        x = torch.randn(r, SHAPES['w1'][1], device=dev, dtype=torch.bfloat16)
        out[r] = per_call(lambda: expert(x, w['w1'], w['w3'], w['w2']), dev) * 1e3
    x = torch.randn(1, SHAPES['w1'][1], device=dev, dtype=torch.bfloat16)
    out['from_fp4_1_row'] = per_call(lambda: expert(x, *(expand(*fp4[k], pairs) for k in ('w1', 'w3', 'w2'))), dev) * 1e3
    return out


def bench_launch(dev):
    z = torch.zeros(1, device=dev)
    op = per_call(lambda: z.add_(1), dev)
    read = per_call(lambda: (z.add_(1), z.item()), dev)
    return dict(op_us=op * 1e6, read_us=read * 1e6, ops_ms_per_token=op * OPS_PER_TOKEN * 1e3,
                reads_ms_per_token=max(0.0, read - op) * READS_PER_TOKEN * 1e3)


def summary(res):
    m, e, x, l = res['meta'], res['expand'], res['expert'], res['launch']
    lines = [f'bench_gpu  host {m["host"]}  {m["time"]}  commit {m["commit"]}  torch {res["torch"]}  {res["gpu"]}'
             + ('  QUICK' if res['quick'] else ''),
             f'expand FP4->BF16  one expert (3 matrices, 25.2M values, {e["bytes_in"] / 1e6:.1f} MB in, {e["bytes_out"] / 1e6:.1f} MB out): '
             f'plain {e["plain"]["ms"]:.3f} ms ({e["plain"]["gbps"]:.0f} GB/s), compiled {e["compiled"]["ms"]:.3f} ms '
             f'({e["compiled"]["gbps"]:.0f} GB/s; first calls incl. compiling {e["compile_s"]:.0f} s); bit-exact vs float64: '
             f'plain {e["exact"]["plain"]}, compiled {e["exact"]["compiled"]}',
             'matmul BF16       x [rows, 4096] @ w1 [2048, 4096]: rows ms TFLOPS (weight read GB/s)']
    lines += [f'                  {r:>6,}  {v["ms"]:8.3f}  {v["tflops"]:6.1f}  ({v["weight_gbps"]:.0f} GB/s)'
              for r, v in res['matmul'].items()]
    rows = [r for r in x if r != 'from_fp4_1_row']
    lines.append('expert BF16       w1, w3, SiLU, w2: ' + ', '.join(f'{r:,} row{"s" if r > 1 else ""} {x[r]:.3f} ms' for r in rows)
                 + f'; 1 row starting from its FP4 bytes {x["from_fp4_1_row"]:.3f} ms')
    lines.append(f'launch            one small GPU op {l["op_us"]:.1f} us, one host read {l["read_us"]:.1f} us -> '
                 f'{OPS_PER_TOKEN:,} ops = {l["ops_ms_per_token"]:.0f} ms/token, {READS_PER_TOKEN} reads = '
                 f'{l["reads_ms_per_token"]:.0f} ms/token more (eager decode, plan section 3.1)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--device', default='cuda', help='cuda (default), or cpu for a dry run')
    ap.add_argument('--quick', action='store_true', help='fewer matmul sizes: checks the code paths, not the numbers')
    ap.add_argument('--out', default=str(ROOT / 'results'), help='where the results go (default: results/ in the repo)')
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        sys.exit('CUDA is not available; use --device cpu for a dry run')
    res = dict(meta=meta(), quick=a.quick, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu')
    print('expanding FP4 experts ...', flush=True)
    res['expand'] = bench_expand(dev)
    print('matmuls ...', flush=True)
    res['matmul'] = bench_matmul(dev, (1, 64, 1024) if a.quick else (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                                                                      4096, 8192, 16384))
    res['expert'] = bench_expert(dev, (1, 512))
    print('launch overhead ...', flush=True)
    res['launch'] = bench_launch(dev)
    text = summary(res)
    stem = save(res, text, 'bench_gpu', a.out)
    print('\n' + text + f'\nsaved {stem}.json and .txt', flush=True)


if __name__ == '__main__':
    main()

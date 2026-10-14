"""Phase 4: DeepSeek's TileLang rounding kernels vs the notebook's, bit for bit, on the GPU.
    /root/tlvenv/bin/python checks/tilelang_check.py    # on the box, in its own venv with TileLang 0.1.8
    venv/bin/python checks/tilelang_check.py --shim     # dry run: the same comparison against the CPU copy
The kernels are act_quant (FP8 E4M3, blocks of 64, power-of-two scales) and fp4_act_quant (FP4 E2M1, blocks of 32)
from deepseek_official/0731/inference/kernel.py, unchanged. The model rounds the attention memory with the first
(448 values per row) and the indexer's keys and queries with the second (128 per row). Ours are the notebook's
fp8_simulate and fp4_simulate. Inputs, all BF16:
  - gate Q's: random values at 5 magnitudes, edge rows (exact power-of-two scales, ties), zero rows;
  - a sweep: every finite BF16 value, in blocks whose first value fixes the block's scale (all ties, tiny values,
    both signs), at two scales.
Bit-identical means the same 16 bits in every value; values that differ only in the sign of zero are counted
separately. --shim compares with checks/shim/kernel.py (the pure-PyTorch copy the CPU checks use) instead, on the
CPU: it needs neither TileLang nor CUDA. Results also go to results/tilelang_check_<host>_<time>.json/.txt."""
import argparse
import importlib.util
import json
import socket
import sys
import time
from datetime import datetime, timezone

import torch

from _paths import ROOT, load_notebook_defs

KERNEL = ROOT / 'deepseek_official' / '0731' / 'inference' / 'kernel.py'


def kernels(shim):
    """DeepSeek's kernel.py with real TileLang, or the CPU copy (already imported as `kernel` by load_notebook_defs)."""
    if shim:
        import kernel
        return kernel
    spec = importlib.util.spec_from_file_location('official_kernel', KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_q_inputs(n):
    """Gate Q's inputs (checks/gates.py), n values per row. Edge rows: exact power-of-two scales and ties."""
    torch.manual_seed(0)
    edge = torch.zeros(32, n)
    edge[0, :6] = torch.tensor([448.0, 1.0625, 1.1875, 17.0, 19.0, 0.0029296875])
    edge[1, :8] = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    return [edge] + [torch.randn(64, n) * sc for sc in (1e-6, 1e-2, 1.0, 300.0, 1e6)] + [torch.zeros(32, n)]


def sweep(anchor, block, n):
    """Every finite BF16 value v with |v| <= anchor, in blocks of `block` values that start with the anchor (so the
    block's largest value, and with it the scale, is fixed), packed into rows of n values; [rows, n]."""
    v = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    v = v[torch.isfinite(v) & (v.abs() <= anchor)]
    k = block - 1
    v = torch.cat([v, v.new_zeros(-len(v) % k)]).view(-1, k)
    blocks = torch.cat([v.new_full((len(v), 1), anchor), v], dim=1)               # [blocks, block]
    per_row = n // block
    blocks = torch.cat([blocks, blocks.new_zeros(-len(blocks) % per_row, block)])
    return blocks.view(-1, n)


def compare(ours, theirs):
    """(values, values with different bits, of which only the sign of a zero differs)."""
    a, b = ours.view(torch.int16), theirs.view(torch.int16)
    bits = a != b
    zero_sign = bits & (ours == theirs)
    return ours.numel(), int(bits.sum()), int(zero_sign.sum())


def check(name, ours_fn, theirs_fn, inputs, dev):
    total = diff = zeros = 0
    for x in inputs:
        x = x.to(dev, torch.bfloat16)
        theirs = theirs_fn(x.clone())
        n, d, z = compare(ours_fn(x), theirs)
        total, diff, zeros = total + n, diff + d, zeros + z
    real = diff - zeros
    ok = real == 0
    text = (f'{total:,} values: ' + ('bit-identical' if diff == 0 else
            f'{real:,} differ' + (f', {zeros:,} differ only in the sign of zero' if zeros else '')))
    return ok, text, dict(values=total, differ=real, zero_sign_only=zeros)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--shim', action='store_true', help='compare with the CPU copy of the kernels (dry run, no TileLang)')
    ap.add_argument('--out', default='results', help='folder for the results, relative to the repo')
    a = ap.parse_args()
    dev = torch.device('cpu' if a.shim else 'cuda')
    if dev.type == 'cuda' and not torch.cuda.is_available():
        sys.exit('CUDA is not available; use --shim for a dry run')
    U = load_notebook_defs()
    fp8, fp4 = U['fp8_simulate'], U['fp4_simulate']
    t0 = time.perf_counter()
    K = kernels(a.shim)
    import_s = time.perf_counter() - t0
    try:
        import tilelang
        tl = tilelang.__version__
    except ImportError:
        tl = None
    gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    meta = dict(host=socket.gethostname(), time=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                torch=torch.__version__, tilelang=tl, gpu=gpu, kernels='shim (CPU copy)' if a.shim else str(KERNEL.relative_to(ROOT)))
    lines = [f'tilelang_check  {meta["host"]}  {meta["time"]}  torch {meta["torch"]}  TileLang {tl}  {gpu}  '
             f'kernels: {meta["kernels"]}' + ('  DRY RUN' if a.shim else '')]
    print(lines[0], flush=True)
    fp8_inputs = gate_q_inputs(448) + [sweep(448.0, 64, 448), sweep(448.0 * 2**-20, 64, 448)]
    fp4_inputs = gate_q_inputs(448) + [sweep(6.0, 32, 448), sweep(6.0 * 2**-20, 32, 448)]
    cases = [
        ('FP8 act_quant, fp32 scales', fp8, lambda x: K.act_quant(x, 64, 'ue8m0', torch.float32, True), fp8_inputs),
        ('FP8 act_quant, E8M0 scales', fp8, lambda x: K.act_quant(x, 64, 'ue8m0', torch.float8_e8m0fnu, True), fp8_inputs),
        ('FP4 fp4_act_quant, 448 per row', fp4, lambda x: K.fp4_act_quant(x, 32, True), fp4_inputs),
        ('FP4 fp4_act_quant, 128 per row', fp4, lambda x: K.fp4_act_quant(x, 32, True),
         gate_q_inputs(128) + [sweep(6.0, 32, 128)]),
    ]
    res, passed = dict(meta=meta, import_s=import_s), {}
    for name, ours, theirs, inputs in cases:
        t = time.perf_counter()
        try:
            ok, text, res[name] = check(name, ours, theirs, inputs, dev)
        except Exception as e:
            ok, text = False, f'ERROR {type(e).__name__}: {str(e)[:300]}'
        line = f'{"PASS" if ok else "FAIL"}  {name:32s} {time.perf_counter() - t:5.1f} s  {text}'
        print(line, flush=True)
        lines.append(line)
        passed[name] = ok
    total = f'{sum(passed.values())}/{len(passed)} PASS'
    print(total)
    lines.append(total)
    res['passed'] = passed
    out_dir = ROOT / a.out
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f'tilelang_check_{meta["host"]}_{meta["time"].replace("-", "").replace(":", "")}'
    stem.with_suffix('.json').write_text(json.dumps(res, indent=1))
    stem.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(f'saved {stem.relative_to(ROOT) if stem.is_relative_to(ROOT) else stem}.json and .txt')
    sys.exit(0 if all(passed.values()) else 1)


if __name__ == '__main__':
    main()

"""Phase 4 (after_arch_plan.md): the notebook's model on the GPU in BF16, one line per gate, PASS or FAIL.
    python checks/gpu_gates.py                         # on the box: a few minutes
    python checks/gpu_gates.py --device cpu --quick    # a dry run of the code paths, e.g. on the Mac
Results also go to results/gpu_gates_<host>_<time>.json/.txt.

  G6    the tiny model in BF16 on the GPU vs float64 on the CPU, prefill 33 + 16 single tokens, 4 seeds: logits.
        Pass: within 2x of the official model's own BF16 error on the same runs (calibration: is ours normal BF16
        rounding, or a bug?), and cosine > 0.999. The plan's fixed bound (relative L2 < 2e-2) is reported, not required:
        on this tiny random model BF16 flips close router / indexer picks, and the official code itself misses it.
        Also: ours vs the official model directly, and how many picks differ from float64.
  G7    layers 0 / 2 / 3 at the real size (random weights, 8 of the 256 experts), BF16 on the GPU: prefill 4,096,
        then 64 single tokens. Time, peak GPU memory, error vs float32 on the CPU. Pass: it runs, outputs finite.
  PROF  torch.profiler over 8 more decode steps of each G7 layer: GPU kernels, CPU-GPU waits, ATen ops per step.
        2 x layer 0 + 21 x layer 2 + 20 x layer 3 (the real model's mix) = the eager cost per token, every expert
        already on the GPU.

Weights get the official model's types: BF16, except what DeepSeek's model.py keeps in float32 (norms, compressors,
mHC, head, sinks, router biases), read from the official model built the way its generate.py builds it. The model
code comes from the notebook (load_notebook_defs), as in checks/gates.py, and runs without a default device, so a
tensor that the code leaves on the CPU shows up as an error."""
import argparse
import collections
import contextlib
import copy
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from _paths import ROOT, load_notebook_defs, set_quant_simulation
from gates import LAYERS, all_position_logits, float64_tiny, layer_type, picks_of, real_args
from load_official import load_official, official_args
from sync_check import counted

MIX = {0: 2, 2: 21, 3: 20}           # how many of the 43 real layers each measured layer stands for
META = None                          # the official code used for meta-device builds only (official_types)


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


@contextlib.contextmanager
def official_defaults(dev):
    """How DeepSeek's generate.py runs its model: BF16 by default, new tensors on the GPU."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(dev):
            yield
    finally:
        torch.set_default_dtype(prev)


def official_types(build):
    """{name: type} of an official module built as generate.py builds it, on the meta device (no memory). It uses its
    own copy of the official code: model.py caches its RoPE and index tables without the device in the cache key."""
    global META
    if META is None:
        META = load_official('0731')
    with official_defaults('meta'):
        return {n: t.dtype for n, t in build(META).state_dict().items()}


def like_official(model, types):
    """Our parameters in the official model's types (same names)."""
    for n, p in model.named_parameters():
        p.data = p.data.to(types[n])
    return model


def metrics(x, ref):
    """Relative L2 error, cosine similarity and top-1 agreement of x against ref (float64)."""
    x, ref = x.double(), ref.double()
    out = dict(rel_l2=((x - ref).norm() / ref.norm()).item(),
               cos=F.cosine_similarity(x.flatten(), ref.flatten(), dim=0).item())
    if x.dim() == 3:                                                  # logits [b, tokens, vocab]
        out['top1'] = (x.argmax(-1) == ref.argmax(-1)).double().mean().item()
    return out


def count_flips(counts, ref, got):
    """Picks (Gate: experts, Indexer: compressed entries) that differ, compared token by token as sets."""
    for name, calls in ref.items():
        kind = 'router' if name.endswith('gate') else 'indexer'
        for a, b in zip(calls, got[name]):
            a = a.cpu().reshape(-1, a.shape[-1]).sort(-1).values
            b = b.cpu().reshape(-1, b.shape[-1]).sort(-1).values
            counts[kind + ' differ'] += (a != b).any(-1).sum().item()
            counts[kind + ' all'] += a.shape[0]


def flips_text(c):
    return ', '.join(f'{k} picks {c[k + " differ"]:,} of {c[k + " all"]:,} differ'
                     for k in ('router', 'indexer') if c[k + ' all'])


def run_ours(U, model, args, ids, P, dtype, dev):
    """Prefill P tokens, then the rest one at a time; logits at every position, [b, tokens, vocab] float64 on the CPU."""
    state = U['SequenceState'](args, ids.shape[0], dtype=dtype, device=dev)
    out = [model(ids[:, :P], state, return_all_logits=True)]
    out += [model(ids[:, t:t + 1], state, return_all_logits=True) for t in range(P, ids.shape[1])]
    return torch.cat(out, 1).double().cpu()


def run_theirs(model, ids, P):
    with official_defaults(ids.device):
        out = [model(ids[:, :P], 0)[1]] + [model(ids[:, t:t + 1], t)[1] for t in range(P, ids.shape[1])]
    return torch.cat([o.reshape(o.shape[0], -1, o.shape[-1]) for o in out], 1).double().cpu()


def gate_g6(U, dev, quick):
    off = load_official('0731')
    P, D, b = 33, 16, 2
    ref, ours, theirs, calibration, flips = [], [], [], None, collections.Counter()
    for seed in range(1 if quick else 4):
        model64, args = float64_tiny(U, seed)          # 64 indexer heads, as in the real model: fewer exact ties
        ids = torch.randint(0, args.vocab_size, (b, P + D))
        with picks_of(model64) as (p_ref,):
            ref.append(run_ours(U, model64, args, ids, P, torch.float64, torch.device('cpu')))
        types = official_types(lambda o: o.Transformer(official_args(o, args, max_batch_size=b)))
        model = like_official(copy.deepcopy(model64), types).to(dev)
        with picks_of(model) as (p_got,):
            ours.append(run_ours(U, model, args, ids.to(dev), P, torch.bfloat16, dev))
        count_flips(flips, p_ref, p_got)
        del model
        if calibration is None:
            try:
                with official_defaults(dev):
                    twin = off.Transformer(official_args(off, args, max_batch_size=b))
                twin.load_state_dict(model64.state_dict())
                twin.temperature = 0
                all_position_logits(twin)
                theirs.append(run_theirs(twin, ids.to(dev), P))
                del twin
            except Exception as e:                     # the official code on this device is the calibration only
                calibration = f'{type(e).__name__}: {str(e)[:120]}'
    ref, ours = torch.cat(ref), torch.cat(ours)
    m = metrics(ours, ref)
    ok = m['cos'] > 0.999
    text = (f'ours vs float64: rel L2 {m["rel_l2"]:.1e} (plan bound 2e-2: {m["rel_l2"] < 2e-2}), cos {m["cos"]:.6f}, '
            f'top-1 {m["top1"]:.1%}')
    res = dict(ours=m, flips=dict(flips))
    if calibration is None:
        t = metrics(torch.cat(theirs), ref)
        within = m['rel_l2'] <= 2 * t['rel_l2']
        ok &= within
        text += (f'; official BF16: rel L2 {t["rel_l2"]:.1e}, cos {t["cos"]:.6f}, top-1 {t["top1"]:.1%} '
                 f'-> ours within 2x: {within}; ours vs official: rel L2 {metrics(ours, torch.cat(theirs))["rel_l2"]:.1e}')
        res['official'] = t
    else:
        ok &= m['rel_l2'] < 2e-2                       # without the calibration, the plan's fixed bound decides
        text += f'; calibration not available ({calibration})'
        res['calibration_error'] = calibration
    return text + f'; {flips_text(flips)}', ok, res


def run_block(U, block, args, layer_id, x, ids, P, dtype, dev):
    """A Block alone: prefill P, then one token at a time. Returns the outputs, prefill s, decode s per token, memory."""
    st = U['SequenceState'](args, 1, dtype=dtype, device=dev).layers[layer_id]
    with torch.inference_mode():
        sync(dev)
        t0 = time.perf_counter()
        outs = [block(x[:, :P], ids[:, :P], st, 0)]
        sync(dev)
        t1 = time.perf_counter()
        outs += [block(x[:, t:t + 1], ids[:, t:t + 1], st, t) for t in range(P, x.shape[1])]
        sync(dev)
        t2 = time.perf_counter()
    return outs, t1 - t0, (t2 - t1) / max(1, x.shape[1] - P), st


def profile_steps(block, x, ids, st, start, dev):
    """GPU kernels, CPU-GPU waits and ATen ops per decode step, over the steps start, start + 1, ... of x."""
    from torch.profiler import ProfilerActivity, profile
    steps = x.shape[1] - start
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == 'cuda' else [])
    with torch.inference_mode():
        with profile(activities=acts, acc_events=True) as prof:
            for t in range(start, x.shape[1] - 1):
                block(x[:, t:t + 1], ids[:, t:t + 1], st, t)
            sync(dev)
        last = x.shape[1] - 1
        ops, reads, copies = counted(lambda: block(x[:, last:last + 1], ids[:, last:last + 1], st, last))
    events = prof.events()
    gpu = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    waits = sum(e.name in ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize') for e in events)
    n = steps - 1
    return dict(gpu_ops=len(gpu) / n, gpu_busy_ms=sum(e.time_range.elapsed_us() for e in gpu) / 1e3 / n, waits=waits / n,
                aten_ops=ops, host_reads=reads, routing_copies=copies)


def gate_g7(U, dev, quick):
    args = real_args(U, max_seq_len=8192)               # 4,096 + 64 positions: longer than the default RoPE table
    P, D, K = (512, 8, 4) if quick else (4096, 64, 9)
    rows, prof_rows, res, ok = [], [], {}, True
    for layer_id in LAYERS:
        torch.manual_seed(layer_id)
        block = U['Block'](layer_id, args)
        x = torch.randn(1, P + D + K, args.hc_mult, args.dim)
        ids = torch.randint(0, args.vocab_size, (1, P + D + K))
        with picks_of(block) as (p_ref,):
            ref = run_block(U, block, args, layer_id, x[:, :P + D], ids[:, :P + D], P, torch.float32, torch.device('cpu'))[0]
        block = like_official(block, official_types(lambda o: o.Block(layer_id, official_args(o, args)))).to(dev)
        xb, idsb = x.to(dev, torch.bfloat16), ids.to(dev)
        run_block(U, block, args, layer_id, xb[:, :258], idsb[:, :258], 256, torch.bfloat16, dev)   # warm-up
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with picks_of(block) as (p_got,):
            outs, t_pre, t_dec, st = run_block(U, block, args, layer_id, xb[:, :P + D], idsb[:, :P + D], P, torch.bfloat16, dev)
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == 'cuda' else float('nan')
        flips = collections.Counter()
        count_flips(flips, p_ref, p_got)
        pre, dec = metrics(outs[0].float().cpu(), ref[0]), metrics(torch.cat(outs[1:], 1).float().cpu(), torch.cat(ref[1:], 1))
        finite = all(torch.isfinite(o).all().item() for o in outs)
        ok &= finite
        p = profile_steps(block, xb, idsb, st, P + D, dev)
        rows.append(f'| {layer_id} | {layer_type(args, layer_id)} | {P / t_pre / 1e3:.1f}K | {t_pre * 1e3 / P * 1e3:.0f} | '
                    f'{t_dec * 1e3:.1f} | {peak:,.0f} | {pre["rel_l2"]:.1e} | {dec["rel_l2"]:.1e} | {flips_text(flips) or "-"} |')
        prof_rows.append(f'| {layer_id} | {t_dec * 1e3:.1f} | {p["gpu_busy_ms"]:.2f} | {p["gpu_ops"]:.0f} | {p["waits"]:.1f} | '
                         f'{p["aten_ops"]:,} | {p["routing_copies"]} | {p["host_reads"] or "none"} |')
        res[layer_id] = dict(type=layer_type(args, layer_id), prefill_s=t_pre, decode_ms=t_dec * 1e3, peak_mib=peak,
                             prefill=pre, decode=dec, flips=dict(flips), finite=finite, profile=p)
        del block, ref, outs, st
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    per_token = sum(MIX[i] * res[i]['decode_ms'] for i in LAYERS)
    busy = sum(MIX[i] * res[i]['profile']['gpu_busy_ms'] for i in LAYERS)
    kernels = sum(MIX[i] * res[i]['profile']['gpu_ops'] for i in LAYERS)
    table = [f'tokens {P:,} + {D}; prefill in tokens/s and us/token; decode ms/token; peak GPU MiB; errors vs float32 on the CPU',
             '| layer | type | prefill tok/s | prefill us/tok | decode ms | peak MiB | prefill rel L2 | decode rel L2 | picks |',
             '|---|---|---|---|---|---|---|---|---|'] + rows
    prof = ['| layer | decode ms | GPU busy ms | GPU kernels | CPU-GPU waits | ATen ops | routing copies | other host reads |',
            '|---|---|---|---|---|---|---|---|'] + prof_rows
    res['eager_ms_per_token'], res['gpu_busy_ms_per_token'], res['gpu_kernels_per_token'] = per_token, busy, kernels
    g7 = (f'all 3 layers ran, outputs finite: {ok}\n' + '\n'.join(table), ok)
    pr = (f'eager decode with every expert on the GPU: {per_token:.0f} ms/token (43 layers: 2 x layer 0 + 21 x layer 2 '
          f'+ 20 x layer 3), GPU busy {busy:.0f} ms, {kernels:,.0f} GPU kernels\n' + '\n'.join(prof), ok)
    return g7, pr, res


def header(dev):
    gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else platform.processor() or 'cpu'
    cuda = torch.backends.cuda.matmul
    flags = dict(tf32_matmul=cuda.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                 bf16_reduced_precision_reduction=cuda.allow_bf16_reduced_precision_reduction,
                 float32_matmul_precision=torch.get_float32_matmul_precision())
    return dict(host=socket.gethostname(), time=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                torch=torch.__version__, cuda=torch.version.cuda, device=str(dev), gpu=gpu, **flags)


def save(res, text, out_dir):
    out_dir = ROOT / out_dir
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f'gpu_gates_{res["meta"]["host"]}_{res["meta"]["time"].replace("-", "").replace(":", "")}'
    stem.with_suffix('.json').write_text(json.dumps(res, indent=1, default=str))
    stem.with_suffix('.txt').write_text(text + '\n')
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--device', default='cuda', help='cuda (default), or cpu for a dry run')
    ap.add_argument('--quick', action='store_true', help='1 seed, prefill 512 + 8: checks the code paths, not the numbers')
    ap.add_argument('--only', help='comma-separated: G6, G7 (G7 includes PROF)')
    ap.add_argument('--out', default='results', help='folder for the results, relative to the repo')
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        if not torch.cuda.is_available():
            sys.exit('CUDA is not available; use --device cpu for a dry run')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False     # the plan: full-precision sums
    U = load_notebook_defs()
    set_quant_simulation('fp8_simulate' in U)
    only = a.only.split(',') if a.only else ['G6', 'G7']
    res = dict(meta=header(dev), quick=a.quick)
    m = res['meta']
    lines = [f'gpu_gates  {m["host"]}  {m["time"]}  torch {m["torch"]}  CUDA {m["cuda"]}  {m["gpu"]}  '
             f'TF32 matmul {m["tf32_matmul"]}, cuDNN {m["tf32_cudnn"]}; BF16 reduced-precision sums {m["bf16_reduced_precision_reduction"]}'
             + ('  QUICK' if a.quick else '')]
    print(lines[0], flush=True)
    passed = {}

    def report(name, what, result, ok, seconds):
        first, *more = result.split('\n')
        out = [f'{name:5s}{"PASS" if ok else "FAIL":6s}{seconds:4.0f} s  {what} | {first}'] + ['      ' + line for line in more]
        print('\n'.join(out), flush=True)
        lines.extend(out)
        passed[name] = ok

    if 'G6' in only:
        t0 = time.perf_counter()
        try:
            text, ok, res['G6'] = gate_g6(U, dev, a.quick)
        except Exception as e:
            text, ok = f'ERROR {type(e).__name__}: {str(e)[:200]}', False
        report('G6', 'tiny model, BF16 vs float64', text, ok, time.perf_counter() - t0)
    if 'G7' in only:
        t0 = time.perf_counter()
        try:
            (g7, ok7), (pr, okp), res['G7'] = gate_g7(U, dev, a.quick)
        except Exception as e:
            g7 = pr = f'ERROR {type(e).__name__}: {str(e)[:200]}'
            ok7 = okp = False
        seconds = time.perf_counter() - t0
        report('G7', 'real-size layers 0 / 2 / 3, BF16', g7, ok7, seconds)
        report('PROF', 'decode step, profiled', pr, okp, 0)
    total = f'{sum(passed.values())}/{len(passed)} PASS' + ('' if all(passed.values()) else f'  FAILED: {[n for n, ok in passed.items() if not ok]}')
    print(total)
    lines.append(total)
    res['passed'] = passed
    stem = save(res, '\n'.join(lines), a.out)
    print(f'saved {stem.relative_to(ROOT) if stem.is_relative_to(ROOT) else stem}.json and .txt')
    sys.exit(0 if passed and all(passed.values()) else 1)


if __name__ == '__main__':
    main()

"""Phase 3, DP3: does this torch build work on this GPU? One line per check (PASS or FAIL), then a verdict.
    python tools/gpu_env_check.py
The checks are what the later phases rely on:
- CUDA works and the build has kernels for the A6000 (sm_86);
- a BF16 matmul (the model runs in BF16 on the GPU);
- a hand-written Triton kernel and torch.compile (both compile GPU code at run time with Triton's own ptxas);
- CUDA-graph capture and replay (Phase 13);
- torch.mm with out_dtype=float32 (BF16 head weights, fp32 logits, D10);
- FP8 E4M3 casts on the GPU bit-identical to the CPU's (the quantization simulation, Phase 4 item 4).
The driver supports CUDA 12.4, the wheels use CUDA 12.6 and Triton's ptxas is 12.8. That works only through CUDA's
minor-version compatibility (a newer 12.x on an older 12.x driver), which is what this tests.
"""
import platform
import subprocess
import sys
import time

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


def rel_err(out, ref):
    return ((out.double() - ref.double()).norm() / ref.double().norm()).item()


def versions():
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        driver = 'no nvidia-smi'
    return (f'python {platform.python_version()}, torch {torch.__version__}, CUDA runtime {torch.version.cuda}, '
            f'cuDNN {torch.backends.cudnn.version()}, triton {triton.__version__ if triton else "missing"}, driver {driver}')


def device():
    assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    assert cap == (8, 6), f'compute capability {cap}, expected (8, 6)'
    assert 'sm_86' in torch.cuda.get_arch_list(), f'no sm_86 kernels in this build: {torch.cuda.get_arch_list()}'
    return f'{p.name}, {p.total_memory / 2**30:.1f} GiB, {p.multi_processor_count} SMs, sm_86 kernels in the build'


def bf16_matmul():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(1024, 4096, generator=g), torch.randn(4096, 2048, generator=g)
    out = (a.cuda().bfloat16() @ b.cuda().bfloat16()).float().cpu()
    err = rel_err(out, a @ b)
    assert err < 1e-2, f'relative error {err:.1e}'
    return f'relative error vs fp32 on the CPU {err:.1e}'


def triton_kernel():
    assert triton is not None, 'triton is not installed'
    n = 1_000_003
    x, y = torch.randn(n, device='cuda'), torch.randn(n, device='cuda')
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    assert torch.equal(out, x + y), 'wrong result'
    return 'a hand-written kernel compiled, ran and matched exactly'


def compiled():
    def f(x, w):
        return torch.nn.functional.silu(x @ w) * 2 + 1
    x, w = torch.randn(64, 256, device='cuda'), torch.randn(256, 512, device='cuda')
    out = torch.compile(f)(x, w)
    err = rel_err(out, f(x, w))
    assert err < 1e-5, f'relative error {err:.1e}'
    return f'matmul + fused SiLU, relative error vs eager {err:.1e}'


def cuda_graph():
    x, w = torch.randn(256, 512, device='cuda'), torch.randn(512, 512, device='cuda')
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            (x @ w).relu_()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = (x @ w).relu_()
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    err = rel_err(y, (x @ w).relu_())
    assert err < 1e-6, f'relative error {err:.1e}'
    return f'captured, replayed on a new input, relative error vs eager {err:.1e}'


def mm_out_dtype():
    a = torch.randn(64, 4096, device='cuda').bfloat16()
    b = torch.randn(4096, 1024, device='cuda').bfloat16()
    out = torch.mm(a, b, out_dtype=torch.float32)
    assert out.dtype == torch.float32, f'output dtype {out.dtype}'
    err = rel_err(out, a.float() @ b.float())
    assert err < 1e-4, f'relative error {err:.1e} (an output rounded to BF16 would be ~2e-3)'
    return f'BF16 inputs, fp32 output, relative error vs fp32 math {err:.1e}'


def fp8_casts():
    g = torch.Generator().manual_seed(0)
    edges = torch.tensor([0.0, -0.0, 448.0, 464.0, 480.0, 1000.0, -1000.0, 2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -11])
    x = torch.cat([torch.randn(1 << 16, generator=g) * s for s in (1e-3, 1.0, 30.0, 400.0)] + [edges])
    for src in (x, x.bfloat16()):
        cpu = src.to(torch.float8_e4m3fn).view(torch.uint8)
        gpu = src.cuda().to(torch.float8_e4m3fn).view(torch.uint8).cpu()
        assert torch.equal(cpu, gpu), f'{(cpu != gpu).sum().item()} of {cpu.numel()} {src.dtype} values differ'
    back_cpu = x.to(torch.float8_e4m3fn).float().view(torch.int32)
    back_gpu = x.cuda().to(torch.float8_e4m3fn).float().cpu().view(torch.int32)
    assert torch.equal(back_cpu, back_gpu), 'FP8 -> float32 differs between CPU and GPU'
    extra = ', '.join(d for d in ('float8_e8m0fnu', 'float4_e2m1fn_x2') if hasattr(torch, d)) or 'none'
    return f'{x.numel()} float32 and BF16 values cast bit-identically, and back; other dtypes present: {extra}'


def settings():
    m = torch.backends.cuda.matmul
    return (f'TF32 matmul {m.allow_tf32}, TF32 cuDNN {torch.backends.cudnn.allow_tf32}, '
            f'BF16 reduced-precision reductions {m.allow_bf16_reduced_precision_reduction}, '
            f'float32 matmul precision {torch.get_float32_matmul_precision()}')


CHECKS = [('versions', versions), ('device', device), ('bf16 matmul', bf16_matmul), ('triton kernel', triton_kernel),
          ('torch.compile', compiled), ('cuda graph', cuda_graph), ('mm out_dtype', mm_out_dtype),
          ('fp8 casts', fp8_casts), ('settings', settings)]

if __name__ == '__main__':
    failed = []
    for name, fn in CHECKS:
        t = time.perf_counter()
        try:
            msg, ok = fn(), True
        except Exception as e:
            lines = [line.strip() for line in str(e).splitlines() if line.strip()] or ['']
            msg, ok = f'{type(e).__name__}: {lines[0]}' + (f' ... {lines[-1]}' if len(lines) > 1 else ''), False
            failed.append(name)
        print(f'{"PASS" if ok else "FAIL"}  {name:<14} {msg[:400]}  [{time.perf_counter() - t:.1f} s]')
    if not failed:
        print(f'DP3: torch {torch.__version__} PASS on this box, keep it')
    elif set(failed) <= {'triton kernel', 'torch.compile'}:
        print("DP3: only run-time compilation failed. Next rung: Triton with CUDA 12.4's ptxas (TRITON_PTXAS_PATH)")
    else:
        print(f'DP3: FAIL ({", ".join(failed)}). Next rungs: torch 2.6 + CUDA 12.4, then the image\'s torch 2.1.1')
    sys.exit(1 if failed else 0)

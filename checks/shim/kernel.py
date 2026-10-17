"""Pure-PyTorch stand-ins for the official GPU kernels (deepseek_official/*/inference/kernel.py).

The official model.py does
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
and those are TileLang GPU kernels. This file has the same names and signatures in plain PyTorch,
so the official model can run on a CPU for checks (see checks/load_official.py).

- sparse_attn, hc_split_sinkhorn: the kernels' math, computed in float32 (float64 stays float64),
  output cast back to the input dtype like the kernels do.
- act_quant, fp4_act_quant with inplace=True: the official model's "round to FP8 / FP4 and back" simulations.
  Here they are a literal copy of the TileLang kernels act_quant_kernel / fp4_quant_kernel with inplace=True
  (Phase 2 step 6):
  - per block of block_size values along the last dim, amax in float32, floored at 1e-4 (FP8) or 6 * 2**-126 (FP4);
  - scale = amax * float32(1/448) (FP8) or amax * float32(1/6) (FP4), rounded up to a power of two through the
    float32 bit pattern (fast_log2_ceil + fast_pow2) when scale_fmt is set (FP8) and always for FP4;
  - y = cast(clamp(x / scale, -max, max)) * scale, cast back to x's dtype. The FP8 cast is torch's float8_e4m3fn
    (round to nearest, ties to even); the FP4 cast is a separate round-to-nearest-even onto the e2m1 grid.
  The kernels are compiled for bf16 inputs; float32 inputs use the same float32 math, and float64 inputs (never used
  by the official model) the same rules in float64. With SIMULATE_QUANT = False both leave x unchanged.
  checks/_paths.load_notebook_defs sets SIMULATE_QUANT from the notebook: on once it defines fp8_simulate.
- act_quant with inplace=False, fp8_gemm, fp4_gemm: the official matmuls with FP8 / FP4 weights (the checkpoint's
  own, see load_official.official_module). act_quant rounds the matmul's input to FP8 with one scale per 128 values,
  as act_quant_kernel with inplace=False does; the GEMMs dequantize both sides (PyTorch's own FP8 and E8M0
  conversions; FP4 with convert.py's table, the low 4 bits first) and multiply in float32, which is what the kernels
  compute up to the order of the float32 sums. The output has the default dtype, like the kernels.
  FP8_ACTIVATIONS = False skips the input rounding: the plan's XD mode (exactly decoded weights, exact activations);
  True (the default) is the official behaviour, the plan's OE mode.
"""
import torch

SIMULATE_QUANT = True
FP8_ACTIVATIONS = True


def _upcast(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive normal x, like kernel.py fast_round_scale."""
    if x.dtype == torch.float32:
        bits = x.view(torch.int32)
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        return ((e + 127) << 23).view(torch.float32)
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), e - (m == 0.5).to(e.dtype))


_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _fp4_round(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value, ties to the even code (codes 0..7 are the grid indices). v is already clamped to [-6, 6]."""
    a = v.abs().double()
    grid = _FP4_GRID.to(a.device)
    lo = (a.unsqueeze(-1) >= grid).sum(-1) - 1
    hi = (lo + 1).clamp(max=7)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    pick = torch.where(d_lo < d_hi, lo, torch.where(d_hi < d_lo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return (torch.sign(v.double()) * grid[pick]).to(v.dtype)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    if not inplace:
        return _fp8_matmul_input(x, block_size, scale_fmt, scale_dtype)
    if not SIMULATE_QUANT:
        return x
    assert x.size(-1) % block_size == 0
    z = _upcast(x.contiguous()).unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True).clamp(min=torch.tensor(1e-4, dtype=z.dtype))
    s = amax * torch.tensor(1 / 448.0, dtype=z.dtype)
    if scale_fmt is not None:
        s = _pow2_ceil(s)
    y = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(z.dtype) * s
    x.copy_(y.flatten(-2).to(x.dtype))
    return x


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    if not inplace:
        raise NotImplementedError("fp4_act_quant(inplace=False) is only used with FP4 weights")
    if not SIMULATE_QUANT:
        return x
    assert x.size(-1) % block_size == 0
    z = _upcast(x.contiguous()).unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True).clamp(min=torch.tensor(6 * 2.0 ** -126, dtype=z.dtype))
    s = _pow2_ceil(amax * torch.tensor(1 / 6.0, dtype=z.dtype))
    y = _fp4_round((z / s).clamp(-6.0, 6.0)) * s
    x.copy_(y.flatten(-2).to(x.dtype))
    return x


def _fp8_matmul_input(x: torch.Tensor, block_size: int, scale_fmt, scale_dtype: torch.dtype):
    """act_quant(inplace=False): x as FP8 values [..., K] and one scale per block [..., K / block_size]."""
    if not FP8_ACTIVATIONS:
        return x.float(), torch.ones(*x.shape[:-1], x.size(-1) // block_size, device=x.device)
    z = _upcast(x.contiguous()).unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True).clamp(min=torch.tensor(1e-4, dtype=z.dtype))
    s = amax * torch.tensor(1 / 448.0, dtype=z.dtype)
    if scale_fmt is not None:
        s = _pow2_ceil(s)
    y = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return y.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C [..., N] = A [..., K] @ B [N, K]^T. A has a scale per 128 values along K, B one per 128 x 128 tile."""
    n, k = b.shape
    A = a.float() * a_s.float().repeat_interleave(128, dim=-1)[..., :k]
    B = b.float() * b_s.float().repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)[:, :k]
    return (A @ B.T).to(torch.get_default_dtype())


_FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])   # convert.py's FP4_TABLE


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C [..., N] = A [..., K] @ B [N, K]^T. A has a scale per 128 values along K. B is [N, K / 2] float4_e2m1fn_x2:
    two FP4 values per byte along K, the low 4 bits first (as convert.py unpacks them), and a scale per 32 values."""
    u = b.view(torch.uint8)
    table = _FP4_TABLE.to(u.device)
    w = torch.stack([table[(u & 0x0F).long()], table[(u >> 4).long()]], dim=-1).flatten(-2)   # [N, K]
    A = a.float() * a_s.float().repeat_interleave(128, dim=-1)
    B = w * b_s.float().repeat_interleave(32, dim=-1)
    return (A @ B.T).to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float, chunk: int = 256) -> torch.Tensor:
    """Attention over a chosen list of KV rows per query, with an attention sink.
    q: [b, s, h, d]   kv: [b, n, d] (one shared head, used as both K and V)
    attn_sink: [h]    topk_idxs: [b, s, k] row indices into kv, -1 = empty slot
    Queries are processed `chunk` at a time, so the gathered keys take [b, chunk, k, d], not [b, s, k, d].
    Returns [b, s, h, d] in q's dtype."""
    b, s = q.shape[:2]
    batch = torch.arange(b, device=kv.device).view(b, 1, 1)
    out = torch.empty_like(q)
    for i in range(0, s, chunk):
        idxs = topk_idxs[:, i:i + chunk]                                    # [b, c, k]
        keys = _upcast(kv[batch, idxs.clamp(min=0)])                        # [b, c, k, d]
        logits = torch.einsum("bshd,bskd->bshk", _upcast(q[:, i:i + chunk]), keys) * softmax_scale
        logits = logits.masked_fill(~(idxs >= 0).unsqueeze(2), float("-inf"))
        # the sink: one extra logit per head with a zero value vector, so it only adds exp(sink) to the denominator
        sink = attn_sink.to(logits.dtype).view(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
        probs = torch.cat([logits, sink], dim=-1).softmax(dim=-1)[..., :-1]  # [b, c, h, k]
        out[:, i:i + chunk] = torch.einsum("bshk,bskd->bshd", probs, keys).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes: [b, s, (2 + hc) * hc] -> pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc]."""
    hc = hc_mult
    pre, post, comb = mixes.split([hc, hc, hc * hc], dim=-1)
    base_pre, base_post, base_comb = hc_base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre * hc_scale[0] + base_pre) + eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + base_post)
    comb = (comb * hc_scale[2] + base_comb).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps                        # row softmax
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)     # columns
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # rows
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # columns
    return pre, post, comb

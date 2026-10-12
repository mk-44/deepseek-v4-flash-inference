from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import math, ast, io, contextlib, torch
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step6_check import tiny_args, upcast, ns as off_ns

# ---------------- Step 9 (as given to the user) ----------------
def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes: [b, s, (2 + hc) * hc] raw mixing numbers for every token (official: a TileLang kernel).
    Returns pre [b, s, hc] (A), post [b, s, hc] (C), comb [b, s, hc, hc] (B, doubly stochastic)."""
    hc = hc_mult
    pre, post, comb = mixes.split([hc, hc, hc * hc], dim=-1)
    base_pre, base_post, base_comb = hc_base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre * hc_scale[0] + base_pre) + eps                  # how much each stream feeds the sublayer
    post = 2 * torch.sigmoid(post * hc_scale[1] + base_post)                 # how much of the output each stream gets
    comb = (comb * hc_scale[2] + base_comb).unflatten(-1, (hc, hc))          # [b, s, hc, hc], row-major
    # Sinkhorn: alternate row and column normalization until every row and column sums to 1
    comb = comb.softmax(dim=-1) + eps                                        # exp + first row normalization
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)                     # columns
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)                 # rows
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)                 # columns
    return pre, post, comb


def hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
           norm_eps: float, hc_mult: int, sinkhorn_iters: int, hc_eps: float):
    """Before a sublayer. x: [b, s, hc, d], the residual streams.
    Returns the sublayer input [b, s, d] (weighted sum of the streams), plus post and comb for hc_post."""
    shape, dtype = x.shape, x.dtype
    x = upcast(x.flatten(2))                                                  # [b, s, hc*d]: streams side by side
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)         # RMS norm of the flattened streams, no gain
    mixes = F.linear(x, hc_fn) * rsqrt                                        # [b, s, (2+hc)*hc]: depends on the token
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)                        # sum_i pre[i] * stream_i -> [b, s, d]
    return y.to(dtype), post, comb


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """After a sublayer. x: [b, s, d] the sublayer output; residual: [b, s, hc, d] the streams before it.
    New stream j = post[j] * x + sum_i comb[i, j] * stream_i."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
    return y.type_as(x)


@torch.no_grad()
def naive_mhc(x, sublayer, hc_fn, hc_scale, hc_base, args):
    # one token at a time, straight from the formulas: X' = B X + C F(A X)  (paper eqs. 1, 3-8, code's orders)
    b, s, hc, d = x.shape
    y, new = torch.zeros(b, s, d), torch.zeros(b, s, hc, d)
    for bi in range(b):
        for t in range(s):
            X = x[bi, t]                                                       # [hc, d]: this token's streams
            v = X.flatten()
            m = hc_fn @ (v / torch.sqrt(v.square().mean() + args.norm_eps))   # 24 numbers
            A = torch.sigmoid(m[:hc] * hc_scale[0] + hc_base[:hc]) + args.hc_eps
            C = 2 * torch.sigmoid(m[hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
            B = (m[2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).view(hc, hc)
            B = torch.softmax(B, dim=1) + args.hc_eps
            B = B / (B.sum(0, keepdim=True) + args.hc_eps)
            for _ in range(args.hc_sinkhorn_iters - 1):
                B = B / (B.sum(1, keepdim=True) + args.hc_eps)
                B = B / (B.sum(0, keepdim=True) + args.hc_eps)
            a = (A[:, None] * X).sum(0)                                        # sublayer input
            y[bi, t] = a
            out = sublayer(a)
            for j in range(hc):
                new[bi, t, j] = C[j] * out + sum(B[i, j] * X[i] for i in range(hc))
    return y, new

# ---------------- literal transcription of the TileLang kernel (kernel.py lines 386-425), one token at a time ----------------
def kernel_transcription(mixes, hc_scale, hc_base, hc, iters, eps):
    n = mixes.shape[0] * mixes.shape[1]
    M = mixes.reshape(n, -1)
    pre = torch.empty(n, hc); post = torch.empty(n, hc); comb = torch.empty(n, hc, hc)
    for i in range(n):
        for j in range(hc):
            pre[i, j] = torch.sigmoid(M[i, j] * hc_scale[0] + hc_base[j]) + eps
            post[i, j] = 2 * torch.sigmoid(M[i, j + hc] * hc_scale[1] + hc_base[j + hc])
        cf = torch.empty(hc, hc)
        for j in range(hc):
            for k in range(hc):
                cf[j, k] = M[i, j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]
        row_max = cf.max(dim=1).values
        cf = torch.exp(cf - row_max[:, None]); row_sum = cf.sum(dim=1)
        cf = cf / row_sum[:, None] + eps
        col_sum = cf.sum(dim=0); cf = cf / (col_sum[None, :] + eps)
        for _ in range(iters - 1):
            row_sum = cf.sum(dim=1); cf = cf / (row_sum[:, None] + eps)
            col_sum = cf.sum(dim=0); cf = cf / (col_sum[None, :] + eps)
        comb[i] = cf
    return pre.view(*mixes.shape[:2], hc), post.view(*mixes.shape[:2], hc), comb.view(*mixes.shape[:2], hc, hc)

# ---------------- official Block.hc_pre / hc_post (verbatim), with our split+sinkhorn as the kernel ----------------
src = open(OFFICIAL / 'model.py').read(); tree = ast.parse(src)
blk = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Block')
class OffBlock: pass
g = dict(off_ns); g['hc_split_sinkhorn'] = hc_split_sinkhorn
for fn in blk.body:
    if isinstance(fn, ast.FunctionDef) and fn.name in ('hc_pre', 'hc_post'):
        exec(compile(ast.Module([fn], []), 'official', 'exec'), g); setattr(OffBlock, fn.name, g[fn.name])

if __name__ == "__main__":
    torch.manual_seed(0)
    args = tiny_args()
    hc, d = args.hc_mult, args.dim
    mix_hc = (2 + hc) * hc
    hc_fn = torch.randn(mix_hc, hc * d) * 0.02
    hc_scale = torch.randn(3) * 0.5 + 1
    hc_base = torch.randn(mix_hc) * 0.5
    x = torch.randn(2, 5, hc, d)
    # 1) split+sinkhorn vs kernel transcription
    mixes = torch.randn(2, 5, mix_hc)
    a = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, args.hc_sinkhorn_iters, args.hc_eps)
    k = kernel_transcription(mixes, hc_scale, hc_base, hc, args.hc_sinkhorn_iters, args.hc_eps)
    print("split+sinkhorn vs kernel transcription:", [torch.allclose(u, v, atol=1e-6) for u, v in zip(a, k)],
          "max diff", max((u - v).abs().max().item() for u, v in zip(a, k)))
    # 2) hc_pre / hc_post vs official methods
    ob = OffBlock(); ob.norm_eps = args.norm_eps; ob.hc_mult = hc; ob.hc_sinkhorn_iters = args.hc_sinkhorn_iters; ob.hc_eps = args.hc_eps
    y1, p1, c1 = hc_pre(x, hc_fn, hc_scale, hc_base, args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
    y2, p2, c2 = ob.hc_pre(x, hc_fn, hc_scale, hc_base)
    print("hc_pre vs official:", torch.equal(y1, y2), torch.equal(p1, p2), torch.equal(c1, c2))
    out = torch.randn(2, 5, d)
    print("hc_post vs official:", torch.equal(hc_post(out, x, p1, c1), ob.hc_post(out, x, p2, c2)))
    # bf16 path like official
    xb = x.bfloat16()
    yb1, pb1, cb1 = hc_pre(xb, hc_fn, hc_scale, hc_base, args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
    yb2, pb2, cb2 = ob.hc_pre(xb, hc_fn, hc_scale, hc_base)
    print("bf16 hc_pre vs official:", torch.equal(yb1, yb2), torch.equal(pb1, pb2), yb1.dtype,
          "| hc_post:", torch.equal(hc_post(out.bfloat16(), xb, pb1, cb1), ob.hc_post(out.bfloat16(), xb, pb2, cb2)))

    print("--- user-facing check")
    torch.manual_seed(0)
    args = tiny_args()
    hc, d = args.hc_mult, args.dim
    mix_hc = (2 + hc) * hc                                    # 24 = 4 (pre) + 4 (post) + 16 (comb)
    hc_fn = torch.randn(mix_hc, hc * d) * 0.02                # these three are learned weights in the real model
    hc_scale = torch.randn(3) * 0.5 + 1
    hc_base = torch.randn(mix_hc) * 0.5
    x = torch.randn(2, 5, hc, d)                              # 4 residual streams for each of 5 tokens

    y, post, comb = hc_pre(x, hc_fn, hc_scale, hc_base, args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
    print(y.shape, post.shape, comb.shape)
    print(comb.sum(-1).sub(1).abs().max().item(), comb.sum(-2).sub(1).abs().max().item())
    print(torch.linalg.matrix_norm(comb, ord=2).max().item())
    print(bool((comb >= 0).all()), bool((post > 0).all() and (post < 2).all()))

    W = torch.randn(d, d) / d ** 0.5
    sublayer = lambda a: torch.tanh(a @ W)                    # stand-in for attention / MoE
    new = hc_post(sublayer(y), x, post, comb)
    y_ref, new_ref = naive_mhc(x, sublayer, hc_fn, hc_scale, hc_base, args)
    print(new.shape, torch.allclose(y, y_ref, atol=1e-5), torch.allclose(new, new_ref, atol=1e-5))

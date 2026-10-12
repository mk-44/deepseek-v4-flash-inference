from _paths import ROOT, OFFICIAL, NOTEBOOK, load_notebook_defs
import math, ast, io, contextlib, sys, torch
sys.path.insert(0, str(ROOT / 'reference' / 'shim'))
import kernel as shim_kernel   # the official FP8 / FP4 rounding
shim_kernel.SIMULATE_QUANT = False   # the per-step reference checks in these files predate it; load_notebook_defs turns it on when the notebook has it
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step3_check import ModelArgs, tiny_args, layer_freqs_cis, apply_rope, upcast
    from step45_check import rotate_activation

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dtype = x.dtype; x = upcast(x)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)

# ---------------- Step 6 (as given to the user) ----------------
class Compressor(nn.Module):
    """Turns every `ratio` tokens into one KV entry: a learned, per-feature softmax-weighted sum.
    ratio 4 (CSA): overlapping blocks, each entry mixes its own 4 tokens with the previous block's 4.
    other ratios (HCA): plain, non-overlapping blocks."""

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int, rotate: bool = False):
        super().__init__()
        self.ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.rotate = rotate                  # Hadamard after RoPE: only the indexer's compressor
        coff = 2 if self.overlap else 1       # overlap: each token gives 2 feature sets (for the next block, for its own block)
        self.wkv = nn.Linear(args.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(args.dim, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.randn(compress_ratio, coff * head_dim) * 0.1)   # learned bias per slot in a block
        self.norm = RMSNorm(head_dim, args.norm_eps)

    def overlap_transform(self, t: torch.Tensor, fill: float) -> torch.Tensor:
        # t: [b, n, r, 2d] -> [b, n, 2r, d]
        # rows 0..r-1  = FIRST half of the previous block's rows (block 0 has no previous block -> fill)
        # rows r..2r-1 = SECOND half of this block's rows
        b, n, r, _ = t.shape
        d = self.head_dim
        out = t.new_full((b, n, 2 * r, d), fill)
        out[:, :, r:] = t[:, :, :, d:]
        out[:, 1:, :r] = t[:, :-1, :, :d]
        return out

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """x: [b, s, dim], tokens at positions 0..s-1; freqs_cis: the layer's RoPE table.
        Returns [b, s // ratio, head_dim]: one entry per COMPLETE block (an unfinished tail gives nothing yet)."""
        b, s, _ = x.shape
        r = self.ratio
        dtype = x.dtype
        x = upcast(x)                                                        # compression runs in >= fp32
        cutoff = s - s % r                                                   # tokens that fill complete blocks
        kv = self.wkv(x[:, :cutoff]).unflatten(1, (-1, r))                   # [b, n, r, coff*d]
        score = self.wgate(x[:, :cutoff]).unflatten(1, (-1, r)) + self.ape   # [b, n, r, coff*d]
        if self.overlap:
            kv = self.overlap_transform(kv, 0.0)                             # [b, n, 2r, d]
            score = self.overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)                          # [b, n, d]: softmax over rows, per feature
        kv = self.norm(kv.to(dtype))
        kv = apply_rope(kv, freqs_cis[:cutoff:r])                            # entry i sits at position i*r
        if self.rotate:
            kv = rotate_activation(kv)
        return kv


def naive_compress(comp: Compressor, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # block by block, straight from the paper: eqs. (11)-(12) for CSA, (22)-(23) for HCA
    r, d = comp.ratio, comp.head_dim
    kv_all, score_all = comp.wkv(x), comp.wgate(x)                      # [b, s, coff*d]
    entries = []
    for i in range(x.size(1) // r):
        kv_rows = kv_all[:, i * r:(i + 1) * r]                          # this block's r rows
        sc_rows = score_all[:, i * r:(i + 1) * r] + comp.ape
        if comp.overlap:
            kv_rows, sc_rows = kv_rows[..., d:], sc_rows[..., d:]       # own block: SECOND half
            if i > 0:                                                   # previous block: FIRST half
                kv_prev = kv_all[:, (i - 1) * r:i * r, :d]
                sc_prev = score_all[:, (i - 1) * r:i * r, :d] + comp.ape[:, :d]
                kv_rows = torch.cat([kv_prev, kv_rows], dim=1)
                sc_rows = torch.cat([sc_prev, sc_rows], dim=1)
        w = sc_rows.softmax(dim=1)                                      # over the rows, separately per feature
        entries.append((w * kv_rows).sum(dim=1))                        # [b, d]
    kv = comp.norm(torch.stack(entries, dim=1))
    kv = apply_rope(kv, freqs_cis[:kv.size(1) * r:r])
    return rotate_activation(kv) if comp.rotate else kv


# ---------------- official Compressor prefill, extracted verbatim (kernels: reference/shim/kernel.py rounding, our hadamard) ----------------
src = open(OFFICIAL / 'model.py').read(); tree = ast.parse(src)
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
ns = dict(torch=torch, nn=nn, F=F, math=math, Optional=Optional, Tuple=Tuple, Literal=Literal, dataclass=dataclass, world_size=1, rank=0, block_size=128,
          fp4_block_size=32, default_dtype=torch.float32, scale_fmt='ue8m0', scale_dtype=torch.float32,
          act_quant=shim_kernel.act_quant, fp4_act_quant=shim_kernel.fp4_act_quant, rotate_activation=rotate_activation)
from functools import lru_cache; ns['lru_cache'] = lru_cache
for node in tree.body:
    if (isinstance(node, ast.ClassDef) and node.name in ('ModelArgs', 'Linear', 'RMSNorm', 'Compressor')) or \
       (isinstance(node, ast.FunctionDef) and node.name in ('linear', 'precompute_freqs_cis', 'apply_rotary_emb')):
        exec(compile(ast.Module([node], []), 'official', 'exec'), ns)

def official_compress(ours: Compressor, args, x, freqs):
    oargs = ns['ModelArgs'](dim=args.dim, rope_head_dim=args.rope_head_dim, norm_eps=args.norm_eps, max_batch_size=x.size(0), max_seq_len=args.max_seq_len)
    oc = ns['Compressor'](oargs, ours.ratio, ours.head_dim, ours.rotate)
    with torch.no_grad():
        oc.wkv.weight.copy_(ours.wkv.weight); oc.wgate.weight.copy_(ours.wgate.weight)
        oc.ape.copy_(ours.ape); oc.norm.weight.copy_(ours.norm.weight)
    oc.kv_cache = torch.zeros(x.size(0), args.max_seq_len // ours.ratio, ours.head_dim)
    oc.freqs_cis = freqs
    out = oc(x, 0)
    return out

# ---------------- tests ----------------
torch.manual_seed(0)
args = tiny_args()
x = torch.randn(2, 19, args.dim)                 # 19 tokens = 4 complete blocks of 4 + a tail of 3
f = layer_freqs_cis(args, 4)
with torch.no_grad():
    for ratio, rotate, hd in [(4, False, args.head_dim), (4, True, args.index_head_dim), (16, False, args.head_dim)]:
        comp = Compressor(args, ratio, hd, rotate)
        out = comp(x, f)
        offi = official_compress(comp, args, x, f)
        print(f"ratio={ratio:2d} rotate={rotate!s:5}: shape {tuple(out.shape)}  naive match={torch.allclose(out, naive_compress(comp, x, f), atol=1e-5)}  official match={torch.allclose(out, offi, atol=1e-5)} (max diff {(out-offi).abs().max().item():.1e})")

    comp = Compressor(args, 4, args.head_dim)
    x2 = x.clone(); x2[:, 16:] = torch.randn(2, 3, args.dim)    # change only the unfinished tail
    print(torch.equal(comp(x, f), comp(x2, f)))                  # True
    print(comp(x[:, :3], f).shape)                               # torch.Size([2, 0, 64])
    x3 = x.clone(); x3[:, 8] += 1.0                              # change one token in block 2 (tokens 8..11)
    print((comp(x3, f) - comp(x, f)).abs().amax(dim=(0, 2)) > 0) # tensor([False, False,  True,  True])
    # float64 path
    c64 = Compressor(args, 4, args.head_dim).double()
    print("f64:", c64(x.double(), f).dtype, torch.allclose(c64(x.double(), f), naive_compress(c64, x.double(), f), atol=1e-12))

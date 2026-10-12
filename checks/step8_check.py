from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import math, ast, io, contextlib, torch
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step6_check import ModelArgs, tiny_args, layer_freqs_cis, apply_rope, upcast, rotate_activation, Compressor, RMSNorm, ns as off_ns
    from step7_check import Indexer
    from step45_check import sparse_attn

# ---------------- Step 8 (as given to the user) ----------------
def window_topk_idxs(window_size: int, s: int, device=None) -> torch.Tensor:
    """Prefill: for query t, the rows of tokens max(0, t-window+1) .. t (its own row included).
    Short rows (early tokens) are padded with -1. Returns [s, min(s, window_size)]."""
    t = torch.arange(s, device=device).unsqueeze(1)                                              # [s, 1]
    idx = (t - window_size + 1).clamp(min=0) + torch.arange(min(s, window_size), device=device)  # [s, w]
    return torch.where(idx > t, -1, idx)


def compress_topk_idxs(ratio: int, s: int, offset: int, device=None) -> torch.Tensor:
    """Prefill, HCA: for query t, EVERY visible block (i < (t+1)//ratio) as row offset + i; the rest -1.
    Returns [s, s // ratio]."""
    blocks = torch.arange(s // ratio, device=device)                              # [n]
    n_visible = torch.arange(1, s + 1, device=device).unsqueeze(1) // ratio       # [s, 1]
    return torch.where(blocks < n_visible, blocks + offset, -1)


class Attention(nn.Module):
    """One attention layer: a sliding window, plus compressed memory in CSA/HCA layers.
    Shared-KV multi-query attention (one KV head, K = V) with an attention sink."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.eps = args.norm_eps
        self.compress_ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim ** -0.5

        self.attn_sink = nn.Parameter(torch.randn(self.n_heads))                            # one sink logit per head
        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)                       # x -> query latent
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)   # latent -> all heads
        self.wkv = nn.Linear(args.dim, self.head_dim, bias=False)                           # x -> the one shared KV head
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # grouped output projection: wo_a holds one [o_lora_rank, heads_per_group * head_dim] matrix per group
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, args.dim, bias=False)
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None
        # RoPE table of this layer: a buffer moves with .to(device) but is not a learned weight
        self.register_buffer("freqs_cis", layer_freqs_cis(args, self.compress_ratio), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, dim], tokens at positions 0..s-1 (prefill). Returns [b, s, dim]."""
        b, s, _ = x.shape
        freqs = self.freqs_cis[:s]

        # queries: x -> latent qr (shared with the indexer) -> all heads; per-head RMS norm (no gain); RoPE
        qr = self.q_norm(self.wq_a(x))                                                  # [b, s, q_lora_rank]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))                  # [b, s, h, c]
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = apply_rope(q, freqs)

        # window KV: one shared head per token (normalized, RoPE); kv rows 0..s-1 are the s tokens
        kv = apply_rope(self.kv_norm(self.wkv(x)), freqs)                               # [b, s, c]
        idxs = window_topk_idxs(self.window_size, s, x.device).expand(b, -1, -1)         # [b, s, w]

        # compressed memory: entries are appended after the s window rows, so they start at row `s`
        if self.compress_ratio:
            if self.indexer is not None:                                                  # CSA: top-k blocks per query
                comp_idxs = self.indexer(x, qr, self.freqs_cis, offset=s)                 # [b, s, k]
            else:                                                                         # HCA: every visible block
                comp_idxs = compress_topk_idxs(self.compress_ratio, s, offset=s, device=x.device).expand(b, -1, -1)
            idxs = torch.cat([idxs, comp_idxs], dim=-1)                                   # [b, s, w + k]
            kv = torch.cat([kv, self.compressor(x, self.freqs_cis)], dim=1)               # [b, s + n, c]

        o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)                  # [b, s, h, c]
        o = apply_rope(o, freqs, inverse=True)                                           # undo the keys' absolute positions

        # grouped output projection: g groups of heads; group i uses only its own slice of wo_a
        o = o.reshape(b, s, self.n_groups, -1)                                           # [b, s, g, h*c/g]
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)                # [g, r, h*c/g]
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)                                      # [b, s, g, r]
        return self.wo_b(o.flatten(2))                                                   # [b, s, dim]


@torch.no_grad()
def naive_attention(attn: Attention, x: torch.Tensor) -> torch.Tensor:
    # one query at a time with an explicit memory list; same weights as `attn`
    b, s, _ = x.shape
    f = attn.freqs_cis
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_heads, attn.head_dim))
    q = apply_rope(q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps), f[:s])
    kv = apply_rope(attn.kv_norm(attn.wkv(x)), f[:s])                      # window rows, one per token
    if attn.compress_ratio:
        comp = attn.compressor(x, f)                                        # compressed entries
        if attn.indexer is not None:
            chosen = attn.indexer(x, qr, f, offset=0)                       # chosen block numbers (tested in Step 7)
    o = x.new_zeros(b, s, attn.n_heads, attn.head_dim)
    for bi in range(b):
        for t in range(s):
            mem = [kv[bi, j] for j in range(max(0, t - attn.window_size + 1), t + 1)]
            if attn.compress_ratio:
                if attn.indexer is not None:
                    blocks = [i for i in chosen[bi, t].tolist() if i >= 0]
                else:
                    blocks = range((t + 1) // attn.compress_ratio)
                mem += [comp[bi, i] for i in blocks]
            mem = torch.stack(mem)                                          # [m, c]: everything query t may read
            for h in range(attn.n_heads):
                e = torch.exp(mem @ q[bi, t, h] * attn.softmax_scale)
                o[bi, t, h] = (e / (e.sum() + torch.exp(attn.attn_sink[h]))) @ mem
    o = apply_rope(o, f[:s], inverse=True)
    # grouped output projection, one group at a time
    hpg = attn.n_heads // attn.n_groups                                     # heads per group
    wo_a = attn.wo_a.weight.view(attn.n_groups, attn.o_lora_rank, -1)
    parts = [o[:, :, g * hpg:(g + 1) * hpg].flatten(2) @ wo_a[g].T for g in range(attn.n_groups)]
    return attn.wo_b(torch.cat(parts, dim=-1))


# ---------------- official Attention (verbatim), our sparse_attn as the kernel ----------------
src = open(OFFICIAL / 'model.py').read(); tree = ast.parse(src)
off_ns['sparse_attn'] = sparse_attn
for node in tree.body:
    if (isinstance(node, ast.ClassDef) and node.name in ('RowParallelLinear', 'Attention')) or \
       (isinstance(node, ast.FunctionDef) and node.name in ('get_window_topk_idxs', 'get_compress_topk_idxs')):
        exec(compile(ast.Module([node], []), 'official', 'exec'), off_ns)

def official_attention(ours: Attention, layer_id, args, x):
    oargs = off_ns['ModelArgs'](dim=args.dim, n_heads=args.n_heads, head_dim=args.head_dim, rope_head_dim=args.rope_head_dim,
        q_lora_rank=args.q_lora_rank, o_groups=args.o_groups, o_lora_rank=args.o_lora_rank, window_size=args.window_size,
        compress_ratios=args.compress_ratios, norm_eps=args.norm_eps, max_batch_size=x.size(0), max_seq_len=args.max_seq_len,
        index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim, index_topk=args.index_topk,
        rope_theta=args.rope_theta, compress_rope_theta=args.compress_rope_theta, original_seq_len=args.original_seq_len,
        rope_factor=args.rope_factor, beta_fast=args.beta_fast, beta_slow=args.beta_slow)
    oa = off_ns['Attention'](layer_id, oargs)
    oa.wo_a.weight = nn.Parameter(torch.empty_like(ours.wo_a.weight))                     # official keeps wo_a in bf16
    if getattr(oa, 'indexer', None) is not None:
        oa.indexer.weights_proj.weight = nn.Parameter(torch.empty_like(ours.indexer.weights_proj.weight))
    oa.load_state_dict(ours.state_dict(), strict=True)                                    # same names + shapes, or this fails
    return oa(x, 0)

if __name__ == "__main__":
    torch.manual_seed(0)
    args = tiny_args()
    with torch.no_grad():
        for s in (1, 5, 8, 9, 16, 17, 37, 64):
            x = torch.randn(2, s, args.dim)
            res = []
            for layer_id in (0, 2, 3):
                attn = Attention(layer_id, args)
                y = attn(x); ref = official_attention(attn, layer_id, args, x)
                res.append(f"L{layer_id}: {'bit-identical' if torch.equal(y, ref) else f'maxdiff {(y-ref).abs().max().item():.1e}'}")
            print(f"s={s:2d} vs official: " + " | ".join(res))
    print("--- user-facing check")
    torch.manual_seed(0)
    args = tiny_args()
    x = torch.randn(2, 37, args.dim)
    for layer_id in (0, 2, 3):
        attn = Attention(layer_id, args)
        y = attn(x)
        print(layer_id, attn.compress_ratio, tuple(y.shape), torch.allclose(y, naive_attention(attn, x), atol=1e-5))
    for layer_id in (0, 2, 3):
        attn = Attention(layer_id, args)
        x2 = x.clone(); x2[:, 20] += 1.0
        d = (attn(x2) - attn(x)).abs().amax(dim=(0, 2))
        print(layer_id, bool((d[:20] == 0).all()), bool((d[20:] > 0).any()))
    # float64 model
    attn = Attention(2, args).double()
    print("f64:", attn(x.double()).dtype, torch.allclose(attn(x.double()), naive_attention(attn, x.double()), atol=1e-10))

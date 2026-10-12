from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import math, ast, io, contextlib, torch
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step6_check import ModelArgs, tiny_args, layer_freqs_cis, apply_rope, upcast, rotate_activation, Compressor, ns as off_ns

# ---------------- Step 7 (as given to the user) ----------------
class Indexer(nn.Module):
    """CSA only: for every query token, score each visible compressed block and keep the top-k."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.ratio = compress_ratio
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)  # queries from the shared latent
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False)                 # per-head weights, from x
        self.softmax_scale = self.head_dim ** -0.5
        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)     # the indexer's own keys

    def forward(self, x: torch.Tensor, qr: torch.Tensor, freqs_cis: torch.Tensor, offset: int) -> torch.Tensor:
        """x: [b, s, dim], tokens at positions 0..s-1; qr: [b, s, q_lora_rank], the attention's query latent.
        Returns [b, s, k]: chosen blocks as row numbers in the attention's kv array (block + offset), -1 = empty."""
        b, s, _ = x.shape
        # queries: RoPE at the token's position, then Hadamard
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))            # [b, s, h, c_I]
        q = rotate_activation(apply_rope(q, freqs_cis[:s]))
        # keys: one compressed indexer key per complete block
        k = self.compressor(x, freqs_cis)                                           # [b, n, c_I]
        # score[t, i] = sum_h w[t, h] * relu(q[t, h] . k[i])      (paper eq. 16)
        w = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)     # [b, s, h]
        score = torch.einsum("bshd,bnd->bshn", q, k).relu()                         # [b, s, h, n]
        score = (score * w.unsqueeze(-1)).sum(dim=2)                                # [b, s, n]
        # causality: query t may use block i only once that block is complete: i < (t + 1) // ratio
        n_visible = torch.arange(1, s + 1, device=x.device) // self.ratio           # [s]
        visible = torch.arange(k.size(1), device=x.device) < n_visible.unsqueeze(1)  # [s, n]
        score = score.masked_fill(~visible, float("-inf"))
        topk = score.topk(min(self.index_topk, k.size(1)), dim=-1).indices         # [b, s, k], best first
        # slots that landed on an invisible block (fewer than k visible) become -1
        return torch.where(topk < n_visible.unsqueeze(1), topk + offset, -1)


@torch.no_grad()
def naive_indexer(idx: Indexer, x, qr, freqs_cis):
    # one query at a time: score every visible block, keep the best k; returns sorted block numbers
    b, s, _ = x.shape
    q = rotate_activation(apply_rope(idx.wq_b(qr).unflatten(-1, (idx.n_heads, idx.head_dim)), freqs_cis[:s]))
    k = idx.compressor(x, freqs_cis)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    result = []
    for bi in range(b):
        for t in range(s):
            n_vis = (t + 1) // idx.ratio
            scores = [sum(w[bi, t, h] * torch.relu(q[bi, t, h] @ k[bi, i]) for h in range(idx.n_heads))
                      for i in range(n_vis)]
            best = sorted(range(n_vis), key=lambda i: -float(scores[i]))[:idx.index_topk]
            result.append(sorted(best))
    return result

# ---------------- official Indexer (verbatim) ----------------
src = open(OFFICIAL / 'model.py').read(); tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in ('ColumnParallelLinear', 'Indexer'):
        exec(compile(ast.Module([node], []), 'official', 'exec'), off_ns)

def official_indexer(ours: Indexer, args, x, qr, freqs, offset):
    oargs = off_ns['ModelArgs'](dim=args.dim, rope_head_dim=args.rope_head_dim, norm_eps=args.norm_eps, max_batch_size=x.size(0),
                                max_seq_len=args.max_seq_len, index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim,
                                index_topk=args.index_topk, q_lora_rank=args.q_lora_rank)
    oi = off_ns['Indexer'](oargs, ours.ratio)
    with torch.no_grad():
        oi.wq_b.weight.copy_(ours.wq_b.weight)
        oi.weights_proj.weight = nn.Parameter(ours.weights_proj.weight.detach().clone())   # official keeps this one in bf16
        for n in ('wkv', 'wgate'):
            getattr(oi.compressor, n).weight.copy_(getattr(ours.compressor, n).weight)
        oi.compressor.ape.copy_(ours.compressor.ape); oi.compressor.norm.weight.copy_(ours.compressor.norm.weight)
    oi.freqs_cis = freqs
    return oi(x, qr, 0, offset)

if __name__ == "__main__":
    torch.manual_seed(0)
    args = tiny_args()
    f = layer_freqs_cis(args, 4)
    with torch.no_grad():
        for s in (3, 4, 8, 19, 40, 77):
            idx = Indexer(args)
            x, qr = torch.randn(2, s, args.dim), torch.randn(2, s, args.q_lora_rank)
            ours = idx(x, qr, f, offset=s)
            ref = official_indexer(idx, args, x, qr, f, offset=s)
            print(f"s={s:2d}: shape {tuple(ours.shape)}  identical to official = {torch.equal(ours, ref)}")
    print("--- user-facing check")
    torch.manual_seed(0)
    args = tiny_args()                                   # ratio 4, index_topk 4
    indexer = Indexer(args)
    s = 40
    x = torch.randn(2, s, args.dim)
    qr = torch.randn(2, s, args.q_lora_rank)             # stands in for the attention's query latent
    f = layer_freqs_cis(args, 4)
    topk = indexer(x, qr, f, offset=0)                   # offset 0 -> values are plain block numbers
    print(topk.shape)
    print(topk[0, :12])
    print(topk[0, 39])
    ours = [sorted(i for i in row.tolist() if i >= 0) for row in topk.reshape(-1, topk.size(-1))]
    print(ours == naive_indexer(indexer, x, qr, f))
    print(all(i < (t + 1) // 4 for t in range(s) for i in topk[0, t].tolist()))

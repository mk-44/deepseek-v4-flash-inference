from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import math, torch
import numpy as np

def upcast(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

# ---------------- Step 4 ----------------
def hadamard_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Sylvester construction, normalized so that H @ H.T = I. n must be a power of 2."""
    assert n & (n - 1) == 0
    H = torch.ones(1, 1, dtype=dtype)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation of the last dim (official: fast_hadamard_transform with scale n ** -0.5)."""
    xu = upcast(x)
    H = hadamard_matrix(x.size(-1), xu.dtype).to(x.device)
    return (xu @ H).to(x.dtype)

# reference: scipy.linalg.hadamard construction (what fast_hadamard_transform's reference uses), numpy-only
def scipy_style_hadamard(n):
    H = np.array([[1]], dtype=np.float64)
    for _ in range(int(math.log2(n))):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H

# reference 2: in-place fast Walsh-Hadamard butterfly (natural order), independent of matrix construction
def fwht(x):
    x = x.clone(); n = x.size(-1); hlen = 1
    while hlen < n:
        x = x.view(*x.shape[:-1], n // (2 * hlen), 2, hlen)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).view(*x.shape[:-3], n)
        hlen *= 2
    return x / math.sqrt(n)

print("== Step 4")
print("matches scipy-style construction:", np.array_equal(hadamard_matrix(16, torch.float64).numpy() * 4, scipy_style_hadamard(16)))
xx = torch.randn(3, 5, 32, dtype=torch.float64)
print("matches butterfly FWHT:", torch.allclose(rotate_activation(xx), fwht(xx), atol=1e-12))
H = hadamard_matrix(8)
print(H * math.sqrt(8))
print(torch.allclose(H @ H.T, torch.eye(8), atol=1e-6))
q, k = torch.randn(3, 32), torch.randn(5, 32)
print(torch.allclose(rotate_activation(q) @ rotate_activation(k).T, q @ k.T, atol=1e-5))
print("bf16 in -> dtype:", rotate_activation(torch.randn(2, 32).bfloat16()).dtype, "| f64 in -> dtype:", rotate_activation(xx).dtype)

# ---------------- Step 5 ----------------
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Attention over a chosen list of KV rows per query, with an attention sink.
    q: [b, s, h, d]   kv: [b, n, d] (one shared head, used as both K and V)
    attn_sink: [h]    topk_idxs: [b, s, k] row indices into kv, -1 = empty slot
    returns [b, s, h, d]"""
    b = q.size(0)
    valid = topk_idxs >= 0                                              # [b, s, k]
    batch = torch.arange(b, device=kv.device).view(b, 1, 1)
    keys = upcast(kv[batch, topk_idxs.clamp(min=0)])                    # [b, s, k, d]: gathered rows
    logits = torch.einsum("bshd,bskd->bshk", upcast(q), keys) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))     # empty slots get no weight
    # the sink is one extra logit per head whose value vector is zero: it only adds exp(sink) to the denominator
    sink = attn_sink.to(logits.dtype).view(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    probs = torch.cat([logits, sink], dim=-1).softmax(dim=-1)[..., :-1]  # [b, s, h, k]
    out = torch.einsum("bshk,bskd->bshd", probs, keys)
    return out.to(q.dtype)


def naive_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    # one query, one head at a time: the formula written out directly
    b, s, h, d = q.shape
    out = torch.zeros_like(q)
    for bi in range(b):
        for t in range(s):
            rows = [i for i in topk_idxs[bi, t].tolist() if i >= 0]
            keys = kv[bi, rows]                                         # [n_valid, d]
            for hi in range(h):
                e = torch.exp(keys @ q[bi, t, hi] * softmax_scale)      # exp(logits), [n_valid]
                w = e / (e.sum() + torch.exp(attn_sink[hi]))
                out[bi, t, hi] = w @ keys
    return out

print("== Step 5")
torch.manual_seed(0)
b, s, h, d, n, k = 2, 5, 3, 8, 7, 4
q, kv, sink = torch.randn(b, s, h, d), torch.randn(b, n, d), torch.randn(h)
idx = torch.randint(0, n, (b, s, k))
idx[0, 0, 2:] = -1          # some empty slots
idx[1, 3, 1:] = -1
print(torch.allclose(sparse_attn(q, kv, sink, idx, d ** -0.5), naive_sparse_attn(q, kv, sink, idx, d ** -0.5), atol=1e-5))

q1 = torch.zeros(1, 1, 1, d); q1[..., 0] = 2.0
kv1 = torch.zeros(1, 1, d); kv1[..., 0] = 1.0
out = sparse_attn(q1, kv1, torch.tensor([0.5]), torch.zeros(1, 1, 1, dtype=torch.long), 1.0)
print(out[0, 0, 0, 0].item(), math.exp(2.0) / (math.exp(2.0) + math.exp(0.5)))

# extra: float64 kept, bf16 path, int32 indices (official passes .int())
print("f64:", sparse_attn(q.double(), kv.double(), sink, idx, 0.3).dtype,
      "| bf16:", sparse_attn(q.bfloat16(), kv.bfloat16(), sink, idx, 0.3).dtype,
      "| int32 idx ok:", torch.allclose(sparse_attn(q, kv, sink, idx.int(), 0.3), sparse_attn(q, kv, sink, idx, 0.3)))
# extra: very negative sink -> ordinary softmax; sink large -> output near 0
plain = torch.softmax(torch.einsum("bshd,bskd->bshk", q, kv[torch.arange(b).view(b,1,1), idx.clamp(min=0)]) * 0.3
                      + torch.where(idx >= 0, 0.0, float('-inf')).unsqueeze(2), -1)
print("sink=-inf equals plain softmax attention:",
      torch.allclose(sparse_attn(q, kv, torch.full((h,), -1e9), idx, 0.3),
                     torch.einsum("bshk,bskd->bshd", plain, kv[torch.arange(b).view(b,1,1), idx.clamp(min=0)]), atol=1e-6))

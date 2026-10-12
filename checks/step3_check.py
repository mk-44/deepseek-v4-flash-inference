from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import math, ast, sys
from functools import lru_cache
import torch
sys.path.insert(0, '.')
from modelargs_check import ModelArgs, tiny_args

# ---- official functions, extracted verbatim from model.py ----
src = open(OFFICIAL / 'model.py').read()
tree = ast.parse(src)
ns = {'torch': torch, 'math': math, 'lru_cache': lru_cache, 'F': torch.nn.functional}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('precompute_freqs_cis', 'apply_rotary_emb'):
        exec(compile(ast.Module([node], []), 'official', 'exec'), ns)
off_freqs, off_apply = ns['precompute_freqs_cis'], ns['apply_rotary_emb']

# ---- our code (as given to the user) ----
def upcast(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))   # theta_j, [dim/2]
    if original_seq_len > 0:
        def pair_index(n_turns):
            return dim * math.log(original_seq_len / (n_turns * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(pair_index(beta_fast)), 0)
        high = min(math.ceil(pair_index(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(angles), angles)

def layer_freqs_cis(args, ratio):
    if ratio:
        return precompute_freqs_cis(args.rope_head_dim, args.max_seq_len, args.original_seq_len,
                                    args.compress_rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
    return precompute_freqs_cis(args.rope_head_dim, args.max_seq_len, 0,
                                args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)

def apply_rope(x, freqs_cis, inverse=False):
    rd = 2 * freqs_cis.size(-1)
    x_pass, x_rot = x[..., :-rd], x[..., -rd:]
    z = torch.view_as_complex(upcast(x_rot).unflatten(-1, (-1, 2)).contiguous())
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 4:
        freqs_cis = freqs_cis.unsqueeze(1)
    x_rot = torch.view_as_real(z * freqs_cis).flatten(-2).to(x.dtype)
    return torch.cat([x_pass, x_rot], dim=-1)

# ---- 1. tables bit-identical to official (real + tiny configs, both table types) ----
for name, a in [('real', ModelArgs()), ('tiny', tiny_args())]:
    for ratio in (0, 4):
        ours = layer_freqs_cis(a, ratio)
        ol, th = (a.original_seq_len, a.compress_rope_theta) if ratio else (0, a.rope_theta)
        ref = off_freqs(a.rope_head_dim, a.max_seq_len, ol, th, a.rope_factor, a.beta_fast, a.beta_slow)
        print(f"{name} ratio={ratio}: table bit-identical={torch.equal(ours, ref)} shape={tuple(ours.shape)} {ours.dtype}")

# ---- 2. apply_rope vs official apply_rotary_emb (float32 and bf16, 3D and 4D) ----
a = tiny_args(); f = layer_freqs_cis(a, 4); rd = a.rope_head_dim
for dt in (torch.float32, torch.bfloat16):
    for shape in [(2, 10, 3, a.head_dim), (2, 10, a.head_dim)]:
        x = torch.randn(shape).to(dt)
        ref = x.clone(); off_apply(ref[..., -rd:], f[:10])
        ours = apply_rope(x, f[:10])
        refi = ref.clone(); off_apply(refi[..., -rd:], f[:10], True)
        oursi = apply_rope(ours, f[:10], inverse=True)
        print(f"{dt} {shape}: fwd equal={torch.equal(ours, ref)}  inverse equal={torch.equal(oursi, refi)}")

# ---- 3. float64 input stays float64 ----
x64 = torch.randn(2, 10, 3, a.head_dim, dtype=torch.float64)
print("float64 kept:", apply_rope(x64, f[:10]).dtype)

# ---- 4. the user-facing check cell ----
args = tiny_args()
f_swa = layer_freqs_cis(args, 0)
f_cmp = layer_freqs_cis(args, 4)
print(f_swa.shape, f_swa.dtype)
x = torch.randn(2, 10, 3, args.head_dim)
y = apply_rope(x, f_cmp[:10])
rd = args.rope_head_dim
print(torch.equal(y[..., :-rd], x[..., :-rd]))
print(torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5))
print(torch.allclose(apply_rope(y, f_cmp[:10], inverse=True), x, atol=1e-5))
q, k = torch.randn(rd), torch.randn(rd)
def score(t, s):
    qt = apply_rope(q.view(1, 1, rd), f_cmp[t:t + 1])
    ks = apply_rope(k.view(1, 1, rd), f_cmp[s:s + 1])
    return (qt * ks).sum().item()
print(score(5, 2), score(105, 102))
e = torch.zeros(1, 2, rd); e[..., 0] = 1.0
r = apply_rope(e, f_swa[:2])
print(r[0, 1, :2], torch.tensor([math.cos(1.0), math.sin(1.0)]))

from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import ast, io, contextlib, torch
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step6_check import tiny_args, upcast, ns as off_ns, ModelArgs

# ================= Step 10a: Gate (as it will be given to the user) =================
class Gate(nn.Module):
    """Router: for every token, picks n_activated_experts of the routed experts and their weights.
    Hash layers (layer_id < n_hash_layers): the experts come from a fixed table, indexed by the token id.
    Other layers: top-k of (scores + bias). The bias only changes WHICH experts, never their weights."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.dim) * args.dim ** -0.5)
        if self.hash:
            # stand-in table until real weights: every token id gets k distinct random experts
            table = torch.rand(args.vocab_size, args.n_routed_experts).argsort(dim=-1)[:, :self.topk]
            self.register_buffer("tid2eid", table.int())                       # [vocab, k]
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.randn(args.n_routed_experts) * 0.1)   # [E]

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        # x: [T, dim] (all tokens, flattened), input_ids: [T]
        scores = F.softplus(F.linear(upcast(x), upcast(self.weight))).sqrt()    # [T, E]: sqrt(softplus(logits))
        if self.hash:
            indices = self.tid2eid[input_ids].long()                           # [T, k]: fixed by the token id
        else:
            indices = (scores + self.bias).topk(self.topk, dim=-1).indices     # [T, k]: bias only for choosing
        weights = scores.gather(1, indices)                                     # [T, k]: from the UNbiased scores
        weights = weights / weights.sum(dim=-1, keepdim=True) * self.route_scale
        return weights, indices


@torch.no_grad()
def naive_gate(gate: Gate, x: torch.Tensor, input_ids: torch.Tensor):
    # one token at a time, straight from the formulas
    all_w, all_idx = [], []
    for t in range(x.size(0)):
        s = torch.sqrt(F.softplus(gate.weight @ x[t]))                          # [E]: affinity to every expert
        if gate.hash:
            chosen = gate.tid2eid[input_ids[t]].long()                          # [k]
        else:
            chosen = torch.argsort(s + gate.bias, descending=True)[:gate.topk]  # [k]
        w = s[chosen] / s[chosen].sum() * gate.route_scale                      # [k], sums to route_scale
        all_w.append(w); all_idx.append(chosen)
    return torch.stack(all_w), torch.stack(all_idx)

# ================= Step 10b: Expert + MoE =================
class Expert(nn.Module):
    """One SwiGLU feed-forward: w2(silu(w1 x) * (w3 x)), with both inner values clamped."""
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)       # gate projection (goes through silu)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)       # down projection
        self.w3 = nn.Linear(dim, inter_dim, bias=False)       # up projection (the linear part)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        # x: [n, dim], weights: [n, 1] routing weight of each token (None for the shared expert)
        dtype = x.dtype
        gate = upcast(self.w1(x))                                 # [n, inter]
        up = upcast(self.w3(x))                                   # [n, inter]
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)  # linear part: [-10, 10]
            gate = gate.clamp(max=self.swiglu_limit)              # gate part: only capped from above
        h = F.silu(gate) * up                                     # [n, inter]
        if weights is not None:
            h = weights * h                                       # routing weight goes in before w2
        return self.w2(h.to(dtype))                               # [n, dim]


class MoE(nn.Module):
    """Routed experts (each token uses n_activated_experts of them) + 1 shared expert (every token)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
                                      for _ in range(args.n_routed_experts)])
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b, s, dim], input_ids: [b, s]
        shape = x.shape
        x = x.view(-1, self.dim)                                          # [T, dim], T = b * s
        weights, indices = self.gate(x, input_ids.flatten())              # [T, k], [T, k]
        y = torch.zeros_like(x, dtype=torch.promote_types(x.dtype, torch.float32))   # sum in >= fp32
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i, expert in enumerate(self.experts):                         # expert by expert, all its tokens at once
            if counts[i] == 0:
                continue
            idx, slot = torch.where(indices == i)                         # tokens that chose expert i, and in which slot
            y[idx] += expert(x[idx], weights[idx, slot, None])            # [n_i, dim]
        y += self.shared_experts(x)                                       # every token, weight 1
        return y.type_as(x).view(shape)


@torch.no_grad()
def naive_moe(moe: MoE, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # token by token, straight from the formulas: out = shared(x) + sum over chosen e of w_e * expert_e(x)
    L = moe.shared_experts.swiglu_limit
    def ffn(e, v):
        g = (e.w1.weight @ v).clamp(max=L)                    # gate part, capped above
        u = (e.w3.weight @ v).clamp(-L, L)                    # linear part
        return e.w2.weight @ (g * torch.sigmoid(g) * u)       # silu(g) = g * sigmoid(g)
    out = torch.zeros_like(x)
    for bi in range(x.size(0)):
        for t in range(x.size(1)):
            v = x[bi, t]
            w, chosen = naive_gate(moe.gate, v[None], input_ids[bi, t][None])
            y = ffn(moe.shared_experts, v)
            for e, we in zip(chosen[0].tolist(), w[0]):
                y = y + we * ffn(moe.experts[e], v)
            out[bi, t] = y
    return out

# ================= official Gate / Expert / MoE (verbatim, world_size 1) =================
src = open(OFFICIAL / 'model.py').read(); tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in ('Gate', 'Expert', 'MoE'):
        exec(compile(ast.Module([node], []), 'official', 'exec'), off_ns)

def official_args(args):
    return off_ns['ModelArgs'](dim=args.dim, vocab_size=args.vocab_size, n_routed_experts=args.n_routed_experts,
        n_activated_experts=args.n_activated_experts, n_hash_layers=args.n_hash_layers, moe_inter_dim=args.moe_inter_dim,
        score_func=args.score_func, route_scale=args.route_scale, swiglu_limit=args.swiglu_limit, expert_dtype=None)

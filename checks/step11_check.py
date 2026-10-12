from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import json, ast, io, contextlib, copy, torch
from contextlib import contextmanager
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    import step6_check as s6, step8_check as s8, step9_check as s9, step10_check as s10

# every definition cell of the user's saved notebook (imports, functions, classes)
U = load_notebook_defs()

STEP11 = r'''
class Block(nn.Module):
    """One transformer layer: attention, then MoE, each wrapped in mHC (4 residual streams)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult                        # 24 = 4 pre + 4 post + 16 comb
        hc_dim = args.hc_mult * args.dim                                  # the 4 streams side by side
        # random stand-ins until real weights (the official code creates these empty, in float32)
        self.hc_attn_fn = nn.Parameter(torch.randn(mix_hc, hc_dim) * 0.02)
        self.hc_attn_base = nn.Parameter(torch.randn(mix_hc) * 0.5)
        self.hc_attn_scale = nn.Parameter(torch.randn(3) * 0.5 + 1)
        self.hc_ffn_fn = nn.Parameter(torch.randn(mix_hc, hc_dim) * 0.02)
        self.hc_ffn_base = nn.Parameter(torch.randn(mix_hc) * 0.5)
        self.hc_ffn_scale = nn.Parameter(torch.randn(3) * 0.5 + 1)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b, s, hc, dim] the 4 streams, input_ids: [b, s]
        residual = x
        x, post, comb = hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                               self.norm_eps, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)   # [b, s, dim]
        x = self.attn(self.attn_norm(x))                                  # [b, s, dim]
        x = hc_post(x, residual, post, comb)                              # [b, s, hc, dim]

        residual = x
        x, post, comb = hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                               self.norm_eps, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)   # [b, s, dim]
        x = self.ffn(self.ffn_norm(x), input_ids)                         # [b, s, dim]
        return hc_post(x, residual, post, comb)                           # [b, s, hc, dim]


def hc_head(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            norm_eps: float, hc_eps: float) -> torch.Tensor:
    """Merge the 4 streams into one vector per token, before the final norm and head.
    Same as hc_pre, but only the pre weights: no post, no comb, because no sublayer comes after it."""
    shape, dtype = x.shape, x.dtype
    x = upcast(x.flatten(2))                                              # [b, s, hc * dim]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)     # [b, s, 1]
    mixes = F.linear(x, hc_fn) * rsqrt                                    # [b, s, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps              # [b, s, hc]
    y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)                    # [b, s, dim]
    return y.to(dtype)


class Transformer(nn.Module):
    """Whole model: embed -> 4 streams -> blocks -> merge streams -> norm -> head -> logits."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(layer_id, args) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.randn(args.hc_mult, hc_dim) * 0.02)   # [hc, hc * dim]
        self.hc_head_base = nn.Parameter(torch.randn(args.hc_mult) * 0.5)          # [hc]
        self.hc_head_scale = nn.Parameter(torch.randn(1) * 0.5 + 1)                # [1]

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, return_all_logits: bool = False) -> torch.Tensor:
        # input_ids: [b, s] -> logits [b, vocab] for the last position (like official), or [b, s, vocab]
        h = self.embed(input_ids)                                         # [b, s, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)                  # [b, s, hc, dim]: 4 identical copies
        for layer in self.layers:
            h = layer(h, input_ids)                                       # [b, s, hc, dim]
        h = hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm_eps, self.hc_eps)  # [b, s, dim]
        h = self.norm(h)                                                  # [b, s, dim]
        if not return_all_logits:
            h = h[:, -1]                                                  # [b, dim]: only the last position
        return self.head(upcast(h))                                       # [b, vocab] or [b, s, vocab]
'''
exec(STEP11, U)

# ---- official Block / ParallelEmbedding / ParallelHead / Transformer, verbatim ----
off = s6.ns
off.update(contextmanager=contextmanager, dist=dist, hc_split_sinkhorn=s9.hc_split_sinkhorn)
tree = ast.parse(open(OFFICIAL / 'model.py').read())
for node in tree.body:
    if (isinstance(node, ast.ClassDef) and node.name in ('Block', 'ParallelEmbedding', 'ParallelHead', 'Transformer')) or \
       (isinstance(node, ast.FunctionDef) and node.name == 'set_dtype'):
        exec(compile(ast.Module([node], []), 'official', 'exec'), off)

def official_model(args, b):
    oargs = off['ModelArgs'](
        max_batch_size=b, max_seq_len=args.max_seq_len, dtype='bf16', scale_fmt='ue8m0', expert_dtype=None, scale_dtype='fp32',
        vocab_size=args.vocab_size, dim=args.dim, moe_inter_dim=args.moe_inter_dim, n_layers=args.n_layers,
        n_hash_layers=args.n_hash_layers, n_mtp_layers=0, n_heads=args.n_heads, n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts, n_activated_experts=args.n_activated_experts, score_func=args.score_func,
        route_scale=args.route_scale, swiglu_limit=args.swiglu_limit, q_lora_rank=args.q_lora_rank, head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim, norm_eps=args.norm_eps, o_groups=args.o_groups, o_lora_rank=args.o_lora_rank,
        window_size=args.window_size, compress_ratios=args.compress_ratios, compress_rope_theta=args.compress_rope_theta,
        original_seq_len=args.original_seq_len, rope_theta=args.rope_theta, rope_factor=args.rope_factor,
        beta_fast=args.beta_fast, beta_slow=args.beta_slow, index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim, index_topk=args.index_topk, hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps)
    om = off['Transformer'](oargs).float()           # its __init__ sets global default_dtype=bf16; .float() undoes that
    off['default_dtype'] = torch.float32
    return om

if __name__ == '__main__':
    args = U['tiny_args']()
    # hc_head vs official ParallelHead.hc_head
    torch.manual_seed(0)
    x = torch.randn(2, 5, args.hc_mult, args.dim); fn = torch.randn(4, 4 * args.dim) * 0.02; sc = torch.randn(1); ba = torch.randn(4)
    ph = off['ParallelHead'](args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    print('hc_head vs official:', torch.equal(U['hc_head'](x, fn, sc, ba, args.norm_eps, args.hc_eps), ph.hc_head(x, fn, sc, ba)),
          '| bf16:', torch.equal(U['hc_head'](x.bfloat16(), fn, sc, ba, args.norm_eps, args.hc_eps), ph.hc_head(x.bfloat16(), fn, sc, ba)))
    # whole model vs official, last-position and all-position logits
    get_logits_last = off['ParallelHead'].get_logits
    for s in (1, 5, 8, 9, 16, 17, 37, 64):
        torch.manual_seed(s)
        b = 2
        model = U['Transformer'](args)
        om = official_model(args, b)
        missing = om.load_state_dict(model.state_dict(), strict=True)
        ids = torch.randint(0, args.vocab_size, (b, s))
        last_ok = torch.equal(model(ids), om(ids, 0))
        off['ParallelHead'].get_logits = lambda self, h: F.linear(h.float(), self.weight)    # all positions
        all_ok = torch.equal(model(ids, return_all_logits=True), om(ids, 0))
        off['ParallelHead'].get_logits = get_logits_last
        print(f's={s:2d}: last-position logits identical {last_ok} | all positions identical {all_ok}')
    print('our keys == official keys:', list(model.state_dict().keys()) == list(om.state_dict().keys()),
          '| same set:', set(model.state_dict()) == set(om.state_dict()))

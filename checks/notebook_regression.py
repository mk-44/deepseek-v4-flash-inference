# Full regression: every module in the user's SAVED arch.ipynb vs the official model.py (with the official FP8/FP4 rounding once the notebook has it).
# run(U) also works on another namespace, e.g. the notebook's globals() (checks/gates.py, gate M).
from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import json, io, contextlib, copy, torch
from torch import nn
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    import step3_check as s3, step45_check as s45, step6_check as s6, step7_check as s7, step8_check as s8, step9_check as s9, step10_check as s10


def run(U, verbose=True):
    """Returns [(check name, passed), ...]; prints one PASS / FAIL line per check when verbose."""
    args = U['tiny_args'](); real = U['ModelArgs']()
    eq = torch.equal
    results = []
    def report(name, ok, extra=''):
        results.append((name, ok))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name} {extra}")

    # 1) RoPE tables
    ok = True
    for a in (args, real):
        for r in (0, 4):
            p = (a.rope_head_dim, a.max_seq_len, a.original_seq_len if r else 0, a.compress_rope_theta if r else a.rope_theta,
                 a.rope_factor, a.beta_fast, a.beta_slow)
            ok &= eq(U['layer_freqs_cis'](a, r), s3.off_freqs(*p)) and eq(U['precompute_theta_cis'](*p), s3.off_freqs(*p))
    report('RoPE tables (tiny + real, SWA + compressed)', ok)

    # 2) apply_rope: 3-D and 4-D, forward and inverse, fp32 and bf16
    torch.manual_seed(0); ok = True
    f = U['layer_freqs_cis'](args, 4)[:7]; rd = 2 * f.size(-1)
    for shape in ((2, 7, 64), (2, 7, 4, 64)):
        for dt in (torch.float32, torch.bfloat16):
            for inv in (False, True):
                x = torch.randn(*shape).to(dt); ref = x.clone(); s3.off_apply(ref[..., -rd:], f, inv)
                ok &= eq(U['apply_rope'](x, f, inv), ref)
    report('apply_rope vs official apply_rotary_emb', ok)

    # 3) Hadamard rotation
    torch.manual_seed(0); x = torch.randn(3, 5, 32)
    H = torch.tensor(s45.scipy_style_hadamard(32), dtype=torch.float32) / 32 ** 0.5
    report('rotate_activation', eq(U['rotate_activation'](x), s45.rotate_activation(x)) and torch.allclose(U['rotate_activation'](x), x @ H, atol=1e-6))

    # 4) sparse_attn: float64 vs naive, bf16 vs canonical
    torch.manual_seed(0)
    q, kv, sink = torch.randn(2, 5, 3, 8, dtype=torch.float64), torch.randn(2, 7, 8, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
    idx = torch.randint(0, 7, (2, 5, 4)); idx[0, 0, 2:] = -1; idx[1, 3, :] = -1
    d64 = (U['sparse_attn'](q, kv, sink, idx, 0.3) - s45.naive_sparse_attn(q, kv, sink, idx, 0.3)).abs().max().item()
    qb, kb = q.float().bfloat16(), kv.float().bfloat16()
    report('sparse_attn', d64 < 1e-12 and eq(U['sparse_attn'](qb, kb, sink.float(), idx, 0.3), s45.sparse_attn(qb, kb, sink.float(), idx, 0.3)), f'(float64 diff {d64:.0e})')

    with torch.no_grad():
        # 5) Compressor
        ok = True
        for ratio, rotate, hd in ((4, False, args.head_dim), (4, True, args.index_head_dim), (16, False, args.head_dim)):
            fr = U['layer_freqs_cis'](args, ratio)
            for s in (8, 19, 37, 64):
                if s < ratio: continue
                torch.manual_seed(s); comp = U['Compressor'](args, ratio, hd, rotate); x = torch.randn(2, s, args.dim)
                ok &= eq(comp(x, fr), s6.official_compress(comp, args, x, fr))
        report('Compressor (CSA, CSA+rotate, HCA; s = 8, 19, 37, 64)', ok)

        # 6) Indexer
        ok = True; fr = U['layer_freqs_cis'](args, 4)
        for s in (3, 4, 8, 19, 40, 77):
            torch.manual_seed(s); ix = U['Indexer'](args); x, qr = torch.randn(2, s, args.dim), torch.randn(2, s, args.q_lora_rank)
            ok &= eq(ix(x, qr, fr, offset=s), s7.official_indexer(ix, args, x, qr, fr, offset=s))
        report('Indexer (s = 3, 4, 8, 19, 40, 77)', ok)

        # 7) Attention (strict load_state_dict into the official class)
        ok = True
        for s in (1, 5, 8, 9, 16, 17, 37, 64):
            torch.manual_seed(s); x = torch.randn(2, s, args.dim)
            for layer_id in (0, 2, 3):
                at = U['Attention'](layer_id, args)
                ok &= eq(at(x), s8.official_attention(at, layer_id, args, x))
        report('Attention (SWA/CSA/HCA layers, s = 1..64)', ok)

    # 8) mHC
    torch.manual_seed(0); hc, d = args.hc_mult, args.dim
    fn, sc, ba = torch.randn(24, hc * d) * 0.02, torch.randn(3) * 0.5 + 1, torch.randn(24) * 0.5
    ob = s9.OffBlock(); ob.norm_eps, ob.hc_mult, ob.hc_sinkhorn_iters, ob.hc_eps = args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 5, hc, d).to(dt); out = torch.randn(2, 5, d).to(dt)
        y1, p1, c1 = U['hc_pre'](x, fn, sc, ba, args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        y2, p2, c2 = ob.hc_pre(x, fn, sc, ba)
        ok &= eq(y1, y2) and eq(p1, p2) and eq(c1, c2) and eq(U['hc_post'](out, x, p1, c1), ob.hc_post(out, x, p2, c2))
    report('mHC hc_pre / hc_post (fp32 + bf16)', ok)

    # 9) Gate / Expert / MoE (strict load into official classes), fp32 + bf16, clamp active at x*20
    oargs = s10.official_args(args); ok = True
    def bf16_like_official(m):
        for n, p in m.named_parameters():
            if p.is_floating_point() and not n.endswith('gate.bias') and n != 'bias':
                p.data = p.data.bfloat16()
        return m
    torch.manual_seed(0); x = torch.randn(3, 9, args.dim); ids = torch.randint(0, args.vocab_size, (3, 9))
    for layer_id in (0, 2, 3, 5):
        g = U['Gate'](layer_id, args); og = s10.off_ns['Gate'](layer_id, oargs); og.load_state_dict(g.state_dict(), strict=True)
        w1, i1 = g(x.view(-1, args.dim), ids.flatten()); w2, i2 = og(x.view(-1, args.dim), ids.flatten())
        ok &= eq(w1, w2) and eq(i1, i2.long())
        for scale in (1.0, 20.0):
            m = U['MoE'](layer_id, args); om = s10.off_ns['MoE'](layer_id, oargs); om.load_state_dict(m.state_dict(), strict=True)
            ok &= eq(m(x * scale, ids), om(x * scale, ids))
            mb, omb = bf16_like_official(copy.deepcopy(m)), bf16_like_official(copy.deepcopy(om))
            ok &= eq(mb((x * scale).bfloat16(), ids), omb((x * scale).bfloat16(), ids))
    report('Gate + Expert + MoE (hash + learned, fp32 + bf16, clamp active)', ok)
    return results


if __name__ == '__main__':
    # every definition cell of the user's saved notebook (imports, functions, classes)
    results = run(load_notebook_defs())
    print(f'\n{sum(ok for _, ok in results)}/{len(results)} passed')

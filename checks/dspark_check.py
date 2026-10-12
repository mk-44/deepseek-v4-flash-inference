"""Phase 2, Steps 20-22: the DSpark drafter of the 0731 checkpoint (plan item 8).

DSpark guesses a block of 5 tokens in one pass, from the main model's hidden states at layers 40-42 (tiny: 3-5).
The official code (0731 model.py) drafts after one-token decode steps; the loop that checks the guesses is not in
the official repo (vLLM / SGLang have it), so it comes later (Phase 8).
A. tiny, float32, vs official 0731: prefill P (the drafter only fills its memory), then D decode steps, each one
   followed by a draft. Main logits, the hidden states the drafter reads, draft ids, draft logits, confidence and
   the drafter's 3 window memories are identical (torch.equal), at temperature 0 and 1 (same random numbers).
   Our weights load into the official model with only its aliases missing (mtp.N.embed.weight and
   mtp.N.head.weight point at embed.weight and head.weight; the checkpoint does not store them).
B. tiny, float64: an extend of s main tokens, then a draft == the same tokens one at a time, each followed by a
   draft (the official code drafts only after one token; ours also after an extend, which checking guesses needs).
C. snapshot / restore also covers the drafter's memory: prefill, clone, wrong tokens (main + draft), restore,
   right tokens == the direct run (draft outputs and every memory tensor).
D. the real config: ModelArgs() defaults equal the 0731 inference/config.json in every field it sets (this catches a
   mistyped default), and the model built on the meta device has exactly the names in the 0731 checkpoint index
   (model.safetensors.index.json without the .scale tensors), main model and drafter.
E. real size, DSpark block level (layer 43 = stage 0, 8 experts), float32: prefill 300 (memory only), then 4 draft
   steps, identical to official. ~3 GiB.
"""
import json
import resource
import sys
from _paths import ROOT, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'reference'))
from load_official import load_official, official_args

ALIASES = [f'mtp.{i}.{m}.weight' for i in range(3) for m in ('embed', 'head')]


def memory(state):
    out = []
    for layer in state.layers + state.mtp:
        out.append(layer.window)
        for c in (layer.compressor, layer.indexer):
            if c is not None:
                out += [c.entries, c.kv, c.score]
    return out


def run(U, real_size=True):
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    off = load_official('0731')
    args = U['tiny_args'](dspark=True)
    w, b, D = args.window_size, 2, 12

    print('A. tiny model + drafter vs official 0731 (float32): prefill P, then 12 decode steps, each with a draft')
    for P in (1, 3, 8, 9, 17):
        for T in (0.0, 1.0):
            torch.manual_seed(P)
            model = Transformer(args)
            theirs = off.Transformer(official_args(off, args, max_batch_size=b, n_mtp_layers=args.n_mtp_layers)).float()
            missing, unexpected = theirs.load_state_dict(model.state_dict(), strict=False)
            names_ok = sorted(missing) == sorted(ALIASES) and not unexpected
            for m in theirs.mtp:
                m.temperature = T
            ids = torch.randint(0, args.vocab_size, (b, P + D))
            state = SequenceState(args, b)
            lo, ho = model(ids[:, :P], state, return_hidden=True)
            _, lt, ht = theirs(ids[:, :P], 0)
            ok = [torch.equal(lo, lt), torch.equal(ho, ht)]
            ok.append(model.forward_spec(lo.argmax(-1), ho, state, T) is None and theirs.forward_spec(lo.argmax(-1), ht, 0) is None)
            for t in range(P, P + D):
                lo, ho = model(ids[:, t:t + 1], state, return_hidden=True)
                _, lt, ht = theirs(ids[:, t:t + 1], t)
                ok += [torch.equal(lo, lt), torch.equal(ho, ht)]
                torch.manual_seed(1000 + t)
                ours = model.forward_spec(lo.argmax(-1), ho, state, T)
                torch.manual_seed(1000 + t)
                ref = theirs.forward_spec(lo.argmax(-1), ht, t)
                ok += [torch.equal(x, y) for x, y in zip(ours, ref)]
            rings = all(torch.equal(s.window, m.attn.kv_cache[:b, :w]) for s, m in zip(state.mtp, theirs.mtp))
            print(f'  P = {P:2d}  T = {T}: names {names_ok}  all {len(ok)} outputs identical {all(ok)}  '
                  f'drafter memory identical {rings}  last guess {ours[0][0].tolist()}')

    print('B. extend s main tokens, then draft == the same tokens one at a time (float64)')
    args64 = U['tiny_args'](dspark=True)
    args64.index_n_heads = 64
    torch.manual_seed(0)
    model = Transformer(args64).double()
    worst, same_ids, same_mem = 0.0, True, True
    for P in (1, 3, 4, 8, 9, 17):
        for s in (2, 3, 5, 8, 9, 16):
            ids = torch.randint(0, args64.vocab_size, (b, P + s))
            one_go = SequenceState(args64, b, dtype=torch.float64)
            one_by_one = SequenceState(args64, b, dtype=torch.float64)
            for st in (one_go, one_by_one):
                lo, ho = model(ids[:, :P], st, return_hidden=True)
                model.forward_spec(lo.argmax(-1), ho, st, 0.0)
            lo, ho = model(ids[:, P:], one_go, return_hidden=True)
            ra = model.forward_spec(lo.argmax(-1), ho, one_go, 0.0)
            for t in range(P, P + s):
                lo, ho = model(ids[:, t:t + 1], one_by_one, return_hidden=True)
                rb = model.forward_spec(lo.argmax(-1), ho, one_by_one, 0.0)
            same_ids &= torch.equal(ra[0], rb[0])
            for x, y in zip(ra[1:], rb[1:]):
                worst = max(worst, ((x - y).abs().max() / y.abs().max()).item())
            same_mem &= all(torch.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in zip(memory(one_go), memory(one_by_one)))
    print(f'  36 cases: guesses identical {same_ids}  worst relative difference (logits, confidence) {worst:.1e}  '
          f'memory equal {same_mem}')

    print('C. snapshot -> wrong tokens -> restore -> right tokens, vs right tokens directly (float32)')
    torch.manual_seed(0)
    model = Transformer(args)
    print(f'  SequenceState.tensors(): {len(SequenceState(args, b).tensors())} tensors (24 main + 3 drafter)')

    def steps(state, toks):
        outs = []
        for t in range(toks.shape[1]):
            lo, ho = model(toks[:, t:t + 1], state, return_hidden=True)
            outs += list(model.forward_spec(lo.argmax(-1), ho, state, 0.0))
        return outs

    for P in (3, 9, 17):
        for k in (1, 6):
            right = torch.randint(0, args.vocab_size, (b, P + 6))
            wrong = torch.randint(0, args.vocab_size, (b, k))
            direct, state = SequenceState(args, b), SequenceState(args, b)
            for st in (direct, state):
                lo, ho = model(right[:, :P], st, return_hidden=True)
                model.forward_spec(lo.argmax(-1), ho, st, 0.0)
            direct_out = steps(direct, right[:, P:])
            snap = state.clone()
            steps(state, wrong)
            state.restore(snap)
            out = steps(state, right[:, P:])
            same = all(torch.equal(x, y) for x, y in zip(out, direct_out))
            same &= state.pos == direct.pos and all(torch.equal(x, y) for x, y in zip(memory(state), memory(direct)))
            print(f'  P = {P:2d}  {k} wrong tokens: identical to the direct run {same}')

    print('D. real config: ModelArgs() defaults vs the 0731 config.json, and weight names (meta device) vs the 0731 checkpoint index')
    cfg = json.load(open(ROOT / 'reference' / 'official' / '0731' / 'inference' / 'config.json'))
    default, loaded = U['ModelArgs'](), U['ModelArgs'](**cfg)
    diff = {k: (getattr(default, k), getattr(loaded, k)) for k in cfg if getattr(default, k) != getattr(loaded, k)}
    print(f'  ModelArgs() equals config.json in all {len(cfg)} fields: {not diff}  {diff if diff else ""}')
    with torch.device('meta'):
        real = Transformer(U['ModelArgs']())
    ours = set(real.state_dict())
    index = json.load(open(ROOT / 'reference' / 'official' / '0731' / 'model.safetensors.index.json'))['weight_map']
    ckpt = {k for k in index if not k.endswith('.scale')}
    print(f'  ours {len(ours):,}  checkpoint {len(ckpt):,}  only ours {sorted(ours - ckpt)[:3]}  '
          f'only checkpoint {sorted(ckpt - ours)[:3]}  identical {ours == ckpt}')
    del real
    U['precompute_theta_cis'].cache_clear()   # the meta build cached meta RoPE tables (the cache key has no device)

    if real_size:
        print('E. real size, DSpark block (layer 43, stage 0, 8 experts), float32: prefill 300, then 4 draft steps')
        rargs = U['ModelArgs'](n_routed_experts=8)
        oargs = official_args(off, rargs)
        torch.manual_seed(43)
        ours = U['DSparkBlock'](43, rargs)
        theirs = off.DSparkBlock(43, oargs).float()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        k, P, wr = rargs.dspark_block_size, 300, rargs.window_size
        state = SequenceState(rargs, 1).mtp[0]
        ids = torch.randint(0, rargs.vocab_size, (1, k))
        with torch.inference_mode():
            x = torch.randn(1, k, rargs.hc_mult, rargs.dim)
            main_x = torch.randn(1, P, rargs.dim)
            ok = [torch.equal(ours(x, ids, state, 0, main_x), theirs(x, 0, ids, main_x))]
            for t in range(P, P + 4):
                x = torch.randn(1, k, rargs.hc_mult, rargs.dim)
                main_x = torch.randn(1, 1, rargs.dim)
                ok.append(torch.equal(ours(x, ids, state, t, main_x), theirs(x, t, ids, main_x)))
        mem = torch.equal(state.window, theirs.attn.kv_cache[:1, :wr])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f'  outputs identical {ok}  memory identical {mem}  peak {peak / 2**30 if sys.platform == "darwin" else peak / 2**20:.1f} GiB')


if __name__ == '__main__':
    U = load_notebook_defs()
    if 'DSparkBlock' not in U:
        print('the notebook has no DSpark drafter yet (Steps 20-22)')
    else:
        run(U, real_size='--no-real-size' not in sys.argv)

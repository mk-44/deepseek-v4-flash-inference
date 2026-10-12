"""Phase 2, Step 15: extend (s new tokens at start_pos > 0, any s >= 1).

The official code has no extend: it prefills at position 0, then decodes one token at a time. So:
A. tiny Transformer, float64: prefill P, then extends of T tokens up to 64 tokens, equals one prefill of
   all 64 (logits at every position, ~1e-15); 9 P x 10 T. index_n_heads = 64, like the real model: with
   the tiny config's 4 heads, relu often zeroes every head, two entries tie at exactly 0, and topk may
   keep a different one of them in the two runs (the official code does the same).
B. tiny, float64: the memory after one extend of T tokens equals the memory after the same T tokens one
   at a time (every buffer: window ring, entries, compressor tails, including rows not yet in use).
C. tiny, float64: 30 random chunk schedules (chunks of 1 to 40 tokens) vs one prefill.
D. extend with s = 1 is decode: reruns checks/decode_check.py (official decode, bit for bit).
E. real size, Block level, float64: one prefill of 2,304 vs the same tokens split into chunks
   (2048+256, 1000+1304, 2047+1+128+128, 130+129+2045), layers 0 / 2 / 3. Peak ~10 GiB.
"""
import random
import resource
import sys
import time
from _paths import load_notebook_defs
import torch
import decode_check


def memory(state):
    out = []
    for layer in state.layers:
        out.append(layer.window)
        for c in (layer.compressor, layer.indexer):
            if c is not None:
                out += [c.entries, c.kv, c.score]
    return out


def run(U, real_size=True):
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    args = U['tiny_args']()
    args.index_n_heads = 64
    b, L = 2, 64

    print('A. prefill P, then extends of T tokens, vs one prefill of all 64 (float64)')
    torch.manual_seed(0)
    model = Transformer(args).double()
    ids = torch.randint(0, args.vocab_size, (b, L))
    full = model(ids, return_all_logits=True)
    for P in (1, 3, 4, 5, 8, 9, 16, 17, 33):
        rels = []
        for T in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17):
            state = SequenceState(args, b, dtype=torch.float64)
            chunks = [model(ids[:, :P], state, return_all_logits=True)]
            while state.pos < L:
                chunks.append(model(ids[:, state.pos:state.pos + T], state, return_all_logits=True))
            rels.append(((torch.cat(chunks, 1) - full).abs().max() / full.abs().max()).item())
        print(f'  P = {P:2d}  T = 1..17: worst relative difference {max(rels):.1e}')

    print('B. memory after one extend of T tokens vs after the same tokens one at a time (float64)')
    torch.manual_seed(1)
    model = Transformer(args).double()
    bad = []
    for P in (1, 3, 4, 5, 8, 9, 16, 17, 33):
        for T in (2, 3, 4, 5, 7, 8, 9, 16, 17):
            ids = torch.randint(0, args.vocab_size, (b, P + T))
            one_go = SequenceState(args, b, dtype=torch.float64)
            one_by_one = SequenceState(args, b, dtype=torch.float64)
            model(ids[:, :P], one_go)
            model(ids[:, :P], one_by_one)
            model(ids[:, P:], one_go)
            for t in range(P, P + T):
                model(ids[:, t:t + 1], one_by_one)
            if not all(torch.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in zip(memory(one_go), memory(one_by_one))):
                bad.append((P, T))
    print(f'  81 cases, memory equal in all: {not bad}  {bad[:5] if bad else ""}')

    print('C. random chunk schedules vs one prefill (float64)')
    torch.manual_seed(2)
    model = Transformer(args).double()
    rng = random.Random(0)
    worst = 0.0
    for _ in range(30):
        n = rng.randint(20, 120)
        ids = torch.randint(0, args.vocab_size, (b, n))
        full = model(ids, return_all_logits=True)
        state = SequenceState(args, b, dtype=torch.float64)
        chunks = []
        while state.pos < n:
            T = rng.choice((1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17, 31, 40))
            chunks.append(model(ids[:, state.pos:state.pos + T], state, return_all_logits=True))
        worst = max(worst, ((torch.cat(chunks, 1) - full).abs().max() / full.abs().max()).item())
    print(f'  30 schedules, worst relative difference {worst:.1e}')

    print('D. extend with s = 1 vs the official decode (checks/decode_check.py)')
    decode_check.run(U, real_size=real_size)

    if not real_size:
        return
    print('E. real size, Block level: one prefill of 2,304 vs chunks (float64)')
    args = U['ModelArgs'](n_routed_experts=8)
    L = 2304
    for layer_id in (0, 2, 3):
        torch.manual_seed(layer_id)
        block = U['Block'](layer_id, args).double()
        x = torch.randn(1, L, args.hc_mult, args.dim, dtype=torch.float64)
        ids = torch.randint(0, args.vocab_size, (1, L))
        with torch.inference_mode():
            full = block(x, ids)
            for splits in ((2048, 256), (1000, 1304), (2047, 1, 128, 128), (130, 129, 2045)):
                st = SequenceState(args, 1, dtype=torch.float64).layers[layer_id]
                outs, p, t0 = [], 0, time.perf_counter()
                for n in splits:
                    outs.append(block(x[:, p:p + n], ids[:, p:p + n], st, p))
                    p += n
                rel = ((torch.cat(outs, 1) - full).abs().max() / full.abs().max()).item()
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20)
                print(f'  layer {layer_id}  chunks {str(splits):22s} max relative difference {rel:.1e}  '
                      f'({time.perf_counter() - t0:.1f} s, peak {peak:.1f} GiB)')
        del block


if __name__ == '__main__':
    U = load_notebook_defs()
    if 'window_extend_idxs' not in U or not hasattr(U['Compressor'], 'extend'):
        print('the notebook has no extend yet (Step 15)')
    else:
        run(U)

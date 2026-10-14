"""Phase 2, Step 14: decode (one token at start_pos > 0) vs the official code.

A. tiny Transformer: prefill P, then decode 40 tokens one at a time; logits must equal the official
   model's at every step (float32, torch.equal), and the memory must equal its buffers at the end.
B. tiny Transformer in float64: prefill P + decode 40 must equal one prefill of all P + 40 tokens
   (up to rounding, ~1e-15). index_topk is raised to 64 so the indexer keeps every entry: with only
   4 indexer heads, two entries can tie at exactly 0 (relu zeroes every head) and topk may pick a
   different one of them in prefill and in decode. The official code behaves the same way.
C. real size, Block level (layers 0 / 2 / 3, 8 experts): prefill 125 + decode 6 (window fills,
   first HCA entry) and prefill 2300 + decode 8 (CSA + HCA entries, real top-512 picks); outputs and
   memory equal to the official Block.
"""
import sys, time
from _paths import ROOT, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'checks'))
from load_official import load_official, official_args


def memory_pairs(layer_states, official_attns, w):
    pairs = []
    for st, a in zip(layer_states, official_attns):
        pairs.append((st.window, a.kv_cache[:, :w]))
        if st.compressor is not None:
            pairs += [(st.compressor.entries, a.kv_cache[:, w:]), (st.compressor.kv, a.compressor.kv_state),
                      (st.compressor.score, a.compressor.score_state)]
        if st.indexer is not None:
            pairs += [(st.indexer.entries, a.indexer.kv_cache), (st.indexer.kv, a.indexer.compressor.kv_state),
                      (st.indexer.score, a.indexer.compressor.score_state)]
    return pairs


def run(U, real_size=True):
    off = load_official('april')
    args = U['tiny_args']()
    b, D = 2, 40
    print('A. decode vs official, token by token (float32)')
    for P in (1, 3, 4, 5, 8, 9, 16, 17, 33):
        torch.manual_seed(P)
        model = U['Transformer'](args)
        theirs = off.Transformer(official_args(off, args, max_batch_size=b)).float()
        theirs.load_state_dict(model.state_dict(), strict=True)
        ids = torch.randint(0, args.vocab_size, (b, P + D))
        state = U['SequenceState'](args, b)
        same = [torch.equal(model(ids[:, :P], state), theirs(ids[:, :P], 0))]
        for t in range(P, P + D):
            same.append(torch.equal(model(ids[:, t:t + 1], state), theirs(ids[:, t:t + 1], t)))
        mem = all(torch.equal(x, y) for x, y in memory_pairs(state.layers, [l.attn for l in theirs.layers], args.window_size))
        print(f'  P = {P:2d}  logits identical at all {len(same)} steps: {all(same)}  memory identical: {mem}')

    print('B. prefill + decode vs one prefill (float64, index_topk 64)')
    args_b = U['tiny_args']()
    args_b.index_topk = 64
    for P in (1, 4, 5, 9, 17):
        torch.manual_seed(100 + P)
        model = U['Transformer'](args_b).double()
        ids = torch.randint(0, args_b.vocab_size, (b, P + D))
        full = model(ids, return_all_logits=True)
        state = U['SequenceState'](args_b, b, dtype=torch.float64)
        steps = [model(ids[:, :P], state)] + [model(ids[:, t:t + 1], state) for t in range(P, P + D)]
        rel = ((torch.stack(steps, 1) - full[:, P - 1:]).abs().max() / full.abs().max()).item()
        print(f'  P = {P:2d}  max relative difference {rel:.1e}')

    if not real_size:
        return
    print('C. real size, Block level, decode vs official')
    args = U['ModelArgs'](n_routed_experts=8)
    oargs = official_args(off, args)
    for layer_id in (0, 2, 3):
        torch.manual_seed(layer_id)
        ours = U['Block'](layer_id, args)
        for P, D in ((125, 6), (2300, 8)):
            theirs = off.Block(layer_id, oargs).float()
            theirs.load_state_dict(ours.state_dict(), strict=True)
            st = U['SequenceState'](args, 1).layers[layer_id]
            x = torch.randn(1, P + D, args.hc_mult, args.dim)
            ids = torch.randint(0, args.vocab_size, (1, P + D))
            with torch.inference_mode():
                same = [torch.equal(ours(x[:, :P], ids[:, :P], st, 0), theirs(x[:, :P], 0, ids[:, :P]))]
                t0 = time.perf_counter()
                for t in range(P, P + D):
                    same.append(torch.equal(ours(x[:, t:t + 1], ids[:, t:t + 1], st, t), theirs(x[:, t:t + 1], t, ids[:, t:t + 1])))
                ms = (time.perf_counter() - t0) / D * 1000
            mem = all(torch.equal(p, q) for p, q in memory_pairs([st], [theirs.attn], args.window_size))
            print(f'  layer {layer_id}  prefill {P:4d} + decode {D}: outputs identical {all(same)}  memory identical {mem}  ours {ms:.0f} ms/token')
            del theirs


if __name__ == '__main__':
    U = load_notebook_defs()
    if not any(hasattr(U['Compressor'], m) for m in ('decode', 'extend')):   # Step 15 replaces decode with extend
        print('the notebook has no decode yet (Step 14)')
    else:
        run(U)

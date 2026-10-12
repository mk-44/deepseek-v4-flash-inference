"""Phase 2, real-size prefill: the notebook's Block vs the official Block at the real V4-Flash sizes.

Uses reference/load_official.py (the official model.py on CPU, GPU kernels replaced by
reference/shim/). Random weights, 8 experts instead of 256, float32.
- layers 0 (window only, hash routing), 2 (CSA, hash), 3 (HCA, learned routing)
- lengths 1, 127, 128, 129, 600, 2304 (2304 is past 2052 tokens, so the indexer really picks 512 of 576 entries)
Also checks that the RoPE tables are shared, that sparse_attn gives the same bits for any chunk size,
and, if the notebook has a Transformer, the tiny whole model vs the official one.
Once the notebook has SequenceState (Step 13), it also compares the memory a prefill writes
(window ring, compressed entries, indexer keys, compressor tails) with the official buffers.
"""
import inspect, sys, time, resource
from _paths import ROOT, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'reference'))
from load_official import load_official, official_args

U = load_notebook_defs()
off = load_official('april')
ModelArgs, Block = U['ModelArgs'], U['Block']
args = ModelArgs(n_routed_experts=8)

# RoPE tables shared (one per kind of layer)
a2, a3, a4 = (U['Attention'](i, args) for i in (2, 3, 4))
print('RoPE table shared between CSA/HCA layers:', a2.freqs_cis is a4.freqs_cis and a3.freqs_cis is a4.freqs_cis)
del a2, a3, a4

# sparse_attn: same bits for any chunk size
if 'chunk' in inspect.signature(U['sparse_attn']).parameters:
    torch.manual_seed(0)
    s, n = 600, 750
    q, kv, sink = torch.randn(1, s, 64, 512), torch.randn(1, n, 512), torch.randn(64)
    idx = torch.randint(0, n, (1, s, 278)); idx[torch.rand(idx.shape) < 0.2] = -1
    outs = [U['sparse_attn'](q, kv, sink, idx, 512 ** -0.5, chunk=c) for c in (1, 7, 256, 4096)]
    print('sparse_attn identical for chunk 1 / 7 / 256 / 4096:', all(torch.equal(outs[0], o) for o in outs))
else:
    print('sparse_attn has no chunk parameter yet')

def state_pairs(st, attn, w):
    """(ours, official) tensor pairs: the window ring, compressed entries, indexer keys, compressor tails."""
    pairs = [(st.window, attn.kv_cache[:, :w])]
    if st.compressor is not None:
        pairs += [(st.compressor.entries, attn.kv_cache[:, w:]), (st.compressor.kv, attn.compressor.kv_state),
                  (st.compressor.score, attn.compressor.score_state)]
    if st.indexer is not None:
        pairs += [(st.indexer.entries, attn.indexer.kv_cache), (st.indexer.kv, attn.indexer.compressor.kv_state),
                  (st.indexer.score, attn.indexer.compressor.score_state)]
    return pairs

HAS_STATE = 'SequenceState' in U
oargs = official_args(off, args)
for layer_id in (0, 2, 3):
    torch.manual_seed(layer_id)
    ours = Block(layer_id, args)
    for s in (1, 127, 128, 129, 600, 2304):
        theirs = off.Block(layer_id, oargs).float()          # fresh official buffers for every length
        theirs.load_state_dict(ours.state_dict(), strict=True)
        x = torch.randn(1, s, args.hc_mult, args.dim)
        ids = torch.randint(0, args.vocab_size, (1, s))
        st = U['SequenceState'](args, 1).layers[layer_id] if HAS_STATE else None
        with torch.inference_mode():
            t0 = time.perf_counter(); a = ours(x, ids, st) if HAS_STATE else ours(x, ids); t1 = time.perf_counter()
            b = theirs(x, 0, ids)
        extra = ''
        if HAS_STATE:
            pairs = state_pairs(st, theirs.attn, args.window_size)
            extra = f'  state identical {all(torch.equal(p, q) for p, q in pairs)} ({len(pairs)} tensors)'
        print(f'layer {layer_id}  s = {s:4d}  identical {torch.equal(a, b)}{extra}  ours {t1 - t0:5.2f} s')
        del theirs
    del ours
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"peak memory {peak / 2**30 if sys.platform == 'darwin' else peak / 2**20:.1f} GiB")

if 'Transformer' in U:
    tiny = U['tiny_args']()
    ok = []
    for s in (1, 5, 8, 9, 16, 17, 37, 64):
        torch.manual_seed(s)
        model = U['Transformer'](tiny)
        om = off.Transformer(official_args(off, tiny, max_batch_size=2)).float()
        om.load_state_dict(model.state_dict(), strict=True)
        ids = torch.randint(0, tiny.vocab_size, (2, s))
        if HAS_STATE:
            st = U['SequenceState'](tiny, 2)
            ok.append(torch.equal(model(ids, st), om(ids, 0)) and st.pos == s and
                      all(torch.equal(p, q) for ls, layer in zip(st.layers, om.layers)
                          for p, q in state_pairs(ls, layer.attn, tiny.window_size)))
        else:
            ok.append(torch.equal(model(ids), om(ids, 0)))
    print('tiny Transformer vs official, 8 lengths:', all(ok), '(logits + full state)' if HAS_STATE else '(logits)')
else:
    print('notebook has no Transformer class')

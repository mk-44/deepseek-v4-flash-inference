"""Phase 2, Step 16: snapshot / restore / fork of SequenceState (gate RB).

A. rollback, 12 cases (float32): prefill P, clone, feed 1 or 13 wrong tokens, restore, then the right tokens
   (an extend of 5, then 7 single tokens). Every output must equal a direct run that never saw the wrong
   tokens (torch.equal), and so must the final memory. It also checks that the wrong tokens really changed
   the memory and that restore brings back every tensor and the position.
B. fork: two clones of one prefix continue with different tokens; each equals its own direct run, and the
   parent state stays unchanged.
C. restore works on a state whose tensors were created inside inference mode (as generate does). PyTorch
   refuses in-place writes to such tensors outside inference mode, so restore needs @torch.inference_mode().

Memory is compared with its own walk over the layers, not with state.tensors(), so a tensor missing from
tensors() can't hide itself.
"""
from _paths import load_notebook_defs
import torch


def same_memory(s1, s2):
    pairs = []
    for l1, l2 in zip(s1.layers, s2.layers):
        pairs.append((l1.window, l2.window))
        for c1, c2 in ((l1.compressor, l2.compressor), (l1.indexer, l2.indexer)):
            if c1 is not None:
                pairs += [(c1.entries, c2.entries), (c1.kv, c2.kv), (c1.score, c2.score)]
    return s1.pos == s2.pos and all(torch.equal(x, y) for x, y in pairs)


def run(U):
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    args = U['tiny_args']()
    b = 2
    torch.manual_seed(0)
    model = Transformer(args)

    print('A. snapshot -> wrong tokens -> restore -> right tokens, vs right tokens directly (float32)')
    for P in (3, 4, 9, 16, 17, 33):
        for k in (1, 13):
            right = torch.randint(0, args.vocab_size, (b, P + 12))
            wrong = torch.randint(0, args.vocab_size, (b, k))
            direct = SequenceState(args, b)
            model(right[:, :P], direct)
            direct_out = [model(right[:, P:P + 5], direct, return_all_logits=True)]
            for t in range(P + 5, P + 12):
                direct_out.append(model(right[:, t:t + 1], direct, return_all_logits=True))
            state = SequenceState(args, b)
            model(right[:, :P], state)
            snap = state.clone()
            model(wrong, state)
            changed = not same_memory(state, snap)
            state.restore(snap)
            back = same_memory(state, snap)
            out = [model(right[:, P:P + 5], state, return_all_logits=True)]
            for t in range(P + 5, P + 12):
                out.append(model(right[:, t:t + 1], state, return_all_logits=True))
            same = all(torch.equal(x, y) for x, y in zip(out, direct_out)) and same_memory(state, direct)
            print(f'  P = {P:2d}  {k:2d} wrong tokens: memory changed {changed}  back to snapshot {back}  '
                  f'then identical to direct run {same}')

    print('B. fork: two continuations from one prefix')
    for P in (5, 17):
        prefix = torch.randint(0, args.vocab_size, (b, P))
        parent = SequenceState(args, b)
        model(prefix, parent)
        before = parent.clone()
        results = []
        for _ in range(2):
            cont = torch.randint(0, args.vocab_size, (b, 9))
            fork = parent.clone()
            direct = SequenceState(args, b)
            model(prefix, direct)
            results.append(torch.equal(model(cont, fork, return_all_logits=True),
                                       model(cont, direct, return_all_logits=True)))
        print(f'  P = {P:2d}: fork 1 identical to direct run {results[0]}  fork 2 {results[1]}  '
              f'parent unchanged {same_memory(parent, before)}')

    print('C. restore on a state created inside inference mode (as generate does)')
    with torch.inference_mode():
        state = SequenceState(args, b)
    model(torch.randint(0, args.vocab_size, (b, 5)), state)
    snap = state.clone()
    model(torch.randint(0, args.vocab_size, (b, 13)), state)
    state.restore(snap)
    print(f'  back to snapshot: {same_memory(state, snap)}')
    n = len(SequenceState(args, b).tensors())
    print(f'  tensors() lists {n} tensors (tiny config: expected 24)')


if __name__ == '__main__':
    U = load_notebook_defs()
    if not all(hasattr(U['SequenceState'], m) for m in ('tensors', 'clone', 'restore')):
        print('the notebook has no snapshot / restore yet (Step 16)')
    else:
        run(U)

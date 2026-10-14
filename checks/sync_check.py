"""Phase 2, Step 18: no host syncs (plan item 7, decision D23).

A host sync is a point where Python needs a value from a GPU tensor and has to wait for the GPU to finish
everything queued so far. On the Mac nothing waits, so this script counts the operations that would wait on a GPU:
- aten ops that read values on the host: _local_scalar_dense (.item(), int(t), `if t`), nonzero, one-argument
  torch.where(cond), bincount, unique, masked_select. A TorchDispatchMode sees every aten op;
- .tolist() calls, which dispatch no aten op, so they are counted by wrapping torch.Tensor.tolist.

Allowed (D23): one routing copy per MoE layer (counts.tolist()), because the engine needs it on the CPU anyway
to know which experts to fetch. Everything else must be zero.

A. tiny model: prefill 20 tokens and one decode step; ATen ops, other host reads, routing copies.
B. real size, Block level (layer 3, 8 experts with 6 active): one decode step after a prefill of 125.
C. the cached Hadamard matrix works outside inference mode with gradients, even when it was first built inside it.
D. whole model vs official (float32): prefill P, then 8 decode steps, identical.
"""
import collections
import sys
from _paths import ROOT, load_notebook_defs
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(ROOT / 'checks'))
from load_official import load_official, official_args

READS = {'_local_scalar_dense', 'nonzero', 'bincount', 'masked_select', 'unique', '_unique', '_unique2', 'unique_consecutive'}


class HostReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops, self.reads = 0, collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops += 1
        name = func.overloadpacket.__name__
        if name in READS or func is torch.ops.aten.where.default:
            self.reads[name] += 1
        return func(*args, **(kwargs or {}))


def counted(fn):
    """Run fn() and return (aten ops, other host reads, tolist calls)."""
    calls = [0]
    original = torch.Tensor.tolist

    def tolist(self):
        calls[0] += 1
        return original(self)

    torch.Tensor.tolist = tolist
    try:
        with HostReads() as c:
            fn()
    finally:
        torch.Tensor.tolist = original
    return c.ops, dict(c.reads), calls[0]


def run(U, real_size=True):
    print('A. tiny model (b = 1): host reads per forward')
    args = U['tiny_args']()
    torch.manual_seed(0)
    model = U['Transformer'](args)
    ids = torch.randint(0, args.vocab_size, (1, 21))
    state = U['SequenceState'](args, 1)
    for label, chunk in (('prefill 20 tokens', ids[:, :20]), ('decode 1 token   ', ids[:, 20:])):
        ops, reads, copies = counted(lambda: model(chunk, state))
        print(f'  {label}: {ops} ATen ops, other host reads {reads}, routing copies {copies} '
              f'(expected: none, {args.n_layers})')

    if real_size:
        print('B. real size, Block level (layer 3, 8 experts, 6 active): one decode step after a prefill of 125')
        rargs = U['ModelArgs'](n_routed_experts=8)
        torch.manual_seed(3)
        block = U['Block'](3, rargs)
        st = U['SequenceState'](rargs, 1).layers[3]
        x = torch.randn(1, 126, rargs.hc_mult, rargs.dim)
        rid = torch.randint(0, rargs.vocab_size, (1, 126))
        with torch.inference_mode():
            block(x[:, :125], rid[:, :125], st, 0)
            ops, reads, copies = counted(lambda: block(x[:, 125:], rid[:, 125:], st, 125))
        print(f'  {ops} ATen ops, other host reads {reads}, routing copies {copies} (expected: none, 1)')
        del block

    print('C. cached Hadamard matrix, used with gradients after a model run in inference mode')
    w = torch.nn.Parameter(torch.randn(args.index_head_dim, args.index_head_dim))
    try:
        U['rotate_activation'](torch.randn(4, args.index_head_dim) @ w).sum().backward()
        print('  backward ran:', w.grad is not None)
    except RuntimeError as e:
        print('  ERROR:', str(e)[:100])

    print('D. whole model vs official (float32): prefill P, then 8 decode steps')
    off = load_official('april')
    b = 2
    for P in (5, 17, 37):
        torch.manual_seed(P)
        model = U['Transformer'](args)
        theirs = off.Transformer(official_args(off, args, max_batch_size=b)).float()
        theirs.load_state_dict(model.state_dict(), strict=True)
        ids = torch.randint(0, args.vocab_size, (b, P + 8))
        st = U['SequenceState'](args, b)
        same = [torch.equal(model(ids[:, :P], st), theirs(ids[:, :P], 0))]
        for t in range(P, P + 8):
            same.append(torch.equal(model(ids[:, t:t + 1], st), theirs(ids[:, t:t + 1], t)))
        print(f'  P = {P:2d}: identical at all {len(same)} steps {all(same)}')


if __name__ == '__main__':
    run(load_notebook_defs())

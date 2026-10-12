"""Phase 2, Step 17: the official FP8 / FP4 rounding ("quantization simulation").

A. fp8_simulate / fp4_simulate vs the official kernels (act_quant(..., 64, "ue8m0", inplace=True) and
   fp4_act_quant(..., 32, inplace=True), copied bit for bit in reference/shim/kernel.py), float32 and bf16:
   random blocks at 5 magnitudes, all-zero blocks, and two edge rows. The edge rows have amax exactly 448 / 6,
   so the scale is exactly a power of two, and values exactly halfway between two grid points (ties go to the even one).
   This catches mistakes that the whole-model comparison almost never sees (power-of-two rounding, ties, the 1e-4 floor).
B. the dry run from chat: one block before and after.
C. whole model vs official with the rounding on, prefill + 8 decode steps (float32, torch.equal). This catches
   missing or misplaced calls. The full set: notebook_regression.py, step11_user_check.py, decode_check.py,
   realsize_check.py (they switch the rounding on once the notebook defines fp8_simulate).
"""
import sys
from _paths import ROOT, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'reference'))
from load_official import load_official, official_args


def run(U):
    off = load_official('april')
    fp8, fp4 = U['fp8_simulate'], U['fp4_simulate']

    print('A. fp8_simulate / fp4_simulate vs the official kernels')
    torch.manual_seed(0)
    edge = torch.zeros(2, 448)
    edge[0, :6] = torch.tensor([448.0, 1.0625, 1.1875, 17.0, 19.0, 0.0029296875])
    edge[1, :8] = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    for dtype in (torch.float32, torch.bfloat16):
        same8 = same4 = True
        for x in [edge] + [torch.randn(64, 448) * sc for sc in (1e-6, 1e-2, 1.0, 300.0, 1e6)] + [torch.zeros(2, 448)]:
            x = x.to(dtype)
            r8, r4 = x.clone(), x.clone()
            off.act_quant(r8, 64, 'ue8m0', torch.float32, True)
            off.fp4_act_quant(r4, 32, True)
            same8 &= torch.equal(fp8(x), r8)
            same4 &= torch.equal(fp4(x), r4)
        print(f'  {str(dtype):15s} FP8 identical {same8}  FP4 identical {same4}')

    print('B. one block, before and after')
    x = torch.zeros(448)
    x[:6] = torch.tensor([0.3, -1.7, 2.2, 0.05, 10.0, -3.3])
    print('  x  ', [round(v, 4) for v in x[:6].tolist()])
    print('  FP8', [round(v, 4) for v in fp8(x)[:6].tolist()], '  expected [0.3125, -1.75, 2.25, 0.0508, 10.0, -3.25]')
    print('  FP4', [round(v, 4) for v in fp4(x)[:6].tolist()], '  expected [0.0, -2.0, 2.0, 0.0, 8.0, -3.0]')

    print('C. whole model vs official with the rounding on (float32): prefill P, then 8 decode steps')
    args = U['tiny_args']()
    b = 2
    for P in (5, 17, 37):
        torch.manual_seed(P)
        model = U['Transformer'](args)
        theirs = off.Transformer(official_args(off, args, max_batch_size=b)).float()
        theirs.load_state_dict(model.state_dict(), strict=True)
        ids = torch.randint(0, args.vocab_size, (b, P + 8))
        state = U['SequenceState'](args, b)
        same = [torch.equal(model(ids[:, :P], state), theirs(ids[:, :P], 0))]
        for t in range(P, P + 8):
            same.append(torch.equal(model(ids[:, t:t + 1], state), theirs(ids[:, t:t + 1], t)))
        print(f'  P = {P:2d}: identical at all {len(same)} steps {all(same)}')


if __name__ == '__main__':
    U = load_notebook_defs()
    if 'fp8_simulate' not in U or 'fp4_simulate' not in U:
        print('the notebook has no FP8 / FP4 rounding yet (Step 17)')
    else:
        run(U)

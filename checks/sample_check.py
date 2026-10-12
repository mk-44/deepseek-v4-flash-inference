"""Phase 2, Step 19: the seeded sampler (plan item 8).

sample(logits, temperature, generator) picks the next token. Temperature 0 is greedy (argmax). Otherwise it is the
official "exponential race" (a form of the Gumbel-max trick): probs / noise, argmax, with noise ~ Exp(1). Token i
then wins with probability probs[i], and nothing has to be read back to the CPU.
A. vs the official sample (0731 model.py): with the global random generator seeded the same, identical picks for
   4 temperatures and 2 shapes. Also: a torch.Generator seeded with 7 gives the same picks as seeding the global
   generator with 7 (on the CPU; a CUDA generator gives other numbers).
B. each token wins with probability softmax(logits / T), measured over 200,000 draws.
C. generate: the same seed gives the same tokens and another seed different ones; temperature 0 is identical
   to a plain argmax loop.
"""
import sys
from _paths import ROOT, load_notebook_defs
import torch

sys.path.insert(0, str(ROOT / 'reference'))
from load_official import load_official


def run(U):
    sample, generate = U['sample'], U['generate']
    off = load_official('0731')

    print('A. sample vs the official sample (0731), same random numbers')
    torch.manual_seed(0)
    same, same_gen = [], []
    for T in (0.0, 0.6, 1.0, 1.5):
        for shape in ((3, 512), (2, 5, 512)):
            logits = torch.randn(shape) * 3
            torch.manual_seed(7)
            ours = sample(logits, T)
            torch.manual_seed(7)
            same.append(torch.equal(ours, off.sample(logits, T)))
            same_gen.append(torch.equal(sample(logits, T, torch.Generator().manual_seed(7)), ours))
    print(f'  identical in all {len(same)} cases: {all(same)}   own generator (seed 7) == global seed 7: {all(same_gen)}')

    print('B. how often each token wins, vs softmax(logits / T), 200,000 draws')
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    g = torch.Generator().manual_seed(0)
    for T in (1.0, 0.5):
        wins = sample(logits.expand(200_000, 4), T, g)
        want = [round(p, 3) for p in torch.softmax(logits / T, -1).tolist()]
        got = [round(f, 3) for f in (torch.bincount(wins, minlength=4) / 200_000).tolist()]
        print(f'  T = {T}: softmax {want}  measured {got}')

    print('C. generate (tiny model, b = 2, 30 new tokens)')
    torch.manual_seed(0)
    args = U['tiny_args']()
    model = U['Transformer'](args)
    prompt = torch.randint(0, args.vocab_size, (2, 20))
    a = generate(model, args, prompt, 30, temperature=1.0, seed=1)
    b = generate(model, args, prompt, 30, temperature=1.0, seed=1)
    c = generate(model, args, prompt, 30, temperature=1.0, seed=2)
    greedy = generate(model, args, prompt, 30)
    state = U['SequenceState'](args, 2)
    logits, ref = model(prompt, state), []
    for _ in range(30):
        ref.append(logits.argmax(dim=-1, keepdim=True))
        logits = model(ref[-1], state)
    print(f'  seed 1 twice identical {torch.equal(a, b)}  seed 2 different {not torch.equal(a, c)}  '
          f'T = 0 == argmax loop {torch.equal(greedy, torch.cat(ref, dim=1))}')


if __name__ == '__main__':
    U = load_notebook_defs()
    if 'sample' not in U:
        print('the notebook has no sample() yet (Step 19)')
    else:
        run(U)

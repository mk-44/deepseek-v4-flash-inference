from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import json, io, contextlib, torch
import torch.nn.functional as F
with contextlib.redirect_stdout(io.StringIO()):
    from step11_check import off, official_model

# every definition cell of the user's saved notebook (imports, functions, classes)
U = load_notebook_defs()
args = U['tiny_args']()

torch.manual_seed(0)
x = torch.randn(2, 5, args.hc_mult, args.dim); fn = torch.randn(4, 4 * args.dim) * 0.02; sc = torch.randn(1); ba = torch.randn(4)
ph = off['ParallelHead'](args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
print('hc_head vs official: fp32', torch.equal(U['hc_head'](x, fn, sc, ba, args.norm_eps, args.hc_eps), ph.hc_head(x, fn, sc, ba)),
      '| bf16', torch.equal(U['hc_head'](x.bfloat16(), fn, sc, ba, args.norm_eps, args.hc_eps), ph.hc_head(x.bfloat16(), fn, sc, ba)))

get_last = off['ParallelHead'].get_logits
results = []
for s in (1, 5, 8, 9, 16, 17, 37, 64):
    for seed in (0, 1):
        torch.manual_seed(100 * s + seed)
        b = 2
        model = U['Transformer'](args)
        om = official_model(args, b)
        om.load_state_dict(model.state_dict(), strict=True)           # same names and shapes, or this raises
        ids = torch.randint(0, args.vocab_size, (b, s))
        last_ok = torch.equal(model(ids), om(ids, 0))
        off['ParallelHead'].get_logits = lambda self, h: F.linear(h.float(), self.weight)
        all_ok = torch.equal(model(ids, return_all_logits=True), om(ids, 0))
        off['ParallelHead'].get_logits = get_last
        results.append((s, seed, last_ok, all_ok))
print('whole model vs official (s = 1..64, 2 seeds each): last-position identical', all(r[2] for r in results),
      '| all positions identical', all(r[3] for r in results), f'({len(results)} runs)')
print('same parameter names as official:', set(model.state_dict()) == set(om.state_dict()), '|', len(model.state_dict()), 'tensors')

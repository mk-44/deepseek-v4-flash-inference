# Exact parameter/byte inventory of the official V4-Flash model (real config), built on the meta device (no memory used).
from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import sys, types, json, re, collections, torch
sys.path.insert(0, str(OFFICIAL))
k = types.ModuleType('kernel')
for n in ('act_quant', 'fp4_act_quant', 'fp8_gemm', 'fp4_gemm', 'sparse_attn', 'hc_split_sinkhorn'):
    setattr(k, n, lambda *a, **kw: None)
sys.modules['kernel'] = k
import model as M

cfg = json.load(open(OFFICIAL / 'config.json'))
args = M.ModelArgs(**cfg)
args.max_batch_size = 1
torch.set_default_dtype(torch.bfloat16)                     # as in official generate.py
with torch.device('meta'):
    m = M.Transformer(args)

def cat(name):
    if name.startswith('mtp.'): return 'MTP block (not loaded)'
    if name.startswith('embed.'): return 'embedding'
    if name.startswith('head.'): return 'head'
    if re.match(r'layers\.\d+\.ffn\.experts\.', name): return 'routed experts'
    if '.ffn.shared_experts.' in name: return 'shared experts'
    if '.ffn.gate.' in name: return 'router (gate weight, bias, tid2eid)'
    if '.attn.indexer.' in name: return 'indexer (incl. its compressor)'
    if '.attn.compressor.' in name: return 'attention compressor'
    if '.attn.' in name: return 'attention projections + norms + sink'
    if '.hc_' in name or name.startswith('hc_'): return 'mHC'
    return 'norms (block + final)'

rows = collections.defaultdict(lambda: [0, 0, collections.Counter()])
per_layer = collections.defaultdict(int)
params = list(m.named_parameters())
param_names = {n for n, _ in params}
persistent = set(m.state_dict(keep_vars=True).keys())          # persistent buffers + params
for name, t in params + [(n, b) for n, b in m.named_buffers() if n in persistent and n not in param_names]:
    nbytes = t.numel() * t.element_size()
    c = cat(name)
    rows[c][0] += t.numel() * (2 if t.dtype == torch.float4_e2m1fn_x2 else 1)
    rows[c][1] += nbytes
    rows[c][2][str(t.dtype).replace('torch.', '')] += nbytes
    mm = re.match(r'layers\.(\d+)\.', name)
    if mm and c != 'routed experts': per_layer[int(mm.group(1))] += nbytes

tot = sum(v[1] for v in rows.values())
print(f"{'class':40s} {'params':>16s} {'bytes':>16s}  dtypes (bytes)")
for c, (p, b, d) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
    print(f"{c:40s} {p:16,d} {b:16,d}  {dict(d)}")
print(f"{'TOTAL':40s} {'':16s} {tot:16,d}  = {tot/2**30:.2f} GiB")
main = tot - rows['MTP block (not loaded)'][1]
print(f"main model (no MTP): {main:,d} B = {main/2**30:.2f} GiB; without routed experts: {main - rows['routed experts'][1]:,d} B = {(main - rows['routed experts'][1])/2**30:.2f} GiB")
# one expert
e = m.layers[5].ffn.experts[0]
eb = sum(p.numel() * p.element_size() for p in e.parameters())
print('one routed expert:', eb, 'B;', {n: (tuple(p.shape), str(p.dtype)) for n, p in e.named_parameters()})
se = m.layers[5].ffn.shared_experts
print('shared expert:', sum(p.numel() * p.element_size() for p in se.parameters()), 'B;', {n: (tuple(p.shape), str(p.dtype)) for n, p in se.named_parameters()})
# static (non-routed-expert) bytes per layer type
kinds = {0: 'SWA', 4: 'CSA', 128: 'HCA'}
by_kind = collections.defaultdict(list)
for l, b in per_layer.items():
    by_kind[kinds[args.compress_ratios[l]] + (' hash' if l < args.n_hash_layers else '')].append(b)
for kname, v in sorted(by_kind.items()):
    print(f'static bytes per layer, {kname:9s}: {v[0]:,d} B  (x{len(v)} layers)')
print('head dtype at runtime:', m.head.weight.dtype, '| embed dtype:', m.embed.weight.dtype)
# attention submodule dtypes for one CSA layer
for n, p in m.layers[2].attn.named_parameters():
    print('   L2.attn.' + n, tuple(p.shape), str(p.dtype).replace('torch.', ''))
for n, p in m.layers[2].ffn.gate.named_parameters():
    print('   L2.ffn.gate.' + n, tuple(p.shape), str(p.dtype).replace('torch.', ''))
for n, p in m.layers[0].ffn.gate.named_parameters():
    print('   L0.ffn.gate.' + n, tuple(p.shape), str(p.dtype).replace('torch.', ''))
for n in ('hc_attn_fn', 'hc_attn_base', 'hc_attn_scale'):
    p = getattr(m.layers[0], n); print('   L0.' + n, tuple(p.shape), str(p.dtype).replace('torch.', ''))

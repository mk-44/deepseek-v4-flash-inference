# Exact decode-state (KV caches + compressor states) bytes from the official model's buffers, batch 1, various max_seq_len.
from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import sys, types, json, re, collections, torch
sys.path.insert(0, str(OFFICIAL))
k = types.ModuleType('kernel')
for n in ('act_quant', 'fp4_act_quant', 'fp8_gemm', 'fp4_gemm', 'sparse_attn', 'hc_split_sinkhorn'):
    setattr(k, n, lambda *a, **kw: None)
sys.modules['kernel'] = k
import model as M
cfg = json.load(open(OFFICIAL / 'config.json'))
torch.set_default_dtype(torch.bfloat16)
for L in (4096, 65536, 262144, 1048576):
    args = M.ModelArgs(**cfg); args.max_batch_size = 1; args.max_seq_len = L; args.n_mtp_layers = 0
    with torch.device('meta'):
        m = M.Transformer(args)
    persistent = set(m.state_dict(keep_vars=True).keys())
    by = collections.defaultdict(int); shapes = {}
    for name, b in m.named_buffers():
        if name in persistent: continue
        key = re.sub(r'layers\.\d+\.', 'layers.N.', name)
        nb = b.numel() * b.element_size()
        if 'freqs_cis' in name: key = 'freqs_cis (RoPE tables, not state)'
        by[key] += nb
        shapes.setdefault(key, (tuple(b.shape), str(b.dtype).replace('torch.', '')))
    state = {kk: v for kk, v in by.items() if 'freqs' not in kk}
    tot = sum(state.values())
    print(f'max_seq_len {L:>9,d}: decode state total {tot:>15,d} B = {tot/2**20:9.1f} MiB  ({tot/L:,.0f} B per token of capacity)')
    if L in (4096, 1048576):
        for kk, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f'     {kk:45s} {v:>15,d} B   e.g. {shapes[kk]}')

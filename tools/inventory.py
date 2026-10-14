"""Phase 3 item 5: an inventory of the 0731 checkpoint, built from the headers of its 48 files on Hugging Face.
    python tools/inventory.py             # about 20 s: ~8 MB of range requests, no weights downloaded, no token sent
    python tools/inventory.py --offline   # the same checks on the saved headers, without the network
1. read each file's header (bytes 0-7 give its length, then the header JSON) and the file's size;
2. a table of every tensor: file, type, shape, offset in the file, bytes, class;
3. checks, each PASS or FAIL:
   - total: the tensor bytes add up to the index's total_size;
   - index: every name is in the file the index names;
   - packing: in every file the tensors sit back to back, from the end of the header to the end of the file;
   - sizes: every tensor's bytes = its shape x the size of its type;
   - formats: FP4 weights are I8 [out, in/2] with E8M0 scales [out, in/32]; FP8 weights are E4M3 with one E8M0 scale
     per 128 x 128 block; every scale belongs to such a weight;
   - experts: every expert is two byte ranges, its 3 weights and its 3 scales;
   - official: shapes and types against the official 0731 model.py, built on the meta device (no memory);
4. docs/INVENTORY.md (tables) and deepseek_official/0731/checkpoint_headers.json.gz (the 48 headers as read, plus file sizes).
It exits with 1 unless every check passed."""
import argparse
import collections
import gzip
import json
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = 'deepseek-ai/DeepSeek-V4-Flash-0731', '7872f01b1d1fe23eabc4c98b48bffcef5a386062'
EXPERT = 13_369_344                  # bytes of one routed expert: FP4 w1, w3, w2 and their E8M0 scales

INDEX = ROOT / 'deepseek_official' / '0731' / 'model.safetensors.index.json'
CONFIG = ROOT / 'deepseek_official' / '0731' / 'inference' / 'config.json'
SAVED = ROOT / 'deepseek_official' / '0731' / 'checkpoint_headers.json.gz'
REPORT = ROOT / 'docs' / 'INVENTORY.md'
TYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1, 'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2,
              'U32': 4, 'I32': 4, 'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8}
TORCH_TYPE = {'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32', 'F64': 'float64', 'I8': 'int8', 'I32': 'int32',
              'I64': 'int64', 'F8_E4M3': 'float8_e4m3fn', 'F8_E8M0': 'float8_e8m0fnu'}
FP4_BLOCK, FP8_BLOCK = 32, 128       # one E8M0 scale per 32 FP4 values along a row; one per 128 x 128 FP8 block
EXPERT_PART = re.compile(r'^(.+\.experts\.(\d+))\.(w[123])\.(weight|scale)$')


def get(fname, start, end, tries=4):
    """Bytes start..end (inclusive) of one checkpoint file, and the file's size. Refuses any answer that isn't a
    partial one (206), so a server that ignores the range can't start a whole-file download."""
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{fname}'
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'}),
                                        timeout=60) as r:
                if r.status != 206:
                    raise IOError(f'{fname}: HTTP {r.status} instead of 206 (partial content)')
                size = int(r.headers['Content-Range'].rsplit('/', 1)[1])
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f'{fname}: asked for {end - start + 1} bytes, got {len(data)}')
            return data, size
        except (urllib.error.URLError, OSError) as e:
            if k == tries - 1:
                raise IOError(f'{fname}: {e}') from e
            time.sleep(2 ** k)


def read_header(fname):
    """One file's header exactly as stored, its length n and the file's size. The tensor data starts at byte 8 + n."""
    raw, size = get(fname, 0, 7)
    n = struct.unpack('<Q', raw)[0]
    text, _ = get(fname, 8, 8 + n - 1)
    return dict(file_bytes=size, header_bytes=n, header=text.decode('utf-8'))


def save_headers(headers, path):
    """gzip with a fixed time stamp and no file name inside, so the same headers always give the same bytes."""
    blob = json.dumps(dict(repo=REPO, revision=REVISION, files=headers), separators=(',', ':')).encode()
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=9, mtime=0) as f:
        f.write(blob)


def load_headers(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        blob = json.load(f)
    if (blob['repo'], blob['revision']) != (REPO, REVISION):
        raise SystemExit(f'{path} holds {blob["repo"]} @ {blob["revision"]}, not {REPO} @ {REVISION}')
    return blob['files']


def tensor_table(headers):
    """One row per tensor: name, file, type, shape, start (absolute byte offset in its file), bytes."""
    rows = []
    for fname, h in sorted(headers.items()):
        base = 8 + h['header_bytes']
        for name, t in json.loads(h['header']).items():
            if name != '__metadata__':
                s, e = t['data_offsets']
                rows.append(dict(name=name, file=fname, dtype=t['dtype'], shape=t['shape'], start=base + s, nbytes=e - s))
    return rows


def part(name):
    """The classes of the plan's section 2.2, plus the drafter (the mtp.K blocks)."""
    if name == 'embed.weight':
        return 'embedding'
    expert = EXPERT_PART.match(name) is not None
    if name.startswith('mtp.'):
        return 'drafter experts' if expert else 'drafter static'
    return 'routed experts' if expert else 'static'


def role(name):
    """The name with its numbers replaced: layers.N (layer), mtp.K (drafter block), experts.E (expert)."""
    name = re.sub(r'\.experts\.\d+\.', '.experts.E.', name)
    return re.sub(r'^mtp\.\d+\.', 'mtp.K.', re.sub(r'^layers\.\d+\.', 'layers.N.', name))


def contents(names, cfg):
    """What one file holds, in words."""
    tops = sorted({'.'.join(n.split('.')[:2]) if re.match(r'(layers|mtp)\.\d+\.', n) else n.split('.')[0] for n in names})
    if len(tops) == 1 and tops[0].startswith('layers.'):
        i = int(tops[0].split('.')[1])
        kind = {0: 'window only', 4: 'CSA', 128: 'HCA'}[cfg['compress_ratios'][i]]
        return f'layer {i} ({kind}{", hash routing" if i < cfg["n_hash_layers"] else ""})'
    if len(tops) == 1 and tops[0].startswith('mtp.'):
        return f'drafter block {tops[0].split(".")[1]}'
    return 'embedding' if tops == ['embed'] else ', '.join(tops)


def check_total(rows, index):
    total, want = sum(r['nbytes'] for r in rows), index['metadata']['total_size']
    return total == want, f'{total:,} B of tensors, the index says {want:,}'


def check_index(rows, index):
    wm, where = index['weight_map'], collections.defaultdict(list)
    for r in rows:
        where[r['name']].append(r['file'])
    missing, extra = set(wm) - set(where), set(where) - set(wm)
    twice = [n for n, fs in where.items() if len(fs) > 1]
    moved = [n for n, fs in where.items() if n in wm and fs != [wm[n]]]
    ok = not (missing or extra or twice or moved)
    return ok, (f'{len(where):,} tensors, each in the file the index names' if ok else
                f'missing {len(missing)}, not in the index {len(extra)}, in two files {len(twice)}, in another file {len(moved)}')


def check_packing(rows, headers):
    by_file, bad = collections.defaultdict(list), []
    for r in rows:
        by_file[r['file']].append(r)
    for fname, h in headers.items():
        pos = 8 + h['header_bytes']
        for r in sorted(by_file[fname], key=lambda r: r['start']):
            if r['start'] != pos:
                bad.append(f'{fname}: {r["name"]} starts at {r["start"]:,}, not {pos:,}')
            pos = r['start'] + r['nbytes']
        if pos != h['file_bytes']:
            bad.append(f'{fname}: the tensors end at {pos:,}, the file at {h["file_bytes"]:,}')
    return not bad, ('in all files, from the end of the header to the end of the file, with no gaps or overlaps'
                     if not bad else '; '.join(bad[:3]))


def check_sizes(rows):
    bad = [r['name'] for r in rows if r['dtype'] not in TYPE_BYTES or r['nbytes'] != math.prod(r['shape']) * TYPE_BYTES[r['dtype']]]
    return not bad, 'bytes = shape x type size for every tensor' if not bad else f'{len(bad)} wrong, e.g. {bad[:3]}'


def check_formats(rows):
    """Every quantized weight has the scale its format needs, and every scale belongs to one."""
    by_name, bad, n = {r['name']: r for r in rows}, [], collections.Counter()
    for r in rows:
        name = r['name']
        if name.endswith('.scale'):
            w = by_name.get(name[:-len('scale')] + 'weight')
            if r['dtype'] != 'F8_E8M0' or w is None or w['dtype'] not in ('I8', 'F8_E4M3'):
                bad.append(name)
            continue
        if EXPERT_PART.match(name) and r['dtype'] != 'I8':
            bad.append(name)
        if r['dtype'] not in ('I8', 'F8_E4M3'):
            continue
        if len(r['shape']) != 2:
            bad.append(name)
            continue
        out, cols = r['shape']
        if r['dtype'] == 'I8':
            want = [out, cols * 2 // FP4_BLOCK]
            if not EXPERT_PART.match(name):
                bad.append(name)
        else:
            want = [-(-out // FP8_BLOCK), -(-cols // FP8_BLOCK)]
        s = by_name.get(name[:-len('weight')] + 'scale') if name.endswith('.weight') else None
        n[r['dtype']] += 1
        if s is None or s['shape'] != want:
            bad.append(name)
    text = (f'FP4: {n["I8"]:,} weights I8 [out, in/2] with E8M0 [out, in/32] scales; '
            f'FP8: {n["F8_E4M3"]:,} weights E4M3 with an E8M0 scale per 128 x 128 block; every scale belongs to one')
    return not bad, text if not bad else f'{len(bad)} break the rules, e.g. {bad[:3]}'


def check_experts(rows):
    """Every expert is two byte ranges: its 3 weights back to back, and its 3 scales back to back."""
    parts, bad, gaps, inner, orders = collections.defaultdict(lambda: collections.defaultdict(list)), [], [], set(), set()
    for r in rows:
        m = EXPERT_PART.match(r['name'])
        if m:
            parts[m.group(1)][m.group(4)].append(r)
    for key, p in parts.items():
        spans = {}
        for kind in ('weight', 'scale'):
            rs = sorted(p[kind], key=lambda r: r['start'])
            if len(rs) != 3 or any(a['start'] + a['nbytes'] != b['start'] for a, b in zip(rs, rs[1:])):
                bad.append(f'{key} {kind}s')
            spans[kind] = (rs[0]['start'], sum(r['nbytes'] for r in rs))
            if kind == 'weight':
                inner.add(', '.join(EXPERT_PART.match(r['name']).group(3) for r in rs))
        if spans['weight'][1] + spans['scale'][1] != EXPERT:
            bad.append(f'{key} is {spans["weight"][1] + spans["scale"][1]:,} B')
        gaps.append(abs(spans['weight'][0] - spans['scale'][0]))
    blocks = collections.defaultdict(list)                        # experts of one layer or drafter block, file order
    for key, p in parts.items():
        blocks[key.rsplit('.experts.', 1)[0]].append((min(r['start'] for r in p['weight']), key.rsplit('.', 1)[1]))
    for ids in blocks.values():
        ids = [i for _, i in sorted(ids)]
        orders.add('name order (0, 1, 10, 100, ...)' if ids == sorted(ids) else
                   'number order' if ids == sorted(ids, key=int) else 'another order')
    weights = {k: sum(r['nbytes'] for r in p['weight']) for k, p in parts.items()}
    text = (f'{len(parts):,} experts, each = its weights ({" / ".join(sorted(inner))}, '
            f'{sorted(set(weights.values()))[0]:,} B) + its scales, {min(gaps) / 1e6:,.0f}-{max(gaps) / 1e6:,.0f} MB apart; '
            f'in each file the experts are in {" / ".join(sorted(orders))}')
    return not bad, text if not bad else f'{len(bad)} problems, e.g. {bad[:3]}'


def against_official(rows):
    """The official 0731 model.py built on the meta device with the real config, as its generate.py builds it."""
    import torch
    sys.path.insert(0, str(ROOT / 'checks'))
    from load_official import load_official
    off = load_official('0731')
    args = off.ModelArgs(**json.load(open(CONFIG)))
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device('meta'):
            model = off.Transformer(args)
    finally:
        torch.set_default_dtype(torch.float32)
    theirs = {n: (list(t.shape), str(t.dtype).replace('torch.', '')) for n, t in model.state_dict(keep_vars=True).items()}
    ours = {r['name']: r for r in rows}
    only_ckpt, only_model = sorted(set(ours) - set(theirs)), sorted(set(theirs) - set(ours))
    shapes = [n for n in set(ours) & set(theirs) if ours[n]['shape'] != theirs[n][0]]
    types = collections.defaultdict(collections.Counter)          # (checkpoint type, model type) -> role -> count
    for n in set(ours) & set(theirs):
        a, b = TORCH_TYPE.get(ours[n]['dtype'], ours[n]['dtype']), theirs[n][1]
        if a != b:
            types[(a, b)][role(n)] += 1
    return dict(shared=len(set(ours) & set(theirs)), only_ckpt=only_ckpt, only_model=only_model, shapes=shapes,
                types=types)


def official_result(o):
    """PASS when the shapes agree and the names differ only where the official code says they do: the model has no
    wo_a scale (convert.py dequantizes wo_a to BF16), and its drafter blocks reuse the main embedding and head."""
    names_ok = (all(n.endswith('.wo_a.scale') for n in o['only_ckpt'])
                and all(re.fullmatch(r'mtp\.\d+\.(embed|head)\.weight', n) for n in o['only_model']))
    shapes = 'all equal' if not o['shapes'] else f'different for {len(o["shapes"])}, e.g. {o["shapes"][:3]}'
    only_ckpt = ', '.join(sorted({role(n) for n in o['only_ckpt']})) or '-'
    only_model = ', '.join(sorted({role(n) for n in o['only_model']})) or '-'
    text = (f'{o["shared"]:,} names in both, shapes {shapes}; only in the checkpoint {len(o["only_ckpt"])} ({only_ckpt}), '
            f'only in the model {len(o["only_model"])} ({only_model})')
    return not o['shapes'] and names_ok, text


def type_lines(o):
    return [f'{a} -> {b}: {sum(c.values()):,} ({", ".join(f"{r} x{k:,}" for r, k in c.most_common())})'
            for (a, b), c in sorted(o['types'].items(), key=lambda kv: -sum(kv[1].values()))]


def files_line(headers, rows, cfg):
    """The files in runs of the same kind, e.g. 00002-00044 layers 0-42."""
    names = collections.defaultdict(list)
    for r in rows:
        names[r['file']].append(r['name'])
    runs = []                                       # [kind, first file, last file, first number, last number]
    for fname in sorted(headers):
        what, num = contents(names[fname], cfg).split(' (')[0], fname[6:11]
        m = re.fullmatch(r'(layer|drafter block) (\d+)', what)
        kind, k = (m.group(1), m.group(2)) if m else (what, None)
        if k is not None and runs and runs[-1][0] == kind:
            runs[-1][2], runs[-1][4] = num, k
        else:
            runs.append([kind, num, num, k, k])
    return ' | '.join(f'{a}-{b} {kind}s {x}-{y}' if a != b else f'{a} {kind}' + (f' {x}' if x else '')
                      for kind, a, b, x, y in runs)


def report(headers, rows, results, o, cfg, seconds, source):
    """docs/INVENTORY.md: the checks, then one table per view."""
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    L = [f'# Checkpoint inventory: {REPO}', '',
         f'Generated by `tools/inventory.py` on {date} from the headers of the {len(headers)} files at revision '
         f'`{REVISION}` ({source}, {seconds:.0f} s); no weights were downloaded. The raw headers are saved in '
         f'`deepseek_official/0731/checkpoint_headers.json.gz`.', '', '## Checks', '']
    L += [f'- **{name}:** {"PASS" if ok else "FAIL"}. {text}' if ok is not None else f'- **{name}:** {text}'
          for name, ok, text in results]
    names = collections.defaultdict(list)
    for r in rows:
        names[r['file']].append(r['name'])
    L += ['', '## Files', '', '| file | contents | bytes | tensors | header bytes |', '|---|---|---|---|---|']
    L += [f'| {f} | {contents(names[f], cfg)} | {h["file_bytes"]:,} | {len(names[f]):,} | {h["header_bytes"]:,} |'
          for f, h in sorted(headers.items())]
    by_part = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_part[part(r['name'])][0] += 1
        by_part[part(r['name'])][1] += r['nbytes']
    L += ['', '## Bytes by class', '', '| class | tensors | bytes | GiB |', '|---|---|---|---|']
    L += [f'| {p} | {n:,} | {b:,} | {b / 2**30:.2f} |' for p, (n, b) in sorted(by_part.items(), key=lambda kv: -kv[1][1])]
    L += [f'| total | {len(rows):,} | {sum(r["nbytes"] for r in rows):,} | {sum(r["nbytes"] for r in rows) / 2**30:.2f} |']
    roles = collections.defaultdict(collections.Counter)
    for r in rows:
        roles[role(r['name'])][(r['dtype'], tuple(r['shape']), r['nbytes'])] += 1
    L += ['', '## Formats', '', 'N = layer, E = expert, K = drafter block.', '',
          '| tensor | count | type | shape | bytes each |', '|---|---|---|---|---|']
    for name in sorted(roles):
        for (dtype, shape, nbytes), k in sorted(roles[name].items(), key=lambda kv: -kv[1]):
            L.append(f'| `{name}` | {k:,} | {dtype} | {list(shape)} | {nbytes:,} |')
    if o:
        L += ['', '## Against the official 0731 model.py', '',
              'The official inference code does not read these files directly: its `convert.py` first rewrites them '
              '(it dequantizes `wo_a` to BF16 and views the I8 expert bytes as FP4), and `load_model` then copies every '
              'tensor into the type that `model.py` declares. Types that differ, checkpoint -> official model:', '']
        L += [f'- {line}' for line in type_lines(o)]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--offline', action='store_true', help=f'use the saved headers ({SAVED.name}) instead of the network')
    a = ap.parse_args()
    index, cfg, t0 = json.load(open(INDEX)), json.load(open(CONFIG)), time.perf_counter()
    files = sorted(set(index['weight_map'].values()))
    if a.offline:
        headers, source = load_headers(SAVED), 'saved headers'
    else:
        with ThreadPoolExecutor(8) as pool:
            headers, source = dict(zip(files, pool.map(read_header, files))), 'Hugging Face'
        save_headers(headers, SAVED)
    seconds = time.perf_counter() - t0
    rows = tensor_table(headers)
    results = [('total', *check_total(rows, index)), ('index', *check_index(rows, index)),
               ('packing', *check_packing(rows, headers)), ('sizes', *check_sizes(rows)),
               ('formats', *check_formats(rows)), ('experts', *check_experts(rows))]
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r['dtype']][0] += 1
        by_type[r['dtype']][1] += r['nbytes']
    results.append(('types', None, ', '.join(f'{t} {n:,} tensors / {b / 1e9:.2f} GB'
                                              for t, (n, b) in sorted(by_type.items(), key=lambda kv: -kv[1][1]))))
    by_part = collections.Counter()
    for r in rows:
        by_part[part(r['name'])] += r['nbytes']
    results.append(('classes', None, ', '.join(f'{p} {b:,} B ({b / 2**30:.2f} GiB)' for p, b in by_part.most_common())))
    aligned = sum(r['start'] % 4096 == 0 for r in rows)
    results.append(('alignment', None, f'{aligned:,} of {len(rows):,} tensors start on a 4 KiB boundary'))
    o = against_official(rows)
    results.append(('official', *official_result(o)))
    results += [('official types', None, line) for line in type_lines(o)]
    report(headers, rows, results, o, cfg, seconds, source)
    print(f'inventory      {REPO} @ {REVISION[:7]}  {len(headers)} files, headers from {source} in {seconds:.0f} s')
    print(f'files          {files_line(headers, rows, cfg)}')
    for name, ok, text in results:
        print(f'{name:14s} {text}' + ('' if ok is None else f': {"PASS" if ok else "FAIL"}'))
    print(f'saved          {REPORT.relative_to(ROOT)}' + ('' if a.offline else
                                                         f', {SAVED.relative_to(ROOT)} ({SAVED.stat().st_size / 1e6:.1f} MB)'))
    sys.exit(0 if all(ok for _, ok, _ in results if ok is not None) else 1)


if __name__ == '__main__':
    main()

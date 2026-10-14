"""Phase 2 gates (after_arch_plan.md, Phase 2 item 9): every Phase 2 check in one run, one line per gate, PASS or FAIL.

A gate is a test that must pass before moving on. In the notebook's last cell:
    import sys
    sys.path.insert(0, "checks")
    from gates import run_gates
    results = run_gates(globals(), real_size = RUN_REAL_SIZE)
From a terminal (reads the saved arch.ipynb, or the copy named by V4_NOTEBOOK):
    venv/bin/python checks/gates.py [--real-size] [--only G5,RB]

Each gate builds its own models from the notebook's definitions (U), and, where it compares, from the official 0731
code on the CPU (checks/load_official.py). It seeds its own random numbers and prints one line. If a gate raises,
it FAILs with the error and the other gates still run. The per-step scripts in checks/ stay as they are; most gates
here re-run their properties on the final code, several with bigger sweeps.

"Picks" are what the model chooses: the indexer's top-k compressed entries and the router's experts. They are read
with forward hooks (a function PyTorch calls with a module's output), so the model code is not changed.

Tiny-model gates (always):
  M     every module vs official, fp32 + bf16 (notebook_regression.py)
  G3    prefill vs official, fp32: logits (last and all positions), memory, picks
  G3d   decode vs official, fp32: prefill P, then 40 single tokens; logits and picks at every step, memory at the end
  G4    prefill + decode == one prefill, fp64: logits, picks, memory
  G4c   extend == one prefill, fp64: prefill P, then chunks of T tokens; logits, picks, memory
  G5    causality and batch isolation, fp64 (new): two sequences in one batch (B = 2) give each exactly what it gets
        alone (B = 1), through prefill, extend, decode, drafts and memory; changing one token changes nothing before
        it and nothing in the other sequence
  RB    snapshot -> wrong tokens -> restore == direct run, and forks; drafter on
  Q     FP8 / FP4 rounding vs the official kernels, fp32 + bf16, edge rows
  SMP   sampler vs official with the same random numbers; same seed -> same tokens
  DSP   DSpark vs official 0731 (fp32, temperature 0 and 1); an extend then a draft == one token at a time (fp64)
  GEN   generate vs the official generate(): mixed lengths, early stops, the max_seq_len cap, end-of-text in a prompt
  TOK   tokenizer + official chat encoding
  CFG   ModelArgs() == the 0731 config.json; the real config's weight names == the 0731 checkpoint index
  cost  one tiny decode step: ATen ops; host reads: one routing copy per MoE layer, nothing else
Real-size gates (real_size=True): one Block per attention type at the real dimensions, random weights, 8 of 256 experts:
  G3r   layers 0 / 2 / 3: prefill 2,304 vs official (output, memory, picks), fp32
  G4r   layers 0 / 2 / 3: prefill 2,300 + 16 decode steps == one prefill of 2,316, fp64 (crosses the HCA entry at 2,304)
  G7m   layers 0 / 2 / 3: prefill 4,096 + 64 decode steps vs official, fp32, max_seq_len 8,192; time and peak memory
  DSPr  DSpark stage 0 (layer 43) vs official: prefill 300 + 4 drafts, fp32
"""
import argparse
import contextlib
import datetime
import importlib
import inspect
import io
import json
import resource
import subprocess
import sys
import threading
import time
from _paths import ROOT, load_notebook_defs, set_quant_simulation
import torch
import torch.nn.functional as F

sys.path.insert(0, str(ROOT / 'checks'))
from load_official import load_official, official_args, official_generate
import notebook_regression          # importing it switches the official rounding off; run_gates switches it back
from chat_check import EXPECTED_IDS
from decode_check import memory_pairs
from dspark_check import ALIASES, memory
from sync_check import counted

LENGTHS = (1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 21, 32, 33, 64, 100)   # both sides of window 8, blocks 4 and 16
TOL = 1e-12                                                                   # float64 checks: rounding is 1e-16 to 1e-14
CHECKPOINT = ROOT / 'deepseek_official' / '0731'


# ---------------------------------------------------------------- helpers

def diff(x, y):
    """Largest |x - y| relative to the largest finite |y|. Equal values, including equal infinities, count as 0.
    Integer tensors (token ids) must be equal: 0 or inf. Different shapes: inf."""
    if x.shape != y.shape:
        return float('inf')
    if not x.is_floating_point():
        return 0.0 if torch.equal(x, y) else float('inf')
    if x.numel() == 0:
        return 0.0
    x, y = x.double(), y.double()
    d = torch.where(x == y, 0.0, (x - y).abs())
    finite = y[torch.isfinite(y)]
    scale = finite.abs().max().item() if finite.numel() else 1.0
    return d.max().item() / (scale or 1.0)


def same_memory(a, b, tol=0.0):
    return len(a) == len(b) and all(diff(x, y) <= tol for x, y in zip(a, b))


def live_rows(ratio, pos):
    """Compressor tail rows that a later step still reads: with overlap (ratio 4) the last full block (rows 0-3, once
    one exists) and the current partial block (from row 4); without overlap, the partial block. The other rows hold
    leftovers that differ between one prefill (never written: zeros, scores -inf) and a run in pieces (an older
    block), and no step reads them before overwriting them."""
    partial = pos % ratio
    if ratio == 4:
        return (list(range(4)) if pos >= 4 else []) + list(range(4, 4 + partial))
    return list(range(partial))


def live_layer(layer, ratio, pos):
    out = [layer.window]
    for c in (layer.compressor, layer.indexer):
        if c is not None:
            rows = live_rows(ratio, pos)
            out += [c.entries, c.kv[:, rows], c.score[:, rows]]
    return out


def live_memory(state, args):
    """All memory a later step can read: windows, compressed entries, live compressor tail rows, drafter windows."""
    out = []
    for layer, ratio in zip(state.layers, args.compress_ratios):
        out += live_layer(layer, ratio, state.pos)
    return out + [layer.window for layer in state.mtp]


class Picks:
    """Records every Indexer's and Gate's picks with forward hooks. Indexer: its top-k entry numbers minus the offset
    it was given, so ours and the official code (which put the compressed entries at different places in their key
    lists) can be compared; -1 (no entry) stays -1. Gate: its expert ids. picks[module name] = one tensor per call."""

    def __init__(self, model):
        self.picks, self.handles = {}, []
        for name, module in model.named_modules():
            if type(module).__name__ in ('Indexer', 'Gate'):
                self.handles.append(module.register_forward_hook(self.recorder(name, module), with_kwargs=True))

    def recorder(self, name, module):
        signature = inspect.signature(module.forward)

        def record(mod, args, kwargs, out):
            if type(mod).__name__ == 'Gate':
                p = out[1].long()
            else:
                offset = signature.bind(*args, **kwargs).arguments['offset']
                p = torch.where(out >= 0, out - offset, -1).long()
            self.picks.setdefault(name, []).append(p.clone())
        return record

    def remove(self):
        for h in self.handles:
            h.remove()


@contextlib.contextmanager
def picks_of(*models):
    recorders = [Picks(m) for m in models]
    try:
        yield [r.picks for r in recorders]
    finally:
        for r in recorders:
            r.remove()


def same_picks(a, b):
    """Same code path (ours vs official): identical, call by call."""
    return bool(a) and a.keys() == b.keys() and all(
        len(a[k]) == len(b[k]) and all(torch.equal(x, y) for x, y in zip(a[k], b[k])) for k in a)


def per_token(picks, b):
    """Per module: [b, tokens, width], each token's picks in call order. Rows are sorted (picks with equal scores may
    come in either order) and padded with -1 to the widest call, so one prefill and a run in chunks compare token by token."""
    out = {}
    for name, calls in picks.items():
        width = max(p.shape[-1] for p in calls)
        rows = [F.pad(p if p.dim() == 3 else p.view(b, -1, p.shape[-1]), (0, width - p.shape[-1]), value=-1)   # Gate: [b * s, k]
                for p in calls]
        out[name] = torch.cat(rows, 1).sort(dim=-1).values
    return out


def same_picks_per_token(a, b, batch):
    a, b = per_token(a, batch), per_token(b, batch)
    return bool(a) and a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def official_twin(off, model, args, b, n_mtp_layers=0):
    """The official 0731 Transformer with our weights, greedy (its forward samples)."""
    theirs = off.Transformer(official_args(off, args, max_batch_size=b, n_mtp_layers=n_mtp_layers)).float()
    missing, unexpected = theirs.load_state_dict(model.state_dict(), strict=not n_mtp_layers)
    theirs.temperature = 0
    return theirs, sorted(missing) == sorted(ALIASES if n_mtp_layers else []) and not unexpected


def all_position_logits(theirs):
    """The official model returns the last position's logits; its head can return all of them."""
    head = theirs.head
    head.forward = lambda x: type(head).forward(head, x, full_logits=True)


class PeakMemory:
    """Peak resident memory (RSS) of this process while the with-block runs, sampled every 5 ms with psutil
    (without psutil: the process's lifetime peak from getrusage)."""

    def __enter__(self):
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
        self.before = self.peak = self.rss()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def rss(self):
        if self.process is not None:
            return self.process.memory_info().rss
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == 'darwin' else peak * 1024

    def watch(self):
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, self.rss())

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.rss())


def yes(ok):
    return 'identical' if ok else 'DIFFERENT'


# ---------------------------------------------------------------- tiny-model gates

def gate_m(U):
    results = notebook_regression.run(U, verbose=False)
    failed = [name for name, ok in results if not ok]
    return f'{len(results) - len(failed)}/{len(results)} identical' + (f', failed: {failed}' if failed else ''), not failed


def gate_g3(U):
    off = load_official('0731')
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    args = U['tiny_args']()
    b, w = 2, args.window_size
    ok = dict(logits=True, memory=True, picks=True)
    for seed in range(5):
        for s in LENGTHS:
            torch.manual_seed(1000 * seed + s)
            model = Transformer(args)
            theirs, _ = official_twin(off, model, args, b)
            ids = torch.randint(0, args.vocab_size, (b, s))
            state = SequenceState(args, b)
            with picks_of(model, theirs) as (ours, ref):
                ok['logits'] &= torch.equal(model(ids, state), theirs(ids, 0)[1])
            ok['picks'] &= same_picks(ours, ref)
            ok['memory'] &= all(torch.equal(x, y) for x, y in memory_pairs(state.layers, [l.attn for l in theirs.layers], w))
            all_position_logits(theirs)
            ok['logits'] &= torch.equal(model(ids, return_all_logits=True), theirs(ids, 0)[1])
    return f"logits {yes(ok['logits'])}, memory {yes(ok['memory'])}, picks {yes(ok['picks'])}", all(ok.values())


def gate_g3d(U):
    off = load_official('0731')
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    args = U['tiny_args']()
    b, D, w = 2, 40, args.window_size
    ok = dict(logits=True, memory=True, picks=True)
    for P in (1, 3, 4, 5, 8, 9, 16, 17, 33):
        torch.manual_seed(P)
        model = Transformer(args)
        theirs, _ = official_twin(off, model, args, b)
        ids = torch.randint(0, args.vocab_size, (b, P + D))
        state = SequenceState(args, b)
        with picks_of(model, theirs) as (ours, ref):
            ok['logits'] &= torch.equal(model(ids[:, :P], state), theirs(ids[:, :P], 0)[1])
            for t in range(P, P + D):
                ok['logits'] &= torch.equal(model(ids[:, t:t + 1], state), theirs(ids[:, t:t + 1], t)[1])
        ok['picks'] &= same_picks(ours, ref)
        ok['memory'] &= all(torch.equal(x, y) for x, y in memory_pairs(state.layers, [l.attn for l in theirs.layers], w))
    return f"logits at all 41 steps {yes(ok['logits'])}, picks {yes(ok['picks'])}, memory {yes(ok['memory'])}", all(ok.values())


def float64_tiny(U, seed, dspark=False):
    """The tiny model in float64 with 64 indexer heads, like the real model. With the tiny config's 4 heads, relu can
    zero every head, two entries tie at exactly 0, and topk may keep a different one of them in two runs that differ
    only by rounding."""
    args = U['tiny_args'](dspark=dspark)
    args.index_n_heads = 64
    torch.manual_seed(seed)
    return U['Transformer'](args).double(), args


def gate_g4(U):
    SequenceState = U['SequenceState']
    model, args = float64_tiny(U, 4)
    b, D = 2, 40
    worst, picks_ok, memory_ok = 0.0, True, True
    for P in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 20, 31, 33):
        ids = torch.randint(0, args.vocab_size, (b, P + D))
        one = SequenceState(args, b, dtype=torch.float64)
        with picks_of(model) as (p_one,):
            full = model(ids, one, return_all_logits=True)
        steps = SequenceState(args, b, dtype=torch.float64)
        with picks_of(model) as (p_steps,):
            outs = [model(ids[:, :P], steps)] + [model(ids[:, t:t + 1], steps) for t in range(P, P + D)]
        worst = max(worst, diff(torch.stack(outs, 1), full[:, P - 1:]))
        picks_ok &= same_picks_per_token(p_one, p_steps, b)
        memory_ok &= same_memory(live_memory(one, args), live_memory(steps, args), TOL)
    ok = worst <= TOL and picks_ok and memory_ok
    return f'logits max rel diff {worst:.0e} (tol 1e-12), picks {"same" if picks_ok else "DIFFERENT"}, memory {"same" if memory_ok else "DIFFERENT"}', ok


def gate_g4c(U):
    SequenceState = U['SequenceState']
    model, args = float64_tiny(U, 14)
    b, L = 2, 64
    worst, picks_ok, memory_ok = 0.0, True, True
    for P in range(1, 35):
        ids = torch.randint(0, args.vocab_size, (b, L))
        one = SequenceState(args, b, dtype=torch.float64)
        with picks_of(model) as (p_one,):
            full = model(ids, one, return_all_logits=True)
        for T in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17):
            state = SequenceState(args, b, dtype=torch.float64)
            with picks_of(model) as (p,):
                chunks = [model(ids[:, :P], state, return_all_logits=True)]
                while state.pos < L:
                    chunks.append(model(ids[:, state.pos:state.pos + T], state, return_all_logits=True))
            worst = max(worst, diff(torch.cat(chunks, 1), full))
            picks_ok &= same_picks_per_token(p_one, p, b)
            memory_ok &= same_memory(live_memory(one, args), live_memory(state, args), TOL)
    ok = worst <= TOL and picks_ok and memory_ok
    return f'logits max rel diff {worst:.0e} (tol 1e-12), picks {"same" if picks_ok else "DIFFERENT"}, memory {"same" if memory_ok else "DIFFERENT"}', ok


def gate_g5(U):
    SequenceState = U['SequenceState']
    model, args = float64_tiny(U, 5, dspark=True)

    def run(ids, L):
        """Prefill L tokens, an extend of 3, then 3 single tokens; a draft (temperature 0) after each.
        Returns every output and every memory tensor, batch dimension first."""
        state = SequenceState(args, ids.shape[0], dtype=torch.float64)
        logits, hidden = model(ids[:, :L], state, return_all_logits=True, return_hidden=True)
        outs = [logits, hidden]
        model.forward_spec(logits[:, -1].argmax(-1), hidden, state, 0.0)        # after a prefill: memory only
        logits, hidden = model(ids[:, L:L + 3], state, return_all_logits=True, return_hidden=True)
        outs += [logits, hidden, *model.forward_spec(logits[:, -1].argmax(-1), hidden, state, 0.0)]
        for t in range(L + 3, L + 6):
            logits, hidden = model(ids[:, t:t + 1], state, return_hidden=True)
            outs += [logits, hidden, *model.forward_spec(logits.argmax(-1), hidden, state, 0.0)]
        return outs + memory(state)

    isolation, guesses, causality, reached = 0.0, True, 0.0, True
    for L in LENGTHS:
        ids = torch.randint(0, args.vocab_size, (2, L + 6))
        together = run(ids, L)
        for r in (0, 1):
            alone = run(ids[r:r + 1], L)
            for x, y in zip(together, alone):
                if x.is_floating_point():
                    isolation = max(isolation, diff(x[r:r + 1], y))
                else:
                    guesses &= torch.equal(x[r:r + 1], y)                                # the drafts' token ids
        full = model(ids[:, :L], return_all_logits=True)
        for j in sorted({L // 2, L - 1}):
            changed = ids[:, :L].clone()
            changed[0, j] = (changed[0, j] + 1) % args.vocab_size
            out = model(changed, return_all_logits=True)
            causality = max(causality, diff(out[0, :j], full[0, :j]), diff(out[1], full[1]))
            reached &= not torch.equal(out[0, j], full[0, j])
    ok = isolation <= TOL and guesses and causality <= TOL and reached
    return (f'B=2 vs B=1: max rel diff {isolation:.0e}, guesses {"same" if guesses else "DIFFERENT"}; after a changed '
            f'token: earlier positions + other row {causality:.0e} (tol 1e-12), its own position changed {reached}'), ok


def gate_rb(U):
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    args = U['tiny_args'](dspark=True)
    b = 2
    torch.manual_seed(16)
    model = Transformer(args)

    def feed(state, toks):
        """Some tokens (a prefill at position 0, else an extend), then a draft."""
        logits, hidden = model(toks, state, return_hidden=True)
        model.forward_spec(logits.argmax(-1), hidden, state, 0.0)

    def go_on(state, toks):
        """An extend of 5, then single tokens; a draft after each."""
        logits, hidden = model(toks[:, :5], state, return_all_logits=True, return_hidden=True)
        outs = [logits, *model.forward_spec(logits[:, -1].argmax(-1), hidden, state, 0.0)]
        for t in range(5, toks.shape[1]):
            logits, hidden = model(toks[:, t:t + 1], state, return_hidden=True)
            outs += [logits, *model.forward_spec(logits.argmax(-1), hidden, state, 0.0)]
        return outs

    def same(s1, s2):
        return s1.pos == s2.pos and same_memory(memory(s1), memory(s2))

    ok = len(SequenceState(args, b).tensors()) == len(memory(SequenceState(args, b)))
    for P in (3, 4, 9, 16, 17, 33):
        for k in (1, 13):
            right = torch.randint(0, args.vocab_size, (b, P + 12))
            wrong = torch.randint(0, args.vocab_size, (b, k))
            direct, state = SequenceState(args, b), SequenceState(args, b)
            feed(direct, right[:, :P])
            feed(state, right[:, :P])
            want = go_on(direct, right[:, P:])
            snapshot = state.clone()
            feed(state, wrong)
            changed = not same(state, snapshot)
            state.restore(snapshot)
            back = same(state, snapshot)
            got = go_on(state, right[:, P:])
            ok &= changed and back and all(torch.equal(x, y) for x, y in zip(got, want)) and same(state, direct)
    for P in (5, 17):
        prefix = torch.randint(0, args.vocab_size, (b, P))
        parent = SequenceState(args, b)
        feed(parent, prefix)
        before = parent.clone()
        for _ in range(2):
            more = torch.randint(0, args.vocab_size, (b, 9))
            fork, direct = parent.clone(), SequenceState(args, b)
            feed(direct, prefix)
            ok &= all(torch.equal(x, y) for x, y in zip(go_on(fork, more), go_on(direct, more)))
        ok &= same(parent, before)
    with torch.inference_mode():
        state = SequenceState(args, b)
    feed(state, torch.randint(0, args.vocab_size, (b, 5)))
    snapshot = state.clone()
    feed(state, torch.randint(0, args.vocab_size, (b, 13)))
    state.restore(snapshot)
    ok &= same(state, snapshot)
    n = len(SequenceState(args, b).tensors())
    return f'outputs and memory ({n} tensors) {yes(ok)} to the direct runs', ok


def gate_q(U):
    off = load_official('0731')
    fp8, fp4 = U['fp8_simulate'], U['fp4_simulate']
    torch.manual_seed(0)
    edge = torch.zeros(2, 448)
    edge[0, :6] = torch.tensor([448.0, 1.0625, 1.1875, 17.0, 19.0, 0.0029296875])
    edge[1, :8] = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    ok = True
    for dtype in (torch.float32, torch.bfloat16):
        for x in [edge] + [torch.randn(64, 448) * sc for sc in (1e-6, 1e-2, 1.0, 300.0, 1e6)] + [torch.zeros(2, 448)]:
            x = x.to(dtype)
            r8, r4 = x.clone(), x.clone()
            off.act_quant(r8, 64, 'ue8m0', torch.float32, True)
            off.fp4_act_quant(r4, 32, True)
            ok &= torch.equal(fp8(x), r8) and torch.equal(fp4(x), r4)
    return f'FP8 and FP4 {yes(ok)}', ok


def gate_smp(U):
    off = load_official('0731')
    sample, generate = U['sample'], U['generate']
    torch.manual_seed(0)
    same = []
    for T in (0.0, 0.6, 1.0, 1.5):
        for shape in ((3, 512), (2, 5, 512)):
            logits = torch.randn(shape) * 3
            torch.manual_seed(7)
            ours = sample(logits, T)
            torch.manual_seed(7)
            same.append(torch.equal(ours, off.sample(logits, T)))
            same.append(torch.equal(sample(logits, T, torch.Generator().manual_seed(7)), ours))
    torch.manual_seed(0)
    args = U['tiny_args']()
    model = U['Transformer'](args)
    prompt = torch.randint(0, args.vocab_size, (2, 20))
    a, b = (generate(model, args, prompt, 30, temperature=1.0, seed=1) for _ in range(2))
    c = generate(model, args, prompt, 30, temperature=1.0, seed=2)
    greedy = generate(model, args, prompt, 30)
    state = U['SequenceState'](args, 2)
    logits, ref = model(prompt, state), []
    for _ in range(30):
        ref.append(logits.argmax(dim=-1, keepdim=True))
        logits = model(ref[-1], state)
    seeds = torch.equal(a, b) and not torch.equal(a, c) and torch.equal(greedy, torch.cat(ref, dim=1))
    return f'chosen tokens {yes(all(same))}; same seed same tokens, other seed different, T = 0 == argmax loop: {seeds}', all(same) and seeds


def gate_dsp(U):
    Transformer, SequenceState = U['Transformer'], U['SequenceState']
    off = load_official('0731')
    args = U['tiny_args'](dspark=True)
    w, b, D = args.window_size, 2, 12
    ok = True
    for P in (1, 3, 8, 9, 17):
        for T in (0.0, 1.0):
            torch.manual_seed(P)
            model = Transformer(args)
            theirs, names = official_twin(off, model, args, b, n_mtp_layers=args.n_mtp_layers)
            for stage in theirs.mtp:
                stage.temperature = T
            ids = torch.randint(0, args.vocab_size, (b, P + D))
            state = SequenceState(args, b)
            lo, ho = model(ids[:, :P], state, return_hidden=True)
            _, lt, ht = theirs(ids[:, :P], 0)
            ok &= names and torch.equal(lo, lt) and torch.equal(ho, ht)
            ok &= model.forward_spec(lo.argmax(-1), ho, state, T) is None and theirs.forward_spec(lo.argmax(-1), ht, 0) is None
            for t in range(P, P + D):
                lo, ho = model(ids[:, t:t + 1], state, return_hidden=True)
                _, lt, ht = theirs(ids[:, t:t + 1], t)
                ok &= torch.equal(lo, lt) and torch.equal(ho, ht)
                torch.manual_seed(1000 + t)
                ours = model.forward_spec(lo.argmax(-1), ho, state, T)
                torch.manual_seed(1000 + t)
                ref = theirs.forward_spec(lo.argmax(-1), ht, t)
                ok &= all(torch.equal(x, y) for x, y in zip(ours, ref))
            ok &= all(torch.equal(s.window, m.attn.kv_cache[:b, :w]) for s, m in zip(state.mtp, theirs.mtp))
    model, args64 = float64_tiny(U, 0, dspark=True)
    worst, guesses, memory_ok = 0.0, True, True
    for P in (1, 3, 4, 8, 9, 17):
        for s in (2, 3, 5, 8, 9, 16):
            ids = torch.randint(0, args64.vocab_size, (b, P + s))
            one_go, one_by_one = (SequenceState(args64, b, dtype=torch.float64) for _ in range(2))
            for st in (one_go, one_by_one):
                lo, ho = model(ids[:, :P], st, return_hidden=True)
                model.forward_spec(lo.argmax(-1), ho, st, 0.0)
            lo, ho = model(ids[:, P:], one_go, return_hidden=True)
            ra = model.forward_spec(lo.argmax(-1), ho, one_go, 0.0)
            for t in range(P, P + s):
                lo, ho = model(ids[:, t:t + 1], one_by_one, return_hidden=True)
                rb = model.forward_spec(lo.argmax(-1), ho, one_by_one, 0.0)
            guesses &= torch.equal(ra[0], rb[0])
            worst = max([worst] + [diff(x, y) for x, y in zip(ra[1:], rb[1:])])
            memory_ok &= same_memory(memory(one_go), memory(one_by_one), TOL)
    ok64 = guesses and worst <= TOL and memory_ok
    return (f'vs official: names, outputs, drafter memory {yes(ok)}; extend then draft: guesses {yes(guesses)}, '
            f'max rel diff {worst:.0e}, memory {"same" if memory_ok else "DIFFERENT"}'), ok and ok64


def gate_gen(U):
    off = load_official('0731')
    their_generate = official_generate(off)
    args = U['tiny_args']()
    torch.manual_seed(1)
    prompts = [torch.randint(0, args.vocab_size, (n,)).tolist() for n in (11, 12, 21)]

    def case(batch, eos_id):
        torch.manual_seed(0)
        model = U['Transformer'](args)
        theirs, _ = official_twin(off, model, args, len(batch))
        ours = U['generate'](model, args, batch, 20, eos_id=eos_id)
        ref = their_generate(theirs, batch, 20, eos_id)
        cut = [[t for t in row if t >= 0] for row in ours.tolist()]
        cut = [(row[: row.index(eos_id)] if eos_id in row else row) + [eos_id] for row in cut]
        return ours, cut == ref

    free, ok = case(prompts, args.vocab_size)
    stop = free[0, 2].item()
    for batch, eos_id in ((prompts, free[0, 4].item()), (prompts[2:], free[2, 6].item()), ([prompts[0] * 23], args.vocab_size),
                          ([prompts[0], prompts[2][:15] + [stop] + prompts[2][16:]], stop)):
        ok &= case(batch, eos_id)[1]
    return f'token for token {yes(ok)}', ok


def gate_tok(U):
    chat = U['ChatTokenizer'](str(CHECKPOINT / 'tokenizer.json'))
    parse = U['parse_message_from_completion_text']
    tests = CHECKPOINT / 'encoding' / 'tests'
    same = [chat.tokenizer.decode(chat.tokenizer.encode(text).ids, skip_special_tokens=False) == text
            for text in ((tests / f'test_output_{k}.txt').read_text() for k in (1, 2, 3, 4))]
    messages = [{'role': 'user', 'content': 'What is 2+2?'}]
    ids_ok = chat.encode_chat(messages) == EXPECTED_IDS and chat.encode_chat(messages, 'thinking')[-1] == 128821
    reply = chat.tokenizer.encode('2 + 2 = 4.').ids
    decoded = [chat.decode_completion(row) for row in (reply + [chat.eos_id, 77, 88, -1, -1], reply + [-1, -1])]
    decode_ok = all(t == '2 + 2 = 4.<｜end▁of▁sentence｜>' and parse(t, 'chat')['content'] == '2 + 2 = 4.' for t in decoded)
    with contextlib.redirect_stdout(io.StringIO()):
        official = importlib.import_module('test_encoding_dsv4')
        for k in (1, 2, 3, 4):
            getattr(official, f'test_case_{k}')()           # raises if the official encoding changed
    ok = all(same) and ids_ok and decode_ok
    return f'round trips {all(same)}, ids {ids_ok}, decode_completion {decode_ok}, official tests pass', ok


def gate_cfg(U):
    cfg = json.loads((CHECKPOINT / 'inference' / 'config.json').read_text())
    default, loaded = U['ModelArgs'](), U['ModelArgs'](**cfg)
    fields_ok = all(getattr(default, k) == getattr(loaded, k) for k in cfg)
    try:
        with torch.device('meta'):
            real = U['Transformer'](U['ModelArgs']())
        ours = set(real.state_dict())
        del real
    finally:
        U['precompute_theta_cis'].cache_clear()   # the meta build cached meta RoPE tables (the cache key has no device)
    index = json.loads((CHECKPOINT / 'model.safetensors.index.json').read_text())['weight_map']
    names = {k for k in index if not k.endswith('.scale')}
    return f'{len(cfg)} config fields equal {fields_ok}; names {len(ours):,} vs checkpoint {len(names):,}: {yes(ours == names)}', fields_ok and ours == names


def gate_cost(U):
    args = U['tiny_args']()
    torch.manual_seed(0)
    model = U['Transformer'](args)
    ids = torch.randint(0, args.vocab_size, (1, 21))
    state = U['SequenceState'](args, 1)
    counts = [counted(lambda: model(chunk, state)) for chunk in (ids[:, :20], ids[:, 20:])]
    ok = all(not reads and copies == args.n_layers for _, reads, copies in counts)
    (_, _, _), (ops, reads, copies) = counts
    return f'{ops:,} ATen ops; host reads: {copies} routing copies (1 per MoE layer), other {reads or "none"}', ok


# ---------------------------------------------------------------- real-size gates

LAYERS = (0, 2, 3)


def layer_type(args, layer_id):
    kind = {0: 'window only', 4: 'CSA', 128: 'HCA'}[args.compress_ratios[layer_id]]
    return kind + (', hash' if layer_id < args.n_hash_layers else ', learned')


def real_args(U, **kw):
    return U['ModelArgs'](n_routed_experts=8, **kw)


def gate_g3r(U):
    off = load_official('0731')
    args = real_args(U)
    oargs = official_args(off, args)
    s, w, bad = 2304, args.window_size, []
    for layer_id in LAYERS:
        torch.manual_seed(layer_id)
        ours = U['Block'](layer_id, args)
        theirs = off.Block(layer_id, oargs).float()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        x = torch.randn(1, s, args.hc_mult, args.dim)
        ids = torch.randint(0, args.vocab_size, (1, s))
        st = U['SequenceState'](args, 1).layers[layer_id]
        with torch.inference_mode(), picks_of(ours, theirs) as (po, pt):
            same = torch.equal(ours(x, ids, st, 0), theirs(x, 0, ids))
        mem = all(torch.equal(p, q) for p, q in memory_pairs([st], [theirs.attn], w))
        bad += [f'layer {layer_id} {what}' for what, ok in (('output', same), ('memory', mem), ('picks', same_picks(po, pt))) if not ok]
        del ours, theirs
    return 'output, memory and picks identical' if not bad else f'DIFFERENT: {bad}', not bad


def gate_g4r(U):
    args = real_args(U)
    P, D = 2300, 16
    worst, bad = 0.0, []
    for layer_id in LAYERS:
        torch.manual_seed(layer_id)
        block = U['Block'](layer_id, args).double()
        x = torch.randn(1, P + D, args.hc_mult, args.dim, dtype=torch.float64)
        ids = torch.randint(0, args.vocab_size, (1, P + D))
        one = U['SequenceState'](args, 1, dtype=torch.float64).layers[layer_id]
        steps = U['SequenceState'](args, 1, dtype=torch.float64).layers[layer_id]
        with torch.inference_mode():
            with picks_of(block) as (p_one,):
                full = block(x, ids, one, 0)
            with picks_of(block) as (p_steps,):
                outs = [block(x[:, :P], ids[:, :P], steps, 0)]
                outs += [block(x[:, t:t + 1], ids[:, t:t + 1], steps, t) for t in range(P, P + D)]
        worst = max(worst, diff(torch.cat(outs, 1), full))
        bad += [f'layer {layer_id} {what}' for what, ok in (('picks', same_picks_per_token(p_one, p_steps, 1)),
                                                             ('memory', same_memory(live_layer(one, args.compress_ratios[layer_id], P + D), live_layer(steps, args.compress_ratios[layer_id], P + D), TOL))) if not ok]
        del block
    ok = worst <= TOL and not bad
    return f'max rel diff {worst:.0e} (tol 1e-12), picks and memory ' + ('same' if not bad else f'DIFFERENT: {bad}'), ok


def gate_g7m(U):
    off = load_official('0731')
    args = real_args(U, max_seq_len=8192)        # 4,096 + 64 positions: longer than the default RoPE table of 4,096
    oargs = official_args(off, args)
    P, D, w = 4096, 64, args.window_size
    rows, ok = [], True
    for layer_id in LAYERS:
        torch.manual_seed(layer_id)
        ours = U['Block'](layer_id, args)
        x = torch.randn(1, P + D, args.hc_mult, args.dim)
        ids = torch.randint(0, args.vocab_size, (1, P + D))
        st = U['SequenceState'](args, 1).layers[layer_id]
        with torch.inference_mode(), picks_of(ours) as (po,), PeakMemory() as mem:
            t0 = time.perf_counter()
            outs = [ours(x[:, :P], ids[:, :P], st, 0)]
            t1 = time.perf_counter()
            outs += [ours(x[:, t:t + 1], ids[:, t:t + 1], st, t) for t in range(P, P + D)]
            t2 = time.perf_counter()
        theirs = off.Block(layer_id, oargs).float()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        with torch.inference_mode(), picks_of(theirs) as (pt,):
            ref = [theirs(x[:, :P], 0, ids[:, :P])] + [theirs(x[:, t:t + 1], t, ids[:, t:t + 1]) for t in range(P, P + D)]
        same = all(torch.equal(a, b) for a, b in zip(outs, ref)) and all(torch.equal(p, q) for p, q in memory_pairs([st], [theirs.attn], w))
        picks = same_picks(po, pt)
        ok &= same and picks
        rows.append(f'| {layer_id} | {layer_type(args, layer_id)} | 4,096 + 64 | fp32 | {t1 - t0:.1f} | {(t2 - t1) / D * 1000:.0f} | '
                    f'{mem.peak / 2**30:.1f} | {mem.before / 2**30:.1f} | {"equal" if same else "DIFFERENT"} | {"identical" if picks else "DIFFERENT"} |')
        del ours, theirs, outs, ref
    table = ['| layer | type | tokens | dtype | prefill s | decode ms/token | peak RSS GiB | RSS at start GiB | vs official | picks |',
             '|---|---|---|---|---|---|---|---|---|---|'] + rows
    return f'outputs at all 65 steps, memory, picks {yes(ok)}; Mac numbers ({torch.get_num_threads()} threads):\n' + '\n'.join(table), ok


def gate_dspr(U):
    off = load_official('0731')
    args = real_args(U)
    oargs = official_args(off, args)
    torch.manual_seed(43)
    ours = U['DSparkBlock'](43, args)
    theirs = off.DSparkBlock(43, oargs).float()
    theirs.load_state_dict(ours.state_dict(), strict=True)
    k, P, w = args.dspark_block_size, 300, args.window_size
    state = U['SequenceState'](args, 1).mtp[0]
    ids = torch.randint(0, args.vocab_size, (1, k))
    with torch.inference_mode():
        x, main_x = torch.randn(1, k, args.hc_mult, args.dim), torch.randn(1, P, args.dim)
        ok = torch.equal(ours(x, ids, state, 0, main_x), theirs(x, 0, ids, main_x))
        for t in range(P, P + 4):
            x, main_x = torch.randn(1, k, args.hc_mult, args.dim), torch.randn(1, 1, args.dim)
            ok &= torch.equal(ours(x, ids, state, t, main_x), theirs(x, t, ids, main_x))
    ok &= torch.equal(state.window, theirs.attn.kv_cache[:1, :w])
    return f'outputs and drafter memory {yes(ok)}', ok


# ---------------------------------------------------------------- the table and the runner

GATES = [   # name, what, setup, function, real size
    ('M', 'modules vs official', '9 modules, fp32 + bf16', gate_m, False),
    ('G3', 'prefill vs official', f'fp32, 5 seeds x {len(LENGTHS)} lengths (1-100), b = 2', gate_g3, False),
    ('G3d', 'decode vs official', 'fp32, 9 prompt lengths P (1-33) + 40 single tokens, b = 2', gate_g3d, False),
    ('G4', 'prefill + decode == one prefill', 'fp64, 14 prompt lengths P (1-33) + 40 single tokens', gate_g4, False),
    ('G4c', 'extend == one prefill', 'fp64, 34 starts x 10 chunk sizes (1-17), 64 tokens', gate_g4c, False),
    ('G5', 'causality + batch isolation', f'fp64, {len(LENGTHS)} lengths, B=2 vs 2 x B=1, prefill/extend/decode/drafts', gate_g5, False),
    ('RB', 'snapshot / restore', 'fp32, 12 rollbacks + 2 forks, drafter on', gate_rb, False),
    ('Q', 'FP8 / FP4 rounding', 'vs official kernels, fp32 + bf16, edge rows', gate_q, False),
    ('SMP', 'sampler', 'vs official, 4 temperatures x 2 shapes; seeds', gate_smp, False),
    ('DSP', 'DSpark vs official 0731', 'fp32, 5 P x 2 temperatures x 12 drafts; fp64 36 extends', gate_dsp, False),
    ('GEN', 'generate vs official', '5 cases: mixed lengths, stops, max_seq_len, EOS in prompt', gate_gen, False),
    ('TOK', 'tokenizer + chat encoding', '4 official conversations, 11 ids, parser', gate_tok, False),
    ('CFG', 'real config + weight names', '0731 config.json, checkpoint index (meta device)', gate_cfg, False),
    ('cost', 'one tiny decode step', 'ATen ops, host reads', gate_cost, False),
    ('G3r', 'real size: prefill vs official', 'fp32, layers 0 / 2 / 3, 2,304 tokens', gate_g3r, True),
    ('G4r', 'real size: decode == prefill', 'fp64, layers 0 / 2 / 3, 2,300 + 16 single tokens', gate_g4r, True),
    ('G7m', 'real size: long run vs official', 'fp32, layers 0 / 2 / 3, 4,096 + 64 single tokens', gate_g7m, True),
    ('DSPr', 'real size: DSpark stage', 'fp32, layer 43, prefill 300 + 4 drafts', gate_dspr, True),
]


def header(real_size):
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'none (no git yet)'
    return (f'gates  commit {commit}  torch {torch.__version__}  {datetime.date.today()}  {torch.get_num_threads()} threads  '
            f'real size {"on" if real_size else "off"}')


def run_gates(U, real_size=False, only=None):
    """Run the gates on the notebook's definitions U (its globals(), or load_notebook_defs()); print one line per gate.
    real_size: also the real-size gates (minutes, ~10 GiB). only: names to run, e.g. ['G5']. Returns {name: passed}."""
    set_quant_simulation('fp8_simulate' in U)
    print(header(real_size))
    results, skipped = {}, []
    for name, what, setup, gate, big in GATES:
        if only and name not in only:
            continue
        if big and not real_size:
            skipped.append(name)
            continue
        t0 = time.perf_counter()
        try:
            result, ok = gate(U)
        except Exception as e:
            result, ok = f'ERROR {type(e).__name__}: {str(e)[:150]}', False
        first, *more = result.split('\n')
        print(f'{name:5s}{"PASS" if ok else "FAIL":6s}{time.perf_counter() - t0:4.0f} s  {what} | {setup} | {first}', flush=True)
        for line in more:
            print('      ' + line)
        results[name] = ok
    passed = sum(results.values())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20)
    print(f'{passed}/{len(results)} PASS' + ('' if passed == len(results) else f'  FAILED: {[n for n, ok in results.items() if not ok]}')
          + (f'  (skipped, real size off: {", ".join(skipped)})' if skipped else f'  (process peak memory so far {peak:.1f} GiB)'))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2 gates on the saved notebook (or V4_NOTEBOOK)')
    parser.add_argument('--real-size', action='store_true', help='also run the real-size gates (minutes, ~10 GiB)')
    parser.add_argument('--only', help='comma-separated gate names, e.g. G5,RB')
    a = parser.parse_args()
    results = run_gates(load_notebook_defs(), real_size=a.real_size, only=a.only.split(',') if a.only else None)
    sys.exit(0 if results and all(results.values()) else 1)

"""Load an official DeepSeek-V4 model.py so it runs on a CPU, for checks.

    import sys; sys.path.insert(0, "checks")
    from load_official import load_official, official_args
    off = load_official("april")                     # or "0731"
    theirs = off.Block(layer_id, official_args(off, args)).float()

How:
- checks/shim/ goes first on sys.path, so model.py's `from kernel import ...` gets the
  pure-PyTorch stand-ins (shim/kernel.py), and rotate_activation's
  `import fast_hadamard_transform` gets shim/fast_hadamard_transform.py.
- One patch: the official rotate_activation asserts that x is bf16. The patched copy is the same
  function without that assert, so float32 and float64 checks work.
- The official code is otherwise unchanged. It reads its files from deepseek_official/<rev>/.

official_args(off, args) maps our ModelArgs to the official ModelArgs: the same field values,
with float weights (dtype "bf16", no FP8/FP4 formats). The official Linear then creates bf16
weights; call .float() (or .double()) on the built module before running it.

Scales: the real config has scale_fmt "ue8m0" (power-of-two scales in the FP8 simulation).
official_args passes it, and load_official also sets the module-level scale_fmt, because a Block
or Attention built directly (without the official Transformer) reads that global, whose
default in model.py is None. Whether the rounding runs at all is the shim's SIMULATE_QUANT.

Real weights (Phase 5): the checkpoint's own FP8 / FP4 tensors in the official code, as a reference that shares no
loading or decoding code with the notebook.
    off = load_official("0731", real_weights=True)
    oargs = off.ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    theirs = official_module(off, lambda: off.Attention(3, oargs), "~/v4_ckpt", "layers.3.attn.")
    exact, fp8 = official_run(off, lambda: theirs(x, 0))
"""
import ast
import importlib.util
import json
import sys
import time
from typing import List
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "checks" / "shim"


def load_official(rev: str = "april", real_weights: bool = False):
    """real_weights=True: Linear layers get the checkpoint's formats (FP8 weights with E8M0 scales, FP4 experts), as the
    official Transformer.__init__ sets them for the real config (dtype "fp8", scale_dtype "fp8")."""
    path = ROOT / "deepseek_official" / rev / "inference" / "model.py"
    if str(SHIM) not in sys.path:
        sys.path.insert(0, str(SHIM))
    spec = importlib.util.spec_from_file_location(f"official_{rev}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def rotate_activation(x):
        # official body without `assert x.dtype == torch.bfloat16`
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)

    mod.rotate_activation = rotate_activation
    mod.scale_fmt = "ue8m0"
    if real_weights:
        mod.default_dtype, mod.scale_dtype = torch.float8_e4m3fn, torch.float8_e8m0fnu
    return mod


STORED = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8,
          "BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}


def read_checkpoint(folder, names, rev: str = "0731"):
    """{name: tensor} straight from the .safetensors files, in their stored types (FP4 experts are I8, two values per
    byte). Written apart from the notebook's reader, so the reference shares no code with it."""
    folder = Path(folder).expanduser()
    file_of = json.load(open(ROOT / "deepseek_official" / rev / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for file in sorted({file_of[n] for n in names}):
        with open(folder / file, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
            for name in (x for x in names if file_of[x] == file):
                begin, end = header[name]["data_offsets"]
                f.seek(8 + n + begin)
                out[name] = torch.frombuffer(bytearray(f.read(end - begin)), dtype=STORED[header[name]["dtype"]])
                out[name] = out[name].view(header[name]["shape"])
    return out


def official_module(off, build, folder, prefix: str, skip=(), device="cpu"):
    """An official module holding the checkpoint's own tensors, the way DeepSeek's code gets them:
    - generate.py builds the model with BF16 as the default dtype (here directly on `device`);
    - convert.py decodes wo_a to BF16 with its scales, and views the experts' I8 bytes as FP4 (expert_dtype "fp4");
    - generate.py's load_model copies everything else into the module's own types (BF16 norms -> float32, I64
      tid2eid -> int32).
    build: a function that builds the module, e.g. lambda: off.Attention(3, oargs). prefix: its checkpoint names'
    start. skip: name starts left unloaded (uninitialized), e.g. experts that no token picks.
    Every tensor is set with `param.data = ...`, which keeps the Parameter objects (Linear's weight.scale points at its
    scale Parameter), and the 1-byte formats (FP8, FP4, E8M0) travel to the device as plain bytes."""
    with torch.device(device), off.set_dtype(torch.bfloat16):
        m = build()
    params = dict(m.named_parameters())
    names = [n for n in m.state_dict() if not n.startswith(skip)]
    extra = [n.removesuffix("weight") + "scale" for n in names if n.endswith("wo_a.weight")]
    raw = read_checkpoint(folder, [prefix + n for n in names + extra])
    for n in names:
        t, p = raw[prefix + n], params[n]
        if n.endswith("wo_a.weight"):
            s = raw[prefix + n.removesuffix("weight") + "scale"]
            t = t.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * s[:, None, :, None].float()
            t = t.flatten(2, 3).flatten(0, 1).bfloat16()
        elif t.dtype == torch.int8:
            t = t.view(torch.float4_e2m1fn_x2)
        assert t.shape == p.shape, f"{prefix + n}: checkpoint {list(t.shape)}, module {list(p.shape)}"
        if t.element_size() == 1:
            assert t.dtype == p.dtype, f"{prefix + n}: checkpoint {t.dtype}, module {p.dtype}"
            t = t.view(torch.uint8).to(device).view(p.dtype)
        else:
            t = t.to(device, p.dtype)
        p.data = t
    return m


def _run(off, fn, fp8):
    import kernel
    kernel.FP8_ACTIVATIONS = fp8
    try:
        with torch.inference_mode(), off.set_dtype(torch.bfloat16):
            return fn()
    finally:
        kernel.FP8_ACTIVATIONS = True


def official_run(off, fn):
    """fn() run as generate.py runs the model (BF16 default dtype, no gradients), twice: with exact matmul inputs
    (the plan's XD mode) and with the official FP8 rounding of them (OE). Returns (exact, fp8)."""
    return [_run(off, fn, fp8) for fp8 in (False, True)]


def _compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _picks(gate, fn):
    """The experts the router picks while fn() runs, sorted per token: [tokens, 6]."""
    got = []
    hook = gate.register_forward_hook(lambda module, inputs, out: got.append(out[1].long().sort(-1).values))
    try:
        fn()
    finally:
        hook.remove()
    return got[0]


class Reference:
    """DeepSeek's model next to ours, layer by layer: the after_layer of the notebook's prefill_by_layer.
    - At each layer, DeepSeek's Block (built from the checkpoint's own tensors, official_module) runs on our input, with
      exact matmul inputs (XD) and with its FP8 rounding (OE). Both are compared with our output on what the block
      adds (output - input), because the input passes through unchanged and would hide the differences. Its router's
      expert picks are compared with ours (our layer runs once more, with a hook on its gate).
    - DeepSeek also keeps its own chain (XD), from its embedding to its head: finish() prints its top-5 next to ours,
      and how far its chain is from ours after every layer.
    - finish() saves our timings, the comparisons and both top-5 lists to results/first_token_<host>_<time>.json/.txt.
    Needs the embedding and head files on disk when it is created."""

    def __init__(self, folder, prompts, device="cuda", rev="0731"):
        self.off = off = load_official(rev, real_weights=True)
        self.oargs = a = off.ModelArgs(**json.load(open(ROOT / "deepseek_official" / rev / "inference" / "config.json")))
        self.folder, self.device, self.rows = folder, device, []
        emb = official_module(off, lambda: off.ParallelEmbedding(a.vocab_size, a.dim), folder, "embed.", device=device)
        self.r = [_run(off, lambda: emb(ids).unsqueeze(2).repeat(1, 1, a.hc_mult, 1), False) for ids in prompts]
        self.norm = official_module(off, lambda: off.RMSNorm(a.dim, a.norm_eps), folder, "norm.", device=device)
        self.head = official_module(off, lambda: off.ParallelHead(a.vocab_size, a.dim, a.norm_eps, a.hc_eps), folder,
                                    "head.", device=device)
        hc = read_checkpoint(folder, ["hc_head_fn", "hc_head_scale", "hc_head_base"])
        self.hc = [hc[n].to(device, torch.float32) for n in ("hc_head_fn", "hc_head_scale", "hc_head_base")]

    def __call__(self, i, layer, hs, outs, prompts):
        off, a = self.off, self.oargs
        t = time.perf_counter()
        theirs = official_module(off, lambda: off.Block(i, a), self.folder, f"layers.{i}.", device=self.device)
        load = time.perf_counter() - t
        t = time.perf_counter()
        for k, ids in enumerate(prompts):
            got = []
            hook = theirs.ffn.gate.register_forward_hook(lambda module, inputs, out: got.append(out[1].long().sort(-1).values))
            exact, fp8 = official_run(off, lambda: theirs(hs[k], 0, ids))
            hook.remove()
            ours = _picks(layer.ffn.gate, lambda: layer(hs[k], ids))
            self.r[k] = _run(off, lambda: theirs(self.r[k], 0, ids), False)
            row = dict(layer=i, prompt=k, xd=_compare(outs[k] - hs[k], exact - hs[k]),
                       oe=_compare(outs[k] - hs[k], fp8 - hs[k]),
                       same_experts=(ours == got[0]).all(-1).float().mean().item(), tokens=ours.shape[0],
                       chain=_compare(outs[k], self.r[k]), load=load)
            self.rows.append(row)
            print(f"    DeepSeek, prompt {k}: exact rel err {row['xd'][0]:.1e} cos {row['xd'][1]:.6f} | FP8 rounding rel err "
                  f"{row['oe'][0]:.1e} | same experts {row['same_experts'] * 100:.0f}% of {row['tokens']} tokens | "
                  f"its own chain vs ours: rel err {row['chain'][0]:.1e}")
        print(f"    DeepSeek's layer: load {load:.1f} s, runs {time.perf_counter() - t:.1f} s")
        del theirs

    def top(self, k, chat, n=5):
        """DeepSeek's top-n next tokens for prompt k, from its own chain: [(text, log-probability)]."""
        off, a = self.off, self.oargs
        eps = type("Eps", (), dict(norm_eps=a.norm_eps, hc_eps=a.hc_eps))()     # all that Block.hc_head reads from self
        logits = _run(off, lambda: self.head(self.norm(off.Block.hc_head(eps, self.r[k], *self.hc))), False)
        lp = logits.float().log_softmax(-1)[0].topk(n)
        return [(chat.tokenizer.decode([t]), v) for t, v in zip(lp.indices.tolist(), lp.values.tolist())]

    def finish(self, log, logits, chat, n=5, out_dir=ROOT / "results"):
        """Prints both top-n lists per prompt and saves everything to out_dir (default results/)."""
        sys.path.insert(0, str(ROOT / "i_o_check"))
        from bench_storage import meta, save
        res = dict(meta=meta(), torch=torch.__version__, layers=log, deepseek=self.rows, prompts=[])
        lines = []
        for k, l in enumerate(logits):
            lp = l.float().log_softmax(-1)[0].topk(n)
            ours = [(chat.tokenizer.decode([t]), v) for t, v in zip(lp.indices.tolist(), lp.values.tolist())]
            theirs = self.top(k, chat, n)
            res["prompts"].append(dict(ours=ours, deepseek=theirs))
            for who, top in (("ours    ", ours), ("DeepSeek", theirs)):
                lines.append(f"prompt {k} {who}: " + " | ".join(f"{t!r} {v:.2f}" for t, v in top))
        worst = max(self.rows, key=lambda r: r["xd"][0])
        lines.append(f"worst layer vs DeepSeek (exact): L{worst['layer']:02d} prompt {worst['prompt']}, rel err "
                     f"{worst['xd'][0]:.1e}; experts identical in {sum(r['same_experts'] == 1 for r in self.rows)} of "
                     f"{len(self.rows)} layer-prompt pairs")
        lines.append(f"ours: load {sum(r['load'] for r in log):.0f} s, forward {sum(sum(r['forward']) for r in log):.1f} s, "
                     f"waited for downloads {sum(r['waited'] for r in log):.0f} s")
        text = "\n".join(lines)
        stem = save(res, text, "first_token", out_dir)
        print(text + f"\nsaved {stem}.json and .txt")


def official_args(off, args, max_batch_size: int = 1, n_mtp_layers: int = 0):
    names = {f.name for f in fields(off.ModelArgs)}
    values = {k: v for k, v in vars(args).items() if k in names}
    values.update(max_batch_size=max_batch_size, n_mtp_layers=n_mtp_layers,
                  dtype="bf16", scale_fmt="ue8m0", expert_dtype=None, scale_dtype="fp32")
    return off.ModelArgs(**values)


def official_generate(off, rev: str = "0731"):
    """The official generate() from inference/generate.py, bound to the official module `off`.
    generate.py itself can't be imported here (it imports transformers), so only that function is taken
    from its source. With 0731, sampling happens inside Transformer.forward at model.temperature
    (set it to 0 for greedy)."""
    path = ROOT / "deepseek_official" / rev / "inference" / "generate.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    ns = {"torch": torch, "List": List, "Transformer": off.Transformer}
    exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns["generate"]

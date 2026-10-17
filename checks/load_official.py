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


def official_module(off, build, folder, prefix: str, skip=()):
    """An official module holding the checkpoint's own tensors, the way DeepSeek's code gets them:
    - generate.py builds the model with BF16 as the default dtype;
    - convert.py decodes wo_a to BF16 with its scales, and views the experts' I8 bytes as FP4 (expert_dtype "fp4");
    - generate.py's load_model copies everything else into the module's own types (load_state_dict: BF16 norms ->
      float32, I64 tid2eid -> int32).
    build: a function that builds the module, e.g. lambda: off.Attention(3, oargs). prefix: its checkpoint names'
    start. skip: name starts left unloaded (uninitialized), e.g. experts that no token picks."""
    with off.set_dtype(torch.bfloat16):
        m = build()
    names = [n for n in m.state_dict() if not n.startswith(skip)]
    extra = [n.removesuffix("weight") + "scale" for n in names if n.endswith("wo_a.weight")]
    raw = read_checkpoint(folder, [prefix + n for n in names + extra])
    tensors = {}
    for n in names:
        t = raw[prefix + n]
        if n.endswith("wo_a.weight"):
            s = raw[prefix + n.removesuffix("weight") + "scale"]
            t = t.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * s[:, None, :, None].float()
            t = t.flatten(2, 3).flatten(0, 1).bfloat16()
        elif t.dtype == torch.int8:
            t = t.view(torch.float4_e2m1fn_x2)
        tensors[n] = t
    missing, unexpected = m.load_state_dict(tensors, strict=False)
    assert not unexpected and all(n.startswith(skip) for n in missing), (missing[:3], unexpected[:3])
    return m


def official_run(off, fn):
    """fn() run as generate.py runs the model (BF16 default dtype, no gradients), twice: with exact matmul inputs
    (the plan's XD mode) and with the official FP8 rounding of them (OE). Returns (exact, fp8)."""
    import kernel
    outs = []
    try:
        for fp8 in (False, True):
            kernel.FP8_ACTIVATIONS = fp8
            with torch.inference_mode(), off.set_dtype(torch.bfloat16):
                outs.append(fn())
    finally:
        kernel.FP8_ACTIVATIONS = True
    return outs


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

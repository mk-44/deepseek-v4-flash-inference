"""Load an official DeepSeek-V4 model.py so it runs on a CPU, for checks.

    import sys; sys.path.insert(0, "reference")
    from load_official import load_official, official_args
    off = load_official("april")                     # or "0731"
    theirs = off.Block(layer_id, official_args(off, args)).float()

How:
- reference/shim/ goes first on sys.path, so model.py's `from kernel import ...` gets the
  pure-PyTorch stand-ins (shim/kernel.py), and rotate_activation's
  `import fast_hadamard_transform` gets shim/fast_hadamard_transform.py.
- One patch: the official rotate_activation asserts that x is bf16. The patched copy is the same
  function without that assert, so float32 and float64 checks work.
- The official code is otherwise unchanged. It reads its files from reference/official/<rev>/.

official_args(off, args) maps our ModelArgs to the official ModelArgs: the same field values,
with float weights (dtype "bf16", no FP8/FP4 formats). The official Linear then creates bf16
weights; call .float() (or .double()) on the built module before running it.

Scales: the real config has scale_fmt "ue8m0" (power-of-two scales in the FP8 simulation).
official_args passes it, and load_official also sets the module-level scale_fmt, because a Block
or Attention built directly (without the official Transformer) reads that global, whose
default in model.py is None. Whether the rounding runs at all is the shim's SIMULATE_QUANT.
"""
import ast
import importlib.util
import sys
from typing import List
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "reference" / "shim"


def load_official(rev: str = "april"):
    path = ROOT / "reference" / "official" / rev / "inference" / "model.py"
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
    return mod


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
    import torch
    path = ROOT / "reference" / "official" / rev / "inference" / "generate.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    ns = {"torch": torch, "List": List, "Transformer": off.Transformer}
    exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns["generate"]

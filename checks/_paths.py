"""Paths and helpers shared by the check scripts. Run any script from anywhere, for example:
    venv/bin/python checks/notebook_regression.py
"""
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / 'deepseek_official' / 'april' / 'inference'   # the official code the notebook is verified against
NOTEBOOK = Path(os.environ.get('V4_NOTEBOOK', ROOT / 'arch.ipynb'))   # V4_NOTEBOOK: check a copy instead
ENCODING = ROOT / 'deepseek_official' / '0731' / 'encoding'   # official chat encoding (Step 23's tokenizer cell imports it)
if str(ENCODING) not in sys.path:
    sys.path.insert(0, str(ENCODING))

DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_constant(node):
    return isinstance(node, ast.Constant) or (isinstance(node, ast.Tuple) and all(map(_is_constant, node.elts)))


def load_notebook_defs(path=NOTEBOOK):
    """Collect the notebook's definitions: from every code cell, in order, run only the top-level
    imports, functions and classes (decorators included) and constant assignments such as `X = 4`.
    Everything else (check code, prints, loops, model runs) is skipped, so check cells never run here,
    even when they start with an import. An import that fails (e.g. a check cell's
    `from load_official import ...` without its sys.path line) is skipped too, and so is a definition that
    needs a name only check code creates at run time (e.g. Step 21's `class TheirTestBlock(off.Block)`);
    those are listed in ns['_skipped_defs'].
    Returns the namespace as a dict, e.g. U['Transformer']. Cells are found by content, not by number.
    It also switches the official FP8 / FP4 rounding on or off to match the notebook (set_quant_simulation)."""
    ns = {}
    name = Path(path).name
    for i, c in enumerate(json.load(open(path))['cells']):
        src = ''.join(c['source'])
        if c['cell_type'] != 'code' or not src.strip():
            continue
        for node in ast.parse(src).body:
            is_import = isinstance(node, (ast.Import, ast.ImportFrom))
            is_def = isinstance(node, DEF_NODES)
            is_const = (isinstance(node, ast.Assign) and _is_constant(node.value)
                        and all(isinstance(t, ast.Name) for t in node.targets))   # `X = 4`, not `args.x = 4`
            if not (is_import or is_def or is_const):
                continue
            code = compile(ast.Module([node], []), f'{name}:cell{i}', 'exec')
            if is_import:
                try:
                    exec(code, ns)
                except ImportError:
                    pass
            elif is_def:
                try:
                    exec(code, ns)
                except NameError:   # a check cell's class built on a run-time name, e.g. `class T(off.Block)`
                    ns.setdefault('_skipped_defs', []).append(f'cell{i}:{node.name}')
            else:
                exec(code, ns)
    set_quant_simulation('fp8_simulate' in ns)
    return ns


def set_quant_simulation(on):
    """The official model rounds its stored keys to FP8 / FP4 (checks/shim/kernel.py). The notebook does the
    same from Phase 2 step 6 on, so the rounding is switched on exactly when the notebook defines fp8_simulate:
    the checks then compare like with like, before and after that step."""
    shim = str(ROOT / 'checks' / 'shim')
    if shim not in sys.path:
        sys.path.insert(0, shim)
    import kernel
    kernel.SIMULATE_QUANT = on

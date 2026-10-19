import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "checks"))
from _paths import load_notebook_defs

t = time.perf_counter()
globals().update(load_notebook_defs()) # import all from notebook code cells.
LOAD_SECONDS = time.perf_counter() - t

if __name__ == "__main__":
    print(f"arch.ipynb loaded in {LOAD_SECONDS:.1f} s | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    for name in ("ModelArgs", "Transformer", "MoE", "Expert", "Checkpoint", "load_weight", "fp4_decode", "clock"):
        print(f"  {name}: {name in globals()}")
    print("  skipped:", globals().get("_skipped_defs", []))
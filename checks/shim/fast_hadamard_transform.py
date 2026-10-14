"""Stand-in for the fast_hadamard_transform package (a CUDA extension), used by the official
rotate_activation as hadamard_transform(x, scale=n ** -0.5).

It computes x @ (H / sqrt(n)) with the Sylvester matrix H (entries +-1), in float32 (float64
stays float64), and casts back to x's dtype. That is the same formula as the notebook's
rotate_activation, so checks can compare bit for bit. The formula itself was checked separately
against a butterfly transform and the scipy-style matrix (checks/step45_check.py).
"""
import math
import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.size(-1)
    assert n & (n - 1) == 0, "the length must be a power of 2"
    xu = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    h = torch.ones(1, 1, dtype=xu.dtype)
    while h.size(0) < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    if math.isclose(scale, n ** -0.5):
        h = h / math.sqrt(n)        # the official call; same rounding as the notebook
    else:
        h = h * scale
    return (xu @ h.to(x.device)).to(x.dtype)

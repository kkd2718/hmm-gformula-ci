"""Deterministic seeding for reproducibility."""
from __future__ import annotations
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Note: PyTorch operations on GPU may still be non-deterministic for
    some kernels even after seeding. Use `torch.use_deterministic_algorithms(True)`
    for stricter guarantees at performance cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

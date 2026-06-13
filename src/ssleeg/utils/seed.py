"""Reproducibility helpers: global seeding and deterministic data loading."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed all relevant RNGs for reproducible experiments.

    Args:
        seed: The base random seed.
        deterministic: If True, force deterministic cuDNN algorithms. This can
            slow down training but guarantees bit-wise reproducibility on the
            same hardware.

    Returns:
        The seed that was set (useful for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required for deterministic CuBLAS matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Opt-in to deterministic algorithms where available; warn_only avoids
        # hard failures for ops without a deterministic implementation.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """DataLoader ``worker_init_fn`` that derives a unique, reproducible seed per worker."""
    seed = (torch.initial_seed() if base_seed is None else base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded ``torch.Generator`` for DataLoader shuffling reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

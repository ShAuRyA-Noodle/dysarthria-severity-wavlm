"""Deterministic seeding helper.

Call ``set_global_seed()`` at the top of every entry-point script in
``experiments/`` to make results bit-reproducible across runs.

Notes
-----
- ``cudnn.benchmark = True`` (anti-deterministic) is explicitly disabled.
- ``torch.use_deterministic_algorithms(warn_only=True)`` accepts non-deterministic
  ops with a warning rather than crashing — required because some HF model
  ops have no deterministic kernel.
- FP16 inference is not bit-deterministic across GPU drivers; for paper
  numbers, freeze the extracted embeddings on disk and only re-run downstream
  steps.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from src.constants import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and (if available) PyTorch.

    Parameters
    ----------
    seed
        Integer seed. Defaults to project-wide ``RANDOM_SEED`` (42).
    deterministic_torch
        If True, also disable cudnn.benchmark and enable deterministic
        algorithms in torch (warn_only). Set False only for performance
        profiling runs that don't enter results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not _HAS_TORCH:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def auto_device() -> str:
    """Return 'cuda' if available, else 'cpu'. No GPU-name heuristics."""
    if not _HAS_TORCH:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

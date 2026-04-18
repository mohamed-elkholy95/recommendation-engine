"""Deterministic seeding for all training and evaluation scripts.

CODE_STYLE §4 mandates every entry point call ``set_all_seeds`` before any
stochastic work so runs are reproducible bit-for-bit given the same seed.
Torch seeding is applied lazily — we only import torch if it is installed,
so this module stays usable in environments without a GPU stack.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED: int = 42


def set_all_seeds(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG the project touches.

    Seeds Python ``random``, NumPy, and (if importable) PyTorch on CPU and CUDA.
    Also sets ``PYTHONHASHSEED`` so hash-based ordering (e.g. set iteration)
    is stable across processes.

    Args:
        seed: Non-negative integer seed. Defaults to ``DEFAULT_SEED`` (42).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    _seed_torch_if_available(seed)


def _seed_torch_if_available(seed: int) -> None:
    # concept: torch is an optional dependency in this module — the API/eval
    # layers import it, but small utility scripts should not pay the startup
    # cost. Import lazily and fall through cleanly when it is absent.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    # gate: unseeded torch / numpy — make cudnn deterministic too.
    torch.use_deterministic_algorithms(True, warn_only=True)

"""
utils/reproducibility.py
~~~~~~~~~~~~~~~~~~~~~~~~
Centralized seeding for fully deterministic runs.
Paper Section 4.1: fixed seed, reproducible orthogonal rotation matrices.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNG sources for reproducible results.

    Args:
        seed: Integer seed value (paper default: 42).
        deterministic: If True, enables CUDA deterministic mode
            (may reduce throughput slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # NOTE: These flags trade a small throughput penalty for exact reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            logger.warning(
                "torch.use_deterministic_algorithms(True) raised: %s. "
                "Some ops may still be non-deterministic.",
                exc,
            )

    logger.info("Seeded all RNGs with seed=%d (deterministic=%s)", seed, deterministic)
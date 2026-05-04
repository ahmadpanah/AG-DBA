"""
utils/config.py
~~~~~~~~~~~~~~~
Load, merge, and validate AG-DBA YAML configurations.
Supports a base config + one or more override files.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(*yaml_paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML config files, merging left-to-right.

    The first path is treated as the base config; subsequent paths override it.

    Args:
        *yaml_paths: Paths to YAML config files.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If any config file does not exist.
        ValueError: If required keys are missing.
    """
    if not yaml_paths:
        raise ValueError("At least one config path is required.")

    merged: dict[str, Any] = {}
    for path in yaml_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, cfg)
        logger.debug("Loaded config: %s", path)

    _validate(merged)
    return merged


def _validate(cfg: dict[str, Any]) -> None:
    """Basic sanity checks on the merged config.

    Maps to paper constraints:
      - target_bpw must be in [1, 4]  (B_set = {1,2,3,4}, Section 3.3)
      - ema_alpha in [0, 1)            (Eq. 4)
    """
    agdba = cfg.get("agdba", {})

    bpw = agdba.get("target_bpw")
    if bpw is not None and not (1.0 <= bpw <= 4.0):
        raise ValueError(f"target_bpw must be in [1, 4], got {bpw}")

    alpha = agdba.get("ema_alpha")
    if alpha is not None and not (0.0 <= alpha < 1.0):
        raise ValueError(f"ema_alpha must be in [0, 1), got {alpha}")

    bit_widths = agdba.get("bit_widths", [])
    if bit_widths and not all(b in {1, 2, 3, 4} for b in bit_widths):
        raise ValueError(f"bit_widths must be subset of {{1,2,3,4}}, got {bit_widths}")


# ============================================================================
# Reproducibility
# ============================================================================

def set_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds and deterministic behavior for reproducibility.
    
    Paper Section 4.1: "Fixed seeds, controlled RNG, reproducible data loading"
    
    Args:
        seed: Random seed
        deterministic: Whether to enable deterministic CUDA kernels
    """
    import random
    import numpy as np
    import torch
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set reproducibility seed={seed}, deterministic={deterministic}")


def get_device(device_id: int = 0) -> str:
    """Get device string.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        device: "cuda:X" or "cpu"
    """
    import torch
    
    if torch.cuda.is_available():
        return f"cuda:{device_id}"
    else:
        logger.warning("CUDA not available, using CPU (will be slow!)")
        return "cpu"
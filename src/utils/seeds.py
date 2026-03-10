"""
src/utils/seeds.py
Deterministic seed management.

Call ``set_global_seed(seed)`` before any stochastic operation to ensure
reproducibility across NumPy, Python ``random``, PyTorch CPU and CUDA.
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set seeds for all RNG sources used in the project.

    Parameters
    ----------
    seed : Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Sacrifice some speed for full determinism on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    logger.debug("Global seed set to %d", seed)

# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/utils/reproducibility.py
Centralised reproducibility setup for TEKNOFEST 2026.

Call ``setup_reproducibility(seed, deterministic_torch)`` at the start of
any training or inference script to guarantee identical outputs across runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def setup_reproducibility(
    seed: int = 42,
    deterministic_torch: bool = True,
) -> None:
    """
    Set all RNG seeds and determinism flags.

    Parameters
    ----------
    seed               : Global integer seed.
    deterministic_torch: If True, enable torch deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass  # older torch
    except ImportError:
        pass

    logger.info(
        "Reproducibility configured: seed=%d  deterministic_torch=%s",
        seed,
        deterministic_torch,
    )


def snapshot_environment() -> dict:
    """
    Capture current runtime environment metadata (no PII).

    Returns
    -------
    dict with Python version, platform, and seed-relevant env vars.
    """
    import platform

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not_set"),
    }

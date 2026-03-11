"""
src/training/cross_val.py
Standalone leakage-free cross-validation using VariantTrainer.

Delegates per-fold train/eval to the existing VariantTrainer._cross_validate,
keeping a single source of truth for the training logic. This module provides
a convenience wrapper that can be called both programmatically and from main.py.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from src.training.trainer import FoldResult, VariantTrainer

logger = logging.getLogger(__name__)


def leakage_free_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict:
    """
    Run leakage-free stratified K-fold CV via VariantTrainer.

    Parameters
    ----------
    X : Raw feature matrix (pre-preprocessing).
    y : Integer label array.
    n_splits : Number of CV folds (overrides config if provided).

    Returns
    -------
    Dict with 'fold_results', 'mean_f1', 'std_f1'.
    """
    trainer = VariantTrainer()
    # Override fold count if caller specifies
    trainer.cfg.training.cv_folds = n_splits

    fold_results: List[FoldResult] = trainer._cross_validate(X, y)
    f1_scores = [r.f1 for r in fold_results]

    summary = {
        "fold_results": fold_results,
        "fold_f1": f1_scores,
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "std_f1": float(np.std(f1_scores)) if f1_scores else 0.0,
    }

    logger.info(
        "Cross-validation complete: Macro F1 = %.4f ± %.4f (%d folds)",
        summary["mean_f1"], summary["std_f1"], len(f1_scores),
    )
    return summary

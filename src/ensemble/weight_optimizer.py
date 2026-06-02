"""
src/ensemble/weight_optimizer.py
Nelder-Mead optimisation of per-panel ensemble weights.

Weights are learned exclusively from out-of-fold (OOF) predictions or a
held-out validation set.  The blind/test set is NEVER touched here.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_PANELS = ("General", "Hereditary_Cancer", "PAH", "CFTR")


def _f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary F1 without sklearn dependency (avoids import side-effects)."""
    preds = (y_pred >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax to ensure weights sum to 1 and are positive."""
    e = np.exp(x - x.max())
    return cast(np.ndarray, e / e.sum())


def optimize_weights(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    seed: int = 42,
    n_restarts: int = 5,
) -> np.ndarray:
    """
    Find weights that maximise F1 on OOF predictions.

    Parameters
    ----------
    oof_predictions : shape (n_samples, n_models) — probability of Pathogenic.
    y_true          : shape (n_samples,) — ground-truth binary labels.
    seed            : RNG seed for reproducibility.
    n_restarts      : Number of random restarts for Nelder-Mead.

    Returns
    -------
    weights : shape (n_models,), sum to 1, all non-negative.
    """
    rng = np.random.default_rng(seed)
    n_models = oof_predictions.shape[1]

    best_weights: np.ndarray = np.ones(n_models) / n_models
    best_f1 = -1.0

    def _neg_f1(raw_w: np.ndarray) -> float:
        w = _softmax(raw_w)
        blended = oof_predictions @ w
        return -_f1_score_binary(y_true, blended)

    for _ in range(n_restarts):
        x0 = rng.standard_normal(n_models)
        result = minimize(_neg_f1, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6})
        w = _softmax(result.x)
        f1 = _f1_score_binary(y_true, oof_predictions @ w)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = w

    logger.info(
        "WeightOptimizer: best OOF F1=%.4f  weights=%s",
        best_f1,
        np.round(best_weights, 4).tolist(),
    )
    return best_weights

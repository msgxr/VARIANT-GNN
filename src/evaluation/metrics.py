"""
src/evaluation/metrics.py
Comprehensive evaluation metrics:
  - Macro F1, Precision, Recall (primary competition metric: Macro F1)
  - ROC-AUC, PR-AUC
  - Brier Score
  - Matthews Correlation Coefficient (MCC)
  - Expected Calibration Error (ECE)
  - Calibration report (reliability diagram data)
  - Threshold tuning via maximising F1 on validation set
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """All metrics in one dataclass."""

    # Primary
    macro_f1:   float = 0.0
    # Standard
    precision:  float = 0.0
    recall:     float = 0.0
    roc_auc:    Optional[float] = None
    pr_auc:     Optional[float] = None
    # Calibration
    brier_score: float = 0.0
    ece:         float = 0.0
    # Robustness
    mcc:         float = 0.0
    # Confusion matrix
    conf_matrix: Optional[np.ndarray] = None
    # Per-threshold performance for reporting
    threshold_used: float = 0.5
    # Calibration curve data
    calibration_fraction_pos: Optional[np.ndarray] = None
    calibration_mean_pred:    Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "macro_f1":      self.macro_f1,
            "precision":     self.precision,
            "recall":        self.recall,
            "roc_auc":       self.roc_auc,
            "pr_auc":        self.pr_auc,
            "brier_score":   self.brier_score,
            "ece":           self.ece,
            "mcc":           self.mcc,
            "threshold":     self.threshold_used,
        }

    def log(self, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        logger.info("%s=== Evaluation Report ===",          tag)
        logger.info("%s[PRIMARY]  Macro F1      : %.4f",   tag, self.macro_f1)
        logger.info("%s           Precision     : %.4f",   tag, self.precision)
        logger.info("%s           Recall        : %.4f",   tag, self.recall)
        logger.info("%s           MCC           : %.4f",   tag, self.mcc)
        if self.roc_auc is not None:
            logger.info("%s           ROC-AUC       : %.4f", tag, self.roc_auc)
        if self.pr_auc is not None:
            logger.info("%s           PR-AUC        : %.4f", tag, self.pr_auc)
        logger.info("%s[CALIB ]   Brier Score   : %.4f",   tag, self.brier_score)
        logger.info("%s           ECE           : %.4f",   tag, self.ece)
        logger.info("%s           Threshold     : %.3f",   tag, self.threshold_used)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute ECE: weighted average |confidence - accuracy| across probability bins.

    Parameters
    ----------
    y_true : Binary ground-truth labels.
    y_prob : Predicted probability for the positive class.
    n_bins : Number of equal-width bins.

    Returns
    -------
    ECE as a float in [0, 1].
    """
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(conf - acc)

    return float(ece)


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_steps: int = 100,
) -> Tuple[float, float]:
    """
    Sweep classification thresholds and return (best_threshold, best_score).

    Parameters
    ----------
    metric : ``'f1'`` or ``'mcc'``
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_thr   = 0.5
    best_score = -np.inf

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        if score > best_score:
            best_score = score
            best_thr   = float(thr)

    logger.info("Best threshold: %.3f → %s=%.4f", best_thr, metric, best_score)
    return best_thr, float(best_score)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,            # (N, 2) array
    threshold: float    = 0.5,
    n_calibration_bins: int = 10,
) -> EvaluationReport:
    """
    Compute all metrics given ground truth labels and probability matrix.

    Parameters
    ----------
    y_true    : Binary ground-truth labels.
    y_prob    : (N, 2) probability array for [Benign, Pathogenic].
    threshold : Classification threshold applied to Pathogenic prob.
    """
    p1    = y_prob[:, 1]
    preds = (p1 >= threshold).astype(int)

    macro_f1  = float(f1_score(y_true, preds, average="macro", zero_division=0))
    precision = float(precision_score(y_true, preds, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, preds, average="macro", zero_division=0))
    mcc       = float(matthews_corrcoef(y_true, preds))
    brier     = float(brier_score_loss(y_true, p1))
    ece       = expected_calibration_error(y_true, p1, n_bins=n_calibration_bins)
    cm        = confusion_matrix(y_true, preds)

    roc_auc: Optional[float] = None
    pr_auc:  Optional[float] = None
    try:
        roc_auc = float(roc_auc_score(y_true, p1))
        pr_auc  = float(average_precision_score(y_true, p1))
    except ValueError:
        pass

    # Calibration curve
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, p1, n_bins=n_calibration_bins, strategy="uniform"
        )
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])

    report = EvaluationReport(
        macro_f1                  = macro_f1,
        precision                 = precision,
        recall                    = recall,
        roc_auc                   = roc_auc,
        pr_auc                    = pr_auc,
        brier_score               = brier,
        ece                       = ece,
        mcc                       = mcc,
        conf_matrix               = cm,
        threshold_used            = threshold,
        calibration_fraction_pos  = frac_pos,
        calibration_mean_pred     = mean_pred,
    )
    report.log()
    return report

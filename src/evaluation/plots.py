"""
src/evaluation/plots.py
Evaluation visualisation: confusion matrix, ROC, PR, calibration.
All plots are saved to disk (no GUI dependency).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

from src.evaluation.metrics import EvaluationReport

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    report: EvaluationReport,
    output_path: str | Path = "reports/confusion_matrix.png",
    labels: Optional[list] = None,
) -> None:
    if report.conf_matrix is None:
        logger.warning("No confusion matrix in report; skipping plot.")
        return

    labels = labels or ["Benign (0)", "Pathogenic (1)"]
    path   = _ensure_dir(Path(output_path))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        report.conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved confusion matrix → %s", path)


# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path = "reports/roc_curve.png",
) -> None:
    path = _ensure_dir(Path(output_path))
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_val      = np.trapz(tpr, fpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})", color="steelblue")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved ROC curve → %s", path)


# ---------------------------------------------------------------------------
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path = "reports/pr_curve.png",
) -> None:
    path = _ensure_dir(Path(output_path))
    prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved PR curve → %s", path)


# ---------------------------------------------------------------------------
def plot_calibration(
    report: EvaluationReport,
    output_path: str | Path = "reports/calibration.png",
    cal_frac_pos: Optional[np.ndarray] = None,
    cal_mean_pred: Optional[np.ndarray] = None,
) -> None:
    """
    Reliability diagram comparing uncalibrated vs calibrated probabilities.
    Pass ``cal_frac_pos`` / ``cal_mean_pred`` for the calibrated curve.
    """
    path = _ensure_dir(Path(output_path))

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    if (
        report.calibration_fraction_pos is not None
        and len(report.calibration_fraction_pos) > 0
    ):
        plt.plot(
            report.calibration_mean_pred,
            report.calibration_fraction_pos,
            "o-", color="steelblue", label="Before calibration",
        )

    if cal_frac_pos is not None and len(cal_frac_pos) > 0:
        plt.plot(
            cal_mean_pred, cal_frac_pos,
            "s--", color="darkorange", label="After calibration",
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Plot (ECE={report.ece:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved calibration plot → %s", path)


# ---------------------------------------------------------------------------
def save_all_plots(
    report: EvaluationReport,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path = "reports",
    cal_frac_pos: Optional[np.ndarray] = None,
    cal_mean_pred: Optional[np.ndarray] = None,
) -> None:
    """Save all evaluation plots to ``output_dir``."""
    d = Path(output_dir)
    plot_confusion_matrix(report, d / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, d / "roc_curve.png")
    plot_pr_curve(y_true, y_prob,  d / "pr_curve.png")
    plot_calibration(report, d / "calibration.png", cal_frac_pos, cal_mean_pred)

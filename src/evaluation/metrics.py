"""
src/evaluation/metrics.py
Comprehensive evaluation metrics.

Primary competition metric (TEKNOFEST 2026 §7.3):
  binary_f1 = 2*TP / (2*TP + FP + FN)  — F1 for the Pathogenic (positive) class.
  This is the ranking metric as defined by the spec ("TP, FP, FN üzerinden F1 Skoru").

Additional metrics: Macro F1, Precision, Recall, ROC-AUC, PR-AUC, Brier Score, MCC, ECE.
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

    # TEKNOFEST 2026 §7.3 primary ranking metric: F1 = 2*TP/(2*TP+FP+FN)
    binary_f1:  float = 0.0   # binary F1 for Pathogenic class (positive=1)
    # Supporting metrics
    macro_f1:   float = 0.0
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
            # §7.3 primary competition metric first
            "binary_f1":     self.binary_f1,
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
        logger.info("%s=== Evaluation Report ===",                   tag)
        logger.info("%s[§7.3 PRIMARY] Binary F1     : %.4f",        tag, self.binary_f1)
        logger.info("%s               Macro F1      : %.4f",        tag, self.macro_f1)
        logger.info("%s               Precision     : %.4f",        tag, self.precision)
        logger.info("%s               Recall        : %.4f",        tag, self.recall)
        logger.info("%s               MCC           : %.4f",        tag, self.mcc)
        if self.roc_auc is not None:
            logger.info("%s               ROC-AUC       : %.4f",    tag, self.roc_auc)
        if self.pr_auc is not None:
            logger.info("%s               PR-AUC        : %.4f",    tag, self.pr_auc)
        logger.info("%s[CALIB ]       Brier Score   : %.4f",        tag, self.brier_score)
        logger.info("%s               ECE           : %.4f",        tag, self.ece)
        logger.info("%s               Threshold     : %.3f",        tag, self.threshold_used)


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
    """Sweep thresholds; return (best_threshold, best_score)."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_thr, best_score = 0.5, -np.inf
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, average="binary", pos_label=1, zero_division=0)
        elif metric == "macro_f1":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if score > best_score:
            best_score, best_thr = score, float(thr)
    logger.info("Best threshold: %.3f -> %s=%.4f", best_thr, metric, best_score)
    return best_thr, float(best_score)


# Alias used throughout the codebase
find_f1_optimal_threshold = find_best_threshold


def find_panel_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    panels: np.ndarray,
    n_steps: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Panel bazlı F1-optimal eşik optimizasyonu.

    Her panel için ayrı ayrı P-R eğrisi üzerinde F1'i maksimize eden
    eşik değerini bulur.  Genel eşik ("General") de hesaplanır.

    Parameters
    ----------
    y_true   : Gerçek etiketler [N]
    y_prob   : P(Pathogenic) olasılıkları [N]
    panels   : Panel adları [N]  (str array)
    output_path : Kayıt yolu (None = kaydetme)

    Returns
    -------
    Dict[panel_name -> optimal_threshold]
    """
    import json
    from pathlib import Path

    panel_names = ["__all__"] + sorted(set(panels))
    results: Dict[str, float] = {}

    for panel in panel_names:
        if panel == "__all__":
            mask = np.ones(len(y_true), dtype=bool)
            label = "General"
        else:
            mask  = np.array(panels) == panel
            label = panel

        if mask.sum() < 10:
            logger.warning("Panel %s: yetersiz örnek (%d), eşik=0.5 kullanılıyor", label, mask.sum())
            results[label] = 0.5
            continue

        thr, score = find_best_threshold(y_true[mask], y_prob[mask], metric="macro_f1")
        results[label] = thr
        logger.info("Panel %-20s → eşik=%.3f  Macro-F1=%.4f", label, thr, score)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump({"panel_thresholds": results}, fh, indent=2)
        logger.info("Panel eşikleri → %s", output_path)

    return results


def save_threshold_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    panels: Optional[np.ndarray] = None,
    reports_dir: str = "reports",
) -> None:
    """
    Tam eşik raporu: JSON + P-R eğrisi grafiği üretir.

    Çıktılar:
      reports/threshold_report.json
      reports/figures/precision_recall_curve.png
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from pathlib import Path

    rep_dir = Path(reports_dir)
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Global optimal threshold
    global_thr, global_f1 = find_best_threshold(y_true, y_prob, metric="macro_f1")

    # P-R eğrisi
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # F1 eğrisi (her threshold için)
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rec_arr, prec_arr, lw=2, color="#e63946", label=f"PR Eğrisi (AUC={pr_auc:.4f})")
    ax.axvline(rec_arr[np.argmax(f1_arr)], color="#2563eb", ls="--", lw=1.2,
               label=f"Optimal Eşik ({global_thr:.3f}) F1={global_f1:.4f}")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Eğrisi — TEKNOFEST 2026", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig_path = fig_dir / "precision_recall_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("P-R eğrisi → %s", fig_path)

    # Panel eşikleri
    panel_thresholds: Dict[str, float] = {"General": global_thr}
    if panels is not None:
        panel_thresholds = find_panel_thresholds(
            y_true, y_prob, panels,
            output_path=str(rep_dir / "panel_thresholds.json"),
        )

    report = {
        "global_optimal_threshold": global_thr,
        "global_macro_f1_at_threshold": global_f1,
        "pr_auc": float(pr_auc),
        "panel_thresholds": panel_thresholds,
        "pr_curve_path": str(fig_path),
    }
    json_path = rep_dir / "threshold_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Threshold raporu → %s", json_path)


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

    # §7.3 primary competition metric: binary F1 (TP/FP/FN for Pathogenic class)
    binary_f1 = float(f1_score(y_true, preds, average="binary", pos_label=1, zero_division=0))
    macro_f1  = float(f1_score(y_true, preds, average="macro", zero_division=0))
    precision = float(precision_score(y_true, preds, average="binary", pos_label=1, zero_division=0))
    recall    = float(recall_score(y_true, preds, average="binary", pos_label=1, zero_division=0))
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
        binary_f1                 = binary_f1,
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


# ---------------------------------------------------------------------------
# Panel-based evaluation — TEKNOFEST 2026
# ---------------------------------------------------------------------------


def evaluate_per_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    panels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, EvaluationReport]:
    """
    Her panel için ayrı değerlendirme raporu üretir.

    Parameters
    ----------
    y_true : Binary ground-truth labels.
    y_prob : (N, 2) probability array.
    panels : Panel labels for each sample (e.g., 'General', 'CFTR').

    Returns
    -------
    Dict mapping panel name → EvaluationReport.
    """
    unique_panels = np.unique(panels)
    reports = {}

    for panel in unique_panels:
        mask = panels == panel
        if mask.sum() < 2:
            logger.warning("Panel '%s' has too few samples (%d), skipping.", panel, mask.sum())
            continue

        panel_report = evaluate(
            y_true[mask],
            y_prob[mask],
            threshold=threshold,
        )
        reports[str(panel)] = panel_report
        logger.info(
            "Panel %s | n=%d | F1=%.4f | AUC=%s | Brier=%.4f",
            panel, mask.sum(), panel_report.macro_f1,
            f"{panel_report.roc_auc:.4f}" if panel_report.roc_auc else "N/A",
            panel_report.brier_score,
        )

    return reports


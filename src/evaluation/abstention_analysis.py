"""
src/evaluation/abstention_analysis.py
Analysis of abstention / triage performance.

Reports per-flag statistics and how well uncertainty correlates with
prediction errors.  Never trains or fits on blind data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.inference.triage import ALL_FLAGS

logger = logging.getLogger(__name__)


def compute_abstention_stats(
    preds_df: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute per-flag statistics from a predictions DataFrame.

    Parameters
    ----------
    preds_df : Predictions DataFrame with 'Clinical_Flag' and 'Prediction' columns.
    y_true   : Optional ground-truth labels (0=Benign, 1=Pathogenic).
               When provided, per-flag accuracy and error rate are added.

    Returns
    -------
    stats_df : Per-flag summary DataFrame.
    """
    if "Clinical_Flag" not in preds_df.columns:
        raise ValueError("preds_df must contain 'Clinical_Flag' column.")

    rows = []
    for flag in ALL_FLAGS:
        mask = preds_df["Clinical_Flag"] == flag
        n = int(mask.sum())
        if n == 0:
            rows.append({"flag": flag, "count": 0, "fraction": 0.0})
            continue

        row: dict = {
            "flag": flag,
            "count": n,
            "fraction": n / len(preds_df),
        }

        if y_true is not None and "Prediction" in preds_df.columns:
            pred_bin = (preds_df.loc[mask, "Prediction"] == "Pathogenic").astype(int).values
            true_bin = y_true[mask.values].astype(int)
            correct = (pred_bin == true_bin).sum()
            row["accuracy"] = correct / n
            row["error_rate"] = 1.0 - row["accuracy"]

        if "Uncertainty" in preds_df.columns:
            row["mean_uncertainty"] = float(preds_df.loc[mask, "Uncertainty"].mean())

        if "Pathogenic_Probability" in preds_df.columns:
            probs = preds_df.loc[mask, "Pathogenic_Probability"]
            row["mean_prob"] = float(probs.mean())
            row["std_prob"] = float(probs.std())

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    logger.info("Abstention stats:\n%s", stats_df.to_string(index=False))
    return stats_df

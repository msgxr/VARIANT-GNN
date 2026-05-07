"""
src/inference/prediction_schema.py
Fixed output schema for competition submission predictions.

All inference output DataFrames must conform to this schema before being
written to disk.  Column order is canonical and enforced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

PREDICTION_COLUMNS: List[str] = [
    "Variant_ID",
    "Panel",
    "Prediction",
    "Pathogenic_Probability",
    "Benign_Probability",
    "Calibrated_Risk",
    "Uncertainty",
    "Clinical_Flag",
    "Model_Version",
    "Inference_Timestamp",
]

VALID_PREDICTIONS = frozenset({"Pathogenic", "Benign"})


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_prediction_frame(
    variant_ids: pd.Series,
    panels: pd.Series,
    pathogenic_prob: np.ndarray,
    calibrated_risk: np.ndarray,
    uncertainty: np.ndarray,
    clinical_flags: np.ndarray,
    threshold: float = 0.5,
    model_version: str = "unknown",
) -> pd.DataFrame:
    """
    Construct a prediction DataFrame conforming to PREDICTION_COLUMNS.

    Parameters
    ----------
    variant_ids       : Variant identifiers.
    panels            : Panel labels.
    pathogenic_prob   : P(Pathogenic) for each sample, in [0, 1].
    calibrated_risk   : Calibrated probability, in [0, 1].
    uncertainty       : Epistemic/aleatoric uncertainty per sample.
    clinical_flags    : String triage flag per sample.
    threshold         : Decision threshold for binary prediction.
    model_version     : Artifact version string.

    Returns
    -------
    pd.DataFrame with exactly PREDICTION_COLUMNS columns.
    """
    n = len(variant_ids)
    if not (len(panels) == len(pathogenic_prob) == len(calibrated_risk)
            == len(uncertainty) == len(clinical_flags) == n):
        raise ValueError("All input arrays must have the same length.")

    pathogenic_prob = np.clip(np.asarray(pathogenic_prob, dtype=float), 0.0, 1.0)
    benign_prob = 1.0 - pathogenic_prob
    calibrated_risk = np.clip(np.asarray(calibrated_risk, dtype=float), 0.0, 1.0)
    uncertainty = np.clip(np.asarray(uncertainty, dtype=float), 0.0, 1.0)

    predictions = np.where(pathogenic_prob >= threshold, "Pathogenic", "Benign")

    ts = _now_utc()
    df = pd.DataFrame(
        {
            "Variant_ID": variant_ids.reset_index(drop=True),
            "Panel": panels.reset_index(drop=True),
            "Prediction": predictions,
            "Pathogenic_Probability": np.round(pathogenic_prob, 4),
            "Benign_Probability": np.round(benign_prob, 4),
            "Calibrated_Risk": np.round(calibrated_risk, 4),
            "Uncertainty": np.round(uncertainty, 4),
            "Clinical_Flag": clinical_flags,
            "Model_Version": model_version,
            "Inference_Timestamp": ts,
        }
    )
    return df[PREDICTION_COLUMNS]


def validate_prediction_frame(df: pd.DataFrame) -> None:
    """
    Raise ValueError if df does not conform to PREDICTION_COLUMNS schema.
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction DataFrame missing columns: {missing}")

    invalid_preds = ~df["Prediction"].isin(VALID_PREDICTIONS)
    if invalid_preds.any():
        bad = df.loc[invalid_preds, "Prediction"].unique().tolist()
        raise ValueError(f"Invalid Prediction values: {bad}. Must be one of {VALID_PREDICTIONS}.")

    for col in ("Pathogenic_Probability", "Benign_Probability", "Calibrated_Risk", "Uncertainty"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric.")
        if (df[col] < 0).any() or (df[col] > 1).any():
            raise ValueError(f"Column '{col}' values must be in [0, 1].")

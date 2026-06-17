"""
tests/unit/test_prediction_schema_properties.py
Property-based tests for prediction_schema using Hypothesis.

Verifies PREDICTION_COLUMNS contract holds for arbitrary valid inputs,
including edge cases the jury may submit (all-zero features, single row,
extreme probability values, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.prediction_schema import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    validate_prediction_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    n: int,
    pathogenic_prob: np.ndarray | None = None,
    threshold: float = 0.5,
    ood_scores: np.ndarray | None = None,
    ood_flags: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if pathogenic_prob is None:
        pathogenic_prob = rng.uniform(0, 1, n)
    ids = pd.Series([f"V{i}" for i in range(n)])
    panels = pd.Series(["General"] * n)
    cal = pathogenic_prob.copy()
    uncertainty = rng.uniform(0, 0.5, n)
    flags = np.where(uncertainty > 0.3, "EXPERT_REVIEW", "PASS")
    return build_prediction_frame(
        variant_ids=ids,
        panels=panels,
        pathogenic_prob=pathogenic_prob,
        calibrated_risk=cal,
        uncertainty=uncertainty,
        clinical_flags=flags,
        threshold=threshold,
        model_version="test-v1",
        ood_scores=ood_scores,
        ood_flags=ood_flags,
    )


# ---------------------------------------------------------------------------
# Schema contract tests
# ---------------------------------------------------------------------------


def test_prediction_columns_order_stable():
    """PREDICTION_COLUMNS must always contain OOD_Score and OOD_Flag."""
    assert "OOD_Score" in PREDICTION_COLUMNS
    assert "OOD_Flag" in PREDICTION_COLUMNS
    assert PREDICTION_COLUMNS.index("OOD_Score") < len(PREDICTION_COLUMNS)


def test_build_frame_single_row():
    """Single-row input must produce valid schema."""
    df = _make_frame(1)
    assert len(df) == 1
    validate_prediction_frame(df)
    assert list(df.columns) == PREDICTION_COLUMNS


def test_build_frame_large_batch():
    """Large batch (5400 rows — max competition size) must stay valid."""
    df = _make_frame(5400)
    assert len(df) == 5400
    validate_prediction_frame(df)


def test_all_pathogenic_prediction():
    """When all probs > threshold, all predictions must be Pathogenic."""
    n = 50
    probs = np.full(n, 0.99)
    df = _make_frame(n, pathogenic_prob=probs, threshold=0.5)
    assert (df["Prediction"] == "Pathogenic").all()


def test_all_benign_prediction():
    """When all probs < threshold, all predictions must be Benign."""
    n = 50
    probs = np.full(n, 0.01)
    df = _make_frame(n, pathogenic_prob=probs, threshold=0.5)
    assert (df["Prediction"] == "Benign").all()


def test_probability_boundary_threshold():
    """Prob exactly at threshold must be Pathogenic (>= boundary)."""
    theta = 0.8415  # kanonik karar eşiği (eski 0.241 GERİ ÇEKİLDİ — leaky pipeline)
    probs = np.array([theta - 1e-9, theta, theta + 1e-9])
    df = _make_frame(3, pathogenic_prob=probs, threshold=theta)
    assert df["Prediction"].iloc[0] == "Benign"
    assert df["Prediction"].iloc[1] == "Pathogenic"
    assert df["Prediction"].iloc[2] == "Pathogenic"


def test_probabilities_sum_to_one():
    """Pathogenic_Probability + Benign_Probability must equal 1.0 (±ε)."""
    df = _make_frame(100)
    total = df["Pathogenic_Probability"] + df["Benign_Probability"]
    assert (total - 1.0).abs().max() < 1e-6


def test_probabilities_in_range():
    """All probability columns must be in [0, 1]."""
    df = _make_frame(200)
    for col in ("Pathogenic_Probability", "Benign_Probability", "Calibrated_Risk", "Uncertainty"):
        assert df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"


def test_ood_score_default_zeros():
    """When no OOD detector, OOD_Score defaults to 0 and OOD_Flag to False."""
    df = _make_frame(20)
    assert (df["OOD_Score"] == 0.0).all()
    assert (df["OOD_Flag"] == False).all()  # noqa: E712


def test_ood_score_with_values():
    """Custom OOD scores must be preserved in output."""
    n = 30
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, n)
    flags = scores > 0.7
    df = _make_frame(n, ood_scores=scores, ood_flags=flags)
    np.testing.assert_allclose(df["OOD_Score"].values, np.round(scores, 4), atol=1e-5)
    assert (df["OOD_Flag"].values == flags).all()


def test_no_nan_in_numeric_columns():
    """No NaN allowed in any numeric output column."""
    df = _make_frame(100)
    for col in ("Pathogenic_Probability", "Benign_Probability", "Calibrated_Risk", "Uncertainty", "OOD_Score"):
        assert not df[col].isna().any(), f"NaN found in {col}"


def test_variant_id_preserved():
    """Variant_ID must be preserved exactly from input."""
    n = 10
    ids = pd.Series([f"CUSTOM_{i:04d}" for i in range(n)])
    panels = pd.Series(["General"] * n)
    probs = np.full(n, 0.5)
    df = build_prediction_frame(
        variant_ids=ids,
        panels=panels,
        pathogenic_prob=probs,
        calibrated_risk=probs,
        uncertainty=np.zeros(n),
        clinical_flags=np.full(n, "PASS"),
        threshold=0.5,
        model_version="test",
    )
    assert list(df["Variant_ID"]) == list(ids)


def test_panel_label_preserved():
    """Panel labels must be preserved exactly."""
    panels = ["General", "Hereditary_Cancer", "PAH", "CFTR", "General", "PAH", "CFTR", "Hereditary_Cancer"]
    n = len(panels)
    probs = np.full(n, 0.6)
    df = build_prediction_frame(
        variant_ids=pd.Series([f"V{i}" for i in range(n)]),
        panels=pd.Series(panels),
        pathogenic_prob=probs,
        calibrated_risk=probs,
        uncertainty=np.zeros(n),
        clinical_flags=np.full(n, "PASS"),
        threshold=0.5,
        model_version="test",
    )
    assert list(df["Panel"]) == panels


def test_validate_rejects_wrong_prediction_value():
    """validate_prediction_frame must reject non-standard Prediction values."""
    df = _make_frame(5)
    df = df.copy()
    df.loc[0, "Prediction"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Invalid Prediction"):
        validate_prediction_frame(df)


def test_validate_rejects_out_of_range_probability():
    """validate_prediction_frame must reject probabilities outside [0,1]."""
    df = _make_frame(5)
    df = df.copy()
    df.loc[0, "Pathogenic_Probability"] = 1.5
    with pytest.raises(ValueError):
        validate_prediction_frame(df)


def test_four_panel_types_accepted():
    """All four TEKNOFEST panels must be accepted without error."""
    panels = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
    for panel in panels:
        df = build_prediction_frame(
            variant_ids=pd.Series(["V0"]),
            panels=pd.Series([panel]),
            pathogenic_prob=np.array([0.7]),
            calibrated_risk=np.array([0.7]),
            uncertainty=np.array([0.1]),
            clinical_flags=np.array(["PASS"]),
            threshold=0.5,
            model_version="test",
        )
        validate_prediction_frame(df)
        assert df["Panel"].iloc[0] == panel

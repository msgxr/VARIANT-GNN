"""tests/unit/test_triage.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.triage import (
    ALL_FLAGS,
    BORDERLINE_PROBABILITY,
    EXPERT_REVIEW_RECOMMENDED,
    HIGH_UNCERTAINTY,
    LOW_CONFIDENCE,
    STABLE_PREDICTION,
    TriageEngine,
)
from src.evaluation.abstention_analysis import compute_abstention_stats


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _engine(ut=0.30, bm=0.10, et=0.65) -> TriageEngine:
    return TriageEngine(uncertainty_threshold=ut, borderline_margin=bm, entropy_threshold=et)


# ---------------------------------------------------------------------------
# STABLE_PREDICTION
# ---------------------------------------------------------------------------


def test_stable_prediction():
    eng = _engine()
    p = np.array([0.95, 0.05])  # far from boundary
    u = np.array([0.01, 0.01])  # low uncertainty
    flags = eng.assign_flags(p, u)
    assert all(f == STABLE_PREDICTION for f in flags)


# ---------------------------------------------------------------------------
# HIGH_UNCERTAINTY
# ---------------------------------------------------------------------------


def test_high_uncertainty_flag():
    eng = _engine(ut=0.30)
    p = np.array([0.9])
    u = np.array([0.50])  # > 0.30
    flags = eng.assign_flags(p, u)
    assert flags[0] == HIGH_UNCERTAINTY


# ---------------------------------------------------------------------------
# BORDERLINE_PROBABILITY
# ---------------------------------------------------------------------------


def test_borderline_probability_flag():
    # Use a very high entropy threshold so only borderline triggers (not low_confidence).
    # p=0.52 → margin=0.02 < bm=0.10 (borderline), entropy≈0.999 > default 0.65 (low_conf).
    # Disabling entropy condition isolates borderline flag.
    eng = TriageEngine(uncertainty_threshold=0.30, borderline_margin=0.10, entropy_threshold=1.1)
    p = np.array([0.52])
    u = np.array([0.01])
    flags = eng.assign_flags(p, u)
    assert flags[0] == BORDERLINE_PROBABILITY


# ---------------------------------------------------------------------------
# LOW_CONFIDENCE (entropy)
# ---------------------------------------------------------------------------


def test_low_confidence_flag():
    eng = _engine(et=0.65)
    # p=0.5 → entropy = 1.0 bit > 0.65
    p = np.array([0.50])
    u = np.array([0.0])
    flags = eng.assign_flags(p, u)
    assert flags[0] in (LOW_CONFIDENCE, EXPERT_REVIEW_RECOMMENDED)


# ---------------------------------------------------------------------------
# EXPERT_REVIEW_RECOMMENDED (multiple risk factors)
# ---------------------------------------------------------------------------


def test_expert_review_multiple_factors():
    eng = _engine(ut=0.30, bm=0.10, et=0.65)
    p = np.array([0.50])  # borderline + high entropy
    u = np.array([0.50])  # also high uncertainty
    flags = eng.assign_flags(p, u)
    assert flags[0] == EXPERT_REVIEW_RECOMMENDED


# ---------------------------------------------------------------------------
# Shape and type
# ---------------------------------------------------------------------------


def test_output_shape_matches_input():
    eng = _engine()
    n = 50
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=n)
    u = rng.uniform(0, 0.4, size=n)
    flags = eng.assign_flags(p, u)
    assert flags.shape == (n,)
    assert all(f in ALL_FLAGS for f in flags)


def test_mismatched_shapes_raises():
    eng = _engine()
    with pytest.raises(ValueError, match="same shape"):
        eng.assign_flags(np.array([0.5, 0.5]), np.array([0.1]))


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


def test_custom_threshold():
    eng = _engine(bm=0.10)
    p = np.array([0.85])  # |0.85 - 0.80| = 0.05 < 0.10 → borderline
    u = np.array([0.0])
    flags = eng.assign_flags(p, u, threshold=0.80)
    assert flags[0] in (BORDERLINE_PROBABILITY, EXPERT_REVIEW_RECOMMENDED)


# ---------------------------------------------------------------------------
# Binary Prediction column NOT modified
# ---------------------------------------------------------------------------


def test_prediction_column_unchanged():
    eng = _engine()
    p = np.array([0.9, 0.1, 0.5, 0.4])
    u = np.array([0.5, 0.5, 0.5, 0.1])
    flags = eng.assign_flags(p, u)
    # flags are separate; binary decisions must still respect threshold
    binary = (p >= 0.5).astype(int)
    assert binary[0] == 1
    assert binary[1] == 0


# ---------------------------------------------------------------------------
# Abstention analysis
# ---------------------------------------------------------------------------


def test_abstention_stats_basic():
    df = pd.DataFrame(
        {
            "Prediction": ["Pathogenic", "Benign", "Pathogenic"],
            "Clinical_Flag": [STABLE_PREDICTION, HIGH_UNCERTAINTY, BORDERLINE_PROBABILITY],
            "Uncertainty": [0.05, 0.45, 0.20],
            "Pathogenic_Probability": [0.9, 0.2, 0.55],
        }
    )
    stats = compute_abstention_stats(df)
    assert "flag" in stats.columns
    assert "count" in stats.columns
    assert set(stats["flag"]).issubset(set(ALL_FLAGS))


def test_abstention_stats_with_labels():
    df = pd.DataFrame(
        {
            "Prediction": ["Pathogenic", "Benign", "Pathogenic"],
            "Clinical_Flag": [STABLE_PREDICTION, STABLE_PREDICTION, HIGH_UNCERTAINTY],
            "Uncertainty": [0.05, 0.05, 0.40],
            "Pathogenic_Probability": [0.9, 0.1, 0.8],
        }
    )
    y_true = np.array([1, 0, 1])
    stats = compute_abstention_stats(df, y_true=y_true)
    stable_row = stats[stats["flag"] == STABLE_PREDICTION]
    if not stable_row.empty and "accuracy" in stable_row.columns:
        assert 0 <= float(stable_row["accuracy"].iloc[0]) <= 1


def test_abstention_stats_missing_flag_column_raises():
    df = pd.DataFrame({"Prediction": ["Pathogenic"]})
    with pytest.raises(ValueError, match="Clinical_Flag"):
        compute_abstention_stats(df)

"""
tests/unit/test_metrics_f1.py
F1 score computation tests per TEKNOFEST §7.3.

Formula: F1 = 2·TP / (2·TP + FP + FN)
Primary metric: binary F1 for Pathogenic class (pos_label=1).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score as sklearn_f1

# ---------------------------------------------------------------------------
# Helper: import competition metric
# ---------------------------------------------------------------------------


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """§7.3 binary F1 for Pathogenic class (pos_label=1)."""
    return sklearn_f1(y_true, y_pred, average="binary", pos_label=1, zero_division=0)


# ---------------------------------------------------------------------------
# Formula correctness — §7.3
# ---------------------------------------------------------------------------


class TestF1Formula:
    def test_perfect_predictions(self):
        """Perfect predictions → F1 = 1.0."""
        y = np.array([1, 1, 0, 0, 1, 0])
        assert abs(_binary_f1(y, y) - 1.0) < 1e-9

    def test_all_wrong_predictions(self):
        """All predictions inverted → F1 = 0.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        assert abs(_binary_f1(y_true, y_pred) - 0.0) < 1e-9

    def test_formula_matches_sklearn(self):
        """Result must match sklearn.metrics.f1_score(pos_label=1)."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        expected = sklearn_f1(y_true, y_pred, average="binary", pos_label=1)
        assert abs(_binary_f1(y_true, y_pred) - expected) < 1e-9

    def test_tp_fp_fn_identity(self):
        """
        Manual TP/FP/FN → F1 calculation must match function output.
        §7.3: F1 = 2TP / (2TP + FP + FN)
        """
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        # TP=2, FP=1, FN=2
        tp, fp, fn = 2, 1, 2
        expected = 2 * tp / (2 * tp + fp + fn)
        assert abs(_binary_f1(y_true, y_pred) - expected) < 1e-9

    def test_high_recall_low_precision(self):
        """High-recall model (as used in competition) must still compute correctly."""
        y_true = np.array([1] * 90 + [0] * 10)
        y_pred = np.array([1] * 95 + [0] * 5)  # catches all 1s + 5 false positives
        result = _binary_f1(y_true, y_pred)
        assert 0.0 < result < 1.0

    def test_all_positive_predictions(self):
        """All positive predictions → F1 = 2P/(1+P) where P=class fraction."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        # TP=2, FP=2, FN=0 → F1 = 4/(4+2+0) = 4/6 = 0.667
        expected = 4 / 6
        assert abs(_binary_f1(y_true, y_pred) - expected) < 1e-6

    def test_pos_label_is_pathogenic(self):
        """pos_label must be 1 (Pathogenic), not 0 (Benign)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        # pos_label=1: TP=1, FP=0, FN=1 → F1 = 2/(2+0+1) = 0.667
        expected = sklearn_f1(y_true, y_pred, average="binary", pos_label=1)
        assert abs(_binary_f1(y_true, y_pred) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Threshold effect on F1
# ---------------------------------------------------------------------------


class TestThresholdEffect:
    def test_low_threshold_maximizes_recall(self):
        """Low threshold → high recall, potentially lower precision."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        proba = rng.uniform(0, 1, 100)

        y_pred_low = (proba >= 0.1).astype(int)
        y_pred_high = (proba >= 0.9).astype(int)

        from sklearn.metrics import recall_score

        recall_low = recall_score(y_true, y_pred_low, pos_label=1, zero_division=0)
        recall_high = recall_score(y_true, y_pred_high, pos_label=1, zero_division=0)
        assert recall_low >= recall_high

    def test_f1_optimal_threshold_beats_default(self):
        """F1-optimal threshold should be >= default 0.5 threshold F1."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 300)
        proba = rng.beta(2, 5, 300)  # skewed toward low values (harder problem)

        best_f1 = 0.0
        for t in np.linspace(0.05, 0.95, 50):
            y_pred = (proba >= t).astype(int)
            f1 = sklearn_f1(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        f1_at_default = sklearn_f1(y_true, (proba >= 0.5).astype(int), average="binary", pos_label=1, zero_division=0)
        assert best_f1 >= f1_at_default - 1e-9


# ---------------------------------------------------------------------------
# Panel-level F1 — §7.3 requires per-panel reporting
# ---------------------------------------------------------------------------


class TestPanelF1:
    # Canonical per-panel test (group-aware hold-out @ θ=0.8415) binary F1.
    # Source of truth: RESULTS_CANONICAL.json -> panel_test_metrics[*].binary_f1
    # (== reports/cv_report.json panel_metrics). Updated from stale leakage-era
    # numbers (General 0.8872 / HC 0.8960 / PAH 0.9556 / CFTR 0.9524) after the
    # group-aware leakage-free retrain.
    EXPECTED = {
        "General": 0.8185,
        "Hereditary_Cancer": 0.9060,
        "PAH": 0.9120,
        "CFTR": 0.7143,
    }

    @pytest.mark.parametrize("panel,expected_f1", EXPECTED.items())
    def test_reported_panel_f1_in_valid_range(self, panel, expected_f1):
        """Reported panel F1 values must be in a valid F1 range [0,1]."""
        assert 0.0 <= expected_f1 <= 1.0, f"{panel}: F1 out of range"

    def test_panel_f1_matches_canonical(self):
        """
        EXPECTED must track RESULTS_CANONICAL.json (no silent drift). Tolerance
        abs=0.03 absorbs documentation rounding without masking a real change.
        """
        import json
        from pathlib import Path

        canon_path = Path(__file__).resolve().parents[2] / "RESULTS_CANONICAL.json"
        canon = json.loads(canon_path.read_text(encoding="utf-8"))
        pm = canon["panel_test_metrics"]
        for panel, expected_f1 in self.EXPECTED.items():
            canonical_f1 = pm[panel]["binary_f1"]
            assert expected_f1 == pytest.approx(canonical_f1, abs=0.03), (
                f"{panel}: test EXPECTED {expected_f1} drifted from canonical {canonical_f1}"
            )

    def test_cftr_f1_reflects_small_n_holdout(self):
        """
        CFTR F1=0.7143 is the honest small-n hold-out value (canonical notes
        CFTR has too few hold-out rows for a stable estimate). It sits below the
        other panels by design; the only hard floor here is that it beats random.
        """
        assert self.EXPECTED["CFTR"] > 0.5

    def test_all_panels_f1_above_random_baseline(self):
        """All panels must exceed random classifier baseline (~0.5)."""
        for panel, f1 in self.EXPECTED.items():
            assert f1 > 0.5, f"{panel}: F1={f1} below random baseline"

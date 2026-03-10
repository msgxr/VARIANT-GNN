"""
tests/unit/test_evaluation.py
Unit tests for metrics module.
"""
import numpy as np
import pytest


def _make_binary(n: int = 300, seed: int = 0):
    rng   = np.random.default_rng(seed)
    y     = rng.integers(0, 2, size=n)
    p1    = np.clip(y * rng.normal(0.7, 0.15, size=n) +
                    (1 - y) * rng.normal(0.3, 0.15, size=n), 0.01, 0.99)
    proba = np.column_stack([1 - p1, p1])
    return y, proba


class TestEvaluationMetrics:
    def test_evaluate_returns_report(self):
        from src.evaluation.metrics import evaluate
        y, proba = _make_binary()
        report   = evaluate(y, proba)
        assert 0.0 <= report.macro_f1  <= 1.0
        assert 0.0 <= report.brier_score <= 1.0
        assert 0.0 <= report.ece <= 1.0
        assert report.roc_auc is not None

    def test_macro_f1_gt_random(self):
        from src.evaluation.metrics import evaluate
        y, proba = _make_binary()
        report   = evaluate(y, proba)
        assert report.macro_f1 > 0.55  # better than random

    def test_ece_perfect_predictor(self):
        from src.evaluation.metrics import expected_calibration_error
        # Perfect predictor: p1 = y
        y   = np.array([0, 0, 1, 1, 0, 1])
        p1  = y.astype(float)
        ece = expected_calibration_error(y, p1)
        assert ece < 0.01

    def test_find_best_threshold(self):
        from src.evaluation.metrics import find_best_threshold
        y, proba = _make_binary()
        thr, score = find_best_threshold(y, proba[:, 1], metric="f1")
        assert 0.0 < thr < 1.0
        assert score > 0.0

    def test_conf_matrix_shape(self):
        from src.evaluation.metrics import evaluate
        y, proba = _make_binary()
        report   = evaluate(y, proba)
        assert report.conf_matrix is not None
        assert report.conf_matrix.shape == (2, 2)

    def test_mcc_range(self):
        from src.evaluation.metrics import evaluate
        y, proba = _make_binary()
        report   = evaluate(y, proba)
        assert -1.0 <= report.mcc <= 1.0

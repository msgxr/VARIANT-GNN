"""
tests/unit/test_calibration.py
Unit tests for EnsembleCalibrator.
"""
import numpy as np
import pytest


def _fake_proba(n: int = 200, seed: int = 0) -> tuple:
    rng    = np.random.default_rng(seed)
    p1     = rng.uniform(0.1, 0.9, size=n)
    p0     = 1 - p1
    proba  = np.column_stack([p0, p1])
    y      = (p1 > 0.5).astype(int)
    return proba, y


class TestEnsembleCalibrator:
    def test_isotonic_fit_transform(self):
        from src.calibration.calibrator import EnsembleCalibrator
        proba, y = _fake_proba()
        cal      = EnsembleCalibrator(method="isotonic")
        cal_p    = cal.fit_transform(proba, y)
        assert cal_p.shape == proba.shape
        np.testing.assert_allclose(cal_p.sum(axis=1), 1.0, atol=1e-6)

    def test_sigmoid_fit_transform(self):
        from src.calibration.calibrator import EnsembleCalibrator
        proba, y = _fake_proba(seed=5)
        cal      = EnsembleCalibrator(method="sigmoid")
        cal_p    = cal.fit_transform(proba, y)
        assert cal_p.shape == proba.shape
        np.testing.assert_allclose(cal_p.sum(axis=1), 1.0, atol=1e-6)

    def test_values_in_range(self):
        from src.calibration.calibrator import EnsembleCalibrator
        proba, y = _fake_proba()
        cal      = EnsembleCalibrator(method="isotonic")
        cal_p    = cal.fit_transform(proba, y)
        assert (cal_p >= 0).all() and (cal_p <= 1).all()

    def test_unknown_method_raises(self):
        from src.calibration.calibrator import EnsembleCalibrator
        with pytest.raises(ValueError):
            EnsembleCalibrator(method="unknown")

    def test_transform_before_fit_raises(self):
        from src.calibration.calibrator import EnsembleCalibrator
        proba, _ = _fake_proba()
        cal = EnsembleCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(proba)

    def test_calibration_curve_data(self):
        from src.calibration.calibrator import EnsembleCalibrator
        proba, y = _fake_proba()
        cal      = EnsembleCalibrator(method="isotonic")
        cal.fit(proba, y)
        data = cal.calibration_curve_data(proba, y, n_bins=5)
        assert "fraction_of_positives_before" in data
        assert "fraction_of_positives_after" in data

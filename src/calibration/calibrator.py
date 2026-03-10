"""
src/calibration/calibrator.py
Post-hoc calibration for the VARIANT-GNN ensemble.

Supports:
  - Platt Scaling  (method='sigmoid')
  - Isotonic Regression (method='isotonic')

Usage pattern:
    calibrator = EnsembleCalibrator(method='isotonic')
    calibrator.fit(proba_val, y_val)           # fit on held-out validation set
    cal_proba = calibrator.transform(proba)    # calibrated probabilities
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

CalibrationMethod = Literal["isotonic", "sigmoid"]


class EnsembleCalibrator:
    """
    Calibrates the Pathogenic-class probability of the VARIANT-GNN ensemble.

    The calibrator is fit on a held-out *validation* set (never the test set)
    to avoid optimistic calibration estimates.

    Parameters
    ----------
    method : ``'isotonic'`` (Isotonic Regression) or ``'sigmoid'`` (Platt Scaling).
    """

    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        self.method = method
        self._calibrator = None
        self._is_fitted  = False

    # ------------------------------------------------------------------
    def fit(self, proba: np.ndarray, y: np.ndarray) -> "EnsembleCalibrator":
        """
        Fit calibrator on validation-set probabilities.

        Parameters
        ----------
        proba : (N, 2) probability array from the ensemble.
        y     : True binary labels (0/1).
        """
        p1 = proba[:, 1]  # Pathogenic class probability

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(p1, y)
        elif self.method == "sigmoid":
            # Platt scaling: logistic regression on log-odds of p1
            log_odds = np.log(np.clip(p1, 1e-7, 1 - 1e-7) / (1 - np.clip(p1, 1e-7, 1 - 1e-7)))
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self._calibrator.fit(log_odds.reshape(-1, 1), y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method!r}")

        self._is_fitted = True
        logger.info("Calibrator fitted: method=%s, n=%d", self.method, len(y))
        return self

    # ------------------------------------------------------------------
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration and return a new (N, 2) probability array.

        Parameters
        ----------
        proba : (N, 2) uncalibrated probabilities.

        Returns
        -------
        np.ndarray of shape (N, 2) with calibrated probabilities that sum to 1.
        """
        if not self._is_fitted:
            raise RuntimeError("EnsembleCalibrator must be fitted before transform.")

        p1 = proba[:, 1]

        if self.method == "isotonic":
            p1_cal = self._calibrator.transform(p1)
        else:
            log_odds = np.log(np.clip(p1, 1e-7, 1 - 1e-7) / (1 - np.clip(p1, 1e-7, 1 - 1e-7)))
            p1_cal = self._calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]

        p1_cal = np.clip(p1_cal, 0.0, 1.0)
        p0_cal = 1.0 - p1_cal
        return np.column_stack([p0_cal, p1_cal])

    # ------------------------------------------------------------------
    def fit_transform(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on and immediately transform the same data (for inspection only)."""
        self.fit(proba, y)
        return self.transform(proba)

    # ------------------------------------------------------------------
    def calibration_curve_data(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Return before/after calibration curve data for plotting.

        Returns
        -------
        dict with keys:
            ``fraction_of_positives_before``
            ``mean_predicted_value_before``
            ``fraction_of_positives_after``
            ``mean_predicted_value_after``
        """
        frac_before, mean_before = calibration_curve(
            y, proba[:, 1], n_bins=n_bins, strategy="uniform"
        )
        cal_proba = self.transform(proba)
        frac_after, mean_after = calibration_curve(
            y, cal_proba[:, 1], n_bins=n_bins, strategy="uniform"
        )
        return {
            "fraction_of_positives_before": frac_before,
            "mean_predicted_value_before":  mean_before,
            "fraction_of_positives_after":  frac_after,
            "mean_predicted_value_after":   mean_after,
        }

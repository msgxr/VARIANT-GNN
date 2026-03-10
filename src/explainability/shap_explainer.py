"""
src/explainability/shap_explainer.py
SHAP-based XAI with safe fallback.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — SHAP explanations disabled.")


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for the XGBoost sub-model.

    Falls back gracefully if SHAP is not installed or if the model
    is incompatible with TreeExplainer.
    """

    def __init__(
        self,
        xgb_model,
        feature_names: Optional[List[str]] = None,
        training_data: Optional[np.ndarray] = None,
    ) -> None:
        self._model        = xgb_model
        self.feature_names = feature_names
        self.training_data = training_data
        self._explainer    = None

        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(xgb_model)
                logger.info("SHAP TreeExplainer initialised.")
            except Exception as exc:
                logger.warning("SHAP TreeExplainer init failed: %s", exc)

    # ------------------------------------------------------------------
    def _names(self, n: int) -> List[str]:
        if self.feature_names and len(self.feature_names) == n:
            return list(self.feature_names)
        return [f"Feature_{i}" for i in range(n)]

    # ------------------------------------------------------------------
    def explain_instance(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return SHAP values for a single sample; None if unavailable."""
        if self._explainer is None:
            return None
        try:
            return self._explainer.shap_values(x)
        except Exception as exc:
            logger.warning("SHAP explain_instance failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def get_top_features(
        self, X: np.ndarray, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Return list of (feature_name, mean_abs_shap) sorted descending."""
        if self._explainer is None:
            return []
        try:
            sv = self._explainer.shap_values(X)
            if isinstance(sv, list):
                sv = np.array(sv[1])
            mean_abs = np.mean(np.abs(sv), axis=0)
            names    = self._names(len(mean_abs))
            ranked   = sorted(zip(names, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
            return ranked[:top_n]
        except Exception as exc:
            logger.warning("get_top_features failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def plot_summary(self, X: np.ndarray, output_path: str = "reports/shap_summary.png") -> None:
        if self._explainer is None:
            logger.warning("SHAP explainer not available; skipping summary plot.")
            return
        try:
            sv    = self._explainer.shap_values(X)
            names = self._names(X.shape[1])
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X, feature_names=names, show=False)
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("SHAP summary plot saved → %s", output_path)
        except Exception as exc:
            logger.warning("SHAP summary plot failed: %s", exc)

    # ------------------------------------------------------------------
    def plot_waterfall(
        self, x_instance: np.ndarray, output_path: str = "reports/shap_waterfall.png"
    ) -> None:
        if self._explainer is None:
            return
        try:
            sv = self._explainer(x_instance.reshape(1, -1))
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(sv[0], show=False)
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("SHAP waterfall saved → %s", output_path)
        except Exception as exc:
            logger.warning("SHAP waterfall failed: %s", exc)

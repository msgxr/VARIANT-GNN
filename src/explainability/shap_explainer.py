"""
src/explainability/shap_explainer.py
SHAP-based XAI with safe fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, cast

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
        xgb_model: Any,
        feature_names: Optional[List[str]] = None,
        training_data: Optional[np.ndarray] = None,
    ) -> None:
        self._model = xgb_model
        self.feature_names = feature_names
        self.training_data = training_data
        self._explainer = None

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
            return cast("np.ndarray | None", self._explainer.shap_values(x))
        except Exception as exc:
            logger.warning("SHAP explain_instance failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def get_top_features(self, X: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return list of (feature_name, mean_abs_shap) sorted descending."""
        if self._explainer is None:
            return []
        try:
            sv = self._explainer.shap_values(X)
            if isinstance(sv, list):
                sv = np.array(sv[1])
            mean_abs = np.mean(np.abs(sv), axis=0)
            names = self._names(len(mean_abs))
            ranked = sorted(zip(names, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
            return ranked[:top_n]
        except Exception as exc:
            logger.warning("get_top_features failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def explain_instance_table(
        self,
        x_instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        base_value_override: Optional[float] = None,
    ) -> Optional[dict]:
        """PDR §4.4 için yapılandırılmış örnek-bazlı SHAP tablosu üretir.

        Returns
        -------
        Dict:
            {
                "base_value":      float,   # eğitim seti prior
                "shap_sum":        float,   # sum(SHAP) ≈ logit(pred) - base
                "features": [
                    {
                        "name": str, "shap_value": float,
                        "abs_shap": float, "direction": str
                    }, ...
                ],
                "top_positive":  List[dict],   # patojenik yönde top-5
                "top_negative":  List[dict],   # benign yönde top-5
            }
        """
        if self._explainer is None:
            return None
        try:
            sv = self._explainer.shap_values(x_instance.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv).flatten()

            try:
                bv = (
                    base_value_override
                    if base_value_override is not None
                    else float(
                        self._explainer.expected_value
                        if not isinstance(self._explainer.expected_value, (list, np.ndarray))
                        else self._explainer.expected_value[1]
                    )
                )
            except Exception:
                bv = 0.0

            names = feature_names or self._names(len(sv))

            features = [
                {
                    "name": n,
                    "shap_value": round(float(v), 5),
                    "abs_shap": round(abs(float(v)), 5),
                    "direction": "Patojenik" if v > 0.001 else ("Benign" if v < -0.001 else "Nötr"),
                }
                for n, v in zip(names, sv)
            ]
            features_sorted = sorted(features, key=lambda d: d["abs_shap"], reverse=True)
            top_pos = [f for f in features_sorted if f["direction"] == "Patojenik"][:5]
            top_neg = [f for f in features_sorted if f["direction"] == "Benign"][:5]

            return {
                "base_value": round(bv, 5),
                "shap_sum": round(float(sv.sum()), 5),
                "features": features_sorted,
                "top_positive": top_pos,
                "top_negative": top_neg,
            }
        except Exception as exc:
            logger.warning("explain_instance_table failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def plot_summary(self, X: np.ndarray, output_path: str = "reports/shap_summary.png") -> None:
        if self._explainer is None:
            logger.warning("SHAP explainer not available; skipping summary plot.")
            return
        try:
            sv = self._explainer.shap_values(X)
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
    def plot_waterfall(self, x_instance: np.ndarray, output_path: str = "reports/shap_waterfall.png") -> None:
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

"""
src/ensemble/panel_aware_ensemble.py
Panel-aware ensemble for TEKNOFEST 2026 variant classification.

Rules enforced:
  - Weights learned only from OOF / validation predictions (NOT blind/test data).
  - Weights are non-negative and sum to 1.0 per panel.
  - Unknown panels fall back to global weights.
  - Deterministic seed used throughout.
  - JSON save/load for artifact persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from src.ensemble.weight_optimizer import optimize_weights

logger = logging.getLogger(__name__)

KNOWN_PANELS: tuple[str, ...] = ("General", "Hereditary_Cancer", "PAH", "CFTR")


class PanelAwareEnsemble:
    """
    Learns per-panel blending weights from OOF predictions.

    Parameters
    ----------
    n_models : Number of base models in the ensemble.
    seed     : Random seed for reproducibility.
    """

    def __init__(self, n_models: int, seed: int = 42) -> None:
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        self.n_models = n_models
        self.seed = seed

        self._panel_weights: Dict[str, np.ndarray] = {}
        self._global_weights: np.ndarray = np.ones(n_models) / n_models
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        panels: Sequence[str],
    ) -> "PanelAwareEnsemble":
        """
        Learn per-panel weights from OOF predictions.

        Parameters
        ----------
        oof_predictions : shape (n_samples, n_models) — Pathogenic probability per model.
        y_true          : shape (n_samples,) — binary ground-truth labels.
        panels          : length n_samples — panel name for each sample.

        Returns
        -------
        self
        """
        oof_predictions = np.asarray(oof_predictions, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        panels = list(panels)

        if oof_predictions.ndim != 2:
            raise ValueError(f"oof_predictions must be 2-D, got shape {oof_predictions.shape}")
        if oof_predictions.shape[1] != self.n_models:
            raise ValueError(
                f"oof_predictions has {oof_predictions.shape[1]} columns; expected n_models={self.n_models}"
            )
        if len(y_true) != len(panels) or len(y_true) != oof_predictions.shape[0]:
            raise ValueError("oof_predictions, y_true and panels must all have the same length.")

        panels_arr = np.array(panels)

        # ── Global weights (all panels combined) ────────────────────────
        self._global_weights = optimize_weights(oof_predictions, y_true, seed=self.seed)

        # ── Per-panel weights ────────────────────────────────────────────
        for panel in KNOWN_PANELS:
            mask = panels_arr == panel
            if mask.sum() < 10:
                logger.warning("Panel '%s' has only %d samples; using global weights.", panel, mask.sum())
                self._panel_weights[panel] = self._global_weights.copy()
                continue
            w = optimize_weights(
                oof_predictions[mask],
                y_true[mask],
                seed=self.seed,
            )
            self._panel_weights[panel] = w
            logger.info("Panel '%s': n=%d  weights=%s", panel, int(mask.sum()), np.round(w, 4).tolist())

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        model_predictions: np.ndarray,
        panels: Sequence[str],
    ) -> np.ndarray:
        """
        Blend model predictions using per-panel weights.

        Parameters
        ----------
        model_predictions : shape (n_samples, n_models) — Pathogenic probability.
        panels            : length n_samples — panel names.

        Returns
        -------
        blended : shape (n_samples,) — blended Pathogenic probability.
        """
        if not self._is_fitted:
            raise RuntimeError("PanelAwareEnsemble must be fitted before calling predict_proba.")

        model_predictions = np.asarray(model_predictions, dtype=float)
        if model_predictions.ndim != 2 or model_predictions.shape[1] != self.n_models:
            raise ValueError(
                f"model_predictions must be shape (n_samples, {self.n_models}), got {model_predictions.shape}"
            )

        panels = list(panels)
        if len(panels) != model_predictions.shape[0]:
            raise ValueError("len(panels) must equal model_predictions.shape[0]")

        blended = np.empty(model_predictions.shape[0], dtype=float)
        panels_arr = np.array(panels)

        for panel in KNOWN_PANELS:
            mask = panels_arr == panel
            if not mask.any():
                continue
            w = self._panel_weights.get(panel, self._global_weights)
            blended[mask] = model_predictions[mask] @ w

        # Unknown panels → global weights
        unknown_mask = ~np.isin(panels_arr, list(KNOWN_PANELS))
        if unknown_mask.any():
            logger.warning(
                "Unknown panels encountered: %s — using global weights.",
                np.unique(panels_arr[unknown_mask]).tolist(),
            )
            blended[unknown_mask] = model_predictions[unknown_mask] @ self._global_weights

        return blended

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialise weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "n_models": self.n_models,
            "seed": self.seed,
            "global_weights": self._global_weights.tolist(),
            "panel_weights": {k: v.tolist() for k, v in self._panel_weights.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("PanelAwareEnsemble saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "PanelAwareEnsemble":
        """Load weights from JSON produced by save()."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PanelAwareEnsemble artifact not found: {path}")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        obj = cls(n_models=payload["n_models"], seed=payload["seed"])
        obj._global_weights = np.array(payload["global_weights"])
        obj._panel_weights = {k: np.array(v) for k, v in payload["panel_weights"].items()}
        obj._is_fitted = True
        logger.info("PanelAwareEnsemble loaded from %s", path)
        return obj

"""
src/models/ensemble.py
Multi-modal ensemble: XGBoost + GNN + DNN.

Ensemble weights are loaded from config and can optionally be optimised on
a held-out validation set via ``optimise_weights``.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from src.config import get_settings
from src.models.dnn import VariantDNN
from src.models.gnn import FeatureGNN

logger = logging.getLogger(__name__)


def _gnn_predict_proba(
    model: FeatureGNN, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Return (N, num_classes) probability array from a GNN DataLoader."""
    model.eval()
    probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out  = model(data)
            probs_list.append(F.softmax(out, dim=1).cpu().numpy())
    return np.vstack(probs_list)


def _dnn_predict_proba(
    model: VariantDNN, X: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return (N, num_classes) probability array from a DNN."""
    model.eval()
    with torch.no_grad():
        tensor_x = torch.FloatTensor(X).to(device)
        out       = model(tensor_x)
        return F.softmax(out, dim=1).cpu().numpy()


class HybridEnsemble:
    """
    Weighted ensemble of XGBoost, GNN, and DNN models.

    Weights can be:
      - loaded from config (default)
      - manually set
      - optimised on validation data via ``optimise_weights``

    Labels in output:
      index 0 → Benign
      index 1 → Pathogenic
    Extensible to multi-class by adding label indices.
    """

    # VUS-ready label map — extend here for multi-class / VUS support
    LABEL_MAP = {0: "Benign", 1: "Pathogenic"}

    def __init__(
        self,
        xgb_model:   Optional[xgb.XGBClassifier] = None,
        gnn_model:   Optional[FeatureGNN]         = None,
        dnn_model:   Optional[VariantDNN]         = None,
        weights:     Optional[List[float]]        = None,
        device:      Optional[torch.device]       = None,
    ) -> None:
        cfg = get_settings()
        self.xgb    = xgb_model
        self.gnn    = gnn_model
        self.dnn    = dnn_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Read weights from config, allow override
        self.weights = list(weights) if weights is not None else list(cfg.ensemble.weights)
        assert len(self.weights) == 3 and abs(sum(self.weights) - 1.0) < 0.01, (
            "Ensemble weights must be a 3-element list summing to 1.0"
        )

    # ------------------------------------------------------------------
    # Probability collection
    # ------------------------------------------------------------------

    def predict_proba_all(
        self,
        X_scaled:   np.ndarray,
        gnn_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xgb_proba, gnn_proba, dnn_proba) each shape (N, C)."""
        if self.xgb is None or self.gnn is None or self.dnn is None:
            raise RuntimeError("All three sub-models must be set before inference.")

        xgb_probs = self.xgb.predict_proba(X_scaled)
        gnn_probs = _gnn_predict_proba(
            self.gnn.to(self.device), gnn_loader, self.device
        )
        dnn_probs = _dnn_predict_proba(self.dnn.to(self.device), X_scaled, self.device)

        return xgb_probs, gnn_probs, dnn_probs

    # ------------------------------------------------------------------
    # Ensemble combination
    # ------------------------------------------------------------------

    def combine(
        self,
        xgb_probs: np.ndarray,
        gnn_probs: np.ndarray,
        dnn_probs: np.ndarray,
    ) -> np.ndarray:
        """Weighted average of three probability matrices."""
        w = self.weights
        return w[0] * xgb_probs + w[1] * gnn_probs + w[2] * dnn_probs

    def predict(
        self,
        X_scaled:    np.ndarray,
        gnn_loader:  DataLoader,
        threshold:   float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, ensemble_proba).
        ``threshold`` applied to the Pathogenic class (index 1).
        """
        xp, gp, dp    = self.predict_proba_all(X_scaled, gnn_loader)
        proba          = self.combine(xp, gp, dp)
        preds          = (proba[:, 1] >= threshold).astype(int)
        return preds, proba

    # ------------------------------------------------------------------
    # Weight optimisation on validation data
    # ------------------------------------------------------------------

    def optimise_weights(
        self,
        X_val:      np.ndarray,
        gnn_loader: DataLoader,
        y_val:      np.ndarray,
    ) -> List[float]:
        """
        Find weights [w_xgb, w_gnn, w_dnn] that maximise validation Macro F1.
        Uses scipy Nelder-Mead on the 3-simplex.
        Weights are updated in place and returned.
        """
        xp, gp, dp = self.predict_proba_all(X_val, gnn_loader)

        def neg_f1(w: np.ndarray) -> float:
            w_norm = w / w.sum()
            proba  = w_norm[0] * xp + w_norm[1] * gp + w_norm[2] * dp
            preds  = np.argmax(proba, axis=1)
            return -f1_score(y_val, preds, average="macro", zero_division=0)

        x0     = np.array(self.weights)
        result = minimize(neg_f1, x0, method="Nelder-Mead")
        w_opt  = result.x / result.x.sum()
        self.weights = w_opt.tolist()
        logger.info("Optimised ensemble weights: %s (F1=%.4f)", self.weights, -result.fun)
        return self.weights

    # ------------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------------

    @staticmethod
    def pathogenic_risk_score(ensemble_proba: np.ndarray) -> np.ndarray:
        """Convert Pathogenic class probability to a 0–100 risk score."""
        return (ensemble_proba[:, 1] * 100).round(2)

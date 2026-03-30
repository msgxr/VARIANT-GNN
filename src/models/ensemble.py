"""
src/models/ensemble.py
Multi-modal ensemble: XGBoost + LightGBM + GNN + DNN.

Ensemble weights are loaded from config and can optionally be optimised on
a held-out validation set via ``optimise_weights``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from src.config import get_settings
from src.models.dnn import VariantDNN
from src.models.gnn import FeatureGNN, VariantSAGEGNN

logger = logging.getLogger(__name__)


def _gnn_predict_proba(
    model: Union[FeatureGNN, VariantSAGEGNN], 
    loader: DataLoader, 
    device: torch.device
) -> np.ndarray:
    """Return (N, num_classes) probability array from a GNN DataLoader."""
    model.eval()
    probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs_list.append(F.softmax(out, dim=1).cpu().numpy())
    return np.vstack(probs_list)


def _dnn_predict_proba(
    model: VariantDNN, 
    X: np.ndarray, 
    device: torch.device
) -> np.ndarray:
    """Return (N, num_classes) probability array from a DNN."""
    model.eval()
    with torch.no_grad():
        tensor_x = torch.FloatTensor(X).to(device)
        out = model(tensor_x)
        return F.softmax(out, dim=1).cpu().numpy()


class HybridEnsemble:
    """
    Weighted ensemble of XGBoost, LightGBM, GNN, and DNN models.

    Weights can be:
      - loaded from config (default)
      - manually set
      - optimised on validation data via ``optimise_weights``

    Labels in output:
      index 0 → Benign
      index 1 → Pathogenic
    """

    LABEL_MAP: Dict[int, str] = {0: "Benign", 1: "Pathogenic"}

    def __init__(
        self,
        xgb_model: Optional[xgb.XGBClassifier] = None,
        lgbm_model: Optional[Any] = None,  # LightGBM model or wrapper
        gnn_model: Optional[nn.Module] = None,  # FeatureGNN or VariantSAGEGNN
        dnn_model: Optional[VariantDNN] = None,
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        cfg = get_settings()
        self.xgb = xgb_model
        self.lgbm = lgbm_model
        self.gnn = gnn_model
        self.dnn = dnn_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Read weights from config, allow override; normalise automatically
        raw_w = list(weights) if weights is not None else list(cfg.ensemble.weights)
        if len(raw_w) != 4:
            logger.warning("Ensemble weights length != 4 (%d). Using default [0.35, 0.30, 0.25, 0.10]", len(raw_w))
            raw_w = [0.35, 0.30, 0.25, 0.10]
            
        w_sum = sum(raw_w)
        self.weights = [w / w_sum for w in raw_w]
        
        self.meta_learner: Optional[Any] = None

    def predict_proba_all(
        self,
        X_scaled: np.ndarray,
        gnn_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (xgb_probs, lgb_probs, gnn_probs, dnn_probs) each shape (N, C)."""
        xgb_probs = self.xgb.predict_proba(X_scaled) if self.xgb is not None else None
        lgb_probs = self.lgbm.predict_proba(X_scaled) if self.lgbm is not None else None

        gnn_probs = None
        if self.gnn is not None:
            if isinstance(self.gnn, VariantSAGEGNN):
                gnn_probs = self._sage_predict_proba(X_scaled, **kwargs)
            elif gnn_loader is not None:
                gnn_probs = _gnn_predict_proba(
                    self.gnn.to(self.device), gnn_loader, self.device
                )

        dnn_probs = _dnn_predict_proba(self.dnn.to(self.device), X_scaled, self.device) if self.dnn is not None else None

        return xgb_probs, lgb_probs, gnn_probs, dnn_probs

    def _sage_predict_proba(
        self,
        X_scaled: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run VariantSAGEGNN inference on the full feature matrix."""
        from src.graph.builder import SampleKNNGraphBuilder
        knn_k = getattr(get_settings().gnn, "knn_k", 5)
        data = SampleKNNGraphBuilder(k=knn_k).build(X_scaled, y=None)
        
        model = self.gnn.to(self.device)
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            logits = model(data.x, data.edge_index, **kwargs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def combine(
        self,
        xgb_proba: Optional[np.ndarray],
        lgb_proba: Optional[np.ndarray],
        gnn_proba: Optional[np.ndarray],
        dnn_proba: Optional[np.ndarray],
    ) -> np.ndarray:
        """Weighted average of available probability matrices."""
        pairs = [
            (xgb_proba, self.weights[0]),
            (lgb_proba, self.weights[1]),
            (gnn_proba, self.weights[2]),
            (dnn_proba, self.weights[3]),
        ]
        available = [(p, w) for p, w in pairs if p is not None]
        if not available:
            raise ValueError("At least one probability array must be provided to combine().")
        
        total_w = sum(w for _, w in available)
        return sum((w / total_w) * p for p, w in available)

    def predict(
        self,
        X_scaled_or_proba: np.ndarray,
        gnn_loader: Optional[DataLoader] = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict classes or (classes, probabilities).
        """
        is_sage = isinstance(self.gnn, VariantSAGEGNN)

        if gnn_loader is None and not is_sage:
            # Pre-computed (N, 2) probability array supplied directly
            proba = X_scaled_or_proba
            return (proba[:, 1] >= threshold).astype(int)

        # Full prediction
        xp, lp, gp, dp = self.predict_proba_all(
            X_scaled_or_proba, gnn_loader, **kwargs
        )
        proba = self.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gp, dnn_proba=dp)
        preds = (proba[:, 1] >= threshold).astype(int)
        return preds, proba

    def optimise_weights(
        self,
        X_val: np.ndarray,
        gnn_loader: DataLoader,
        y_val: np.ndarray,
    ) -> List[float]:
        """Find weights that maximise validation Macro F1."""
        xp, lp, gp, dp = self.predict_proba_all(X_val, gnn_loader)

        def neg_f1(w: np.ndarray) -> float:
            w_norm = w / w.sum()
            proba = w_norm[0] * xp + w_norm[1] * lp + w_norm[2] * gp + w_norm[3] * dp
            preds = np.argmax(proba, axis=1)
            return -f1_score(y_val, preds, average="macro", zero_division=0)

        x0 = np.array(self.weights)
        result = minimize(neg_f1, x0, method="Nelder-Mead")
        w_opt = result.x / result.x.sum()
        self.weights = w_opt.tolist()
        logger.info("Optimised ensemble weights: %s (F1=%.4f)", self.weights, -result.fun)
        return self.weights

    @staticmethod
    def pathogenic_risk_score(ensemble_proba: np.ndarray) -> np.ndarray:
        """Convert Pathogenic class probability to a 0–100 risk score."""
        return (ensemble_proba[:, 1] * 100).round(2)

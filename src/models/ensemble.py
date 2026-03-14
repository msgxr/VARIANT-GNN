"""
src/models/ensemble.py
Multi-modal ensemble: XGBoost + LightGBM + GNN + DNN.

Ensemble weights can be:
  - loaded from config (default: [0.35, 0.30, 0.25, 0.10])
  - manually set
  - optimised on a held-out validation set via ``optimise_weights``
    (scipy Nelder-Mead on the N-simplex)
  - replaced by a stacking meta-learner via ``fit_meta_learner``
    (LogisticRegression trained on the inner validation set probabilities)

Labels in output:
  index 0 → Benign
  index 1 → Pathogenic

Backward compatible: if lgbm_model is None, falls back to 3-model
[XGB, GNN, DNN] ensemble and re-normalises weights automatically.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
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


def _lgbm_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return (N, 2) probability array from a LightGBM classifier."""
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    return proba


class HybridEnsemble:
    """
    Weighted ensemble of XGBoost, LightGBM, GNN, and DNN models.

    Weights [w_xgb, w_lgbm, w_gnn, w_dnn] are automatically normalised.
    If lgbm_model is None (backward compat), weights are treated as
    [w_xgb, w_gnn, w_dnn] and re-normalised.

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
        lgbm_model:  Optional[object]             = None,  # LGBMClassifier
        gnn_model:   Optional[nn.Module]          = None,  # FeatureGNN or VariantSAGEGNN
        dnn_model:   Optional[VariantDNN]         = None,
        weights:     Optional[List[float]]        = None,
        device:      Optional[torch.device]       = None,
    ) -> None:
        cfg = get_settings()
        self.xgb    = xgb_model
        self.lgbm   = lgbm_model
        self.gnn    = gnn_model
        self.dnn    = dnn_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Stacking meta-learner (optional; fitted via fit_meta_learner)
        self.meta_learner: Optional[LogisticRegression] = None

        # Read weights from config, allow override; normalise automatically.
        # Support both 3-weight (legacy) and 4-weight (new) configs.
        if weights is not None:
            raw_w = list(weights)
        else:
            raw_w = list(cfg.ensemble.weights)

        # Backward compat: if LGBM absent and 4 weights given, merge LGB weight into XGB
        if lgbm_model is None and len(raw_w) == 4:
            raw_w = [raw_w[0] + raw_w[1], raw_w[2], raw_w[3]]  # [xgb+lgbm, gnn, dnn]
        elif lgbm_model is not None and len(raw_w) == 3:
            # Redistribute: split first weight between XGB and LGBM equally
            raw_w = [raw_w[0] * 0.55, raw_w[0] * 0.45, raw_w[1], raw_w[2]]

        w_sum = sum(raw_w)
        self.weights = [w / w_sum for w in raw_w]

    # ------------------------------------------------------------------
    # Probability collection
    # ------------------------------------------------------------------

    def predict_proba_all(
        self,
        X_scaled:   np.ndarray,
        gnn_loader: Optional[DataLoader] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Return (xgb_proba, lgbm_proba, gnn_proba, dnn_proba) each shape (N, 2).
        lgbm_proba is None when lgbm_model is not set."""
        if self.xgb is None or self.gnn is None or self.dnn is None:
            raise RuntimeError("xgb, gnn, and dnn sub-models must be set before inference.")

        xgb_probs  = self.xgb.predict_proba(X_scaled)
        lgbm_probs = _lgbm_predict_proba(self.lgbm, X_scaled) if self.lgbm is not None else None

        # Dispatch: VariantSAGEGNN uses its own sample-graph inference
        from src.models.gnn import VariantSAGEGNN
        if isinstance(self.gnn, VariantSAGEGNN):
            gnn_probs = self._sage_predict_proba(X_scaled, **kwargs)
        else:
            gnn_probs = _gnn_predict_proba(
                self.gnn.to(self.device), gnn_loader, self.device
            )

        dnn_probs = _dnn_predict_proba(self.dnn.to(self.device), X_scaled, self.device)

        return xgb_probs, lgbm_probs, gnn_probs, dnn_probs

    def _sage_predict_proba(self, X_scaled: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run VariantSAGEGNN inference on the full feature matrix.

        Builds a coordinate-free cosine k-NN sample graph (all N variants as
        nodes) and returns per-node softmax probabilities of shape [N, 2].
        """
        from src.config import get_settings
        from src.graph.builder import SampleKNNGraphBuilder
        knn_k = getattr(get_settings().gnn, "knn_k", 5)
        data  = SampleKNNGraphBuilder(k=knn_k).build(X_scaled, y=None)
        model = self.gnn.to(self.device)
        model.eval()
        data  = data.to(self.device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, **kwargs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()   # [N, 2]
        return probs

    # ------------------------------------------------------------------
    # Ensemble combination
    # ------------------------------------------------------------------

    def combine(
        self,
        xgb_proba:  Optional[np.ndarray],
        gnn_proba:  Optional[np.ndarray],
        dnn_proba:  Optional[np.ndarray],
        lgbm_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combine probability matrices from base models.

        If a stacking meta-learner has been fitted via ``fit_meta_learner``,
        it is used instead of fixed weights for superior generalisation.
        Otherwise falls back to weighted average.
        """
        # Stacking path — meta-learner takes precedence over fixed weights
        if self.meta_learner is not None and xgb_proba is not None and gnn_proba is not None:
            return self._meta_predict_proba(xgb_proba, gnn_proba, dnn_proba, lgbm_proba)

        # Weighted-average path (fallback)
        if lgbm_proba is not None:
            # 4-model mode
            pairs = [
                (xgb_proba,  self.weights[0]),
                (lgbm_proba, self.weights[1]),
                (gnn_proba,  self.weights[2]),
                (dnn_proba,  self.weights[3]),
            ]
        else:
            # 3-model mode (backward compat)
            w = self.weights
            if len(w) == 4:
                # Re-normalise dropping LGBM weight
                w3 = [w[0] + w[1], w[2], w[3]]
                t  = sum(w3)
                w3 = [x / t for x in w3]
            else:
                w3 = w
            pairs = [
                (xgb_proba, w3[0]),
                (gnn_proba, w3[1]),
                (dnn_proba, w3[2]),
            ]

        available = [(p, w) for p, w in pairs if p is not None]
        if not available:
            raise ValueError("At least one probability array must be provided to combine().")
        total_w = sum(w for _, w in available)
        return sum((w / total_w) * p for p, w in available)

    def predict(
        self,
        X_scaled_or_proba: np.ndarray,
        gnn_loader:        Optional[DataLoader] = None,
        threshold:         float = 0.5,
        **kwargs,
    ) -> "np.ndarray | Tuple[np.ndarray, np.ndarray]":
        """
        Two call modes:
          - ``predict(proba)``                     → class-index array (for testing / simple use)
          - ``predict(X_scaled, gnn_loader, thr)`` → (predictions, proba) tuple (inference pipeline)

        When the GNN member is a ``VariantSAGEGNN``, the loader argument is
        accepted but ignored — SAGE builds its own sample graph from X_scaled.
        """
        from src.models.gnn import VariantSAGEGNN
        is_sage = isinstance(self.gnn, VariantSAGEGNN)

        if gnn_loader is None and not is_sage:
            # Pre-computed (N, 2) probability array supplied directly
            proba = X_scaled_or_proba
            return (proba[:, 1] >= threshold).astype(int)

        # Full prediction: collect proba from all sub-models
        xp, lp, gp, dp = self.predict_proba_all(X_scaled_or_proba, gnn_loader, **kwargs)
        proba           = self.combine(xgb_proba=xp, gnn_proba=gp, dnn_proba=dp, lgbm_proba=lp)
        preds           = (proba[:, 1] >= threshold).astype(int)
        return preds, proba

    # ------------------------------------------------------------------
    # Weight optimisation on validation data
    # ------------------------------------------------------------------

    def optimise_weights(
        self,
        X_val:      np.ndarray,
        gnn_loader: Optional[DataLoader],
        y_val:      np.ndarray,
    ) -> List[float]:
        """
        Find weights that maximise validation Macro F1.
        Uses scipy Nelder-Mead on the N-simplex (N=3 or N=4 depending on LGBM).
        Weights are updated in place and returned.
        """
        xp, lp, gp, dp = self.predict_proba_all(X_val, gnn_loader)
        use_lgbm = lp is not None

        if use_lgbm:
            def neg_f1(w: np.ndarray) -> float:
                w_norm = np.abs(w) / np.abs(w).sum()
                proba  = w_norm[0]*xp + w_norm[1]*lp + w_norm[2]*gp + w_norm[3]*dp
                preds  = np.argmax(proba, axis=1)
                return -f1_score(y_val, preds, average="macro", zero_division=0)
            x0 = np.array(self.weights[:4])
        else:
            w3 = self.weights[:3] if len(self.weights) == 3 else [self.weights[0]+self.weights[1], self.weights[2], self.weights[3]]
            def neg_f1(w: np.ndarray) -> float:
                w_norm = np.abs(w) / np.abs(w).sum()
                proba  = w_norm[0]*xp + w_norm[1]*gp + w_norm[2]*dp
                preds  = np.argmax(proba, axis=1)
                return -f1_score(y_val, preds, average="macro", zero_division=0)
            x0 = np.array(w3)

        result = minimize(neg_f1, x0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5})
        w_opt  = np.abs(result.x) / np.abs(result.x).sum()
        self.weights = w_opt.tolist()
        logger.info("Optimised ensemble weights: %s  (val Macro F1=%.4f)",
                    [f"{v:.3f}" for v in self.weights], -result.fun)
        return self.weights

    # ------------------------------------------------------------------
    # Stacking meta-learner
    # ------------------------------------------------------------------

    def fit_meta_learner(
        self,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        gnn_loader: Optional[DataLoader] = None,
    ) -> "HybridEnsemble":
        """
        Fit a LogisticRegression meta-learner on the inner validation set.

        The meta-learner takes [xgb_p1, lgbm_p1, gnn_p1, dnn_p1] as input
        and predicts the true label, replacing fixed-weight combination.

        Call AFTER ``optimise_weights`` for best results — weight optimisation
        first gives a good starting-point ensemble, then stacking refines it.

        Returns self for chaining.
        """
        xp, lp, gp, dp = self.predict_proba_all(X_val, gnn_loader)
        # Build meta-feature matrix: pathogenic probability from each model
        cols = [xp[:, 1], gp[:, 1], dp[:, 1]]
        if lp is not None:
            cols.insert(1, lp[:, 1])
        meta_X = np.column_stack(cols)

        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", random_state=42
        )
        self.meta_learner.fit(meta_X, y_val)
        meta_preds = self.meta_learner.predict(meta_X)
        meta_f1    = f1_score(y_val, meta_preds, average="macro", zero_division=0)
        logger.info("Meta-learner fitted. Val Macro F1 = %.4f", meta_f1)
        return self

    def _meta_predict_proba(
        self,
        xgb_proba:  np.ndarray,
        gnn_proba:  np.ndarray,
        dnn_proba:  np.ndarray,
        lgbm_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Use stacking meta-learner to combine base model probabilities."""
        cols = [xgb_proba[:, 1], gnn_proba[:, 1], dnn_proba[:, 1]]
        if lgbm_proba is not None:
            cols.insert(1, lgbm_proba[:, 1])
        meta_X = np.column_stack(cols)
        meta_p = self.meta_learner.predict_proba(meta_X)  # (N, 2)
        return meta_p

    # ------------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------------

    @staticmethod
    def pathogenic_risk_score(ensemble_proba: np.ndarray) -> np.ndarray:
        """Convert Pathogenic class probability to a 0–100 risk score."""
        return (ensemble_proba[:, 1] * 100).round(2)

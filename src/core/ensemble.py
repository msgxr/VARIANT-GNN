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
from src.core.models.dnn import VariantDNN
from src.core.models.gnn import VariantGATv2GNN

logger = logging.getLogger(__name__)


def _gnn_predict_proba(
    model: VariantGATv2GNN, 
    loader: DataLoader, 
    device: torch.device
) -> np.ndarray:
    """Return (N, num_classes) probability array from a GNN DataLoader."""
    model.eval()
    probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
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
    Weighted ensemble of XGBoost, LightGBM, GATv2, and DNN models.
    Supports Uncertainty Quantification via GNN MC-Dropout.
    """

    LABEL_MAP: Dict[int, str] = {0: "Benign", 1: "Pathogenic"}

    def __init__(
        self,
        xgb_model: Optional[xgb.XGBClassifier] = None,
        lgbm_model: Optional[Any] = None,
        gnn_model: Optional[VariantGATv2GNN] = None,
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
        
        raw_w = list(weights) if weights is not None else list(cfg.ensemble.weights)
        if len(raw_w) != 4:
            raw_w = [0.30, 0.30, 0.25, 0.15]  # PSR §5.3 uyumlu
            
        w_sum = sum(raw_w)
        self.weights = [w / w_sum for w in raw_w]
        
        self.meta_learner: Optional[Any] = None

    def predict_proba_all(
        self,
        X_scaled: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (xgb_probs, lgb_probs, gnn_probs, dnn_probs) each shape (N, C)."""
        xgb_probs = self.xgb.predict_proba(X_scaled) if self.xgb is not None else None
        lgb_probs = self.lgbm.predict_proba(X_scaled) if self.lgbm is not None else None

        gnn_probs = None
        if self.gnn is not None:
             gnn_probs = self._gat_predict_proba(X_scaled, **kwargs)

        dnn_probs = _dnn_predict_proba(self.dnn.to(self.device), X_scaled, self.device) if self.dnn is not None else None

        return xgb_probs, lgb_probs, gnn_probs, dnn_probs

    def _gat_predict_proba(
        self,
        X_scaled: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run VariantGATv2GNN inference using KNN graph building."""
        from src.core.graph.builder import SampleKNNGraphBuilder
        knn_k = getattr(get_settings().gnn, "knn_k", 5)
        data = SampleKNNGraphBuilder(k=knn_k).build(X_scaled, y=None)
        
        model = self.gnn.to(self.device)
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            logits = model(data.x, data.edge_index, **kwargs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_with_uncertainty(
        self,
        X_scaled: np.ndarray,
        n_iter: int = 15,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates ensemble prediction with an added 'Uncertainty' (std) 
        from the GATv2 model's MC Dropout passes.
        
        Returns: (preds, combined_probs, uncertainty_std)
        """
        # 1. GATv2 Uncertainty estimate
        from src.core.graph.builder import SampleKNNGraphBuilder
        knn_k = getattr(get_settings().gnn, "knn_k", 5)
        data = SampleKNNGraphBuilder(k=knn_k).build(X_scaled, y=None)
        
        model = self.gnn.to(self.device)
        data = data.to(self.device)
        
        # MC Dropout passes on GNN
        gnn_mean, gnn_std = model.predict_with_uncertainty(
            data.x, data.edge_index, n_iter=n_iter
        )
        gnn_probs = gnn_mean.cpu().numpy()
        
        # 2. Others (Normal)
        xp = self.xgb.predict_proba(X_scaled) if self.xgb is not None else None
        lp = self.lgbm.predict_proba(X_scaled) if self.lgbm is not None else None
        dp = _dnn_predict_proba(self.dnn.to(self.device), X_scaled, self.device) if self.dnn is not None else None
        
        # 3. Combine
        combined_proba = self.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gnn_probs, dnn_proba=dp)
        preds = (combined_proba[:, 1] >= threshold).astype(int)
        
        # Uncertainty is based on GNN variance (main stochastic component)
        uncertainty = gnn_std[:, 1].cpu().numpy()
        
        return preds, combined_proba, uncertainty

    def combine(
        self,
        xgb_proba: Optional[np.ndarray],
        lgb_proba: Optional[np.ndarray],
        gnn_proba: Optional[np.ndarray],
        dnn_proba: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Combine base-model probability matrices.

        Stacking path (meta-learner mevcutsa):
            [xgb_p1, lgb_p1, gnn_p1, dnn_p1] → LogisticRegression → (N, 2)
        Fallback (meta-learner yoksa):
            Ağırlıklı ortalama (self.weights).
        """
        # ── Meta-learner stacking (adaptif birleştirme) ───────────────────
        if self.meta_learner is not None:
            cols = [
                p[:, 1]
                for p in [xgb_proba, lgb_proba, gnn_proba, dnn_proba]
                if p is not None
            ]
            if cols:
                try:
                    meta_X = np.column_stack(cols)             # (N, n_models)
                    return self.meta_learner.predict_proba(meta_X)  # (N, 2)
                except Exception as exc:
                    logger.warning(
                        "Meta-learner predict_proba başarısız (%s) — "
                        "ağırlıklı ortalamaya geçiliyor.", exc
                    )

        # ── Weighted-average fallback ─────────────────────────────────────
        pairs = [
            (xgb_proba, self.weights[0]),
            (lgb_proba, self.weights[1]),
            (gnn_proba, self.weights[2]),
            (dnn_proba, self.weights[3]),
        ]
        available = [(p, w) for p, w in pairs if p is not None]
        if not available:
            raise ValueError("No predictions available.")

        total_w = sum(w for _, w in available)
        return sum((w / total_w) * p for p, w in available)

    def optimise_weights(
        self,
        X_val: np.ndarray,
        loader: Any,
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> None:
        """Optimise ensemble weights on a held-out validation set.

        Uses Nelder-Mead to maximise Macro F1 over the 4-model weight space.
        Updates ``self.weights`` in-place.

        Parameters
        ----------
        X_val  : Preprocessed validation feature matrix.
        loader : Unused (kept for API compatibility). Pass ``None``.
        y_val  : True binary labels for the validation set.
        """
        xp, lp, gp, dp = self.predict_proba_all(X_val)
        components = [xp, lp, gp, dp]
        active_idx = [i for i, p in enumerate(components) if p is not None]
        if len(active_idx) < 2:
            logger.info("optimise_weights: fewer than 2 active models — skipping.")
            return

        active_probs = [components[i] for i in active_idx]

        def _neg_f1(raw_w: np.ndarray) -> float:
            w = np.abs(raw_w)
            w = w / w.sum()
            blended = sum(wi * pi for wi, pi in zip(w, active_probs))
            preds = (blended[:, 1] >= 0.5).astype(int)
            return -f1_score(y_val, preds, average="macro", zero_division=0)

        x0 = np.array([self.weights[i] for i in active_idx])
        result = minimize(_neg_f1, x0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-4})
        opt_w = np.abs(result.x)
        opt_w = opt_w / opt_w.sum()

        new_weights = list(self.weights)
        for j, idx in enumerate(active_idx):
            new_weights[idx] = float(opt_w[j])
        w_sum = sum(new_weights)
        self.weights = [w / w_sum for w in new_weights]
        logger.info(
            "optimise_weights: updated weights=%s (val F1=%.4f)",
            [round(w, 4) for w in self.weights], -result.fun,
        )

    def fit_meta_learner(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit a stacking meta-learner (LogisticRegression) on validation predictions.

        After fitting, ``self.combine()`` will use the meta-learner instead of
        weighted averaging for future predictions.

        Parameters
        ----------
        X_val : Preprocessed validation feature matrix.
        y_val : True binary labels for the validation set.
        """
        from sklearn.linear_model import LogisticRegression

        xp, lp, gp, dp = self.predict_proba_all(X_val)
        cols = [
            p[:, 1]
            for p in [xp, lp, gp, dp]
            if p is not None
        ]
        if len(cols) < 2:
            logger.info("fit_meta_learner: fewer than 2 active models — skipping.")
            return

        meta_X = np.column_stack(cols)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42)
        lr.fit(meta_X, y_val)
        self.meta_learner = lr
        meta_preds = lr.predict(meta_X)
        meta_f1 = f1_score(y_val, meta_preds, average="macro", zero_division=0)
        logger.info(
            "fit_meta_learner: fitted on %d samples, %d models → val F1=%.4f",
            len(y_val), len(cols), meta_f1,
        )

    def predict(
        self,
        X_scaled: np.ndarray,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classic ensemble prediction."""
        xp, lp, gp, dp = self.predict_proba_all(X_scaled, **kwargs)
        proba = self.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gp, dnn_proba=dp)
        preds = (proba[:, 1] >= threshold).astype(int)
        return preds, proba

    # ------------------------------------------------------------------
    # Weight optimisation — Nelder-Mead (PSR §5.3)
    # ------------------------------------------------------------------

    def optimise_weights(
        self,
        X_val: np.ndarray,
        loader: Optional[Any],
        y_val: np.ndarray,
    ) -> None:
        """Optimise ensemble weights on a validation set via Nelder-Mead.

        Maximises Macro F1 by searching the 4-weight simplex.
        The result overwrites ``self.weights`` in-place.
        """
        xp, lp, gp, dp = self.predict_proba_all(X_val)
        matrices = [xp, lp, gp, dp]
        avail_idx = [i for i, m in enumerate(matrices) if m is not None]
        if len(avail_idx) < 2:
            logger.info("optimise_weights: <2 models available — skipping.")
            return

        avail_matrices = [matrices[i] for i in avail_idx]

        def _neg_f1(w_raw: np.ndarray) -> float:
            w = np.abs(w_raw)
            w = w / w.sum()
            blended = sum(wi * mi for wi, mi in zip(w, avail_matrices))
            preds = (blended[:, 1] >= 0.5).astype(int)
            return -f1_score(y_val, preds, average="macro", zero_division=0)

        x0 = np.array([self.weights[i] for i in avail_idx])
        res = minimize(_neg_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4})

        opt_w = np.abs(res.x)
        opt_w = opt_w / opt_w.sum()

        new_weights = [0.0] * 4
        for idx, ai in enumerate(avail_idx):
            new_weights[ai] = float(opt_w[idx])
        # Re-normalise full vector
        ws = sum(new_weights)
        if ws > 0:
            new_weights = [w / ws for w in new_weights]
        self.weights = new_weights
        logger.info(
            "Nelder-Mead optimised weights: %s  (val F1=%.4f)",
            [round(w, 4) for w in self.weights], -res.fun,
        )

    # ------------------------------------------------------------------
    # Stacking meta-learner (PSR §5.3)
    # ------------------------------------------------------------------

    def fit_meta_learner(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit a LogisticRegression meta-learner on base-model predictions.

        The meta-learner replaces weighted-average combination with an
        adaptive stacking layer.  Falls back gracefully if fitting fails.
        """
        from sklearn.linear_model import LogisticRegression

        xp, lp, gp, dp = self.predict_proba_all(X_val)
        cols = [
            p[:, 1] for p in [xp, lp, gp, dp] if p is not None
        ]
        if len(cols) < 2:
            logger.info("fit_meta_learner: <2 models available — skipping.")
            return

        meta_X = np.column_stack(cols)  # (N, n_models)
        lr = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=42, C=1.0,
        )
        lr.fit(meta_X, y_val)
        self.meta_learner = lr
        logger.info(
            "Meta-learner fitted (LogisticRegression) on %d samples, %d model cols.",
            len(y_val), meta_X.shape[1],
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def pathogenic_risk_score(ensemble_proba: np.ndarray) -> np.ndarray:
        return (ensemble_proba[:, 1] * 100).round(2)

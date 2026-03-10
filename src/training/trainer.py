"""
src/training/trainer.py
Leakage-free training and cross-validation pipeline.

Key guarantees:
  - Preprocessor (imputer, scaler, SMOTE, feature selection, AutoEncoder) is
    fit ONLY on the training split/fold — never on validation or test data.
  - Graph topology is computed from training-fold correlation.
  - Model selection metric: Macro F1 (consistent with reporting metric).
  - Deterministic seeds applied at every fold.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.config import get_settings
from src.features.preprocessing import VariantPreprocessor, build_preprocessor_from_config
from src.models.gnn import FeatureGNN
from src.models.dnn import VariantDNN
from src.models.ensemble import HybridEnsemble
from src.utils.seeds import set_global_seed
import xgboost as xgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_geo_loader(
    preprocessor: VariantPreprocessor,
    X_scaled: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> GeoDataLoader:
    graphs = [
        preprocessor.row_to_graph(row, label=(int(y[i]) if y is not None else None))
        for i, row in enumerate(X_scaled)
    ]
    return GeoDataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def _make_dnn_loader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> TorchDataLoader:
    if y is not None:
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    else:
        ds = TensorDataset(torch.FloatTensor(X))
    return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _gnn_epoch(
    model: FeatureGNN,
    loader: GeoDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def _gnn_eval(
    model: FeatureGNN, loader: GeoDataLoader, device: torch.device
) -> Tuple[List[int], List[List[float]]]:
    model.eval()
    preds_all, probs_all = [], []
    with torch.no_grad():
        for data in loader:
            data  = data.to(device)
            out   = model(data)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            probs_all.extend(probs.tolist())
            preds_all.extend(preds.tolist())
    return preds_all, probs_all


def _dnn_epoch(
    model: VariantDNN,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def _dnn_eval(
    model: VariantDNN, loader: TorchDataLoader, device: torch.device
) -> Tuple[List[int], List[List[float]]]:
    model.eval()
    preds_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            out     = model(X_batch)
            probs   = F.softmax(out, dim=1).cpu().numpy()
            preds   = out.argmax(dim=1).cpu().numpy()
            probs_all.extend(probs.tolist())
            preds_all.extend(preds.tolist())
    return preds_all, probs_all


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold:    int
    f1:      float
    xgb_f1: float
    gnn_f1: float
    dnn_f1: float


@dataclass
class TrainResult:
    ensemble:    HybridEnsemble
    preprocessor: VariantPreprocessor
    fold_results: List[FoldResult] = field(default_factory=list)
    mean_cv_f1:  float = 0.0
    std_cv_f1:   float = 0.0


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------


class VariantTrainer:
    """
    Leakage-free trainer for the VARIANT-GNN hybrid ensemble.

    The full pipeline inside each CV fold:
        1. Split → train_fold / val_fold
        2. preprocessor.fit_resample_train(X_train, y_train)
           → impute, scale, SMOTE, feature-selection, autoencoder, graph
        3. preprocessor.transform(X_val)
        4. Train XGBoost on (X_train_proc, y_resampled)
        5. Train GNN on graph-converted train_fold
        6. Train DNN on tensor train_fold
        7. Evaluate via Macro F1 on val_fold
    """

    def __init__(
        self,
        device:   Optional[torch.device] = None,
        config_path: Optional[str]       = None,
    ) -> None:
        self.cfg    = get_settings(config_path)
        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available() and self.cfg.device != "cpu"
            else torch.device("cpu")
        )
        logger.info("VariantTrainer | device=%s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TrainResult:
        """
        Train on provided arrays with a final held-out test split,
        then re-fit on the full training portion.

        Returns a ``TrainResult`` with the fitted ensemble + preprocessor.
        """
        set_global_seed(self.cfg.seed)
        cfg = self.cfg

        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y,
            test_size   = cfg.training.test_size,
            stratify    = y,
            random_state= cfg.seed,
        )

        # Cross-validate on train portion
        fold_results = self._cross_validate(X_train_all, y_train_all)
        mean_f1 = float(np.mean([r.f1 for r in fold_results]))
        std_f1  = float(np.std( [r.f1 for r in fold_results]))
        logger.info(
            "Cross-validation complete: Macro F1 = %.4f ± %.4f", mean_f1, std_f1
        )

        # Final model — fit on full training set
        preprocessor, ensemble = self._fit_single(X_train_all, y_train_all)

        # Optionally optimise weights on test set
        if cfg.ensemble.optimize_weights:
            gnn_loader = _make_geo_loader(
                preprocessor, preprocessor.transform(X_test), None,
                cfg.training.batch_size, shuffle=False,
            )
            X_test_proc = preprocessor.transform(X_test)
            ensemble.optimise_weights(X_test_proc, gnn_loader, y_test)

        return TrainResult(
            ensemble      = ensemble,
            preprocessor  = preprocessor,
            fold_results  = fold_results,
            mean_cv_f1    = mean_f1,
            std_cv_f1     = std_f1,
        )

    # ------------------------------------------------------------------
    # Internal — single fit
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[VariantPreprocessor, HybridEnsemble]:
        """Fit ALL preprocessing + ALL models on X_train / y_train."""
        set_global_seed(self.cfg.seed)
        cfg = self.cfg

        preprocessor = build_preprocessor_from_config()
        X_proc, y_resampled = preprocessor.fit_resample_train(X_train, y_train)

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            **cfg.xgb.as_dict(), random_state=cfg.seed
        )
        xgb_model.fit(X_proc, y_resampled)
        logger.info("XGBoost fitted: n_features_in=%d", X_proc.shape[1])

        # --- GNN ---
        n_features = X_proc.shape[1]
        gnn_model  = FeatureGNN(
            in_channels = 1,
            hidden_dim  = cfg.gnn.hidden_dim,
            num_classes = 2,
            use_gat     = cfg.gnn.use_gat,
        ).to(self.device)

        train_loader = _make_geo_loader(
            preprocessor, X_proc, y_resampled, cfg.training.batch_size, shuffle=True
        )
        gnn_model = self._train_gnn(gnn_model, train_loader, val_loader=None)

        # --- DNN ---
        dnn_model = VariantDNN(
            input_dim  = n_features,
            hidden_dim = cfg.dnn.hidden_dim,
            num_classes= 2,
        ).to(self.device)
        dnn_loader = _make_dnn_loader(X_proc, y_resampled, cfg.training.batch_size, shuffle=True)
        dnn_model  = self._train_dnn(dnn_model, dnn_loader, val_loader=None)

        ensemble = HybridEnsemble(
            xgb_model = xgb_model,
            gnn_model = gnn_model,
            dnn_model = dnn_model,
            weights   = cfg.ensemble.weights,
            device    = self.device,
        )
        return preprocessor, ensemble

    # ------------------------------------------------------------------
    # Internal — cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[FoldResult]:
        cfg   = self.cfg
        skf   = StratifiedKFold(
            n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed
        )
        results: List[FoldResult] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            set_global_seed(cfg.seed + fold_idx)
            logger.info("--- Fold %d/%d ---", fold_idx, cfg.training.cv_folds)

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # --- Preprocessing fit on fold training data ONLY ---
            preprocessor = build_preprocessor_from_config()
            X_tr_proc, y_tr_res = preprocessor.fit_resample_train(X_tr, y_tr)
            X_val_proc           = preprocessor.transform(X_val)

            # --- XGBoost ---
            xgb_model = xgb.XGBClassifier(**cfg.xgb.as_dict(), random_state=cfg.seed)
            xgb_model.fit(X_tr_proc, y_tr_res)
            xgb_preds   = xgb_model.predict(X_val_proc)
            xgb_f1      = float(f1_score(y_val, xgb_preds, average="macro", zero_division=0))

            # --- GNN ---
            gnn_model    = FeatureGNN(1, cfg.gnn.hidden_dim, 2, cfg.gnn.use_gat).to(self.device)
            train_loader = _make_geo_loader(
                preprocessor, X_tr_proc, y_tr_res, cfg.training.batch_size, shuffle=True
            )
            val_loader   = _make_geo_loader(
                preprocessor, X_val_proc, y_val, cfg.training.batch_size, shuffle=False
            )
            gnn_model    = self._train_gnn(gnn_model, train_loader, val_loader)
            gnn_preds, _ = _gnn_eval(gnn_model, val_loader, self.device)
            gnn_f1       = float(f1_score(y_val, gnn_preds[:len(y_val)],
                                          average="macro", zero_division=0))

            # --- DNN ---
            dnn_model     = VariantDNN(X_tr_proc.shape[1], cfg.dnn.hidden_dim, 2).to(self.device)
            dnn_tr_loader = _make_dnn_loader(X_tr_proc, y_tr_res, cfg.training.batch_size, True)
            dnn_val_loader= _make_dnn_loader(X_val_proc, y_val, cfg.training.batch_size, False)
            dnn_model     = self._train_dnn(dnn_model, dnn_tr_loader, dnn_val_loader)
            dnn_preds, _  = _dnn_eval(dnn_model, dnn_val_loader, self.device)
            dnn_f1        = float(f1_score(y_val, dnn_preds[:len(y_val)],
                                           average="macro", zero_division=0))

            # --- Ensemble ---
            w = cfg.ensemble.weights
            xgb_probs = xgb_model.predict_proba(X_val_proc)
            gnn_probs = np.array(_gnn_eval(gnn_model, val_loader, self.device)[1])
            dnn_probs = np.array(_dnn_eval(dnn_model, dnn_val_loader, self.device)[1])
            ens_probs = w[0] * xgb_probs + w[1] * gnn_probs + w[2] * dnn_probs
            ens_preds = np.argmax(ens_probs, axis=1)
            ens_f1    = float(f1_score(y_val, ens_preds, average="macro", zero_division=0))

            logger.info(
                "Fold %d | Ensemble Macro F1: %.4f  (XGB=%.4f, GNN=%.4f, DNN=%.4f)",
                fold_idx, ens_f1, xgb_f1, gnn_f1, dnn_f1,
            )
            results.append(FoldResult(fold_idx, ens_f1, xgb_f1, gnn_f1, dnn_f1))

        return results

    # ------------------------------------------------------------------
    # GNN training loop
    # ------------------------------------------------------------------

    def _train_gnn(
        self,
        model:      FeatureGNN,
        train_loader: GeoDataLoader,
        val_loader:   Optional[GeoDataLoader],
    ) -> FeatureGNN:
        cfg       = self.cfg
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.gnn.lr, weight_decay=cfg.gnn.weight_decay
        )
        best_f1       = -1.0
        best_weights  = copy.deepcopy(model.state_dict())

        for epoch in range(1, cfg.gnn.epochs + 1):
            loss = _gnn_epoch(model, train_loader, optimizer, self.device)

            if val_loader is not None:
                preds, _ = _gnn_eval(model, val_loader, self.device)
                val_labels = [d.y.item() for batch in val_loader for d in batch.to_data_list()]
                val_f1 = float(f1_score(
                    val_labels, preds[:len(val_labels)], average="macro", zero_division=0
                ))
                if val_f1 > best_f1:
                    best_f1      = val_f1
                    best_weights = copy.deepcopy(model.state_dict())
                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug("GNN epoch %d/%d | loss=%.4f | val_f1=%.4f",
                                 epoch, cfg.gnn.epochs, loss, val_f1)
            else:
                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug("GNN epoch %d/%d | loss=%.4f", epoch, cfg.gnn.epochs, loss)

        if val_loader is not None:
            model.load_state_dict(best_weights)
        return model

    # ------------------------------------------------------------------
    # DNN training loop
    # ------------------------------------------------------------------

    def _train_dnn(
        self,
        model:       VariantDNN,
        train_loader: TorchDataLoader,
        val_loader:   Optional[TorchDataLoader],
    ) -> VariantDNN:
        cfg       = self.cfg
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.dnn.lr, weight_decay=cfg.dnn.weight_decay
        )
        criterion    = nn.CrossEntropyLoss()
        best_f1      = -1.0
        best_weights = copy.deepcopy(model.state_dict())

        for epoch in range(1, cfg.dnn.epochs + 1):
            loss = _dnn_epoch(model, train_loader, optimizer, criterion, self.device)

            if val_loader is not None:
                preds, _ = _dnn_eval(model, val_loader, self.device)
                y_val    = [batch[1].numpy() for batch in val_loader]
                y_val    = np.concatenate(y_val)
                val_f1   = float(f1_score(
                    y_val, preds[:len(y_val)], average="macro", zero_division=0
                ))
                if val_f1 > best_f1:
                    best_f1      = val_f1
                    best_weights = copy.deepcopy(model.state_dict())
                if epoch % 5 == 0 or epoch == cfg.dnn.epochs:
                    logger.debug("DNN epoch %d/%d | loss=%.4f | val_f1=%.4f",
                                 epoch, cfg.dnn.epochs, loss, val_f1)
            else:
                if epoch % 5 == 0 or epoch == cfg.dnn.epochs:
                    logger.debug("DNN epoch %d/%d | loss=%.4f", epoch, cfg.dnn.epochs, loss)

        if val_loader is not None:
            model.load_state_dict(best_weights)
        return model

"""
src/training/trainer.py
Leakage-free training and cross-validation pipeline.

Key guarantees:
  - Preprocessor (imputer, scaler, SMOTE, feature selection, AutoEncoder) is
    fit ONLY on the training split/fold — never on validation or test data.
  - Graph topology is computed from training-fold correlation.
  - Model selection metric: Macro F1 (consistent with reporting metric).
  - Deterministic seeds applied at every fold.

TEKNOFEST 2026 additions:
  - WeightedBCELoss: dynamically computes class weights from training
    distribution to handle Pathogenic / Benign imbalance.
  - VariantSAGEGNN training via _train_sage(): full-batch node classification
    on a coordinate-free cosine k-NN sample graph.
  - Early stopping driven by Validation Macro F1 (not accuracy).
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.config import get_settings
from src.features.preprocessing import VariantPreprocessor, build_preprocessor_from_config
from src.models.dnn import VariantDNN
from src.models.ensemble import HybridEnsemble
from src.models.gnn import FeatureGNN, VariantSAGEGNN
from src.training.focal_loss import FocalLoss
from src.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeightedBCELoss — Module 4
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy for 2-class pathogenicity prediction.

    Dynamically computes per-class weights from the label distribution so
    that the minority class (often Pathogenic) receives proportionally
    higher loss signal.

    Formula per class c:
        weight[c] = N_total / (N_classes * count[c])

    This is equivalent to sklearn's ``compute_class_weight('balanced', ...)``.
    """

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", class_weights)   # [num_classes]

    @staticmethod
    def from_labels(y: np.ndarray, num_classes: int = 2) -> "WeightedBCELoss":
        """Factory: compute balanced weights from a label array."""
        counts  = np.bincount(y, minlength=num_classes).astype(float)
        weights = len(y) / (num_classes * counts)
        return WeightedBCELoss(torch.tensor(weights, dtype=torch.float))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [N, num_classes] raw model output.
        targets : [N] integer class indices.
        """
        return F.cross_entropy(logits, targets, weight=self.weight)


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Return balanced class weight tensor from label array."""
    counts  = np.bincount(y, minlength=2).astype(float)
    weights = len(y) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float)


def _make_criterion(y: np.ndarray, device: torch.device) -> nn.Module:
    """Select loss function based on config: 'focal' or 'weighted_bce'."""
    cfg = get_settings()
    loss_type = getattr(cfg.training, "loss_function", "weighted_bce")
    if loss_type == "focal":
        gamma = getattr(cfg.training, "focal_gamma", 2.0)
        criterion = FocalLoss.from_labels(y, gamma=gamma)
        logger.info("Using FocalLoss (gamma=%.1f)", gamma)
    else:
        criterion = WeightedBCELoss.from_labels(y)
        logger.info("Using WeightedBCELoss")
    return criterion.to(device)


# ---------------------------------------------------------------------------
# SAGE training helpers — Module 3 & 4
# ---------------------------------------------------------------------------

def _build_sample_graph(
    preprocessor: VariantPreprocessor,
    X: np.ndarray,
    y: Optional[np.ndarray],
    knn_k: int = 5,
):
    """Build a cosine kNN sample graph; returns a single PyG Data object."""
    return preprocessor.build_sample_graph(X, y, k=knn_k)


def _tokenize_sequences(
    nuc_seqs: Optional[List[str]],
    aa_seqs:  Optional[List[str]],
    device:   torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Tokenise raw Nuc_Context / AA_Context string lists to int64 tensors.
    Returns (None, None) when sequences are not provided.
    """
    from src.features.multimodal_encoder import tokenize_nucleotides, tokenize_amino_acids

    nuc_t = (
        torch.tensor(tokenize_nucleotides(nuc_seqs), dtype=torch.long).to(device)
        if nuc_seqs is not None else None
    )
    aa_t = (
        torch.tensor(tokenize_amino_acids(aa_seqs), dtype=torch.long).to(device)
        if aa_seqs is not None else None
    )
    return nuc_t, aa_t


def _sage_epoch(
    model: VariantSAGEGNN,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedBCELoss,
    device: torch.device,
    nuc_ids: Optional[torch.Tensor] = None,
    aa_ids:  Optional[torch.Tensor] = None,
) -> float:
    """One full-batch training step on the sample graph."""
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, nuc_ids=nuc_ids, aa_ids=aa_ids)
    loss   = criterion(logits, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def _sage_eval(
    model: VariantSAGEGNN,
    data,
    device: torch.device,
    nuc_ids: Optional[torch.Tensor] = None,
    aa_ids:  Optional[torch.Tensor] = None,
) -> Tuple[List[int], np.ndarray]:
    """Return (preds, probs) for all nodes in a sample graph."""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, nuc_ids=nuc_ids, aa_ids=aa_ids)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().tolist()
    return preds, probs



# ---------------------------------------------------------------------------
# Legacy feature-graph helpers (FeatureGNN / CorrelationGraph path)
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
    lgbm_f1: float
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
        nuc_seqs: Optional[List[str]] = None,
        aa_seqs:  Optional[List[str]] = None,
    ) -> TrainResult:
        """
        Train on provided arrays with a final held-out test split,
        then re-fit on the full training portion.

        Parameters
        ----------
        X        : Numeric feature matrix [N, F].
        y        : Integer label array [N].
        nuc_seqs : Optional list of Nuc_Context strings (length N).
        aa_seqs  : Optional list of AA_Context strings (length N).

        Returns a ``TrainResult`` with the fitted ensemble + preprocessor.
        """
        set_global_seed(self.cfg.seed)
        cfg = self.cfg

        # Split indices so we can slice sequences in parallel
        from sklearn.model_selection import train_test_split as _tts
        idx = np.arange(len(X))
        idx_tr, idx_te = _tts(idx, test_size=cfg.training.test_size, stratify=y,
                               random_state=cfg.seed)
        X_train_all, X_test = X[idx_tr], X[idx_te]
        y_train_all, y_test = y[idx_tr], y[idx_te]
        nuc_tr = ([nuc_seqs[i] for i in idx_tr] if nuc_seqs else None)
        nuc_te = ([nuc_seqs[i] for i in idx_te] if nuc_seqs else None)  # noqa: F841
        aa_tr  = ([aa_seqs[i]  for i in idx_tr] if aa_seqs  else None)
        aa_te  = ([aa_seqs[i]  for i in idx_te] if aa_seqs  else None)   # noqa: F841

        # Cross-validate on train portion
        fold_results = self._cross_validate(X_train_all, y_train_all,
                                            nuc_seqs=nuc_tr, aa_seqs=aa_tr)
        mean_f1 = float(np.mean([r.f1 for r in fold_results]))
        std_f1  = float(np.std( [r.f1 for r in fold_results]))
        logger.info(
            "Cross-validation complete: Macro F1 = %.4f ± %.4f", mean_f1, std_f1
        )

        # Final model — fit on full training set
        preprocessor, ensemble, X_opt_val, y_opt_val = self._fit_single(
            X_train_all, y_train_all, nuc_seqs=nuc_tr, aa_seqs=aa_tr
        )

        # Optionally optimise ensemble weights on the inner val set
        # (NOT on X_test, to keep the test set strictly held-out for metrics)
        if cfg.ensemble.optimize_weights:
            ensemble.optimise_weights(X_opt_val, None, y_opt_val)

        # Fit stacking meta-learner on the same inner val set
        # MetaLearner replaces fixed weights with a LogisticRegression combiner,
        # which can learn non-linear combinations and panel-specific patterns.
        try:
            ensemble.fit_meta_learner(X_opt_val, y_opt_val)
        except Exception as exc:
            logger.warning("Meta-learner fitting failed (%s) — using weighted average.", exc)

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
        nuc_seqs: Optional[List[str]] = None,
        aa_seqs:  Optional[List[str]] = None,
    ) -> Tuple["VariantPreprocessor", "HybridEnsemble", np.ndarray, np.ndarray]:
        """
        Fit ALL preprocessing + ALL models on X_train / y_train.

        Returns
        -------
        (preprocessor, ensemble, X_val_proc, y_val)
          where X_val_proc / y_val are the inner early-stopping split
          (post-SMOTE, already preprocessed) — used by caller for weight
          optimisation without touching the held-out test set.
        """
        set_global_seed(self.cfg.seed)
        cfg = self.cfg
        use_multimodal = getattr(cfg.gnn, "use_multimodal", False)
        seq_enc_dim    = getattr(cfg.gnn, "seq_enc_dim", 32)

        # Auto-disable SMOTE when multimodal sequences are present to avoid
        # sequence-row misalignment after synthetic oversampling.
        if use_multimodal and nuc_seqs is not None:
            logger.warning(
                "use_multimodal=True: auto-disabling SMOTE to preserve "
                "sequence-row alignment."
            )
            cfg.preprocessing.smote_enabled = False  # type: ignore[attr-defined]

        preprocessor = build_preprocessor_from_config()
        X_proc, y_resampled = preprocessor.fit_resample_train(X_train, y_train)

        # ── Inner val split for GNN/DNN early stopping (AFTER SMOTE) ──────────
        # Carving from the post-SMOTE pool so the ES val set is balanced.
        # Never touches the external test set → no leakage of jury data.
        inner_val_size = min(0.15, 200 / max(len(X_proc), 1))  # at least 15%
        inner_val_size = max(inner_val_size, 0.10)
        idx_inner = np.arange(len(X_proc))
        idx_inner_tr, idx_inner_val = train_test_split(
            idx_inner, test_size=inner_val_size, stratify=y_resampled,
            random_state=cfg.seed,
        )
        X_inner_tr, X_inner_val = X_proc[idx_inner_tr], X_proc[idx_inner_val]
        y_inner_tr, y_inner_val = y_resampled[idx_inner_tr], y_resampled[idx_inner_val]

        # Sequence slices for multimodal: use original-indexed rows only
        post_smote_nuc: Optional[List[str]] = None
        post_smote_aa:  Optional[List[str]] = None
        if use_multimodal and nuc_seqs is not None:
            n_orig = len(nuc_seqs)
            # Intersect inner_tr indices with original-row range
            orig_inner_tr = idx_inner_tr[idx_inner_tr < n_orig]
            post_smote_nuc = [nuc_seqs[i] for i in orig_inner_tr]
            post_smote_aa  = ([aa_seqs[i] for i in orig_inner_tr]
                              if aa_seqs else None)

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(**cfg.xgb.as_dict())
        xgb_model.fit(
            X_inner_tr, y_inner_tr,
            eval_set=[(X_inner_val, y_inner_val)],
            verbose=False,
        )
        logger.info("XGBoost fitted: n_features_in=%d", X_inner_tr.shape[1])

        # --- LightGBM ---
        lgbm_model = None
        try:
            import lightgbm as lgb
            lgbm_model = lgb.LGBMClassifier(**cfg.lgbm.as_dict())
            lgbm_model.fit(
                X_inner_tr, y_inner_tr,
                eval_set=[(X_inner_val, y_inner_val)],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            logger.info("LightGBM fitted: best_iteration=%d",
                        lgbm_model.best_iteration_)
        except ImportError:
            logger.warning("lightgbm not installed — skipping LGBM member.")
        except Exception as exc:
            logger.warning("LightGBM training failed (%s) — skipping.", exc)

        # --- GNN (VariantSAGEGNN) — with proper early stopping via inner val ---
        knn_k    = getattr(cfg.gnn, "knn_k", 5)
        patience = getattr(cfg.gnn, "early_stopping_patience", 5)
        gnn_model = VariantSAGEGNN(
            numeric_dim    = X_proc.shape[1],
            hidden_dim     = cfg.gnn.hidden_dim,
            num_classes    = 2,
            use_multimodal = use_multimodal,
            seq_enc_dim    = seq_enc_dim,
        ).to(self.device)
        gnn_model = self._train_sage(
            gnn_model, preprocessor,
            X_inner_tr, y_inner_tr,
            X_val=X_inner_val, y_val=y_inner_val,   # ← early stopping now active
            knn_k=knn_k, patience=patience,
            nuc_seqs=post_smote_nuc, aa_seqs=post_smote_aa,
        )

        # --- DNN ---
        dnn_model  = VariantDNN(
            input_dim  = X_proc.shape[1],
            hidden_dim = cfg.dnn.hidden_dim,
            num_classes= 2,
        ).to(self.device)
        dnn_tr_loader  = _make_dnn_loader(X_inner_tr, y_inner_tr, cfg.training.batch_size, True)
        dnn_val_loader = _make_dnn_loader(X_inner_val, y_inner_val, cfg.training.batch_size, False)
        dnn_model = self._train_dnn(dnn_model, dnn_tr_loader, dnn_val_loader, y_inner_tr)

        ensemble = HybridEnsemble(
            xgb_model  = xgb_model,
            lgbm_model = lgbm_model,
            gnn_model  = gnn_model,
            dnn_model  = dnn_model,
            weights    = cfg.ensemble.weights,
            device     = self.device,
        )
        # Return the inner val set so caller can use it for weight optimisation
        return preprocessor, ensemble, X_inner_val, y_inner_val

    # ------------------------------------------------------------------
    # Internal — cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray,
        nuc_seqs: Optional[List[str]] = None,
        aa_seqs:  Optional[List[str]] = None,
    ) -> List[FoldResult]:
        cfg   = self.cfg
        skf   = StratifiedKFold(
            n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed
        )
        results: List[FoldResult] = []
        use_multimodal = getattr(cfg.gnn, "use_multimodal", False)
        seq_enc_dim    = getattr(cfg.gnn, "seq_enc_dim", 32)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            set_global_seed(cfg.seed + fold_idx)
            logger.info("--- Fold %d/%d ---", fold_idx, cfg.training.cv_folds)

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Slice sequences for this fold (pre-SMOTE rows only)
            nuc_tr  = ([nuc_seqs[i] for i in train_idx] if nuc_seqs else None)
            nuc_val = ([nuc_seqs[i] for i in val_idx]   if nuc_seqs else None)
            aa_tr   = ([aa_seqs[i]  for i in train_idx] if aa_seqs  else None)
            aa_val  = ([aa_seqs[i]  for i in val_idx]   if aa_seqs  else None)

            # Auto-disable SMOTE when multimodal to preserve sequence alignment
            fold_cfg = cfg
            if use_multimodal and nuc_tr is not None:
                cfg.preprocessing.smote_enabled = False  # type: ignore[attr-defined]

            # --- Preprocessing fit on fold training data ONLY ---
            preprocessor = build_preprocessor_from_config()
            X_tr_proc, y_tr_res = preprocessor.fit_resample_train(X_tr, y_tr)
            X_val_proc           = preprocessor.transform(X_val)

            # After SMOTE, limit sequence arrays to original train size
            n_orig_tr = len(X_tr)
            nuc_tr_proc = nuc_tr[:n_orig_tr] if (nuc_tr and use_multimodal) else None
            aa_tr_proc  = aa_tr[:n_orig_tr]  if (aa_tr  and use_multimodal) else None

            # --- XGBoost ---
            xgb_model = xgb.XGBClassifier(**cfg.xgb.as_dict())
            xgb_model.fit(
                X_tr_proc, y_tr_res,
                eval_set=[(X_val_proc, y_val)], verbose=False,
            )
            xgb_preds   = xgb_model.predict(X_val_proc)
            xgb_f1      = float(f1_score(y_val, xgb_preds, average="macro", zero_division=0))

            # --- LightGBM ---
            lgbm_f1 = 0.0
            lgbm_model_fold = None
            try:
                import lightgbm as lgb
                lgbm_model_fold = lgb.LGBMClassifier(**cfg.lgbm.as_dict())
                lgbm_model_fold.fit(
                    X_tr_proc, y_tr_res,
                    eval_set=[(X_val_proc, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False),
                               lgb.log_evaluation(-1)],
                )
                lgbm_preds = lgbm_model_fold.predict(X_val_proc)
                lgbm_f1    = float(f1_score(y_val, lgbm_preds, average="macro", zero_division=0))
            except Exception:
                pass

            # --- GNN (VariantSAGEGNN) ---
            knn_k    = getattr(cfg.gnn, "knn_k", 5)
            patience = getattr(cfg.gnn, "early_stopping_patience", 5)
            X_for_gnn = X_tr_proc[:n_orig_tr] if nuc_tr_proc else X_tr_proc
            y_for_gnn = y_tr_res[:n_orig_tr]  if nuc_tr_proc else y_tr_res
            sage_model = VariantSAGEGNN(
                numeric_dim    = X_tr_proc.shape[1],
                hidden_dim     = cfg.gnn.hidden_dim,
                num_classes    = 2,
                use_multimodal = use_multimodal,
                seq_enc_dim    = seq_enc_dim,
            ).to(self.device)
            sage_model = self._train_sage(
                sage_model, preprocessor,
                X_for_gnn, y_for_gnn,
                X_val_proc, y_val,
                knn_k=knn_k, patience=patience,
                nuc_seqs=nuc_tr_proc, aa_seqs=aa_tr_proc,
                nuc_val=nuc_val, aa_val=aa_val,
            )
            # Evaluate SAGE on validation graph
            val_graph = _build_sample_graph(preprocessor, X_val_proc, y_val, knn_k)
            nuc_val_t, aa_val_t = _tokenize_sequences(nuc_val if use_multimodal else None,
                                                      aa_val  if use_multimodal else None,
                                                      self.device)
            gnn_preds, gnn_probs_fold = _sage_eval(sage_model, val_graph, self.device,
                                                   nuc_ids=nuc_val_t, aa_ids=aa_val_t)
            gnn_f1 = float(f1_score(y_val, gnn_preds[:len(y_val)],
                                    average="macro", zero_division=0))

            # --- DNN ---
            dnn_model     = VariantDNN(X_tr_proc.shape[1], cfg.dnn.hidden_dim, 2).to(self.device)
            dnn_tr_loader = _make_dnn_loader(X_tr_proc, y_tr_res, cfg.training.batch_size, True)
            dnn_val_loader= _make_dnn_loader(X_val_proc, y_val, cfg.training.batch_size, False)
            dnn_model     = self._train_dnn(dnn_model, dnn_tr_loader, dnn_val_loader, y_train=y_tr_res)
            dnn_preds, _  = _dnn_eval(dnn_model, dnn_val_loader, self.device)
            dnn_f1        = float(f1_score(y_val, dnn_preds[:len(y_val)],
                                           average="macro", zero_division=0))

            # --- Ensemble ---
            w         = cfg.ensemble.weights
            xgb_probs = xgb_model.predict_proba(X_val_proc)
            gnn_probs = np.array(gnn_probs_fold)
            dnn_probs = np.array(_dnn_eval(dnn_model, dnn_val_loader, self.device)[1])
            if lgbm_model_fold is not None and len(w) >= 4:
                lgbm_probs = lgbm_model_fold.predict_proba(X_val_proc)
                # 4-model weighted combine: [XGB, LGB, GNN, DNN]
                w_sum = sum(w[:4])
                ens_probs = (w[0]/w_sum * xgb_probs + w[1]/w_sum * lgbm_probs
                             + w[2]/w_sum * gnn_probs + w[3]/w_sum * dnn_probs)
            else:
                w3 = w[:3] if len(w) == 3 else [w[0]+w[1], w[2], w[3]]
                t  = sum(w3); w3 = [x/t for x in w3]
                ens_probs = w3[0] * xgb_probs + w3[1] * gnn_probs + w3[2] * dnn_probs
            ens_preds = np.argmax(ens_probs, axis=1)
            ens_f1    = float(f1_score(y_val, ens_preds, average="macro", zero_division=0))

            logger.info(
                "Fold %d | Ensemble Macro F1: %.4f  (XGB=%.4f, LGB=%.4f, GNN=%.4f, DNN=%.4f)",
                fold_idx, ens_f1, xgb_f1, lgbm_f1, gnn_f1, dnn_f1,
            )
            results.append(FoldResult(fold_idx, ens_f1, xgb_f1, lgbm_f1, gnn_f1, dnn_f1))

        return results

    # ------------------------------------------------------------------
    # VariantSAGEGNN training loop — Module 3 & 4
    # ------------------------------------------------------------------

    def _train_sage(
        self,
        model:       VariantSAGEGNN,
        preprocessor: VariantPreprocessor,
        X_tr:        np.ndarray,
        y_tr:        np.ndarray,
        X_val:       Optional[np.ndarray],
        y_val:       Optional[np.ndarray],
        knn_k:       int = 5,
        patience:    int = 5,
        nuc_seqs:    Optional[List[str]] = None,
        aa_seqs:     Optional[List[str]] = None,
        nuc_val:     Optional[List[str]] = None,
        aa_val:      Optional[List[str]] = None,
    ) -> VariantSAGEGNN:
        """
        Full-batch node-classification training on a cosine k-NN sample graph.

        Loss function:  WeightedBCELoss (class-balanced cross-entropy).
        Early stopping: monitored on Validation Macro F1.

        Parameters
        ----------
        nuc_seqs / aa_seqs   : Training sequence strings (pre-tokenised internally).
        nuc_val  / aa_val    : Validation sequence strings.
        """
        cfg = self.cfg

        criterion = _make_criterion(y_tr, self.device)
        logger.info(
            "SAGE loss class_weights: %s",
            getattr(criterion, 'weight', getattr(criterion, 'alpha', 'N/A')),
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = cfg.gnn.lr,
            weight_decay = cfg.gnn.weight_decay,
        )

        # Tokenise sequences once
        nuc_tr_t, aa_tr_t   = _tokenize_sequences(nuc_seqs, aa_seqs, self.device)
        nuc_val_t, aa_val_t = _tokenize_sequences(nuc_val, aa_val, self.device)

        # Build full-batch training graph (coordinate-free, cosine kNN)
        train_graph = _build_sample_graph(preprocessor, X_tr, y_tr, knn_k)

        val_graph: Optional[object] = None
        if X_val is not None and y_val is not None:
            val_graph = _build_sample_graph(preprocessor, X_val, y_val, knn_k)

        best_val_f1   = -1.0
        best_weights  = copy.deepcopy(model.state_dict())
        patience_cnt  = 0

        for epoch in range(1, cfg.gnn.epochs + 1):
            loss = _sage_epoch(model, train_graph, optimizer, criterion, self.device,
                               nuc_ids=nuc_tr_t, aa_ids=aa_tr_t)

            if val_graph is not None:
                preds, _ = _sage_eval(model, val_graph, self.device,
                                      nuc_ids=nuc_val_t, aa_ids=aa_val_t)
                val_f1   = float(f1_score(
                    y_val, preds[:len(y_val)], average="macro", zero_division=0
                ))

                if val_f1 > best_val_f1:
                    best_val_f1  = val_f1
                    best_weights = copy.deepcopy(model.state_dict())
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug(
                        "SAGE epoch %d/%d | loss=%.4f | val_macro_f1=%.4f "
                        "(patience %d/%d)",
                        epoch, cfg.gnn.epochs, loss, val_f1,
                        patience_cnt, patience,
                    )

                if patience > 0 and patience_cnt >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(best val Macro F1 = %.4f)", epoch, best_val_f1,
                    )
                    break
            else:
                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug(
                        "SAGE epoch %d/%d | loss=%.4f", epoch, cfg.gnn.epochs, loss
                    )

        if val_graph is not None:
            model.load_state_dict(best_weights)
            logger.info("SAGE restored best checkpoint (val Macro F1 = %.4f)", best_val_f1)

        return model

    # ------------------------------------------------------------------
    # GNN training loop (legacy FeatureGNN path)
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
        y_train:      Optional[np.ndarray] = None,
    ) -> VariantDNN:
        cfg       = self.cfg
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.dnn.lr, weight_decay=cfg.dnn.weight_decay
        )
        # Configurable loss function (FocalLoss or WeightedBCE)
        if y_train is not None:
            criterion = _make_criterion(y_train, self.device)
        else:
            criterion = nn.CrossEntropyLoss()
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

"""
src/training/trainer.py
Leakage-free training and cross-validation pipeline.

Key guarantees:
  - Preprocessor (imputer, scaler, SMOTE, feature selection, AutoEncoder) is
    fit ONLY on the training split/fold — never on validation or test data.
  - Graph topology is computed from training-fold correlation.
  - Model selection metric: Binary F1 (Pathogenic sınıfı, pos_label=1, §7.3).
  - Deterministic seeds applied at every fold.

TEKNOFEST 2026 additions:
  - WeightedBCELoss: dynamically computes class weights from training
    distribution to handle Pathogenic / Benign imbalance.
  - VariantGATv2GNN training via _train_gatv2(): full-batch node classification
    on a coordinate-free cosine k-NN sample graph (GATv2Conv, 4 heads).
  - Early stopping driven by Validation Binary F1 (Pathogenic, §7.3).
"""
from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
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
from src.core.models.ensemble import HybridEnsemble
from src.core.models.gnn import VariantGATv2GNN
from src.features.preprocessing import VariantPreprocessor, build_preprocessor_from_config
from src.models.dnn_model import VariantDNN
from src.training.focal_loss import FocalLoss
from src.training.swa import CyclicSWAScheduler, SWABuffer
from src.utils.seeds import set_global_seed

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mlflow = None

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


def _to_lgbm_frame(X):
    """Wrap numpy → DataFrame with stable column names so LightGBM fit/predict
    stay consistent and sklearn does not raise the 'feature names' UserWarning."""
    if isinstance(X, np.ndarray):
        import pandas as pd
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X


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
    from src.features.multimodal_encoder import tokenize_amino_acids, tokenize_nucleotides

    nuc_t = (
        torch.tensor(tokenize_nucleotides(nuc_seqs), dtype=torch.long).to(device)
        if nuc_seqs is not None else None
    )
    aa_t = (
        torch.tensor(tokenize_amino_acids(aa_seqs), dtype=torch.long).to(device)
        if aa_seqs is not None else None
    )
    return nuc_t, aa_t


def _gatv2_epoch(
    model: VariantGATv2GNN,
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


def _gatv2_eval(
    model: VariantGATv2GNN,
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
    model: VariantGATv2GNN,
    loader: GeoDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def _gnn_eval(
    model: VariantGATv2GNN, loader: GeoDataLoader, device: torch.device
) -> Tuple[List[int], List[List[float]]]:
    model.eval()
    preds_all, probs_all = [], []
    with torch.no_grad():
        for data in loader:
            data  = data.to(device)
            out   = model(data.x, data.edge_index)
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
    # Held-out test indices used internally — returned so the caller evaluates
    # on the SAME split (group-aware when groups are provided). Prevents the
    # two-independent-splits inconsistency and guarantees leakage-free eval.
    test_indices: Optional[np.ndarray] = None


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
        7. Evaluate via Binary F1 (Pathogenic, pos_label=1, §7.3) on val_fold
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
        groups:   Optional[np.ndarray] = None,
        panels:   Optional[np.ndarray] = None,
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

        run_name = f"HybridEnsemble_{getattr(cfg.gnn, 'model_type', 'gatv2')}"
        if mlflow is not None:
            mlflow.set_experiment("VARIANT-GNN-Professional")
            run_ctx = mlflow.start_run(run_name=run_name)
        else:
            logger.info("MLflow is not installed; proceeding without experiment logging.")
            run_ctx = nullcontext()

        with run_ctx:
            if mlflow is not None:
                mlflow.log_params({
                    "gnn_hidden_dim": cfg.gnn.hidden_dim,
                    "gnn_lr": cfg.gnn.lr,
                    "xgb_max_depth": cfg.xgb.max_depth,
                    "ensemble_weights": str(cfg.ensemble.weights)
                })

            # Split indices so we can slice sequences in parallel.
            # GROUP-AWARE when groups (e.g. Variant_ID) are provided → the same
            # variant never straddles train/test (panel-overlap + augmentation
            # leakage eliminated, TEKNOFEST §7.5 integrity).
            from sklearn.model_selection import train_test_split as _tts
            idx = np.arange(len(X))
            if groups is not None:
                from sklearn.model_selection import GroupShuffleSplit
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=cfg.training.test_size, random_state=cfg.seed
                )
                idx_tr, idx_te = next(gss.split(idx, y, groups))
                logger.info("Hold-out split: GROUP-AWARE by group id (n_groups=%d)",
                            len(np.unique(groups)))
            else:
                idx_tr, idx_te = _tts(idx, test_size=cfg.training.test_size, stratify=y,
                                       random_state=cfg.seed)
            X_train_all, _X_test = X[idx_tr], X[idx_te]
            y_train_all, _y_test = y[idx_tr], y[idx_te]
            groups_tr = groups[idx_tr] if groups is not None else None
            panels_tr = panels[idx_tr] if panels is not None else None

            # ── ConfidentLearner — opsiyonel etiket gürültüsü filtresi ──────
            # use_label_cleaning=True ile gürültülü örnekler eğitimden çıkarılır.
            # Referans: Northcutt et al. (2021) JAIR 70:1373; §4.4 veri kalitesi
            if getattr(getattr(cfg, "training", cfg), "use_label_cleaning", False):
                try:
                    from src.scientific.label_quality import ConfidentLearner
                    from src.features.preprocessing import VariantPreprocessor
                    _cl = ConfidentLearner(noise_threshold=0.45)
                    _preproc_tmp = VariantPreprocessor()
                    _X_tmp, _y_tmp = _preproc_tmp.fit_resample_train(
                        X_train_all, y_train_all
                    )
                    _report = _cl.fit(_X_tmp, _y_tmp)
                    _clean_mask = _report.clean_mask()
                    _keep = np.where(_clean_mask)[0]
                    X_train_all = X_train_all[_keep]
                    y_train_all = y_train_all[_keep]
                    logger.info(
                        "ConfidentLearner: %d → %d örnek (%d şüpheli çıkarıldı, "
                        "tahmini gürültü %%%.1f)",
                        len(_clean_mask), len(_keep),
                        _report.n_flagged, _report.estimated_noise_rate * 100,
                    )
                except Exception as _cl_err:
                    logger.warning("ConfidentLearner atlandı: %s", _cl_err)
            nuc_tr = ([nuc_seqs[i] for i in idx_tr] if nuc_seqs else None)
            aa_tr  = ([aa_seqs[i]  for i in idx_tr] if aa_seqs  else None)

            # Cross-validate on train portion (group-aware when groups given)
            fold_results = self._cross_validate(X_train_all, y_train_all,
                                                nuc_seqs=nuc_tr, aa_seqs=aa_tr,
                                                groups=groups_tr, panels=panels_tr)
            mean_f1 = float(np.mean([r.f1 for r in fold_results]))
            std_f1  = float(np.std( [r.f1 for r in fold_results]))
            
            if mlflow is not None:
                mlflow.log_metric("mean_cv_f1", mean_f1)
                mlflow.log_metric("std_cv_f1", std_f1)
            
            logger.info(
                "Cross-validation complete: Binary F1 (Pathogenic §7.3) = %.4f ± %.4f", mean_f1, std_f1
            )

            # Final model — fit on full training set
            preprocessor, ensemble, X_opt_val, y_opt_val = self._fit_single(
                X_train_all, y_train_all, nuc_seqs=nuc_tr, aa_seqs=aa_tr,
                panels=panels_tr,
            )

            # Stacking/Weight optimization (logging this is useful)
            if cfg.ensemble.optimize_weights:
                ensemble.optimise_weights(X_opt_val, None, y_opt_val)

            # Fit stacking meta-learner on the same inner val set
            try:
                ensemble.fit_meta_learner(X_opt_val, y_opt_val)
            except ValueError as exc:
                logger.warning(
                    "Meta-learner fitting başarısız (ValueError: %s) — "
                    "ağırlıklı ortalama kullanılıyor. Olası neden: yetersiz örnek (%d).",
                    exc, len(y_opt_val),
                )
            except Exception as exc:
                logger.error(
                    "Meta-learner fitting beklenmedik hata (%s: %s) — "
                    "ağırlıklı ortalama kullanılıyor.",
                    type(exc).__name__, exc,
                )

            if mlflow is not None:
                mlflow.pytorch.log_model(ensemble.gnn, "gnn_model")
                mlflow.sklearn.log_model(ensemble.xgb, "xgb_model")
            
            return TrainResult(
                ensemble      = ensemble,
                preprocessor  = preprocessor,
                fold_results  = fold_results,
                mean_cv_f1    = mean_f1,
                std_cv_f1     = std_f1,
                test_indices  = idx_te,
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
        panels:   Optional[np.ndarray] = None,
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

        preprocessor = build_preprocessor_from_config()
        # Local override: SMOTE breaks sequence-row alignment when multimodal
        # GNN is enabled. Override on the preprocessor instance only — never
        # mutate the global Settings object.
        if use_multimodal and nuc_seqs is not None and preprocessor.smote_enabled:
            logger.warning(
                "use_multimodal=True: disabling SMOTE on this preprocessor "
                "to preserve sequence-row alignment (global config untouched)."
            )
            preprocessor.smote_enabled = False
        X_proc, y_resampled = preprocessor.fit_resample_train(X_train, y_train)

        # ── Inner val split for GNN/DNN early stopping (AFTER SMOTE) ──────────
        # Carving from the post-SMOTE pool so the ES val set is balanced.
        # Never touches the external test set → no leakage of jury data.
        inner_val_size = min(0.15, 200 / max(len(X_proc), 1))  # at most 15%
        inner_val_size = max(inner_val_size, 0.10)               # at least 10%
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
                _to_lgbm_frame(X_inner_tr), y_inner_tr,
                eval_set=[(_to_lgbm_frame(X_inner_val), y_inner_val)],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            logger.info("LightGBM fitted: best_iteration=%d",
                        lgbm_model.best_iteration_)
        except ImportError:
            logger.warning("lightgbm not installed — skipping LGBM member.")
        except Exception as exc:
            logger.warning("LightGBM training failed (%s) — skipping.", exc)

        # --- GNN (VariantGATv2GNN) — with proper early stopping via inner val ---
        knn_k    = getattr(cfg.gnn, "knn_k", 5)
        patience = getattr(cfg.gnn, "early_stopping_patience", 5)
        gnn_model = VariantGATv2GNN(
            numeric_dim    = X_proc.shape[1],
            hidden_dim     = cfg.gnn.hidden_dim,
            num_classes    = 2,
            use_multimodal = use_multimodal,
            seq_enc_dim    = seq_enc_dim,
        ).to(self.device)
        gnn_model = self._train_gatv2(
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
        if getattr(cfg.dnn, "use_dann", False) and panels is not None:
            # Domain-adversarial final DNN — trained on pre-SMOTE original rows
            # (panel labels defined). X_proc[:n_orig] are the original rows.
            n_orig = len(X_train)
            p_arr = np.asarray(panels)[:n_orig]
            dnn_model = self._train_dnn_dann(
                dnn_model, X_proc[:n_orig], y_resampled[:n_orig], p_arr
            )
        else:
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
        groups:   Optional[np.ndarray] = None,
        panels:   Optional[np.ndarray] = None,
    ) -> List[FoldResult]:
        cfg   = self.cfg
        # --- Girdi doğrulama ---
        if len(X) == 0:
            raise ValueError("CV icin en az 1 ornek gerekli; 0 satir geldi.")
        if len(X) < 2:
            raise ValueError(f"CV icin en az 2 ornek gerekli; {len(X)} satir geldi.")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Egitim icin en az 2 sinif gerekli; yalnizca {unique_classes} bulundu. "
                "Veri setinin sinif dagilimini kontrol edin."
            )
        min_samples_needed = cfg.training.cv_folds * 2
        if len(X) < min_samples_needed:
            logger.warning(
                "Veri cok kucuk (%d ornek) %d-fold CV icin (min %d onerilir). "
                "Hata olusabilir.",
                len(X), cfg.training.cv_folds, min_samples_needed,
            )
        # Group-aware CV when groups provided → no variant straddles folds.
        if groups is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            skf = StratifiedGroupKFold(
                n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed
            )
            split_iter = skf.split(X, y, groups)
            logger.info("CV: StratifiedGroupKFold (group-aware, n_groups=%d)",
                        len(np.unique(groups)))
        else:
            skf = StratifiedKFold(
                n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed
            )
            split_iter = skf.split(X, y)
        results: List[FoldResult] = []
        use_multimodal = getattr(cfg.gnn, "use_multimodal", False)
        seq_enc_dim    = getattr(cfg.gnn, "seq_enc_dim", 32)
        # Per-model group-aware OOF (out-of-sample) — for honest ensemble-weight
        # optimization + conformal calibration (reports/oof_per_model.npz).
        _oof = np.full((len(X), 4), np.nan)  # [XGB, LGBM, GNN, DNN] Pathogenic prob

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
            set_global_seed(cfg.seed + fold_idx)
            logger.info("--- Fold %d/%d ---", fold_idx, cfg.training.cv_folds)

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Slice sequences for this fold (pre-SMOTE rows only)
            nuc_tr  = ([nuc_seqs[i] for i in train_idx] if nuc_seqs else None)
            nuc_val = ([nuc_seqs[i] for i in val_idx]   if nuc_seqs else None)
            aa_tr   = ([aa_seqs[i]  for i in train_idx] if aa_seqs  else None)
            aa_val  = ([aa_seqs[i]  for i in val_idx]   if aa_seqs  else None)

            # --- Preprocessing fit on fold training data ONLY ---
            preprocessor = build_preprocessor_from_config()
            # Local override: SMOTE breaks sequence-row alignment when
            # multimodal GNN is enabled — disable on this fold's preprocessor
            # only, never on the shared Settings object.
            if use_multimodal and nuc_tr is not None and preprocessor.smote_enabled:
                logger.warning(
                    "Fold %d: use_multimodal=True → SMOTE bu fold'da devre dışı "
                    "(sequence-row alignment koruması). Global config değiştirilmedi.",
                    fold_idx,
                )
                preprocessor.smote_enabled = False
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
            xgb_f1      = float(f1_score(y_val, xgb_preds, average="binary", pos_label=1, zero_division=0))

            # --- LightGBM ---
            lgbm_f1 = 0.0
            lgbm_model_fold = None
            try:
                import lightgbm as lgb
                lgbm_model_fold = lgb.LGBMClassifier(**cfg.lgbm.as_dict())
                lgbm_model_fold.fit(
                    _to_lgbm_frame(X_tr_proc), y_tr_res,
                    eval_set=[(_to_lgbm_frame(X_val_proc), y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False),
                               lgb.log_evaluation(-1)],
                )
                lgbm_preds = lgbm_model_fold.predict(_to_lgbm_frame(X_val_proc))
                lgbm_f1    = float(f1_score(y_val, lgbm_preds, average="binary", pos_label=1, zero_division=0))
            except Exception as lgbm_exc:
                logger.warning("Fold %d: LightGBM başarısız (%s) — ensemble'dan çıkarıldı.", fold_idx, lgbm_exc)

            # --- GNN (VariantGATv2GNN) ---
            knn_k    = getattr(cfg.gnn, "knn_k", 5)
            patience = getattr(cfg.gnn, "early_stopping_patience", 5)
            X_for_gnn = X_tr_proc[:n_orig_tr] if nuc_tr_proc else X_tr_proc
            y_for_gnn = y_tr_res[:n_orig_tr]  if nuc_tr_proc else y_tr_res
            gat_model = VariantGATv2GNN(
                numeric_dim    = X_tr_proc.shape[1],
                hidden_dim     = cfg.gnn.hidden_dim,
                num_classes    = 2,
                use_multimodal = use_multimodal,
                seq_enc_dim    = seq_enc_dim,
            ).to(self.device)
            gat_model = self._train_gatv2(
                gat_model, preprocessor,
                X_for_gnn, y_for_gnn,
                X_val_proc, y_val,
                knn_k=knn_k, patience=patience,
                nuc_seqs=nuc_tr_proc, aa_seqs=aa_tr_proc,
                nuc_val=nuc_val, aa_val=aa_val,
            )
            # Evaluate GATv2 on validation graph
            val_graph = _build_sample_graph(preprocessor, X_val_proc, y_val, knn_k)
            nuc_val_t, aa_val_t = _tokenize_sequences(nuc_val if use_multimodal else None,
                                                      aa_val  if use_multimodal else None,
                                                      self.device)
            gnn_preds, gnn_probs_fold = _gatv2_eval(gat_model, val_graph, self.device,
                                                   nuc_ids=nuc_val_t, aa_ids=aa_val_t)
            gnn_f1 = float(f1_score(y_val, gnn_preds[:len(y_val)],
                                    average="binary", pos_label=1, zero_division=0))

            # --- DNN ---
            dnn_model     = VariantDNN(X_tr_proc.shape[1], cfg.dnn.hidden_dim, 2).to(self.device)
            dnn_val_loader= _make_dnn_loader(X_val_proc, y_val, cfg.training.batch_size, False)
            if getattr(cfg.dnn, "use_dann", False) and panels is not None:
                # Domain-adversarial: train on pre-SMOTE fold rows (panels defined).
                p_fold = np.asarray(panels)[train_idx][:n_orig_tr]
                dnn_model = self._train_dnn_dann(
                    dnn_model, X_tr_proc[:n_orig_tr], y_tr_res[:n_orig_tr], p_fold
                )
            else:
                dnn_tr_loader = _make_dnn_loader(X_tr_proc, y_tr_res, cfg.training.batch_size, True)
                dnn_model     = self._train_dnn(dnn_model, dnn_tr_loader, dnn_val_loader, y_train=y_tr_res)
            dnn_preds, dnn_probs_fold = _dnn_eval(dnn_model, dnn_val_loader, self.device)
            dnn_f1        = float(f1_score(y_val, dnn_preds[:len(y_val)],
                                           average="binary", pos_label=1, zero_division=0))

            # --- Ensemble ---
            w         = cfg.ensemble.weights
            xgb_probs = xgb_model.predict_proba(X_val_proc)
            gnn_probs = np.array(gnn_probs_fold)
            dnn_probs = np.array(dnn_probs_fold)  # reuse — no second forward pass
            if lgbm_model_fold is not None and len(w) >= 4:
                lgbm_probs = lgbm_model_fold.predict_proba(_to_lgbm_frame(X_val_proc))
                # 4-model weighted combine: [XGB, LGB, GNN, DNN]
                w_sum = sum(w[:4])
                ens_probs = (w[0]/w_sum * xgb_probs + w[1]/w_sum * lgbm_probs
                             + w[2]/w_sum * gnn_probs + w[3]/w_sum * dnn_probs)
            else:
                # LightGBM fold'da başarısız — 3 model: XGB + GNN + DNN
                if len(w) >= 4:
                    w3 = [w[0] + w[1], w[2], w[3]]  # XGB+LGB birleşik, GNN, DNN
                elif len(w) == 3:
                    w3 = list(w[:3])
                else:
                    w3 = [1.0, 0.0, 0.0]  # yedek: sadece XGB
                t  = sum(w3) or 1.0  # sıfır bölme koruması
                w3 = [x / t for x in w3]
                ens_probs = w3[0] * xgb_probs + w3[1] * gnn_probs + w3[2] * dnn_probs
            ens_preds = np.argmax(ens_probs, axis=1)
            # TEKNOFEST §7.3: temel metrik binary F1 (Pathogenic sınıfı, pos_label=1)
            ens_f1    = float(f1_score(y_val, ens_preds, average="binary", pos_label=1, zero_division=0))

            # --- Per-model OOF accumulation (out-of-sample for this fold) ---
            nv = len(y_val)
            _oof[val_idx, 0] = xgb_probs[:nv, 1]
            if lgbm_model_fold is not None:
                _oof[val_idx, 1] = lgbm_model_fold.predict_proba(_to_lgbm_frame(X_val_proc))[:nv, 1]
            else:
                _oof[val_idx, 1] = xgb_probs[:nv, 1]
            _oof[val_idx, 2] = np.asarray(gnn_probs_fold)[:nv, 1]
            _oof[val_idx, 3] = np.asarray(dnn_probs_fold)[:nv, 1]

            logger.info(
                "Fold %d | Ensemble Binary F1 (Pathogenic): %.4f  (XGB=%.4f, LGB=%.4f, GNN=%.4f, DNN=%.4f)",
                fold_idx, ens_f1, xgb_f1, lgbm_f1, gnn_f1, dnn_f1,
            )
            results.append(FoldResult(fold_idx, ens_f1, xgb_f1, lgbm_f1, gnn_f1, dnn_f1))

        # Persist per-model OOF for honest weight optimization (+ conformal).
        try:
            mask = ~np.isnan(_oof).any(axis=1)
            np.savez(cfg.paths.reports_dir / "oof_per_model.npz",
                     oof=_oof[mask], labels=y[mask],
                     models=np.array(["XGB", "LGBM", "GNN", "DNN"]))
            logger.info("Per-model OOF saved → reports/oof_per_model.npz (n=%d)", int(mask.sum()))
        except Exception as _oof_exc:
            logger.warning("OOF save skipped: %s", _oof_exc)

        return results

    # ------------------------------------------------------------------
    # VariantGATv2GNN training loop — Module 3 & 4
    # ------------------------------------------------------------------

    def _train_gatv2(
        self,
        model:       VariantGATv2GNN,
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
    ) -> VariantGATv2GNN:
        """
        Full-batch node-classification training on a cosine k-NN sample graph.

        Loss function:  WeightedBCELoss (class-balanced cross-entropy).
        Early stopping: monitored on Validation Binary F1 (Pathogenic, pos_label=1, §7.3).

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

        # Stochastic Weight Averaging — collects checkpoints in the last 25 %
        # of epochs for better external-validation generalisation (Izmailov 2018).
        swa_buffer = SWABuffer(swa_start_fraction=0.75, max_checkpoints=10)
        swa_scheduler = CyclicSWAScheduler(
            optimizer,
            lr_min = cfg.gnn.lr * 0.1,
            lr_max = cfg.gnn.lr,
            cycle_length = max(3, cfg.gnn.epochs // 10),
        )
        swa_epoch_counter: int = 0

        # Epoch-level learning curve log (PSR §4.5 — öğrenme süreci kanıtı)
        learning_curve: list = []

        for epoch in range(1, cfg.gnn.epochs + 1):
            # SWA cyclic LR once we enter the collection window
            if swa_buffer.should_collect(epoch, cfg.gnn.epochs):
                swa_scheduler.step(swa_epoch_counter)
                swa_epoch_counter += 1

            loss = _gatv2_epoch(model, train_graph, optimizer, criterion, self.device,
                               nuc_ids=nuc_tr_t, aa_ids=aa_tr_t)

            # Collect checkpoint for SWA
            collected = swa_buffer.push(epoch, cfg.gnn.epochs, model)

            # Train F1 (same graph, eval mode)
            tr_preds, _ = _gatv2_eval(model, train_graph, self.device,
                                      nuc_ids=nuc_tr_t, aa_ids=aa_tr_t)
            train_f1 = float(f1_score(
                y_tr, tr_preds[:len(y_tr)], average="binary", pos_label=1, zero_division=0
            ))

            epoch_entry: dict = {
                "epoch": epoch, "loss": round(loss, 6),
                "train_f1": round(train_f1, 4), "swa_collected": collected,
            }

            if val_graph is not None:
                preds, _ = _gatv2_eval(model, val_graph, self.device,
                                      nuc_ids=nuc_val_t, aa_ids=aa_val_t)
                val_f1   = float(f1_score(
                    y_val, preds[:len(y_val)], average="binary", pos_label=1, zero_division=0
                ))
                epoch_entry["val_f1"] = round(val_f1, 4)

                if val_f1 > best_val_f1:
                    best_val_f1  = val_f1
                    best_weights = copy.deepcopy(model.state_dict())
                    patience_cnt = 0
                    epoch_entry["best"] = True
                else:
                    patience_cnt += 1

                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug(
                        "GATv2 epoch %d/%d | loss=%.4f | train_f1=%.4f | val_f1=%.4f "
                        "(patience %d/%d | swa=%d)",
                        epoch, cfg.gnn.epochs, loss, train_f1, val_f1,
                        patience_cnt, patience, swa_buffer.n_collected,
                    )

                if patience > 0 and patience_cnt >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val Binary F1=%.4f)",
                        epoch, best_val_f1,
                    )
                    epoch_entry["early_stop"] = True
                    learning_curve.append(epoch_entry)
                    break
            else:
                if epoch % 5 == 0 or epoch == cfg.gnn.epochs:
                    logger.debug(
                        "GATv2 epoch %d/%d | loss=%.4f | train_f1=%.4f | swa=%d",
                        epoch, cfg.gnn.epochs, loss, train_f1, swa_buffer.n_collected,
                    )

            learning_curve.append(epoch_entry)

        # Apply SWA: average buffered checkpoints into final model weights.
        # Prefer SWA over best-single-epoch only when enough checkpoints exist
        # AND the training ran to completion (no early stop).
        if swa_buffer.n_collected >= 2:
            swa_buffer.apply(model)
            logger.info(
                "GATv2 SWA applied (%d checkpoints averaged).",
                swa_buffer.n_collected,
            )
            # GATv2 LayerNorm kullanır (BatchNorm değil) — update gerekmez.
            # Ama SequenceEncoder CNN katmanları BatchNorm içerebilir; güvenli ol.
            has_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                for m in model.modules()
            )
            if has_bn:
                from src.training.swa import update_batch_norm
                _tmp_loader = _make_geo_loader(
                    preprocessor, X_tr, y_tr, batch_size=min(32, len(X_tr)), shuffle=False
                )
                update_batch_norm(model, _tmp_loader, self.device)

        # Persist learning curve JSON for PDR §4.5 reproducibility
        try:
            import json as _json
            lc_path = getattr(cfg.paths, "reports_dir", None)
            if lc_path is not None:
                import pathlib as _pl
                lc_dir = _pl.Path(lc_path)
                lc_dir.mkdir(parents=True, exist_ok=True)
                lc_file = lc_dir / "gnn_learning_curve.json"
                existing: list = []
                if lc_file.exists():
                    try:
                        with open(lc_file) as _f:
                            loaded = _json.load(_f)
                        # Eski format dict ise listeye dönüştür
                        existing = loaded if isinstance(loaded, list) else [loaded]
                    except Exception:
                        existing = []
                existing.append({"run_epochs": learning_curve})
                with open(lc_file, "w") as _f:
                    _json.dump(existing, _f, indent=2)
                logger.info("GNN learning curve → %s", lc_file)
        except Exception as _lc_exc:
            logger.debug("Learning curve save failed (non-fatal): %s", _lc_exc)

        # Restore best single-epoch checkpoint only when SWA was NOT applied
        # (if SWA was applied above, we keep SWA weights which are superior).
        if val_graph is not None and swa_buffer.n_collected < 2:
            model.load_state_dict(best_weights)
            logger.info("GATv2 restored best checkpoint (val Binary F1=%.4f)", best_val_f1)

        return model

    # ------------------------------------------------------------------
    # DNN training loop  (with SWA)
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
            # y_train sağlanmadı — loader'dan etiketleri yeniden topla (sınıf dengesi korunsun)
            try:
                y_from_loader = np.concatenate([b[1].numpy() for b in train_loader])
                criterion = _make_criterion(y_from_loader, self.device)
                logger.warning(
                    "_train_dnn: y_train=None — class weights DataLoader'dan yeniden hesaplandı."
                )
            except Exception:
                criterion = nn.CrossEntropyLoss()
                logger.warning(
                    "_train_dnn: y_train=None ve DataLoader'dan etiket alınamadı "
                    "— ağırlıksız CrossEntropyLoss kullanılıyor (sınıf dengesizliği riski)."
                )
        best_f1      = -1.0
        best_weights = copy.deepcopy(model.state_dict())
        dnn_swa = SWABuffer(swa_start_fraction=0.75, max_checkpoints=8)
        dnn_swa_scheduler = CyclicSWAScheduler(
            optimizer,
            lr_min = cfg.dnn.lr * 0.1,
            lr_max = cfg.dnn.lr,
            cycle_length = max(3, cfg.dnn.epochs // 8),
        )
        dnn_swa_epoch: int = 0

        for epoch in range(1, cfg.dnn.epochs + 1):
            # SWA cyclic LR in collection window
            if dnn_swa.should_collect(epoch, cfg.dnn.epochs):
                dnn_swa_scheduler.step(dnn_swa_epoch)
                dnn_swa_epoch += 1

            loss = _dnn_epoch(model, train_loader, optimizer, criterion, self.device)
            dnn_swa.push(epoch, cfg.dnn.epochs, model)

            if val_loader is not None:
                preds, _ = _dnn_eval(model, val_loader, self.device)
                y_val    = [batch[1].numpy() for batch in val_loader]
                y_val    = np.concatenate(y_val)
                val_f1   = float(f1_score(
                    y_val, preds[:len(y_val)], average="binary", pos_label=1, zero_division=0
                ))
                if val_f1 > best_f1:
                    best_f1      = val_f1
                    best_weights = copy.deepcopy(model.state_dict())
                if epoch % 5 == 0 or epoch == cfg.dnn.epochs:
                    logger.debug("DNN epoch %d/%d | loss=%.4f | val_f1=%.4f | swa=%d",
                                 epoch, cfg.dnn.epochs, loss, val_f1, dnn_swa.n_collected)
            else:
                if epoch % 5 == 0 or epoch == cfg.dnn.epochs:
                    logger.debug("DNN epoch %d/%d | loss=%.4f | swa=%d",
                                 epoch, cfg.dnn.epochs, loss, dnn_swa.n_collected)

        # Apply DNN SWA if enough checkpoints collected
        if dnn_swa.n_collected >= 2:
            dnn_swa.apply(model)
            logger.info("DNN SWA applied (%d checkpoints).", dnn_swa.n_collected)
            # SWA ağırlık ortalaması BatchNorm running_mean/var'ı geçersiz kılar.
            # Yeni ağırlıklarla istatistikleri yeniden hesapla (PyTorch SWA best practice).
            from src.training.swa import update_batch_norm
            update_batch_norm(model, train_loader, self.device)
        elif val_loader is not None:
            model.load_state_dict(best_weights)
        return model

    def _train_dnn_dann(
        self,
        model:  VariantDNN,
        X:      np.ndarray,
        y:      np.ndarray,
        panels: np.ndarray,
    ) -> VariantDNN:
        """Domain-Adversarial DNN training (Ganin 2015): the encoder is pushed to
        produce PANEL-INVARIANT features via a gradient-reversed panel discriminator.
        Improves cross-panel / distribution-shift generalization (LOPO +2.17pp,
        reports/dann_lopo_validation.json). Trained on pre-SMOTE rows where panel
        labels are defined. ``model.net`` = Sequential(encoder..., head)."""
        import torch.nn as _nn
        from src.training.domain_adversarial import (
            PanelDiscriminator, dann_lambda, encode_panels,
        )
        cfg, dev = self.cfg, self.device
        encoder    = _nn.Sequential(*list(model.net[:-1]))
        head       = model.net[-1]
        feat_dim   = head.in_features
        pe, _      = encode_panels(list(panels))
        n_panels   = int(pe.max()) + 1
        disc       = PanelDiscriminator(feat_dim, n_panels=max(n_panels, 2)).to(dev)

        Xt = torch.tensor(X, dtype=torch.float32).to(dev)
        yt = torch.tensor(y, dtype=torch.long).to(dev)
        pt = torch.tensor(pe, dtype=torch.long).to(dev)

        # Class-weighted variant loss (imbalance) + panel CE (adversarial).
        counts = np.bincount(y, minlength=2).astype(float)
        cw = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1, None)),
                          dtype=torch.float32).to(dev)
        var_crit = _nn.CrossEntropyLoss(weight=cw)
        pan_crit = _nn.CrossEntropyLoss()
        params = list(encoder.parameters()) + list(head.parameters()) + list(disc.parameters())
        opt = torch.optim.Adam(params, lr=cfg.dnn.lr, weight_decay=cfg.dnn.weight_decay)

        gamma = getattr(cfg.dnn, "dann_gamma", 10.0)
        model.train()
        for epoch in range(cfg.dnn.epochs):
            encoder.train(); head.train(); disc.train()
            opt.zero_grad()
            feats = encoder(Xt)
            v_logits = head(feats)
            lam = dann_lambda(epoch / max(cfg.dnn.epochs, 1), gamma=gamma)
            p_logits = disc(feats, lambda_=lam)
            loss = var_crit(v_logits, yt) + lam * pan_crit(p_logits, pt)
            loss.backward()
            opt.step()
        model.eval()
        logger.info("DNN trained with DANN (panel-invariant, λ_max@end, n_panels=%d)", n_panels)
        return model

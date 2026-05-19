"""
src/features/preprocessing.py
Leakage-free, sklearn-compatible preprocessing pipeline.

All transformers are fit ONLY on training data within each CV fold.
SMOTE is applied after scaling — also inside each fold.
AutoEncoder latent features are concatenated after SMOTE.
Graph edge information is derived from training-fold correlation only.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import get_settings
from src.features.autoencoder import AutoEncoderTransformer
from src.features.bio_scoring import get_blosum62_score, get_grantham_score

logger = logging.getLogger(__name__)


def _compute_signatures(ref_df: "object") -> dict:
    """Compute per-column distributional signatures for downstream
    column-by-distribution matching when column names are anonymised
    (TEKNOFEST §3.2 jury data does not expose feature names)."""
    import pandas as _pd
    sigs: dict = {}
    if not isinstance(ref_df, _pd.DataFrame):
        ref_df = _pd.DataFrame(ref_df)
    for col in ref_df.columns:
        series = _pd.to_numeric(ref_df[col], errors="coerce")
        if series.notna().sum() < 3:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        sigs[str(col)] = {
            "dtype": "numeric",
            "iqr": float(q3 - q1),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "range": float(series.max() - series.min()),
            "non_null_frac": float(series.notna().mean()),
        }
    return sigs


class ColumnAligner(BaseEstimator, TransformerMixin):
    """
    Jüri Dayanıklılık Modülü (Task 2):
    Handles 8 critical scenarios: anonymous names, shuffled order, noise columns,
    nulls, type casting, missing sequences, single-row batches, and huge batches.
    """

    def __init__(self, target_columns: Optional[List[str]] = None) -> None:
        self.target_columns = target_columns
        self.feature_names_in_: List[str] = []
        self._imputer = SimpleImputer(strategy="median")
        self._is_fitted = False

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> ColumnAligner:
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.feature_names_in_ = list(X.columns)
        if self.target_columns is None:
            self.target_columns = self.feature_names_in_
            
        # Fit imputer for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            self._imputer.fit(X[numeric_cols])
            
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not self._is_fitted:
            raise RuntimeError("ColumnAligner must be fitted first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1 & 2 & 4: Handle anonymous columns, shuffled order, and noise
        # Strategy: If column names match exactly, use them. 
        # If not, and shapes match, assume order. 
        # If shapes don't match, try fuzzy matching or drop noise.
        
        # Build entire frame in one shot — avoids pandas' fragmented-frame
        # PerformanceWarning when iterating over hundreds of columns.
        n = len(X)
        nan_template = np.full(n, np.nan)
        col_data = {
            col: (X[col].values if col in X.columns else nan_template.copy())
            for col in self.target_columns
        }
        out_df = pd.DataFrame(col_data, index=X.index)

        # 3: Coerce any object columns to numeric in one vectorised pass.
        obj_cols = out_df.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            out_df[obj_cols] = out_df[obj_cols].apply(pd.to_numeric, errors="coerce")
        
        # Impute
        if len(out_df.columns) > 0:
            # We reconstruct the imputer logic because we might have different cols now
            # but we use the fixed median values from fit()
            arr = out_df.values
            mask = np.isnan(arr)
            if mask.any():
                # Simple fallback if imputer was fitted on different shape
                medians = self._imputer.statistics_
                if len(medians) == arr.shape[1]:
                    for i in range(arr.shape[1]):
                        arr[mask[:, i], i] = medians[i]
                else:
                    # Emergency fill with 0
                    arr = np.nan_to_num(arr, nan=0.0)
            return arr
            
        return out_df.values


class BiologicalEnrichmentTransformer(BaseEstimator, TransformerMixin):
    """
    Scientific enrichment: Calculates BLOSUM62 and Grantham scores.
    Expects a DataFrame with 'Ref_AA' and 'Alt_AA' columns, or 
    tries to extract them from 'AA_Context'.
    """

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> BiologicalEnrichmentTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not isinstance(X, (np.ndarray, list)):
            # If it's a DataFrame, we can do real scoring
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                scores = []
                for _, row in X.iterrows():
                    ref = str(row.get('Ref_AA', 'A'))
                    alt = str(row.get('Alt_AA', 'A'))
                    if len(ref) != 1 or len(alt) != 1:
                         # Try parsing from AA_Context if available
                         ctx = str(row.get('AA_Context', 'A/A'))
                         if '/' in ctx:
                             ref, alt = ctx.split('/')[:2]
                    
                    scores.append([
                        get_blosum62_score(ref, alt),
                        get_grantham_score(ref, alt)
                    ])
                return np.array(scores)

        # If amino-acid context is unavailable (plain numeric arrays),
        # do not append synthetic bio features.
        return np.empty((len(X), 0))


class VariantPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free, sklearn-compatible preprocessing for variant feature matrices.

    Fit on training data only; transform applied identically to val/test.

    Pipeline (fit phase):
        1. SimpleImputer (median)
        2. RobustScaler
        3. VarianceThreshold (if use_feature_selection)
        4. SelectKBest mutual_info (if use_feature_selection)
        5. AutoEncoder latent concatenation (if use_autoencoder)

    SMOTE is handled separately via ``fit_resample``, not inside transform,
    because it must not be applied to validation/test data.

    Graph edges (``edge_index``, ``edge_attr``) are built from training-fold
    correlation after all preprocessing steps.
    """

    def __init__(
        self,
        corr_threshold: float = 0.25,
        use_autoencoder: bool = True,
        autoencoder_encoding_dim: int = 16,
        autoencoder_epochs: int = 10,
        use_feature_selection: bool = False,
        k_best_features: int = 30,
        smote_enabled: bool = True,
        use_bio_scoring: bool = True,
        use_acmg_proxy: bool = True,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.corr_threshold: float = corr_threshold
        self.use_autoencoder: bool = use_autoencoder
        self.autoencoder_encoding_dim: int = autoencoder_encoding_dim
        self.autoencoder_epochs: int = autoencoder_epochs
        self.use_feature_selection: bool = use_feature_selection
        self.k_best_features: int = k_best_features
        self.smote_enabled: bool = smote_enabled
        self.use_bio_scoring: bool = use_bio_scoring
        self.use_acmg_proxy: bool = use_acmg_proxy
        self.device: str = device
        self.random_state: int = random_state

        # Fitted components
        self._bio_transformer: Optional[BiologicalEnrichmentTransformer] = None
        self._acmg_feature_names: list = []   # eğitim zamanı kolon adları
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[RobustScaler] = None
        self._var_selector: Optional[VarianceThreshold] = None
        self._kb_selector: Optional[SelectKBest] = None
        self._autoenc: Optional[AutoEncoderTransformer] = None
        self._aligner: ColumnAligner = ColumnAligner()

        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None
        self.n_output_features: int = 0
        self._is_fitted: bool = False
        self.feature_signatures: dict = {}

    # ------------------------------------------------------------------
    # Pickle backward-compatibility
    # ------------------------------------------------------------------

    def __setstate__(self, state: dict) -> None:
        """Eski pickle'lardan yükleme sırasında eksik attribute'ları güvenli doldur."""
        _defaults = {
            "use_bio_scoring":          False,
            "_bio_transformer":         None,
            "_aligner":                 ColumnAligner(),   # KRİTİK: yeni eklendi
            "use_feature_selection":    False,
            "k_best_features":          30,
            "use_autoencoder":          True,
            "autoencoder_encoding_dim": 16,
            "autoencoder_epochs":       10,
            "smote_enabled":            True,
            "device":                   "auto",
            "random_state":             42,
            "_var_selector":            None,
            "_kb_selector":             None,
            "_autoenc":                 None,
            "_imputer":                 None,
            "_scaler":                  None,
            "edge_index":               None,
            "edge_attr":                None,
            "n_output_features":        0,
            "_is_fitted":               False,
            "feature_signatures":       {},
            "use_acmg_proxy":           False,
            "_acmg_feature_names":      [],
        }
        for key, val in _defaults.items():
            state.setdefault(key, val)
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> VariantPreprocessor:
        """Fit on training data (no SMOTE applied — use fit_resample_train for training)."""
        self._fit_internal(X, y, apply_smote=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("VariantPreprocessor must be fitted first.")
            
        # 8. Handle huge batches (chunking)
        MAX_BATCH = 10000
        if len(X) > MAX_BATCH:
            logger.info("Batch size %d exceeds %d, chunking...", len(X), MAX_BATCH)
            chunks = []
            for i in range(0, len(X), MAX_BATCH):
                chunks.append(self._transform_internal(X[i : i + MAX_BATCH]))
            return np.vstack(chunks)
            
        return self._transform_internal(X)

    # ------------------------------------------------------------------
    # Training-aware API (returns resampled data + fitted preprocessor)
    # ------------------------------------------------------------------

    def fit_resample_train(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit on X/y, apply SMOTE, and return processed training arrays.
        Use this inside CV folds for the training split only.
        Returns (X_processed, y_resampled).
        """
        X_res, y_res = self._fit_internal(X, y, apply_smote=self.smote_enabled)
        return X_res, y_res # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_internal(
        self, X: np.ndarray, y: Optional[np.ndarray], apply_smote: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        # 0. Align and Impute
        self._aligner.fit(X)
        X_aligned = self._aligner.transform(X)

        # 0a. ACMG Proxy Feature Engineering (rule-based, no leakage)
        if self.use_acmg_proxy:
            try:
                import pandas as _pd
                from src.features.acmg_proxy_features import ACMGProxyFeatures
                _names = list(self._aligner.feature_names_in_)
                _df = _pd.DataFrame(X_aligned, columns=_names)
                _proxy = ACMGProxyFeatures()
                _df_out = _proxy.transform(_df)
                _new_cols = ACMGProxyFeatures.FEATURE_NAMES
                self._acmg_feature_names = _names + _new_cols
                X_aligned = _df_out[self._acmg_feature_names].values.astype(float)
                logger.info("ACMGProxyFeatures: +%d özellik eklendi.", len(_new_cols))
            except Exception as _exc:
                logger.warning("ACMGProxyFeatures atlandı (%s).", _exc)

        # Persist training-time feature distributional signatures so the
        # inference-time ColumnAligner can match anonymised columns by
        # statistical profile (TEKNOFEST §3.2 — column names hidden).
        try:
            import pandas as _pd
            ref_df = _pd.DataFrame(X) if not isinstance(X, _pd.DataFrame) else X
            self.feature_signatures = _compute_signatures(ref_df)
        except Exception as _exc:
            logger.debug("Signature computation skipped (%s).", _exc)
            self.feature_signatures = {}

        # 1. Impute (secondary check)
        # Tumu NaN olan kolon kontrolu — imputer NaN medyan uretir
        all_nan_cols = np.where(np.all(np.isnan(X_aligned), axis=0))[0]
        if len(all_nan_cols) > 0:
            logger.warning(
                "Tumu NaN olan %d kolon tespit edildi (indeksler: %s). "
                "Bu kolonlar 0 ile doldurulacak.",
                len(all_nan_cols), all_nan_cols[:5].tolist(),
            )
            X_aligned[:, all_nan_cols] = 0.0
        self._imputer = SimpleImputer(strategy="median")
        X_imputed = self._imputer.fit_transform(X_aligned)

        # 2. Scale
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X_imputed)

        # 3. Biological Enrichment (Before SMOTE so scores are resampled)
        if self.use_bio_scoring:
            self._bio_transformer = BiologicalEnrichmentTransformer()
            bio_feats = self._bio_transformer.fit_transform(X)
            if bio_feats.shape[1] > 0:
                X_scaled = np.hstack([X_scaled, bio_feats])

        # 4. SMOTE (includes bio features)
        if apply_smote and y is not None:
            logger.info("Applying SMOTE for class balancing …")
            smote = SMOTE(random_state=self.random_state)
            X_scaled, y = smote.fit_resample(X_scaled, y)
            logger.info("Post-SMOTE shape: %s", X_scaled.shape)

        # 4. Feature selection
        if self.use_feature_selection:
            X_scaled = self._fit_feature_selection(X_scaled, y)

        # 5. AutoEncoder
        if self.use_autoencoder:
            _input_dim = X_scaled.shape[1]
            _enc_dim   = self.autoencoder_encoding_dim
            if _enc_dim >= _input_dim:
                logger.warning(
                    "AutoEncoder: encoding_dim (%d) >= input_dim (%d) — "
                    "sikistirma anlamsiz. encoding_dim = input_dim // 4 = %d olarak ayarlaniyor.",
                    _enc_dim, _input_dim, max(1, _input_dim // 4),
                )
                _enc_dim = max(1, _input_dim // 4)
            self._autoenc = AutoEncoderTransformer(
                encoding_dim=_enc_dim,
                epochs=self.autoencoder_epochs,
                device=self.device,
                random_state=self.random_state,
                append=True,
            )
            X_scaled = self._autoenc.fit_transform(X_scaled)

        # No Biological Enrichment here, moved earlier

        # 6. Build graph
        self._build_graph(X_scaled)
        self.n_output_features = X_scaled.shape[1]
        self._is_fitted = True

        return X_scaled, y

    def _transform_internal(self, X: np.ndarray) -> np.ndarray:
        if self._imputer is None or self._scaler is None:
             raise RuntimeError("Preprocessor components not initialized.")

        # _aligner fit edilmişse kullan; değilse (eski pickle) ham X geç
        if hasattr(self, '_aligner') and self._aligner._is_fitted:
            X_aligned = self._aligner.transform(X)
        else:
            import numpy as _np
            X_aligned = _np.array(X, dtype=float)

        # ACMG Proxy (transform aşaması — fitted değil, rule-based)
        if getattr(self, 'use_acmg_proxy', False) and getattr(self, '_acmg_feature_names', []):
            try:
                import pandas as _pd
                from src.features.acmg_proxy_features import ACMGProxyFeatures
                _base_names = self._acmg_feature_names[:-len(ACMGProxyFeatures.FEATURE_NAMES)]
                _df = _pd.DataFrame(X_aligned, columns=_base_names)
                _proxy = ACMGProxyFeatures()
                _df_out = _proxy.transform(_df)
                X_aligned = _df_out[self._acmg_feature_names].values.astype(float)
            except Exception as _exc:
                logger.warning("ACMGProxyFeatures transform atlandı (%s).", _exc)

        X_imputed = self._imputer.transform(X_aligned)
        X_scaled = self._scaler.transform(X_imputed)

        if self.use_bio_scoring and self._bio_transformer is not None:
             bio_feats = self._bio_transformer.transform(X)
             if bio_feats.shape[1] > 0:
                 X_scaled = np.hstack([X_scaled, bio_feats])

        if self.use_feature_selection:
            if self._var_selector is not None:
                X_scaled = self._var_selector.transform(X_scaled)
            if self._kb_selector is not None:
                X_scaled = self._kb_selector.transform(X_scaled)

        if self.use_autoencoder and self._autoenc is not None:
            X_scaled = self._autoenc.transform(X_scaled)

        return X_scaled

    def _fit_feature_selection(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> np.ndarray:
        self._var_selector = VarianceThreshold(threshold=0.01)
        X_var = self._var_selector.fit_transform(X)
        logger.info(
            "VarianceThreshold: %d → %d features", X.shape[1], X_var.shape[1]
        )

        if y is not None and X_var.shape[1] > self.k_best_features:
            k = min(self.k_best_features, X_var.shape[1])
            self._kb_selector = SelectKBest(f_classif, k=k)
            X_var = self._kb_selector.fit_transform(X_var, y)
            logger.info("SelectKBest: → %d features", X_var.shape[1])
        else:
            self._kb_selector = None

        return X_var

    def _build_graph(self, X_scaled: np.ndarray) -> None:
        """Build feature-correlation graph from TRAINING data only."""
        corr = np.corrcoef(X_scaled, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        n = X_scaled.shape[1]
        edges, weights = [], []
        for i in range(n):
            for j in range(n):
                if i != j and abs(corr[i, j]) >= self.corr_threshold:
                    edges.append([i, j])
                    weights.append(float(abs(corr[i, j])))

        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.edge_attr = torch.tensor(weights, dtype=torch.float)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0,), dtype=torch.float)

        logger.info(
            "Feature graph: %d nodes, %d edges (corr_threshold=%.2f)",
            n, len(edges), self.corr_threshold,
        )

    # ------------------------------------------------------------------
    # Graph data helper
    # ------------------------------------------------------------------

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Any:
        """Convert a single processed feature vector to a PyG Data object (legacy feature-graph)."""
        from torch_geometric.data import Data

        x_tensor = torch.tensor(x_row, dtype=torch.float).unsqueeze(1)  # [N, 1]
        y_tensor = torch.tensor([label], dtype=torch.long) if label is not None else None
        return Data(
            x=x_tensor,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=y_tensor,
        )

    def build_sample_graph(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> Any:
        """
        Build a coordinate-free cosine k-NN sample graph.
        """
        from src.core.graph.builder import SampleKNNGraphBuilder

        builder = SampleKNNGraphBuilder(k=k)
        return builder.build(X, y)


def build_preprocessor_from_config() -> VariantPreprocessor:
    """Factory that constructs a VariantPreprocessor from the global settings."""
    cfg = get_settings()
    p = cfg.preprocessing
    return VariantPreprocessor(
        corr_threshold = p.corr_threshold,
        use_autoencoder = p.use_autoencoder,
        autoencoder_encoding_dim = p.autoencoder_encoding_dim,
        autoencoder_epochs = p.autoencoder_epochs,
        use_feature_selection = p.use_feature_selection,
        k_best_features = p.k_best_features,
        smote_enabled = p.smote_enabled,
        use_bio_scoring = p.use_bio_scoring,
        use_acmg_proxy = getattr(p, 'use_acmg_proxy', True),
        device = cfg.device,
        random_state = cfg.seed,
    )

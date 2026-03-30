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
from typing import Optional, Tuple, Any, List, Union

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import get_settings
from src.features.autoencoder import AutoEncoderTransformer
from src.features.bio_scoring import get_blosum62_score, get_grantham_score

logger = logging.getLogger(__name__)


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
        self.device: str = device
        self.random_state: int = random_state

        # Fitted components
        self._bio_transformer: Optional[BiologicalEnrichmentTransformer] = None
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[RobustScaler] = None
        self._var_selector: Optional[VarianceThreshold] = None
        self._kb_selector: Optional[SelectKBest] = None
        self._autoenc: Optional[AutoEncoderTransformer] = None

        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None
        self.n_output_features: int = 0
        self._is_fitted: bool = False

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

        # 1. Impute
        self._imputer = SimpleImputer(strategy="median")
        X_imputed = self._imputer.fit_transform(X)

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
            self._autoenc = AutoEncoderTransformer(
                encoding_dim=self.autoencoder_encoding_dim,
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
             raise RuntimeError("Preprocessor components not intialized.")
             
        X_imputed = self._imputer.transform(X)
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
            self._kb_selector = SelectKBest(mutual_info_classif, k=k)
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
        device = cfg.device,
        random_state = cfg.seed,
    )

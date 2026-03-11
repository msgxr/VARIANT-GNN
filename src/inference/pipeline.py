"""
src/inference/pipeline.py
End-to-end inference pipeline.

Preserves Variant_ID and all metadata through the prediction process.
Output columns per variant:
  - Variant_ID          (from metadata)
  - Prediction          (Pathogenic | Benign)
  - Probability         (raw ensemble P(Pathogenic))
  - Calibrated_Risk     (calibrated P(Pathogenic) × 100)
  - Confidence          (max class probability × 100)
  - High_Risk           (bool: calibrated_risk ≥ threshold)
  - <all original metadata columns>
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.calibration.calibrator import EnsembleCalibrator
from src.config import get_settings
from src.data.loader import LoadedDataset, load_predict_csv
from src.features.preprocessing import VariantPreprocessor
from src.models.ensemble import HybridEnsemble
from src.utils.serialization import ModelStore

logger = logging.getLogger(__name__)


def _build_gnn_loader(
    preprocessor: VariantPreprocessor,
    X_scaled:     np.ndarray,
    batch_size:   int,
) -> GeoDataLoader:
    graphs = [preprocessor.row_to_graph(row) for row in X_scaled]
    return GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)


class InferencePipeline:
    """
    Loads serialised models and runs prediction on new variant data.

    Parameters
    ----------
    model_dir : Directory containing saved model artefacts.
    """

    def __init__(self, model_dir: Optional[str | Path] = None) -> None:
        self.cfg       = get_settings()
        model_dir_path = Path(model_dir) if model_dir else self.cfg.paths.models_dir
        self.store     = ModelStore(model_dir_path)

        self._ensemble:    Optional[HybridEnsemble]       = None
        self._preprocessor: Optional[VariantPreprocessor] = None
        self._calibrator:  Optional[EnsembleCalibrator]   = None
        self._loaded       = False

    # ------------------------------------------------------------------

    def load(self) -> "InferencePipeline":
        """Load all model artefacts from disk."""
        self._preprocessor, self._ensemble, self._calibrator = self.store.load_all()
        self._loaded = True
        logger.info("InferencePipeline loaded from %s", self.store.model_dir)
        return self

    # ------------------------------------------------------------------

    def predict_from_dataset(self, dataset: LoadedDataset) -> pd.DataFrame:
        """
        Run inference on a ``LoadedDataset``.

        Returns a DataFrame with prediction columns plus preserved metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before predict_from_dataset().")

        cfg       = self.cfg

        X_np     = dataset.features.values
        X_scaled = self._preprocessor.transform(X_np)

        # VariantSAGEGNN builds its own sample graph; FeatureGNN needs a GeoLoader
        from src.models.gnn import VariantSAGEGNN
        if isinstance(self._ensemble.gnn, VariantSAGEGNN):
            loader = None
        else:
            loader = _build_gnn_loader(
                self._preprocessor, X_scaled, cfg.training.batch_size
            )

        threshold = cfg.thresholds.classification
        preds, raw_proba = self._ensemble.predict(X_scaled, loader, threshold)

        # Calibrated probabilities
        if self._calibrator is not None:
            cal_proba = self._calibrator.transform(raw_proba)
        else:
            cal_proba = raw_proba

        cal_risk   = HybridEnsemble.pathogenic_risk_score(cal_proba)
        confidence = (np.max(raw_proba, axis=1) * 100).round(2)

        # Build output DataFrame
        result = dataset.metadata.copy()
        result["Prediction"]      = np.where(preds == 1, "Pathogenic", "Benign")
        result["Probability"]     = raw_proba[:, 1].round(4)
        result["Calibrated_Risk"] = cal_risk
        result["Confidence"]      = confidence
        result["High_Risk"]       = cal_proba[:, 1] >= cfg.thresholds.high_risk

        return result

    # ------------------------------------------------------------------

    def predict_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a CSV, validate, and run inference."""
        dataset = load_predict_csv(csv_path)
        return self.predict_from_dataset(dataset)

    # ------------------------------------------------------------------

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference directly on a DataFrame (e.g., from Streamlit).

        Metadata / label columns are automatically separated.
        If the uploaded CSV has anonymous / unknown column names but the
        correct number of numeric features, the columns are automatically
        mapped to the expected training-time feature names.
        """
        cfg = self.cfg
        expected_n = self._preprocessor._imputer.n_features_in_

        # Separate metadata cols (id + non-feature columns like Panel, Nuc_Context, AA_Context)
        non_feature_cols = getattr(cfg.schema, 'non_feature_columns', [])
        id_cols  = [c for c in cfg.schema.id_columns if c in df.columns]
        meta_cols = list(dict.fromkeys(id_cols + [c for c in non_feature_cols if c in df.columns]))
        drop     = meta_cols + (
            [cfg.schema.target_column] if cfg.schema.target_column in df.columns else []
        )
        metadata = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
        feature_df = df.drop(columns=drop, errors='ignore').select_dtypes(include=[np.number])

        # Auto-remap anonymous columns when count matches
        if feature_df.shape[1] != expected_n:
            numeric_all = df.select_dtypes(include=[np.number])
            if numeric_all.shape[1] == expected_n:
                feature_df = numeric_all
            else:
                raise ValueError(
                    f"X has {feature_df.shape[1]} features, but "
                    f"the model expects {expected_n} features."
                )

        dummy_dataset = LoadedDataset(
            features       = feature_df,
            labels         = None,
            metadata       = metadata,
            feature_columns= list(feature_df.columns),
        )
        return self.predict_from_dataset(dummy_dataset)

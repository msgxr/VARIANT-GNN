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
from typing import Optional, Any, List, Union, Tuple

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.calibration.calibrator import EnsembleCalibrator
from src.config import get_settings, Settings
from src.data.loader import LoadedDataset, load_predict_csv
from src.features.preprocessing import VariantPreprocessor
from src.models.ensemble import HybridEnsemble
from src.utils.serialization import ModelStore

logger = logging.getLogger(__name__)


def _build_gnn_loader(
    preprocessor: VariantPreprocessor,
    X_scaled: np.ndarray,
    batch_size: int,
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

    def __init__(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        self.cfg: Settings = get_settings()
        model_dir_path = Path(model_dir) if model_dir else self.cfg.paths.models_dir
        self.store = ModelStore(model_dir_path)

        self._ensemble: Optional[HybridEnsemble] = None
        self._preprocessor: Optional[VariantPreprocessor] = None
        self._calibrator: Optional[EnsembleCalibrator] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------

    def load(self) -> InferencePipeline:
        """Load all model artefacts from disk."""
        self._preprocessor, self._ensemble, self._calibrator = self.store.load_all() # type: ignore
        self._loaded = True
        logger.info("InferencePipeline loaded from %s", self.store.model_dir)
        return self

    # ------------------------------------------------------------------

    def predict_from_dataset(self, dataset: LoadedDataset) -> pd.DataFrame:
        """
        Run inference on a ``LoadedDataset``.
        """
        if not self._loaded or self._ensemble is None or self._preprocessor is None:
            raise RuntimeError("Call .load() before predict_from_dataset().")

        cfg = self.cfg

        X_np = dataset.features.values
        X_scaled = self._preprocessor.transform(X_np)

        # ── Build sequence tensors for multimodal GNN (if applicable) ──
        from src.models.gnn import VariantSAGEGNN
        nuc_ids = None
        aa_ids = None
        if (
            isinstance(self._ensemble.gnn, VariantSAGEGNN)
            and getattr(self._ensemble.gnn, "use_multimodal", False)
            and dataset.nuc_sequences is not None
        ):
            import torch
            from src.features.multimodal_encoder import tokenize_amino_acids, tokenize_nucleotides
            
            device = next(self._ensemble.gnn.parameters()).device
            nuc_ids = torch.tensor(
                tokenize_nucleotides(dataset.nuc_sequences), dtype=torch.long
            ).to(device)
            if dataset.aa_sequences is not None:
                aa_ids = torch.tensor(
                    tokenize_amino_acids(dataset.aa_sequences), dtype=torch.long
                ).to(device)
            logger.info(
                "Inference: feeding sequence tokens to multimodal GNN "
                "(nuc=%s, aa=%s)",
                nuc_ids.shape, aa_ids.shape if aa_ids is not None else None,
            )
        elif (
            isinstance(self._ensemble.gnn, VariantSAGEGNN)
            and getattr(self._ensemble.gnn, "use_multimodal", False)
            and dataset.nuc_sequences is None
        ):
            logger.warning(
                "Multimodal GNN is enabled but Nuc_Context/AA_Context "
                "columns are missing from input data. Falling back to "
                "numeric-only features."
            )

        # VariantSAGEGNN builds its own sample graph; FeatureGNN needs a GeoLoader
        if isinstance(self._ensemble.gnn, VariantSAGEGNN):
            loader = None
        else:
            loader = _build_gnn_loader(
                self._preprocessor, X_scaled, cfg.training.batch_size
            )

        threshold = cfg.thresholds.classification
        preds, raw_proba = self._ensemble.predict(
            X_scaled, loader, threshold,
            nuc_ids=nuc_ids, aa_ids=aa_ids,
        ) # type: ignore

        # Calibrated probabilities
        if self._calibrator is not None:
            cal_proba = self._calibrator.transform(raw_proba)
        else:
            cal_proba = raw_proba

        cal_risk = HybridEnsemble.pathogenic_risk_score(cal_proba)
        confidence = (np.max(raw_proba, axis=1) * 100).round(2)

        # Build output DataFrame
        result: pd.DataFrame = dataset.metadata.copy()
        result["Prediction"] = np.where(preds == 1, "Pathogenic", "Benign")
        result["Probability"] = raw_proba[:, 1].round(4)
        result["Calibrated_Risk"] = cal_risk
        result["Confidence"] = confidence
        result["High_Risk"] = cal_proba[:, 1] >= cfg.thresholds.high_risk

        return result

    # ------------------------------------------------------------------

    def predict_from_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Load a CSV, validate, and run inference."""
        dataset = load_predict_csv(csv_path)
        return self.predict_from_dataset(dataset)

    # ------------------------------------------------------------------

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference directly on a DataFrame (e.g., from Streamlit).
        """
        cfg = self.cfg
        if self._ensemble is None or self._preprocessor is None:
             raise RuntimeError("Pipeline must be loaded first.")

        # ── Panel one-hot encoding ──
        KNOWN_PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
        if "Panel" in df.columns:
            panel_series = df["Panel"].astype(str).str.strip()
            for panel_name in KNOWN_PANELS:
                col = f"Panel_{panel_name}"
                if col not in df.columns:
                    df[col] = (panel_series == panel_name).astype(float)

        try:
            expected_features: Optional[List[str]] = self._ensemble.xgb.get_booster().feature_names if self._ensemble.xgb is not None else None
            expected_n: int = len(expected_features) if expected_features else self._preprocessor._imputer.n_features_in_ # type: ignore
        except Exception:
            expected_n = self._preprocessor._imputer.n_features_in_ # type: ignore
            expected_features = None

        # Separate metadata cols
        non_feature_cols: List[str] = getattr(cfg.schema, 'non_feature_columns', [])
        id_cols: List[str] = [c for c in cfg.schema.id_columns if c in df.columns]

        drop_cols: List[str] = list(id_cols)
        if cfg.schema.target_column in df.columns:
            drop_cols.append(cfg.schema.target_column)

        for col in non_feature_cols:
            if col in df.columns and col not in drop_cols:
                drop_cols.append(col)

        for col in df.columns:
            if col not in drop_cols and df[col].dtype == object:
                drop_cols.append(col)

        metadata = df[[c for c in drop_cols if c in df.columns]].copy()

        if expected_features is not None:
            feature_df = df[[c for c in expected_features if c in df.columns]]
            if feature_df.shape[1] != expected_n:
                raise ValueError(f"X has {feature_df.shape[1]} features, model expects {expected_n}")
        else:
            feature_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
            if feature_df.shape[1] != expected_n:
                if feature_df.shape[1] > expected_n:
                    feature_df = feature_df.iloc[:, :expected_n]
                else:
                    raise ValueError(f"X has {feature_df.shape[1]} features, model expects {expected_n}")

        if expected_features is not None:
            feature_df = feature_df[expected_features]
            
        dummy_dataset = LoadedDataset(
            features = feature_df,
            labels = None,
            metadata = metadata,
            feature_columns = list(feature_df.columns),
        )
        return self.predict_from_dataset(dummy_dataset)

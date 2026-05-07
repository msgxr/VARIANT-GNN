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
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.config import Settings, get_settings
from src.core.models.ensemble import HybridEnsemble
from src.data.loader import LoadedDataset, load_predict_csv
from src.features.preprocessing import VariantPreprocessor
from src.scientific.calibration.calibrator import EnsembleCalibrator
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
        from src.core.models.gnn import VariantGATv2GNN
        nuc_ids = None
        aa_ids = None
        if (
            isinstance(self._ensemble.gnn, VariantGATv2GNN)
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
        
        # Load F1-optimal threshold saved during training (TEKNOFEST §7.3)
        # Falls back to config value (0.50) when no saved threshold exists.
        threshold = self.store.load_threshold(default=cfg.thresholds.classification)

        if isinstance(self._ensemble.gnn, VariantGATv2GNN):
            # MC-Dropout ile belirsizlik tahmini (multimodal token'lar dahil)
            preds, raw_proba, uncertainty = self._ensemble.predict_with_uncertainty(
                X_scaled, n_iter=10, threshold=threshold,
                nuc_ids=nuc_ids, aa_ids=aa_ids,
            )
            # Güven skoru: belirsizliğin tersi (yüksek std → düşük güven)
            confidence = ((1.0 - uncertainty) * 100).round(2)
            # Klinik karar bayrağı — belirsizlik eşiklerine göre (Rapor §3.5)
            clinical_flag = np.where(
                uncertainty > 0.30,
                "⚠️ Uzman Değerlendirmesi Gerekli",
                np.where(uncertainty <= 0.15, "✅ Yüksek Güven", "🔶 Orta Güven"),
            )
        else:
            # Klasik tahmin (GATv2 olmayan path)
            preds, raw_proba = self._ensemble.predict(
                X_scaled, threshold=threshold,
                nuc_ids=nuc_ids, aa_ids=aa_ids,
            )
            confidence = (np.max(raw_proba, axis=1) * 100).round(2)
            # Belirsizlik skoru olmadığında max-prob üzerinden bayrak
            conf_frac = np.max(raw_proba, axis=1)
            clinical_flag = np.where(
                conf_frac < 0.70,
                "⚠️ Uzman Değerlendirmesi Gerekli",
                np.where(conf_frac >= 0.90, "✅ Yüksek Güven", "🔶 Orta Güven"),
            )

        # Kalibrasyon
        if self._calibrator is not None:
            cal_proba = self._calibrator.transform(raw_proba)
        else:
            cal_proba = raw_proba

        cal_risk = HybridEnsemble.pathogenic_risk_score(cal_proba)

        # Çıktı DataFrame
        result: pd.DataFrame = dataset.metadata.copy()
        result["Prediction"]    = np.where(preds == 1, "Pathogenic", "Benign")
        result["Probability"]   = raw_proba[:, 1].round(4)
        result["Calibrated_Risk"] = cal_risk
        result["Confidence"]    = confidence
        result["High_Risk"]     = cal_proba[:, 1] >= cfg.thresholds.high_risk
        result["Clinical_Flag"] = clinical_flag

        # ── OOD Tespiti (opsiyonel, hata varsa sessizce atlanır) ─────────────
        try:
            from src.scientific.ood_detector import OODDetector
            _ood_det = OODDetector(z_threshold=3.5, ood_frac_thresh=0.25)
            _ood_det.fit(X_scaled)
            _ood_out = _ood_det.detect(X_scaled)
            result["OOD_Score"] = _ood_out["ood_scores"].round(3)
            result["OOD_Flag"]  = _ood_out["ood_flags"]
        except Exception:
            pass   # OOD modülü opsiyonel — mevcut pipeline etkilenmez

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
            # Multimodal GNN için sekans kolonları varsa pas et
            nuc_sequences = (
                df["Nuc_Context"].astype(str).tolist()
                if "Nuc_Context" in df.columns else None
            ),
            aa_sequences = (
                df["AA_Context"].astype(str).tolist()
                if "AA_Context" in df.columns else None
            ),
        )
        return self.predict_from_dataset(dummy_dataset)

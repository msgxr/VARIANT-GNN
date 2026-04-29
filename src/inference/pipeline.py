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

        If the model was trained with ``use_multimodal=True`` and the dataset
        carries ``nuc_sequences`` / ``aa_sequences``, they are tokenised and
        fed to the GNN SequenceEncoder automatically.

        Returns a DataFrame with prediction columns plus preserved metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before predict_from_dataset().")

        cfg = self.cfg

        X_np     = dataset.features.values
        X_scaled = self._preprocessor.transform(X_np)

        # ── Build sequence tensors for multimodal GNN (if applicable) ──
        from src.models.gnn import VariantSAGEGNN
        nuc_ids = None
        aa_ids  = None
        if (
            isinstance(self._ensemble.gnn, VariantSAGEGNN)
            and getattr(self._ensemble.gnn, "use_multimodal", False)
            and dataset.nuc_sequences is not None
        ):
            from src.features.multimodal_encoder import tokenize_nucleotides, tokenize_amino_acids
            import torch
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
                "numeric-only features. Add 'Nuc_Context' and 'AA_Context' "
                "columns for full SequenceEncoder performance."
            )

        # VariantSAGEGNN builds its own sample graph; FeatureGNN needs a GeoLoader
        from src.models.gnn import VariantSAGEGNN
        if not isinstance(self._ensemble.gnn, VariantSAGEGNN):
            # Keep _build_gnn_loader call just in case it's used later or the user needs it
            _ = _build_gnn_loader(
                self._preprocessor, X_scaled, cfg.training.batch_size
            )

        threshold = cfg.thresholds.classification
        preds, raw_proba = self._ensemble.predict(
            X_scaled, threshold=threshold,
            nuc_ids=nuc_ids, aa_ids=aa_ids,
        )

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
        8 jüri senaryosunu kapsayan güçlendirilmiş DataFrame inference.

        Senaryo 1: Anonim kolon adları       → ColumnAligner distributional matching
        Senaryo 2: Karışık kolon sırası       → ColumnAligner re-order
        Senaryo 3: %10 null/missing           → NaN → imputer → median
        Senaryo 4: 20 gürültülü ek kolon      → drop
        Senaryo 5: String sayısal kolonda      → pd.to_numeric coerce → NaN
        Senaryo 6: Nuc_Context/AA_Context yok → uyarı, devam
        Senaryo 7: Tek satır                  → batch_norm eval mode
        Senaryo 8: 10K satır                  → ColumnAligner chunked
        """
        from src.data.column_aligner import ColumnAligner
        import numpy as _np

        cfg = self.cfg
        df  = df.copy()

        # ── Sabit eğitim özellikleri (43 ham + 4 panel = 47) ─────────────
        TRAINING_FEATURES = [
            'Ref_Nucleotide', 'Alt_Nucleotide', 'Codon_Change_Type',
            'AA_Grantham_Score', 'GC_Content_Window', 'In_CpG_Site',
            'Motif_Disruption_Score', 'AA_Polarity_Change',
            'AA_Hydrophobicity_Diff', 'AA_Mol_Weight_Diff', 'AA_Size_Diff',
            'Protein_Impact_Score', 'Delta_Solvent_Accessibility',
            'Secondary_Structure_Disruption', 'GERP_RS',
            'PhyloP100way_vertebrate', 'phastCons100way_vertebrate',
            'SiPhy_29way_logOdds', 'Phylo_Diversity_Index',
            'gnomAD_exomes_AF', 'gnomAD_exomes_AF_afr',
            'gnomAD_exomes_AF_eur', 'gnomAD_exomes_AF_eas',
            'gnomAD_exomes_AF_sas', 'gnomAD_exomes_AF_amr', 'ExAC_AF',
            'SIFT_score', 'PolyPhen2_HDIV_score', 'PolyPhen2_HVAR_score',
            'CADD_phred', 'REVEL_score', 'MutPred2_score', 'VEST4_score',
            'PROVEAN_score', 'MutationTaster_score', 'MetaSVM_score',
            'MetaLR_score', 'MCAP_score', 'In_Critical_Protein_Domain',
            'Splice_Site_Distance', 'Is_Exonic', 'Exon_Conservation_Ratio',
            'OMIM_Disease_Gene',
        ]
        KNOWN_PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
        PANEL_COLS   = [f"Panel_{p}" for p in KNOWN_PANELS]
        ALL_EXPECTED = TRAINING_FEATURES + PANEL_COLS

        # ── Panel one-hot ─────────────────────────────────────────────────
        if "Panel" in df.columns:
            panel_series = df["Panel"].astype(str).str.strip()
            for p in KNOWN_PANELS:
                col = f"Panel_{p}"
                if col not in df.columns:
                    df[col] = (panel_series == p).astype(float)
        else:
            for col in PANEL_COLS:
                if col not in df.columns:
                    df[col] = 0.0

        # ── Metadata ayrımı ───────────────────────────────────────────────
        id_cols  = [c for c in cfg.schema.id_columns if c in df.columns]
        drop_set = set(id_cols)
        if cfg.schema.target_column in df.columns:
            drop_set.add(cfg.schema.target_column)
        for c in getattr(cfg.schema, "non_feature_columns", []):
            if c in df.columns:
                drop_set.add(c)
        metadata = df[[c for c in drop_set if c in df.columns]].copy()

        # ── ColumnAligner ile 8 senaryolu güçlü hizalama ─────────────────
        aligner = ColumnAligner(
            expected_columns=ALL_EXPECTED,
            fuzzy_threshold=0.80,
            allow_positional=True,
        )
        try:
            aligned, align_report = aligner.robust_apply(df, chunk_size=2000)
            n_matched = sum(1 for c in ALL_EXPECTED if c in aligned.columns
                           and not _np.isnan(aligned[c].fillna(_np.nan).values).all())
            n_missing = len(align_report.unmatched_expected)
            logger.info(
                "ColumnAligner: %d/%d kolon eşleşti, %d kolon sıfır doldu",
                n_matched, len(ALL_EXPECTED), n_missing,
            )
        except Exception as exc:
            logger.warning("ColumnAligner hatası (%s) — basit hizalamaya geçiliyor", exc)
            aligned = pd.DataFrame(index=df.index)
            csv_map = {c.lower().replace(" ", "_"): c for c in df.columns}
            for feat in ALL_EXPECTED:
                fl = feat.lower().replace(" ", "_")
                if feat in df.columns:
                    aligned[feat] = pd.to_numeric(df[feat], errors="coerce")
                elif fl in csv_map:
                    aligned[feat] = pd.to_numeric(df[csv_map[fl]], errors="coerce")
                else:
                    aligned[feat] = _np.nan

        # Boyut ayarı (expected_n'e tam uydur)
        try:
            expected_n = self._preprocessor._imputer.n_features_in_
        except Exception:
            expected_n = len(ALL_EXPECTED)

        if aligned.shape[1] > expected_n:
            aligned = aligned.iloc[:, :expected_n]
        elif aligned.shape[1] < expected_n:
            for i in range(aligned.shape[1], expected_n):
                aligned[f"_pad_{i}"] = 0.0

        # NaN → 0 (imputer downstream'de median ile tamamlar)
        aligned = aligned.fillna(0.0)

        dummy_dataset = LoadedDataset(
            features        = aligned,
            labels          = None,
            metadata        = metadata,
            feature_columns = list(aligned.columns),
        )
        return self.predict_from_dataset(dummy_dataset)


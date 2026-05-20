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


def _compute_clinical_flag(score: np.ndarray, is_mc_dropout: bool = True) -> np.ndarray:
    """
    Klinik karar bayragi hesapla.

    MC-Dropout modunda: score = uncertainty (dusuk = iyi)
    Klasik modunda:     score = confidence fraction (yuksek = iyi)
    """
    if is_mc_dropout:
        return np.where(
            score > 0.30,
            "⚠️ Uzman Degerlendirmesi Gerekli",
            np.where(score <= 0.15, "✅ Yuksek Guven", "\U0001f536 Orta Guven"),
        )
    else:
        return np.where(
            score < 0.70,
            "⚠️ Uzman Degerlendirmesi Gerekli",
            np.where(score >= 0.90, "✅ Yuksek Guven", "\U0001f536 Orta Guven"),
        )


def _build_gnn_loader(
    preprocessor: VariantPreprocessor,
    X_scaled: np.ndarray,
    batch_size: int,
) -> GeoDataLoader:
    graphs = [preprocessor.row_to_graph(row) for row in X_scaled]
    return GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)


def _build_safe_sequence_tensors(
    gnn_model,
    n_samples: int,
    nuc_sequences,
    aa_sequences,
):
    """
    TD-006 fail-safe: multimodal GNN için sekans tensörlerini güvenle üret.

    Aşağıdaki senaryolarda çökmeden devam eder:

      1. ``nuc_sequences`` veya ``aa_sequences`` ``None``       → padding tensor üret
      2. Liste boş veya satır sayısı eşleşmiyor                  → eksik satırları padla
      3. Tokenization sırasında hata oluşuyor                    → padding tensor + uyarı
      4. Sekanslar hatalı tip / NaN / boş string içeriyor         → ``str()`` cast + unknown=5/21
      5. GNN multimodal aktif ama veri yok                       → uyarı logla, padding gönder

    Returns
    -------
    (nuc_ids, aa_ids) : torch.Tensor | None  ikisi de None ise multimodal devre dışı
    """
    import torch

    from src.features.multimodal_encoder import (
        AA_SEQ_LEN,
        NUC_SEQ_LEN,
        tokenize_amino_acids,
        tokenize_nucleotides,
    )

    device = next(gnn_model.parameters()).device

    # Tüm sekanslar yok ya da uzunluk eşleşmiyor → zero padding
    nuc_seqs_safe: list[str] = []
    aa_seqs_safe:  list[str] = []

    if nuc_sequences is None or len(nuc_sequences) != n_samples:
        if nuc_sequences is None:
            logger.warning(
                "TD-006 fail-safe: Nuc_Context yok — multimodal model padding "
                "tokenları kullanacak."
            )
        else:
            logger.warning(
                "TD-006 fail-safe: Nuc_Context uzunluğu (%d) feature satır "
                "sayısından (%d) farklı — eksik satırlar padlenecek.",
                len(nuc_sequences), n_samples,
            )
        nuc_seqs_safe = [""] * n_samples
    else:
        nuc_seqs_safe = [str(s) if s is not None else "" for s in nuc_sequences]

    if aa_sequences is None or len(aa_sequences) != n_samples:
        if aa_sequences is None:
            logger.warning(
                "TD-006 fail-safe: AA_Context yok — multimodal model padding "
                "tokenları kullanacak."
            )
        aa_seqs_safe = [""] * n_samples
    else:
        aa_seqs_safe = [str(s) if s is not None else "" for s in aa_sequences]

    # Tokenize defansif olarak — hata olursa zero tensor üret
    try:
        nuc_arr = tokenize_nucleotides(nuc_seqs_safe)
        nuc_ids = torch.tensor(nuc_arr, dtype=torch.long).to(device)
    except Exception as exc:
        logger.warning(
            "TD-006 fail-safe: Nuc tokenization başarısız (%s) — "
            "zero-padding tensor kullanılacak.", exc,
        )
        nuc_ids = torch.zeros(n_samples, NUC_SEQ_LEN, dtype=torch.long, device=device)

    try:
        aa_arr = tokenize_amino_acids(aa_seqs_safe)
        aa_ids = torch.tensor(aa_arr, dtype=torch.long).to(device)
    except Exception as exc:
        logger.warning(
            "TD-006 fail-safe: AA tokenization başarısız (%s) — "
            "zero-padding tensor kullanılacak.", exc,
        )
        aa_ids = torch.zeros(n_samples, AA_SEQ_LEN, dtype=torch.long, device=device)

    return nuc_ids, aa_ids


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
        self._check_provenance()
        return self

    def _check_provenance(self) -> None:
        """PROVENANCE.json varsa oku; sentetik veriyle eğitilmişse uyar."""
        import json as _json
        prov_path = self.store.model_dir / "PROVENANCE.json"
        if not prov_path.exists():
            return
        try:
            with open(prov_path, encoding="utf-8") as _fh:
                prov = _json.load(_fh)
        except Exception:
            return
        if prov.get("real_data_received") is False:
            logger.warning("=" * 70)
            logger.warning("UYARI: Mevcut model ağırlıkları sentetik/pilot veriyle")
            logger.warning("eğitilmiştir. Tahminler gerçek klinik data üzerinde")
            logger.warning("geçersizdir. Gerçek veri geldiğinde yeniden eğitin:")
            logger.warning("  %s", prov.get("retrain_command", "python main.py --mode train"))
            logger.warning("=" * 70)

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

        # ── Build sequence tensors for multimodal GNN (TD-006 fail-safe) ──
        from src.core.models.gnn import VariantGATv2GNN
        nuc_ids = None
        aa_ids = None

        is_multimodal = (
            isinstance(self._ensemble.gnn, VariantGATv2GNN)
            and getattr(self._ensemble.gnn, "use_multimodal", False)
        )

        if is_multimodal:
            nuc_ids, aa_ids = _build_safe_sequence_tensors(
                gnn_model     = self._ensemble.gnn,
                n_samples     = X_scaled.shape[0],
                nuc_sequences = dataset.nuc_sequences,
                aa_sequences  = dataset.aa_sequences,
            )
        
        # Load F1-optimal threshold saved during training (TEKNOFEST §7.3)
        # Falls back to config value (0.50) when no saved threshold exists.
        threshold = self.store.load_threshold(default=cfg.thresholds.classification)

        if isinstance(self._ensemble.gnn, VariantGATv2GNN):
            # MC-Dropout ile belirsizlik tahmini (multimodal token'lar dahil)
            preds, raw_proba, uncertainty = self._ensemble.predict_with_uncertainty(
                X_scaled, n_iter=10, threshold=threshold,
                nuc_ids=nuc_ids, aa_ids=aa_ids,
            )
            # NaN uncertainty koruması
            if np.isnan(uncertainty).any():
                logger.warning("NaN uncertainty degerleri tespit edildi; 0.0 ile dolduruluyor.")
                uncertainty = np.nan_to_num(uncertainty, nan=0.0)
            confidence = ((1.0 - uncertainty) * 100).round(2)
            clinical_flag = _compute_clinical_flag(uncertainty, is_mc_dropout=True)
        else:
            # Klasik tahmin (GATv2 olmayan path)
            preds, raw_proba = self._ensemble.predict(
                X_scaled, threshold=threshold,
                nuc_ids=nuc_ids, aa_ids=aa_ids,
            )
            confidence = (np.max(raw_proba, axis=1) * 100).round(2)
            conf_frac = np.max(raw_proba, axis=1)
            clinical_flag = _compute_clinical_flag(conf_frac, is_mc_dropout=False)

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
        # F1-optimal threshold'u kullan (eğitimde kayıt edilen) — config sabit değeri değil
        result["High_Risk"]     = cal_proba[:, 1] >= threshold
        result["Clinical_Flag"] = clinical_flag

        # ── OOD Tespiti — eğitim dağılımından sapma kontrolü ─────────────────
        # Referans dağılım: eğitimde kayıt edilen OOD dedektörü (models/ood_detector.pkl).
        # Yoksa devre dışı — inference verisiyle fit etmek YANLIŞTIR (tüm noktalar
        # kendi dağılımında olur → anlamsız skor).
        try:
            from src.scientific.ood_detector import OODDetector
            _ood_path = self.store.model_dir / "ood_detector.pkl"
            if _ood_path.exists():
                import joblib as _jl
                _ood_det = _jl.load(str(_ood_path))
                _ood_out = _ood_det.detect(X_scaled)
                result["OOD_Score"] = _ood_out["ood_scores"].round(3)
                result["OOD_Flag"]  = _ood_out["ood_flags"]
            else:
                logger.info(
                    "OOD dedektörü bulunamadı (%s). "
                    "Eğitimde `python main.py --mode train` çalıştırıldıktan sonra "
                    "kayıt edilecek. OOD_Score NaN olarak bırakılıyor.",
                    _ood_path,
                )
                result["OOD_Score"] = np.nan
                result["OOD_Flag"]  = False
        except Exception as _ood_exc:
            logger.warning(
                "OOD tespiti basarisiz (OOD_Score/OOD_Flag kolonlari olmayacak): %s",
                _ood_exc,
            )
            result["OOD_Score"] = np.nan
            result["OOD_Flag"]  = False

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
                raise ValueError(
                    f"predict_from_dataframe: DataFrame'de {feature_df.shape[1]} sayısal "
                    f"özellik var, model {expected_n} bekliyor. "
                    f"Kolon listesini kontrol edin veya predict_from_csv kullanın."
                )

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

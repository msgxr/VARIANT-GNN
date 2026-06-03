"""
src/inference/external_validation_runner.py
Offline external validation runner for TEKNOFEST 2026 competition.

Reads a blind CSV, sanitizes it, aligns features, runs ensemble inference
without any training, and writes predictions in the canonical schema.

Rules enforced:
  - NO training or fitting on blind data.
  - NO internet access.
  - NO label leakage.
  - Metrics computed only when labels are explicitly present (local validation mode).
  - Output conforms to prediction_schema.PREDICTION_COLUMNS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd

from src.data.column_aligner import ColumnAligner
from src.data.competition_sanitizer import CompetitionSanitizer
from src.inference.artifact_loader import ArtifactLoader
from src.inference.prediction_schema import (
    build_prediction_frame,
    validate_prediction_frame,
)
from src.inference.triage import TriageEngine

logger = logging.getLogger(__name__)

_METADATA_COLS = ("Variant_ID", "Panel", "Nuc_Context", "AA_Context")


class ExternalValidationRunner:
    """
    End-to-end offline inference for competition external validation.

    Parameters
    ----------
    model_dir   : Directory with serialised model artifacts.
    reports_dir : Directory to write leakage and inference reports.
    """

    def __init__(
        self,
        model_dir: Path,
        reports_dir: Path = Path("reports"),
    ) -> None:
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)

        loader = ArtifactLoader(self.model_dir)
        loader.validate_artifacts()

        self._preprocessor = loader.load_preprocessor()
        self._ensemble = loader.load_ensemble()
        self._calibrator = loader.load_calibrator()
        self._threshold = loader.load_threshold(default=0.5)
        self._feature_names: Optional[list[str]] = loader.load_feature_names()
        self._model_version: str = loader.load_model_version()

        self._sanitizer = CompetitionSanitizer(
            competition_mode=True,
            strict_leakage=False,
            reports_dir=self.reports_dir,
        )
        self._triage = TriageEngine()

        # OOD detector — opsiyonel; eksikse uyarı verilir, çıkarım durmaz
        self._ood_detector = None
        _ood_path = self.model_dir / "ood_detector.pkl"
        if _ood_path.exists():
            try:
                from src.scientific.ood_detector import OODDetector

                self._ood_detector = OODDetector.load(_ood_path)
                logger.info("OOD detector yüklendi → %s", _ood_path)
            except Exception as _ood_exc:
                logger.warning("OOD detector yüklenemedi (%s); OOD analizi atlandı.", _ood_exc)
        else:
            logger.info("ood_detector.pkl bulunamadı — OOD analizi devre dışı.")

        logger.info(
            "ExternalValidationRunner ready — model_dir=%s  threshold=%.3f",
            self.model_dir,
            self._threshold,
        )

    # ------------------------------------------------------------------
    def _extract_metadata(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate metadata from feature columns."""
        meta_cols = [c for c in _METADATA_COLS if c in df.columns]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        return df[meta_cols].copy(), df[feat_cols].copy()

    # ------------------------------------------------------------------
    def _align_features(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Align incoming feature columns to training-time feature names."""
        if self._feature_names is None:
            # Best-effort: use numeric columns only
            numeric_df = feat_df.select_dtypes(include="number")
            logger.warning("No feature_names.json found; using %d numeric columns.", numeric_df.shape[1])
            return cast(np.ndarray, numeric_df.values.astype(float))

        aligner = ColumnAligner(expected_columns=self._feature_names)
        aligned_df, report = aligner.apply(feat_df)
        for msg in report.warnings():
            logger.warning(msg)

        # NaN'i KORU — eğitimle simetri (§3.2 'missing≠0'). Eskiden burada
        # aligned_df.fillna(0.0) vardı: feature_names.json shipped edildiğinde bu
        # dal aktifleşir ve VariantPreprocessor.transform'daki eksiklik-göstergesi
        # bloğunu (np.isnan(_Xraw_t), bkz. features/preprocessing.py:496) TÜMÜYLE
        # SIFIR yapardı — eğitim ham NaN görürken çıkarım 0.0 görür (asimetri).
        # LATENT foot-gun: bugün feature_names yok → erken-dönüş dalı (yukarıda)
        # zaten NaN korur, canonical skor bu siteden etkilenmedi. Hizalanmış
        # frame'in NaN'leri (gerçek + eksik-kolon) olduğu gibi geçilir; ana
        # Preprocessor'ın iç imputer'ı eğitim medyanlarıyla doldurur.
        return cast(np.ndarray, aligned_df.values.astype(float))

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: Path,
        output_path: Path,
        *,
        local_validation: bool = False,
    ) -> pd.DataFrame:
        """
        Run external validation: load blind CSV → sanitize → predict → save.

        Parameters
        ----------
        input_path       : Path to blind test CSV.
        output_path      : Path to write predictions CSV.
        local_validation : If True and 'Label' column present, compute metrics
                           but do NOT use labels as model input.

        Returns
        -------
        predictions DataFrame in canonical schema.
        """
        # Block ClinVar API for the duration of inference (TEKNOFEST §3.2)
        from src.explainability.clinvar_api import set_inference_mode

        set_inference_mode(True)

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        logger.info("ExternalValidationRunner: reading %s", input_path)
        raw_df = pd.read_csv(input_path)
        logger.info("Raw shape: %s  columns: %d", raw_df.shape, len(raw_df.columns))

        # ── Labels extracted before sanitization for local_validation ─────
        local_labels: Optional[np.ndarray] = None
        if local_validation and "Label" in raw_df.columns:
            local_labels = raw_df["Label"].values.copy()
            logger.info("Local validation mode: labels extracted (%d rows).", len(local_labels))

        # ── Sanitize (drops coordinate + label-like columns) ──────────────
        clean_df, leakage_report = self._sanitizer.sanitize_inference(raw_df)
        if not leakage_report.is_clean:
            logger.warning("Leakage issues found and resolved: %s", leakage_report.to_dict())

        # ── Extract metadata ───────────────────────────────────────────────
        meta_df, feat_df = self._extract_metadata(clean_df)

        # Ensure Variant_ID present
        if "Variant_ID" not in meta_df.columns:
            meta_df["Variant_ID"] = [f"V{i}" for i in range(len(clean_df))]

        # Ensure Panel present
        if "Panel" not in meta_df.columns:
            meta_df["Panel"] = "General"

        # ── Feature alignment ──────────────────────────────────────────────
        X = self._align_features(feat_df)
        logger.info("Aligned feature matrix shape: %s", X.shape)

        # ── Preprocessing (transform only — NO fit) ───────────────────────
        X_scaled = self._preprocessor.transform(X)

        # ── Ensemble prediction ────────────────────────────────────────────
        try:
            preds_bin, raw_proba, uncertainty = self._ensemble.predict_with_uncertainty(
                X_scaled, n_iter=10, threshold=self._threshold
            )
            pathogenic_prob = raw_proba[:, 1]
        except (AttributeError, TypeError):
            preds_bin, raw_proba = self._ensemble.predict(X_scaled, threshold=self._threshold)
            pathogenic_prob = raw_proba[:, 1] if raw_proba.ndim == 2 else raw_proba
            uncertainty = np.zeros(len(preds_bin), dtype=float)

        # ── NaN uncertainty koruması ──────────────────────────────────────
        if np.isnan(uncertainty).any():
            logger.warning("NaN uncertainty degerleri tespit edildi; 0.0 ile dolduruluyor.")
            uncertainty = np.nan_to_num(uncertainty, nan=0.0)

        # ── Calibration ────────────────────────────────────────────────────
        if self._calibrator is not None:
            try:
                cal_proba = self._calibrator.transform(raw_proba)
                calibrated_risk = cal_proba[:, 1] if cal_proba.ndim == 2 else cal_proba
            except Exception as _cal_exc:
                logger.warning("Kalibrasyon basarisiz (%s); ham olasilik kullaniliyor.", _cal_exc)
                calibrated_risk = pathogenic_prob
        else:
            calibrated_risk = pathogenic_prob

        # ── OOD tespiti ────────────────────────────────────────────────────
        ood_scores = np.zeros(len(X_scaled), dtype=float)
        ood_flags = np.zeros(len(X_scaled), dtype=bool)
        if self._ood_detector is not None:
            try:
                ood_report = self._ood_detector.detect(X_scaled)
                ood_scores = ood_report.get("ood_scores", ood_scores)
                ood_flags = ood_report.get("ood_flags", ood_flags)
                n_ood = int(ood_flags.sum())
                if n_ood > 0:
                    logger.warning(
                        "OOD uyarı: %d/%d varyant eğitim dağılımı dışında "
                        "(ood_score > eşik). Bu varyantlarda tahmin güvenilirliği düşük.",
                        n_ood,
                        len(X_scaled),
                    )
            except Exception as _ood_exc:
                logger.warning("OOD detect başarısız (%s); atlandı.", _ood_exc)

        # ── Triage flags ───────────────────────────────────────────────────
        clinical_flags = self._triage.assign_flags(
            pathogenic_prob=pathogenic_prob,
            uncertainty=uncertainty,
            threshold=self._threshold,
        )

        # ── Build output ───────────────────────────────────────────────────
        predictions_df = build_prediction_frame(
            variant_ids=meta_df["Variant_ID"],
            panels=meta_df["Panel"],
            pathogenic_prob=pathogenic_prob,
            calibrated_risk=calibrated_risk,
            uncertainty=uncertainty,
            clinical_flags=clinical_flags,
            threshold=self._threshold,
            model_version=self._model_version,
            ood_scores=ood_scores,
            ood_flags=ood_flags,
        )
        validate_prediction_frame(predictions_df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        logger.info("Predictions saved to %s  shape=%s", output_path, predictions_df.shape)

        # ── Local validation metrics (optional, labels not used as input) ──
        if local_validation and local_labels is not None:
            self._log_local_metrics(predictions_df, local_labels)

        return predictions_df

    # ------------------------------------------------------------------
    def _log_local_metrics(self, preds_df: pd.DataFrame, labels: np.ndarray) -> None:
        """Compute and log validation metrics — never used for model fitting."""
        from sklearn.metrics import f1_score, roc_auc_score

        pred_binary = (preds_df["Prediction"] == "Pathogenic").astype(int).values
        prob = preds_df["Pathogenic_Probability"].values

        label_map = {"pathogenic": 1, "benign": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
        y = np.array([label_map.get(str(v).strip().lower(), int(float(v))) for v in labels], dtype=int)

        f1 = f1_score(y, pred_binary, average="binary", pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = float("nan")

        logger.info(
            "[LocalValidation] F1=%.4f  AUC=%.4f  n=%d",
            f1,
            auc,
            len(y),
        )

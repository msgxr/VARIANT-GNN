"""
src/api/export.py
TEKNOFEST 2026 — Jüri uyumlu deterministik tahmin exportu.

Garanti edilen 7 kolon (submission/predictions.csv):
  Variant_ID            | Varyant kimliği (yoksa "VAR_<index>")
  prediction_label      | 1 = Pathogenic, 0 = Benign
  pathogenic_probability| Ham ensemble P(Pathogenic) [0-1]
  calibrated_risk       | Kalibre edilmiş risk skoru [0-100]
  confidence_level      | Güven yüzdesi [0-100]
  uncertainty_score     | MC-Dropout std (varsa) yoksa max_prob tabanlı
  expert_review_flag    | True = Uzman değerlendirmesi gerekli
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Jüri sözleşmesi: kesinlikle bu 7 kolon + bu sırada
JURY_COLUMNS = [
    "Variant_ID",
    "prediction_label",
    "pathogenic_probability",
    "calibrated_risk",
    "confidence_level",
    "uncertainty_score",
    "expert_review_flag",
]

# Minimal jüri modu (opt-in): tek doğruluk kaynağı submission_validator.
# Resmi format UNVERIFIED — yalnız Q&A teyitli ikili çekirdek yazılır.
from src.scientific.submission_validator import JURY_MINIMAL_COLUMNS  # noqa: E402


def _ensure_jury_columns(df: pd.DataFrame, minimal: bool = False) -> pd.DataFrame:
    """Pipeline çıktısından garanti edilen kolonları oluşturur.

    minimal=False (varsayılan): 7 garantili JURY_COLUMNS — mevcut davranış.
    minimal=True : yalnız JURY_MINIMAL_COLUMNS = [Variant_ID, prediction_label]
                   (klinik-çağrışımlı kolon İÇERMEZ — §10 etik, opt-in).
    """
    out = pd.DataFrame()

    # 1. Variant_ID
    if "Variant_ID" in df.columns:
        out["Variant_ID"] = df["Variant_ID"].astype(str)
    else:
        out["Variant_ID"] = [f"VAR_{i:06d}" for i in range(len(df))]

    # 2. prediction_label  (1/0)
    if "Prediction" in df.columns:
        out["prediction_label"] = df["Prediction"].map({"Pathogenic": 1, "Benign": 0}).fillna(0).astype(int)
    elif "Predicted_Label" in df.columns:
        out["prediction_label"] = df["Predicted_Label"].astype(int)
    else:
        out["prediction_label"] = 0

    # 3. pathogenic_probability [0-1]
    if "Probability" in df.columns:
        out["pathogenic_probability"] = df["Probability"].round(6)
    else:
        out["pathogenic_probability"] = out["prediction_label"].astype(float)

    # 4. calibrated_risk [0-100]
    if "Calibrated_Risk" in df.columns:
        out["calibrated_risk"] = df["Calibrated_Risk"].round(2)
    else:
        out["calibrated_risk"] = (out["pathogenic_probability"] * 100).round(2)

    # 5. confidence_level [0-100]
    if "Confidence" in df.columns:
        out["confidence_level"] = df["Confidence"].round(2)
    else:
        p = out["pathogenic_probability"].values
        out["confidence_level"] = (np.maximum(p, 1 - p) * 100).round(2)

    # 6. uncertainty_score  [0-1]  (0=kesin, 1=tamamen belirsiz)
    # MC-Dropout std'yi tercih et; yoksa güven tersini kullan.
    # OOD_Score ≠ uncertainty — OOD dağılım dışılığı ölçer, model belirsizliği değil.
    if "Confidence" in df.columns:
        # Güvenden türet: yüksek güven → düşük belirsizlik
        out["uncertainty_score"] = (1 - df["Confidence"].clip(0, 100) / 100).round(4)
    else:
        out["uncertainty_score"] = (1 - out["confidence_level"] / 100).round(4)

    # 7. expert_review_flag  (bool)
    if "Clinical_Flag" in df.columns:
        out["expert_review_flag"] = df["Clinical_Flag"].str.contains("Uzman", na=False)
    else:
        # Belirsizlik > 0.30 veya risk 30-70 arası → gri bölge
        unc = out["uncertainty_score"].values
        risk = out["calibrated_risk"].values
        out["expert_review_flag"] = (unc > 0.30) | ((risk > 30) & (risk < 70))

    if minimal:
        return out[JURY_MINIMAL_COLUMNS]
    return out[JURY_COLUMNS]


def export_predictions(
    df_result: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "predictions",
    submission_path: str | Path | None = None,
    minimal: bool = False,
) -> dict[str, Path | None]:
    """
    TEKNOFEST 2026 jüri uyumlu tahmin exportu.

    Üretilen dosyalar:
      1. ``{prefix}_jury.csv``  — 7 garantili kolon (jüri sözleşmesi)
      2. ``{prefix}_full.csv``  — tüm pipeline çıktısı
      3. submission/predictions.csv — ``--output`` ile belirtilen path (opsiyonel)

    minimal=True (opt-in, §10 etik): jüriye giden çıktılar (jury.csv + submission)
    yalnız [Variant_ID, prediction_label] içerir — klinik-çağrışımlı kolon yok.
    full.csv YEREL tanı dosyasıdır (jüriye gitmez), ham tanı kolonlarını korur.

    Returns dict: {'jury', 'full', 'submission'} → Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jury_df = _ensure_jury_columns(df_result, minimal=minimal)

    # ── 1. Jury CSV (7 garantili kolon) ──────────────────────────────
    jury_path = output_dir / f"{prefix}_jury.csv"
    jury_df.to_csv(jury_path, index=False)
    logger.info("Jury CSV (7-kolon) → %s (%d satır)", jury_path, len(jury_df))

    # ── 2. Full CSV (tüm kolonlar) ────────────────────────────────────
    full_df = df_result.copy()
    # Garantili kolonları öne al
    for col, vals in jury_df.items():
        full_df[col] = vals.values
    lead = [c for c in JURY_COLUMNS if c in full_df.columns]
    rest = [c for c in full_df.columns if c not in lead]
    full_path = output_dir / f"{prefix}_full.csv"
    full_df[lead + rest].to_csv(full_path, index=False)
    logger.info("Full CSV → %s (%d satır, %d kolon)", full_path, len(full_df), len(full_df.columns))

    # ── 3. Submission path (--output argümanı) ────────────────────────
    sub_path = None
    if submission_path is not None:
        sub_path = Path(submission_path)
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        jury_df.to_csv(sub_path, index=False)
        logger.info("Submission CSV → %s", sub_path)

    # ── Özet istatistikler ────────────────────────────────────────────
    n_path = int((jury_df["prediction_label"] == 1).sum())
    n_ben = int((jury_df["prediction_label"] == 0).sum())
    if "expert_review_flag" in jury_df.columns:
        n_exp = int(jury_df["expert_review_flag"].sum())
        logger.info(
            "Özet: %d Patojenik | %d Benign | %d Uzman Değerlendirmesi",
            n_path,
            n_ben,
            n_exp,
        )
    else:
        logger.info("Özet (minimal): %d Patojenik | %d Benign", n_path, n_ben)

    return {"jury": jury_path, "full": full_path, "submission": sub_path}

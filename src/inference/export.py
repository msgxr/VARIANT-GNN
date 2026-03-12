"""
src/inference/export.py
Standardised TEKNOFEST 2026 jury-compliant prediction CSV exporter.

Final output column spec (jüri beklentisi):
  ┌────────────────┬─────────────────────────────────────────────┐
  │ Variant_ID     │ Varyant kimliği (varsa)                     │
  │ Prediction     │ "Pathogenic" veya "Benign"                  │
  │ Predicted_Label│ 1 (Pathogenic) veya 0 (Benign)              │
  │ Probability    │ Patojenik olasılığı [0.0-1.0]               │
  │ Confidence     │ Tahmin güven yüzdesi [0-100]                │
  │ Panel          │ Panel bilgisi (varsa)                       │
  └────────────────┴─────────────────────────────────────────────┘

Ek olarak, jüri için sadece minimum kolonlar içeren
``_jury.csv`` ve tam detaylı ``_full.csv`` dosyaları üretilir.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def export_predictions(
    df_result: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "predictions",
) -> dict[str, Path]:
    """
    Export predictions in standardised formats for TEKNOFEST 2026 jury.

    Generates two files:
        1. ``{prefix}_jury.csv``  — minimal columns (Variant_ID, Prediction, Predicted_Label)
        2. ``{prefix}_full.csv``  — all columns including probabilities, confidence, risk

    Parameters
    ----------
    df_result : Prediction DataFrame from InferencePipeline.
    output_dir : Directory to save output CSVs.
    prefix : Filename prefix (default: "predictions").

    Returns
    -------
    dict with keys 'jury' and 'full' mapping to saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Jury-minimal CSV ──────────────────────────────────────────
    jury_cols = []
    if "Variant_ID" in df_result.columns:
        jury_cols.append("Variant_ID")

    jury_cols.append("Prediction")

    # Add numeric label column for easy metric computation
    df_export = df_result.copy()
    df_export["Predicted_Label"] = df_export["Prediction"].map(
        {"Pathogenic": 1, "Benign": 0}
    )
    jury_cols.append("Predicted_Label")

    jury_path = output_dir / f"{prefix}_jury.csv"
    df_export[jury_cols].to_csv(jury_path, index=False)
    logger.info("Jury-format predictions → %s (%d rows)", jury_path, len(df_export))

    # ── 2. Full-detail CSV ───────────────────────────────────────────
    full_cols = list(jury_cols)
    for col in ["Probability", "Calibrated_Risk", "Confidence", "High_Risk", "Panel"]:
        if col in df_export.columns:
            full_cols.append(col)
    # Include any remaining metadata columns not yet added
    for col in df_export.columns:
        if col not in full_cols:
            full_cols.append(str(col))

    full_path = output_dir / f"{prefix}_full.csv"
    df_export[full_cols].to_csv(full_path, index=False)
    logger.info("Full predictions → %s (%d rows, %d cols)", full_path, len(df_export), len(full_cols))

    # ── 3. Summary statistics  ───────────────────────────────────────
    n_path = len(df_export[df_export["Prediction"] == "Pathogenic"])
    n_ben = len(df_export[df_export["Prediction"] == "Benign"])
    logger.info(
        "Prediction summary: %d Pathogenic, %d Benign (total %d)",
        n_path, n_ben, len(df_export),
    )

    return {"jury": jury_path, "full": full_path}

# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_export_jury_minimal.py
Locks the export_predictions jury-column contract (anonymous jury writer):
  - minimal=True  -> jury.csv + submission = EXACTLY [Variant_ID, prediction_label]
                     (int 0/1), no clinical-flavored cols (§10 etik)
  - default       -> all 7 JURY_COLUMNS (zero-regression guard)

SCOPE NOTE: minimal mode narrows the JÜRİYE GİDEN çıktıları (jury.csv + submission).
*_full.csv is a LOCAL diagnostic (not sent to the jury) and intentionally retains
df_result's raw columns. This test pins the jury-facing outputs only.

Pure unit test: in-memory DataFrame, no model/preprocessor, no skip gate.
"""

import pandas as pd
import pytest  # noqa: F401  (kept for parametrize-friendliness)

from src.api.export import JURY_COLUMNS, export_predictions
from src.scientific.submission_validator import JURY_MINIMAL_COLUMNS


def _df_result() -> pd.DataFrame:
    # Mimics InferencePipeline.predict_from_dataset output keys read by
    # _ensure_jury_columns: Variant_ID, Prediction ({Pathogenic,Benign}), Probability.
    return pd.DataFrame(
        {
            "Variant_ID": ["V1", "V2", "V3"],
            "Prediction": ["Pathogenic", "Benign", "Pathogenic"],
            "Probability": [0.91, 0.07, 0.83],
        }
    )


def test_export_minimal_exactly_two_cols(tmp_path):
    # FAILS pre-fix: no `minimal` kwarg -> TypeError. PASSES post-fix.
    paths = export_predictions(
        _df_result(),
        output_dir=tmp_path,
        prefix="p",
        submission_path=tmp_path / "sub.csv",
        minimal=True,
    )
    sub = pd.read_csv(paths["submission"])
    assert list(sub.columns) == JURY_MINIMAL_COLUMNS
    assert list(sub.columns) == ["Variant_ID", "prediction_label"]
    assert pd.api.types.is_integer_dtype(sub["prediction_label"])
    assert set(sub["prediction_label"].unique()) <= {0, 1}
    assert list(sub["prediction_label"]) == [1, 0, 1]
    for clin in (
        "calibrated_risk",
        "confidence_level",
        "expert_review_flag",
        "pathogenic_probability",
        "uncertainty_score",
    ):
        assert clin not in sub.columns
    # Jury CSV also narrowed (both jury-facing outputs clean).
    jury_csv = pd.read_csv(paths["jury"])
    assert list(jury_csv.columns) == ["Variant_ID", "prediction_label"]


def test_export_default_still_seven_cols(tmp_path):
    # Zero-regression guard: passes before AND after the fix.
    paths = export_predictions(
        _df_result(),
        output_dir=tmp_path,
        prefix="p",
        submission_path=tmp_path / "sub.csv",
    )
    sub = pd.read_csv(paths["submission"])
    assert list(sub.columns) == JURY_COLUMNS
    assert len(sub.columns) == 7
    for clin in ("calibrated_risk", "confidence_level", "expert_review_flag"):
        assert clin in sub.columns
    assert list(sub["prediction_label"]) == [1, 0, 1]

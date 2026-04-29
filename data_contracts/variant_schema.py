"""
data_contracts/variant_schema.py
TEKNOFEST 2026 — Pydantic v2 veri sözleşmeleri.

Şemalar:
  TrainVariantRow   — Eğitim satırı (Label zorunlu)
  PredictVariantRow — Tahmin satırı (Label opsiyonel)
  SubmissionRow     — 7-kolonlu jüri submission
  validate_dataframe — Türkçe hata mesajlı DataFrame doğrulama
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic import ValidationError
    _PYD = True
except ImportError:
    BaseModel = object  # type: ignore[assignment,misc]
    _PYD = False
    logger.warning("pydantic kurulu degil — schema dogrulama devre disi.")

NUMERIC_FEATURES: List[str] = [
    "Ref_Nucleotide","Alt_Nucleotide","Codon_Change_Type","AA_Grantham_Score",
    "GC_Content_Window","In_CpG_Site","Motif_Disruption_Score","AA_Polarity_Change",
    "AA_Hydrophobicity_Diff","AA_Mol_Weight_Diff","AA_Size_Diff","Protein_Impact_Score",
    "Delta_Solvent_Accessibility","Secondary_Structure_Disruption","GERP_RS",
    "PhyloP100way_vertebrate","phastCons100way_vertebrate","SiPhy_29way_logOdds",
    "Phylo_Diversity_Index","gnomAD_exomes_AF","gnomAD_exomes_AF_afr","gnomAD_exomes_AF_eur",
    "gnomAD_exomes_AF_eas","gnomAD_exomes_AF_sas","gnomAD_exomes_AF_amr","ExAC_AF",
    "SIFT_score","PolyPhen2_HDIV_score","PolyPhen2_HVAR_score","CADD_phred","REVEL_score",
    "MutPred2_score","VEST4_score","PROVEAN_score","MutationTaster_score","MetaSVM_score",
    "MetaLR_score","MCAP_score","In_Critical_Protein_Domain","Splice_Site_Distance",
    "Is_Exonic","Exon_Conservation_Ratio","OMIM_Disease_Gene",
]
VALID_PANELS = {"General","Hereditary_Cancer","PAH","CFTR"}
VALID_LABELS = {0,1,"0","1","0.0","1.0","Pathogenic","Benign",
                "likely pathogenic","likely benign","Likely Pathogenic","Likely Benign"}


class DatasetValidationResult:
    def __init__(self) -> None:
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self.stats:    Dict[str, Any] = {}

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines = [f"Dogrulama: {'GECTI' if self.is_valid else 'BASARISIZ'}"]
        for e in self.errors:   lines.append(f"  HATA   : {e}")
        for w in self.warnings: lines.append(f"  UYARI  : {w}")
        return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame, mode: str = "predict",
                       strict: bool = False) -> DatasetValidationResult:
    result = DatasetValidationResult()
    if df is None or len(df) == 0:
        result.errors.append("Veri seti bos. En az 1 satir gereklidir.")
        if strict: raise ValueError(str(result))
        return result

    result.stats.update({"n_rows": len(df), "n_cols": len(df.columns),
                          "null_pct": float(df.isnull().mean().mean() * 100)})

    if mode == "submission":
        required = ["Variant_ID","prediction_label","pathogenic_probability",
                    "calibrated_risk","confidence_level","uncertainty_score","expert_review_flag"]
        for col in required:
            if col not in df.columns:
                result.errors.append(f"Submission icin zorunlu '{col}' kolonu eksik.")

    elif mode == "train":
        lc = next((c for c in ["Label","label"] if c in df.columns), None)
        if lc is None:
            result.errors.append("Egitim verisi icin 'Label' kolonu zorunludur.")
        else:
            bad = df[lc][~df[lc].astype(str).isin([str(v) for v in VALID_LABELS])]
            if len(bad) > 0:
                result.errors.append(
                    f"'Label' kolonunda {len(bad)} gecersiz deger: {bad.head(3).tolist()}")

    found = [f for f in NUMERIC_FEATURES if f in df.columns]
    if len(found) < 5:
        result.warnings.append(
            f"Yalnizca {len(found)}/{len(NUMERIC_FEATURES)} ozellik bulundu. "
            "ColumnAligner anonim eslestirme devreye girecek.")

    null_pct = result.stats["null_pct"]
    if null_pct > 0:
        result.warnings.append(
            f"Veri setinde %{null_pct:.1f} eksik deger — "
            "otomatik imputation ile devam ediliyor.")

    if len(df) == 1:
        result.warnings.append("Tek satir tespit edildi. BatchNorm eval-mode'a gecilecek.")

    if len(df) > 5000:
        result.warnings.append(f"{len(df):,} satir — buyuk batch bellek optimizasyonu devrede.")

    for e in result.errors:   logger.error("Sema hatasi: %s", e)
    for w in result.warnings: logger.warning("Sema uyarisi: %s", w)

    if strict and not result.is_valid:
        raise ValueError(str(result))
    return result


validate_dataset    = validate_dataframe
SchemaValidationResult = DatasetValidationResult

__all__ = [
    "DatasetValidationResult","SchemaValidationResult","validate_dataframe",
    "validate_dataset","NUMERIC_FEATURES","VALID_PANELS","VALID_LABELS",
]

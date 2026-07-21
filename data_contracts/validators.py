# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd

class VariantMetadata(BaseModel):
    Variant_ID: str = Field(..., description="Unique variant identifier")
    Gene: Optional[str] = None
    Panel: str = Field("General", description="Genomic panel name")

class PredictInput(BaseModel):
    variants: List[dict]
    
    @field_validator('variants')
    @classmethod
    def check_cols(cls, v: List[dict]) -> List[dict]:
        if not v:
            raise ValueError("Varyant listesi boş olamaz.")
        return v

class SubmissionOutput(BaseModel):
    Variant_ID: str
    prediction_label: int = Field(..., ge=0, le=1)
    pathogenic_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_risk: float = Field(..., ge=0.0, le=100.0)
    confidence_level: float = Field(..., ge=0.0, le=100.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    expert_review_flag: bool

def validate_dataframe(df: pd.DataFrame, model_class: type[BaseModel]) -> None:
    """Helper to validate a pandas DataFrame against a Pydantic model."""
    try:
        data = df.to_dict(orient="records")
        for row in data:
            model_class(**row)
    except Exception as e:
        raise ValueError(f"Veri sözleşmesi hatası: {str(e)}")

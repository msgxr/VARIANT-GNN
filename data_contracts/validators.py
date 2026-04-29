from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
import pandas as pd

class VariantMetadata(BaseModel):
    Variant_ID: str = Field(..., description="Unique variant identifier")
    Gene: Optional[str] = None
    Panel: str = Field("General", description="Genomic panel name")

class PredictInput(BaseModel):
    variants: List[dict]
    
    @validator('variants')
    def check_cols(cls, v):
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

def validate_dataframe(df: pd.DataFrame, model_class: type[BaseModel]):
    """Helper to validate a pandas DataFrame against a Pydantic model."""
    try:
        data = df.to_dict(orient="records")
        for row in data:
            model_class(**row)
    except Exception as e:
        raise ValueError(f"Veri sözleşmesi hatası: {str(e)}")

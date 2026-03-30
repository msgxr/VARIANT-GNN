"""
data_contracts/variant_schema.py
Pydantic v2 schema for incoming variant CSV rows.
Validates that numeric features are present, metadata columns are preserved,
and target labels (when present) are in a known set.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Row-level model (used for single-sample validation)
# ---------------------------------------------------------------------------

class VariantRow(BaseModel):
    """A single variant row — metadata fields are optional, features are dynamic."""

    model_config = {"extra": "allow"}  # allow arbitrary numeric features

    Variant_ID: Optional[str] = Field(default=None)
    Label: Optional[str] = Field(default=None)

    @field_validator("Label", mode="before")
    @classmethod
    def normalise_label(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v).strip().lower()


# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------

KNOWN_LABEL_VALUES = {
    "pathogenic", "likely pathogenic", "benign", "likely benign",
    "1", "1.0", "0", "0.0",
}

MINIMUM_NUMERIC_FEATURES = 1


class DatasetValidationResult(BaseModel):
    is_valid: bool
    numeric_feature_columns: List[str]
    metadata_columns: List[str]
    label_column: Optional[str]
    n_samples: int
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = "Label",
    id_columns: Optional[List[str]] = None,
    non_feature_columns: Optional[List[str]] = None,
) -> DatasetValidationResult:
    """
    Validate a loaded DataFrame against the variant schema.

    Returns a ``DatasetValidationResult`` — callers should check
    ``result.is_valid`` before proceeding.
    """
    id_columns = id_columns or ["Variant_ID"]
    non_feature_columns = non_feature_columns or []
    warnings: List[str] = []
    errors: List[str] = []

    if df.empty:
        errors.append("DataFrame is empty.")
        return DatasetValidationResult(
            is_valid=False,
            numeric_feature_columns=[],
            metadata_columns=[],
            label_column=None,
            n_samples=0,
            errors=errors,
        )

    # Identify metadata columns (id + label + non_feature_columns)
    metadata_cols: List[str] = []
    for col in id_columns:
        if col in df.columns:
            metadata_cols.append(col)

    # Non-feature columns (Panel, Nuc_Context, AA_Context etc.)
    for col in non_feature_columns:
        if col in df.columns and col not in metadata_cols:
            metadata_cols.append(col)

    label_col: Optional[str] = None
    if target_column in df.columns:
        label_col = target_column
        metadata_cols.append(target_column)

    # Numeric feature columns: everything that is numeric and not metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]

    # Non-numeric, non-metadata columns
    non_numeric_non_meta = [
        c for c in df.columns
        if c not in numeric_cols and c not in metadata_cols
    ]
    if non_numeric_non_meta:
        warnings.append(
            f"Non-numeric, non-metadata columns will be dropped: {non_numeric_non_meta}"
        )

    if len(feature_cols) < MINIMUM_NUMERIC_FEATURES:
        errors.append(
            f"At least {MINIMUM_NUMERIC_FEATURES} numeric feature column(s) required; "
            f"found {len(feature_cols)}."
        )

    # Check for all-NaN feature columns
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        warnings.append(f"All-NaN feature columns detected: {all_nan}")

    # Label validation (if present)
    if label_col is not None:
        unique_labels = set(df[label_col].astype(str).str.strip().str.lower().unique())
        unknown = unique_labels - KNOWN_LABEL_VALUES
        if unknown:
            errors.append(
                f"Unknown label values in '{label_col}': {unknown}. "
                f"Expected one of: {KNOWN_LABEL_VALUES}"
            )

    is_valid = len(errors) == 0

    return DatasetValidationResult(
        is_valid=is_valid,
        numeric_feature_columns=feature_cols,
        metadata_columns=[c for c in metadata_cols if c != label_col],
        label_column=label_col,
        n_samples=len(df),
        warnings=warnings,
        errors=errors,
    )

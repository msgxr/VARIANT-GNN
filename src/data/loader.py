"""
src/data/loader.py
Safe, schema-validated CSV loader.  Separates metadata from numeric features
and returns structured objects that carry Variant_ID alongside feature matrices.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_contracts.variant_schema import validate_dataset
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


class LoadedDataset:
    """Container separating metadata, feature matrix, and optional labels."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: Optional[np.ndarray],
        metadata: pd.DataFrame,
        feature_columns: List[str],
    ) -> None:
        self.features = features          # numeric features only
        self.labels   = labels            # int array or None
        self.metadata = metadata          # id + any non-feature cols
        self.feature_columns = feature_columns

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"LoadedDataset(n={len(self)}, "
            f"n_features={len(self.feature_columns)}, "
            f"has_labels={self.labels is not None})"
        )


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_csv(
    csv_path: str | Path,
    separator: str = ",",
    target_column: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    label_mapping: Optional[dict] = None,
) -> LoadedDataset:
    """
    Load a variant CSV file, validate its schema, and separate metadata
    from numerical feature columns.

    Args:
        csv_path:      Path to CSV file.
        separator:     Column delimiter.
        target_column: Label column name (optional — inferred from config).
        id_columns:    Columns treated as identifiers, not features.
        label_mapping: Dict mapping raw label strings to integers.

    Returns:
        A ``LoadedDataset`` with separated features, labels, and metadata.

    Raises:
        ValueError: if schema validation fails with errors.
    """
    cfg = get_settings()
    target_column = target_column or cfg.schema.target_column
    id_columns    = id_columns    or cfg.schema.id_columns
    label_mapping = label_mapping or cfg.schema.label_mapping

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, sep=separator, low_memory=False)
    logger.info("Loaded %d rows × %d cols from %s", len(df), df.shape[1], path)

    result = validate_dataset(df, target_column=target_column, id_columns=id_columns)

    for w in result.warnings:
        logger.warning("Schema warning: %s", w)
    if not result.is_valid:
        raise ValueError("Schema validation failed:\n" + "\n".join(result.errors))

    # Build metadata frame (preserve original id columns intact)
    meta_cols = [c for c in id_columns if c in df.columns]
    metadata  = df[meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=df.index)

    # Feature matrix
    feature_df = df[result.numeric_feature_columns].reset_index(drop=True)

    # Labels
    labels: Optional[np.ndarray] = None
    if result.label_column is not None:
        raw_labels = df[result.label_column].astype(str).str.strip().str.lower()
        mapped = raw_labels.map(label_mapping)
        unmapped = mapped.isna().sum()
        if unmapped:
            logger.warning("%d rows have unmappable labels — set to NaN then dropped.", unmapped)
            valid_mask = ~mapped.isna()
            feature_df = feature_df[valid_mask].reset_index(drop=True)
            metadata   = metadata[valid_mask].reset_index(drop=True) if len(metadata) else metadata
            mapped     = mapped[valid_mask]
        labels = mapped.astype(int).values

    logger.info(
        "Dataset ready: %d samples, %d features, labels=%s",
        len(feature_df),
        len(result.numeric_feature_columns),
        labels is not None,
    )

    return LoadedDataset(
        features       = feature_df,
        labels         = labels,
        metadata       = metadata,
        feature_columns= result.numeric_feature_columns,
    )


def load_predict_csv(csv_path: str | Path, separator: str = ",") -> LoadedDataset:
    """
    Convenience wrapper for unlabelled prediction CSVs.
    Label column is ignored even if present.
    """
    return load_csv(csv_path, separator=separator)

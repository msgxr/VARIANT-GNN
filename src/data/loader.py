"""
src/data/loader.py
Safe, schema-validated CSV loader.  Separates metadata from numeric features
and returns structured objects that carry Variant_ID alongside feature matrices.

Changes (2026-03-12):
  - LoadedDataset now carries optional nuc_sequences / aa_sequences lists
    (populated when Nuc_Context / AA_Context columns are present) so that the
    SequenceEncoder in the GNN can be fed without extra I/O.
  - load_predict_csv() now delegates column alignment to ColumnAligner instead
    of the unsafe brute-force [:34] fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data_contracts.variant_schema import validate_dataset
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


class LoadedDataset:
    """Container separating metadata, feature matrix, and optional labels.

    Attributes
    ----------
    features        : Numeric-only feature DataFrame.
    labels          : Integer label array, or None.
    metadata        : Non-feature columns (IDs, Panel, context strings …).
    feature_columns : Ordered list of feature column names.
    nuc_sequences   : Raw ``Nuc_Context`` strings aligned to feature rows, or None.
    aa_sequences    : Raw ``AA_Context`` strings aligned to feature rows, or None.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: Optional[np.ndarray],
        metadata: pd.DataFrame,
        feature_columns: List[str],
        nuc_sequences: Optional[List[str]] = None,
        aa_sequences:  Optional[List[str]] = None,
    ) -> None:
        self.features        = features
        self.labels          = labels
        self.metadata        = metadata
        self.feature_columns = feature_columns
        self.nuc_sequences   = nuc_sequences   # raw strings for SequenceEncoder
        self.aa_sequences    = aa_sequences    # raw strings for SequenceEncoder

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        has_seq = self.nuc_sequences is not None
        return (
            f"LoadedDataset(n={len(self)}, "
            f"n_features={len(self.feature_columns)}, "
            f"has_labels={self.labels is not None}, "
            f"has_sequences={has_seq})"
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

    Nuc_Context and AA_Context (when present) are preserved in the returned
    ``LoadedDataset.nuc_sequences`` / ``aa_sequences`` lists so downstream
    SequenceEncoder usage does not require re-reading the file.

    Args:
        csv_path:      Path to CSV file.
        separator:     Column delimiter.
        target_column: Label column name (optional — inferred from config).
        id_columns:    Columns treated as identifiers, not features.
        label_mapping: Dict mapping raw label strings to integers.

    Returns:
        A ``LoadedDataset`` with separated features, labels, metadata, and
        optional sequence strings.

    Raises:
        ValueError: if schema validation fails with errors.
    """
    cfg = get_settings()
    target_column       = target_column or cfg.schema.target_column
    id_columns          = id_columns    or cfg.schema.id_columns
    label_mapping       = label_mapping or cfg.schema.label_mapping
    non_feature_columns = getattr(cfg.schema, "non_feature_columns", [])

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, sep=separator, low_memory=False)
    logger.info("Loaded %d rows × %d cols from %s", len(df), df.shape[1], path)

    result = validate_dataset(
        df,
        target_column=target_column,
        id_columns=id_columns,
        non_feature_columns=non_feature_columns,
    )

    for w in result.warnings:
        logger.warning("Schema warning: %s", w)
    if not result.is_valid:
        raise ValueError("Schema validation failed:\n" + "\n".join(result.errors))

    # Build metadata frame (preserve original id columns + non-feature cols)
    meta_cols = [c for c in (id_columns + non_feature_columns) if c in df.columns]
    meta_cols = list(dict.fromkeys(meta_cols))  # deduplicate preserving order
    metadata  = df[meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=df.index)

    # Feature matrix
    feature_df = df[result.numeric_feature_columns].reset_index(drop=True)

    # Extract sequence context strings before any row-dropping
    nuc_sequences: Optional[List[str]] = (
        df["Nuc_Context"].astype(str).tolist() if "Nuc_Context" in df.columns else None
    )
    aa_sequences: Optional[List[str]] = (
        df["AA_Context"].astype(str).tolist() if "AA_Context" in df.columns else None
    )

    # Labels
    labels: Optional[np.ndarray] = None
    label_valid_mask: Optional[pd.Series] = None   # kept for panel encoding alignment
    if result.label_column is not None:
        raw_labels = df[result.label_column].astype(str).str.strip().str.lower()
        mapped = raw_labels.map(label_mapping)
        unmapped = mapped.isna().sum()
        if unmapped:
            logger.warning("%d rows have unmappable labels — set to NaN then dropped.", unmapped)
            valid_mask      = ~mapped.isna()
            valid_idx       = valid_mask[valid_mask].index.tolist()
            feature_df      = feature_df[valid_mask].reset_index(drop=True)
            metadata        = metadata[valid_mask].reset_index(drop=True) if len(metadata) else metadata
            mapped          = mapped[valid_mask]
            label_valid_mask = valid_mask
            if nuc_sequences is not None:
                nuc_sequences = [nuc_sequences[i] for i in valid_idx]
            if aa_sequences is not None:
                aa_sequences  = [aa_sequences[i] for i in valid_idx]
        labels = mapped.astype(int).values

    logger.info(
        "Dataset ready: %d samples, %d features, labels=%s, sequences=%s",
        len(feature_df),
        len(result.numeric_feature_columns),
        labels is not None,
        nuc_sequences is not None,
    )

    # ── Panel one-hot encoding (TEKNOFEST 2026 Bölüm 3.2) ───────────────
    # Panel information is provided in both training AND test data, so using
    # it as a feature is valid and improves panel-specific predictions.
    KNOWN_PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
    if "Panel" in df.columns:
        panel_series = df["Panel"].astype(str).str.strip().reset_index(drop=True)
        if label_valid_mask is not None:
            panel_series = df.loc[label_valid_mask, "Panel"].astype(str).str.strip().reset_index(drop=True)
        for panel_name in KNOWN_PANELS:
            col = f"Panel_{panel_name}"
            feature_df[col] = (panel_series == panel_name).astype(float)
        logger.info("Panel one-hot features added: %s", [f"Panel_{p}" for p in KNOWN_PANELS])

    return LoadedDataset(
        features        = feature_df,
        labels          = labels,
        metadata        = metadata,
        feature_columns = list(feature_df.columns),
        nuc_sequences   = nuc_sequences,
        aa_sequences    = aa_sequences,
    )


def load_predict_csv(csv_path: str | Path, separator: str = ",") -> LoadedDataset:
    """
    Convenience wrapper for unlabelled prediction CSVs.

    Uses ``ColumnAligner`` to robustly map incoming column names to the
    expected training-time feature names via a 4-stage strategy:
      1. Exact match
      2. Case-insensitive match
      3. Fuzzy match  (difflib, threshold 0.85)
      4. Positional fallback (with a prominent warning)

    A warning is emitted for every non-exact alignment; unmatched expected
    columns are filled with NaN and a warning is logged.
    """
    dataset = load_csv(csv_path, separator=separator)

    # ── Determine expected feature columns ──────────────────────────────
    cfg = get_settings()
    expected_features: Optional[List[str]] = None

    try:
        from src.utils.serialization import ModelStore
        store = ModelStore(cfg.paths.models_dir)
        try:
            xgb_model   = store.load_xgb()
            feat_names  = xgb_model.get_booster().feature_names
            if feat_names:
                expected_features = list(feat_names)
                logger.debug(
                    "ColumnAligner: using %d feature names from XGBoost booster", len(expected_features)
                )
        except Exception:
            pass

        if expected_features is None:
            preprocessor = store.load_preprocessor()
            if (
                hasattr(preprocessor, "_imputer")
                and preprocessor._imputer is not None
                and hasattr(preprocessor._imputer, "n_features_in_")
            ):
                logger.info(
                    "ColumnAligner: no explicit feature names — keeping %d numeric columns as-is",
                    preprocessor._imputer.n_features_in_,
                )
    except Exception:
        pass  # models not on disk yet — keep dataset as-is

    if expected_features is None:
        logger.info(
            "ColumnAligner: no trained model found — keeping %d columns as-is",
            len(dataset.feature_columns),
        )
        return dataset

    # ── Align columns ────────────────────────────────────────────────────
    from src.data.column_aligner import ColumnAligner

    aligner = ColumnAligner(
        expected_columns = expected_features,
        fuzzy_threshold  = 0.85,
        allow_positional = True,
    )

    aligned_features, report = aligner.apply(dataset.features, extra_numeric=False)

    if not report.is_clean:
        logger.warning(
            "ColumnAligner: %d case, %d fuzzy, %d positional alignments — verify above warnings.",
            len(report.case_matches),
            len(report.fuzzy_matches),
            len(report.positional_matches),
        )

    dataset.features        = aligned_features
    dataset.feature_columns = list(aligned_features.columns)
    return dataset

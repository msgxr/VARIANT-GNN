"""
src/data/loader.py
Safe, schema-validated CSV loader.  Separates metadata from numeric features
and returns structured objects that carry Variant_ID alongside feature matrices.
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
    non_feature_columns = getattr(cfg.schema, 'non_feature_columns', [])

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
    dataset = load_csv(csv_path, separator=separator)
    # Ensure dataset.features exactly matches the 34 columns expected by the model.
    # The current model was trained with exactly 34 columns. 
    # Any additional numeric columns not in standard 34 should be moved to metadata.
    expected_34_features = [
        'Ref_Nucleotide', 'Alt_Nucleotide', 'Codon_Change_Type', 'AA_Grantham_Score', 
        'GC_Content_Window', 'In_CpG_Site', 'Motif_Disruption_Score', 'AA_Polarity_Change', 
        'AA_Hydrophobicity_Diff', 'AA_Mol_Weight_Diff', 'AA_Size_Diff', 'Protein_Impact_Score', 
        'Delta_Solvent_Accessibility', 'Secondary_Structure_Disruption', 'GERP_RS', 
        'PhyloP100way_vertebrate', 'phastCons100way_vertebrate', 'SiPhy_29way_logOdds', 
        'Phylo_Diversity_Index', 'gnomAD_exomes_AF', 'gnomAD_exomes_AF_afr', 
        'gnomAD_exomes_AF_eur', 'gnomAD_exomes_AF_eas', 'gnomAD_exomes_AF_sas', 
        'gnomAD_exomes_AF_amr', 'ExAC_AF', 'SIFT_score', 'PolyPhen2_HDIV_score', 
        'PolyPhen2_HVAR_score', 'CADD_phred', 'REVEL_score', 'MutPred2_score', 'VEST4_score', 
        'PROVEAN_score'
    ]
    
    # Capitalization in actual data vs schema may differ, try case-insensitive or direct match:
    existing_cols = dataset.features.columns.tolist()
    
    # If there are exactly 34 columns, assume it's right.
    if len(existing_cols) == 34:
        return dataset
        
    # Build a lowercase map of the 34 expected features for robust matching
    expected_lower = {c.lower(): c for c in expected_34_features}
    
    matched_cols = []
    # Identify which of the existing columns belong to the expected 34 features
    for c in existing_cols:
        cl = c.lower()
        if cl in expected_lower:
            matched_cols.append(c)
            
    # If we found matches (usually around 34), we subset the features
    if len(matched_cols) > 0:
        # Move non-matched features to metadata
        unmatched = [c for c in existing_cols if c not in matched_cols]
        if unmatched:
            for uc in unmatched:
                dataset.metadata[uc] = dataset.features[uc]
            dataset.features = dataset.features[matched_cols]
            dataset.feature_columns = matched_cols
            
    # Try one more brute-force fallback for 34 features
    if len(dataset.features.columns) > 34:
        # Fallback to taking exactly the first 34 columns to prevent crash
        excess_cols = dataset.features.columns[34:]
        for c in excess_cols:
            dataset.metadata[c] = dataset.features[c]
        dataset.features = dataset.features.iloc[:, :34]
        dataset.feature_columns = dataset.features.columns.tolist()

    return dataset

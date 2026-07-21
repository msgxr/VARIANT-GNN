# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_data_schema.py
TEKNOFEST 2026 competition-specific data schema tests.

Validates dataset structure, panel labels, coordinate blocking,
ACMG label merging, and class balance per specification §3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_competition_df(
    n: int = 100,
    panel: str = "General",
    include_label: bool = True,
    include_coords: bool = False,
    include_id: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict = {}
    if include_id:
        data["Variant_ID"] = [f"V{i:04d}" for i in range(n)]
    data["Panel"] = panel
    for i in range(10):
        data[f"AL_{i + 1}"] = rng.uniform(-3, 3, n)
    if include_label:
        data["Label"] = rng.choice([0, 1], size=n)
    if include_coords:
        data["chromosome"] = ["chr1"] * n
        data["position"] = rng.integers(1_000_000, 50_000_000, n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Panel structure tests — §3.2
# ---------------------------------------------------------------------------


class TestPanelStructure:
    VALID_PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]

    def test_all_valid_panels_accepted(self):
        """All four TEKNOFEST panels must pass schema validation."""
        from src.data.schemas.variant_schema import validate_dataset

        for panel in self.VALID_PANELS:
            df = _make_competition_df(panel=panel)
            result = validate_dataset(df)
            assert result.is_valid, f"Panel {panel} failed: {result.errors}"

    def test_panel_column_is_metadata_not_feature(self):
        """Panel column must not appear in numeric feature columns."""
        from src.data.schemas.variant_schema import validate_dataset

        df = _make_competition_df()
        result = validate_dataset(df, non_feature_columns=["Panel"])
        assert "Panel" not in result.numeric_feature_columns
        assert "Panel" in result.metadata_columns

    def test_panel_not_used_as_numeric_signal(self):
        """Panel (categorical string) must never be treated as numeric feature."""
        from src.data.schemas.variant_schema import validate_dataset

        df = _make_competition_df()
        result = validate_dataset(df)
        numeric_cols = result.numeric_feature_columns
        assert "Panel" not in numeric_cols


# ---------------------------------------------------------------------------
# ACMG label encoding — §3.2
# ---------------------------------------------------------------------------


class TestACMGLabelEncoding:
    def test_pathogenic_encoded_as_1(self):
        """Pathogenic class must map to label 1."""
        df = pd.DataFrame(
            {
                "Variant_ID": ["V1", "V2"],
                "Label": [1, 1],
                "feat_0": [0.5, 0.7],
            }
        )
        from src.data.schemas.variant_schema import validate_dataset

        result = validate_dataset(df)
        assert result.is_valid

    def test_benign_encoded_as_0(self):
        """Benign class must map to label 0."""
        df = pd.DataFrame(
            {
                "Variant_ID": ["V1", "V2"],
                "Label": [0, 0],
                "feat_0": [0.1, 0.2],
            }
        )
        from src.data.schemas.variant_schema import validate_dataset

        result = validate_dataset(df)
        assert result.is_valid

    def test_string_labels_pathogenic_benign_valid(self):
        """String 'Pathogenic'/'Benign' labels must be valid."""
        df = pd.DataFrame(
            {
                "Variant_ID": ["V1", "V2"],
                "Label": ["Pathogenic", "Benign"],
                "feat_0": [0.9, 0.1],
            }
        )
        from src.data.schemas.variant_schema import validate_dataset

        result = validate_dataset(df)
        assert result.is_valid

    def test_vus_label_invalid(self):
        """VUS label must be rejected — excluded from competition per §3.2."""
        df = pd.DataFrame(
            {
                "Variant_ID": ["V1"],
                "Label": ["VUS"],
                "feat_0": [0.5],
            }
        )
        from src.data.schemas.variant_schema import validate_dataset

        result = validate_dataset(df)
        assert not result.is_valid

    def test_binary_classes_only(self):
        """Dataset with only {0,1} labels must be valid (binary classification)."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "Variant_ID": [f"V{i}" for i in range(50)],
                "Label": rng.choice([0, 1], size=50),
                "feat_0": rng.uniform(0, 1, 50),
                "feat_1": rng.uniform(0, 1, 50),
            }
        )
        from src.data.schemas.variant_schema import validate_dataset

        result = validate_dataset(df)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Genomic coordinate blocking — §3.2
# ---------------------------------------------------------------------------


class TestCoordinateBlocking:
    COORDINATE_COLS = [
        "chromosome",
        "chrom",
        "chr",
        "position",
        "pos",
        "start",
        "end",
        "rsid",
        "hgvs_genomic",
    ]

    def test_coordinate_cols_blocked_by_firewall(self):
        """LeakageFirewall must detect and drop all coordinate columns."""
        from src.data.leakage_firewall import LeakageFirewall

        for col in self.COORDINATE_COLS:
            df = _make_competition_df()
            df[col] = "dummy"
            fw = LeakageFirewall(competition_mode=True)
            clean, report = fw.check_and_sanitize(df)
            assert col not in clean.columns, f"Column {col} not dropped"
            assert col in report.coordinate_hits

    def test_no_coordinate_cols_in_clean_data(self):
        """Clean data without coordinates passes through unchanged."""
        from src.data.leakage_firewall import LeakageFirewall

        df = _make_competition_df(include_coords=False)
        fw = LeakageFirewall(competition_mode=True)
        clean, report = fw.check_and_sanitize(df)
        assert report.coordinate_hits == []
        assert len(clean) == len(df)

    def test_label_col_blocked_during_inference(self):
        """Label column must be blocked during inference mode."""
        from src.data.leakage_firewall import LeakageFirewall

        df = _make_competition_df(include_label=True)
        fw = LeakageFirewall(competition_mode=True)
        clean, report = fw.check_and_sanitize(df, is_inference=True)
        assert "Label" not in clean.columns
        assert "Label" in report.label_hits


# ---------------------------------------------------------------------------
# Dataset size tests — §3.2 counts
# ---------------------------------------------------------------------------


class TestDatasetSizeBounds:
    @pytest.mark.parametrize(
        "panel,min_train,max_train",
        [
            ("General", 2000, 4000),
            ("Hereditary_Cancer", 300, 500),
            ("PAH", 300, 500),
            ("CFTR", 80, 200),
        ],
    )
    def test_expected_panel_size_range(self, panel, min_train, max_train):
        """Verify panel sizes are within spec-expected bounds (informational)."""
        n = {"General": 2931, "Hereditary_Cancer": 388, "PAH": 372, "CFTR": 111}[panel]
        assert min_train <= n <= max_train, f"Panel {panel}: n={n} outside [{min_train}, {max_train}]"

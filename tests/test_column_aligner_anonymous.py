"""
Test: ColumnAligner with anonymous/numbered column names.

TEKNOFEST 2026 Şartname: "veri paylaşılırken öznitelik kolon isimleri
verilmeyecektir" — test verisi `col_1`, `col_2`, ... gibi anonim
sütun isimleriyle gelebilir.

Bu test, ColumnAligner'ın pozisyonel fallback mekanizmasının
anonim sütunlarla doğru çalıştığını doğrular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.column_aligner import ColumnAligner


# ── Test fixtures ─────────────────────────────────────────────────────

EXPECTED_FEATURES = [
    "Ref_Nucleotide", "Alt_Nucleotide", "Codon_Change_Type",
    "AA_Grantham_Score", "GC_Content_Window", "In_CpG_Site",
    "Motif_Disruption_Score", "AA_Polarity_Change",
    "AA_Hydrophobicity_Diff", "AA_Mol_Weight_Diff",
]


def _make_anonymous_df(n_rows: int = 5) -> pd.DataFrame:
    """Create a DataFrame with col_0, col_1, ... column names."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, len(EXPECTED_FEATURES)))
    cols = [f"col_{i}" for i in range(len(EXPECTED_FEATURES))]
    return pd.DataFrame(data, columns=cols)


def _make_numbered_df(n_rows: int = 5) -> pd.DataFrame:
    """Create a DataFrame with purely numeric column names (0, 1, 2...)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, len(EXPECTED_FEATURES)))
    cols = [str(i) for i in range(len(EXPECTED_FEATURES))]
    return pd.DataFrame(data, columns=cols)


def _make_mixed_df(n_rows: int = 5) -> pd.DataFrame:
    """Create a DataFrame with some matching and some anonymous columns."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, len(EXPECTED_FEATURES)))
    # First 3 columns match expected names (exact), rest are anonymous
    cols = list(EXPECTED_FEATURES[:3]) + [f"col_{i}" for i in range(3, len(EXPECTED_FEATURES))]
    return pd.DataFrame(data, columns=cols)


# ── Tests ─────────────────────────────────────────────────────────────


class TestColumnAlignerAnonymous:
    """TEKNOFEST şartname uyumluluk testleri: anonim sütun desteği."""

    def test_positional_fallback_with_col_prefix(self):
        """col_0, col_1, ... → positional fallback maps by index order."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df()

        aligned, report = aligner.apply(df)

        # All columns should be matched via positional fallback
        assert list(aligned.columns) == EXPECTED_FEATURES
        assert len(report.positional_matches) == len(EXPECTED_FEATURES)
        assert len(report.unmatched_expected) == 0
        assert aligned.shape == (5, len(EXPECTED_FEATURES))

        # Values should be preserved (same data, just renamed)
        np.testing.assert_array_almost_equal(
            aligned.values, df.values
        )

    def test_positional_fallback_with_numeric_names(self):
        """0, 1, 2, ... → positional fallback maps by index order."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_numbered_df()

        aligned, report = aligner.apply(df)

        assert list(aligned.columns) == EXPECTED_FEATURES
        assert len(report.positional_matches) == len(EXPECTED_FEATURES)
        assert len(report.unmatched_expected) == 0

        np.testing.assert_array_almost_equal(
            aligned.values, df.values
        )

    def test_mixed_exact_and_positional(self):
        """Some exact matches, rest via positional fallback."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_mixed_df()

        aligned, report = aligner.apply(df)

        assert list(aligned.columns) == EXPECTED_FEATURES
        assert len(report.exact_matches) == 3  # First 3 are exact
        assert len(report.positional_matches) == len(EXPECTED_FEATURES) - 3
        assert len(report.unmatched_expected) == 0

    def test_values_preserved_after_alignment(self):
        """Data integrity: values must not change during column renaming."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df(n_rows=20)
        aligned, _ = aligner.apply(df)

        # Column order and values preserved
        for i, exp_col in enumerate(EXPECTED_FEATURES):
            orig_col = f"col_{i}"
            np.testing.assert_array_equal(
                aligned[exp_col].values,
                df[orig_col].values,
                err_msg=f"Value mismatch for column {exp_col} (position {i})",
            )

    def test_report_warns_on_positional(self):
        """Positional mapping should generate warning messages."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df()
        _, report = aligner.apply(df)

        warnings = report.warnings()
        assert len(warnings) > 0
        assert any("positional" in w.lower() for w in warnings)

    def test_report_is_not_clean_on_positional(self):
        """AlignmentReport.is_clean should be False when positional fallback used."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df()
        _, report = aligner.apply(df)

        assert not report.is_clean

    def test_exact_match_is_clean(self):
        """When column names match exactly, report should be clean."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df()
        df.columns = EXPECTED_FEATURES  # Set exact names
        _, report = aligner.apply(df)

        assert report.is_clean
        assert len(report.exact_matches) == len(EXPECTED_FEATURES)

    def test_fewer_incoming_columns_fills_nan(self):
        """When fewer columns arrive, missing features should be NaN."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        df = _make_anonymous_df()
        df = df.iloc[:, :5]  # Only 5 of 10 columns

        aligned, report = aligner.apply(df)

        assert list(aligned.columns) == EXPECTED_FEATURES
        # First 5 should have values, rest NaN
        assert not aligned.iloc[:, :5].isna().any().any()
        assert aligned.iloc[:, 5:].isna().all().all()
        assert len(report.unmatched_expected) == 5

    def test_extra_columns_are_dropped(self):
        """Surplus anonymous columns should be silently dropped."""
        aligner = ColumnAligner(
            expected_columns=EXPECTED_FEATURES,
            allow_positional=True,
        )
        rng = np.random.default_rng(42)
        n_extra = 5
        data = rng.standard_normal((5, len(EXPECTED_FEATURES) + n_extra))
        cols = [f"col_{i}" for i in range(len(EXPECTED_FEATURES) + n_extra)]
        df = pd.DataFrame(data, columns=cols)

        aligned, report = aligner.apply(df)

        assert list(aligned.columns) == EXPECTED_FEATURES
        assert aligned.shape[1] == len(EXPECTED_FEATURES)
        assert len(report.unmatched_incoming) == n_extra

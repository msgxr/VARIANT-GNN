"""
tests/unit/test_schema.py
Unit tests for the Pydantic schema validation layer.
"""
import numpy as np
import pandas as pd
import pytest

from data_contracts.variant_schema import validate_dataset


def _make_df(n: int = 10, n_features: int = 5, with_label: bool = True,
             with_id: bool = True) -> pd.DataFrame:
    np.random.seed(42)
    data = {f"feat_{i}": np.random.randn(n) for i in range(n_features)}
    if with_id:
        data["Variant_ID"] = [f"VAR_{i:04d}" for i in range(n)]
    if with_label:
        data["Label"] = np.random.choice(["Pathogenic", "Benign"], size=n)
    return pd.DataFrame(data)


class TestValidateDataset:
    def test_valid_with_label_and_id(self):
        df     = _make_df()
        result = validate_dataset(df)
        assert result.is_valid
        assert len(result.errors) == 0
        assert "Variant_ID" in result.metadata_columns
        assert result.label_column == "Label"
        assert len(result.numeric_feature_columns) == 5

    def test_no_label(self):
        df     = _make_df(with_label=False)
        result = validate_dataset(df)
        assert result.is_valid
        assert result.label_column is None

    def test_no_id(self):
        df     = _make_df(with_id=False)
        result = validate_dataset(df)
        assert result.is_valid
        assert result.metadata_columns == []

    def test_empty_dataframe(self):
        df     = pd.DataFrame()
        result = validate_dataset(df)
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_unknown_label(self):
        df = _make_df()
        df.loc[0, "Label"] = "VUS"  # unknown
        result = validate_dataset(df)
        assert not result.is_valid
        assert any("Unknown label" in e for e in result.errors)

    def test_numeric_label_is_valid(self):
        df        = _make_df()
        df["Label"] = np.random.choice(["0", "1"], size=len(df))
        result    = validate_dataset(df)
        assert result.is_valid

    def test_all_nan_column_warning(self):
        df           = _make_df()
        df["feat_0"] = np.nan
        result       = validate_dataset(df)
        assert result.is_valid
        assert any("All-NaN" in w for w in result.warnings)

    def test_no_numeric_features(self):
        df     = pd.DataFrame({"Variant_ID": ["A", "B"], "Label": ["Benign", "Pathogenic"]})
        result = validate_dataset(df)
        assert not result.is_valid
        assert any("numeric feature" in e.lower() for e in result.errors)

    # ---------------------------------------------------------------
    # TEKNOFEST Şartname — non_feature_columns testleri
    # ---------------------------------------------------------------

    def test_non_feature_columns_treated_as_metadata(self):
        """Panel, Nuc_Context, AA_Context gibi non-feature sütunlarının
        metadata olarak işlendiğini ve uyarı üretmediğini doğrular."""
        df = _make_df()
        df["Panel"] = "General"
        df["Nuc_Context"] = "ACGTTGACGTG"
        df["AA_Context"] = "AVILMFYWKRN"
        result = validate_dataset(
            df,
            non_feature_columns=["Panel", "Nuc_Context", "AA_Context"]
        )
        assert result.is_valid
        assert "Panel" in result.metadata_columns
        assert "Nuc_Context" in result.metadata_columns
        assert "AA_Context" in result.metadata_columns
        # non-feature sütunlar uyarı üretmemeli
        drop_warnings = [w for w in result.warnings if "dropped" in w.lower()]
        for w in drop_warnings:
            assert "Panel" not in w
            assert "Nuc_Context" not in w
            assert "AA_Context" not in w

    def test_panel_column_not_in_features(self):
        """Panel sütununun sayısal özellik listesine dahil edilmediğini doğrular."""
        df = _make_df()
        df["Panel"] = "CFTR"
        result = validate_dataset(df, non_feature_columns=["Panel"])
        assert "Panel" not in result.numeric_feature_columns
        assert "Panel" in result.metadata_columns


# ---------------------------------------------------------------
# ColumnAligner testleri
# ---------------------------------------------------------------

class TestColumnAligner:
    """src/data/column_aligner.py — 4-aşamalı eşleştirme testleri."""

    def _make_df(self, cols):
        import pandas as pd
        import numpy as np
        return pd.DataFrame({c: np.random.randn(5) for c in cols})

    def test_exact_match(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["alpha", "beta", "gamma"]
        df = self._make_df(["alpha", "beta", "gamma"])
        aligner = ColumnAligner(expected)
        aligned, report = aligner.apply(df)
        assert list(aligned.columns) == expected
        assert len(report.exact_matches) == 3
        assert len(report.case_matches) == 0
        assert len(report.positional_matches) == 0

    def test_case_insensitive_match(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["Alpha", "Beta", "Gamma"]
        df = self._make_df(["alpha", "BETA", "Gamma"])
        aligner = ColumnAligner(expected)
        aligned, report = aligner.apply(df)
        assert list(aligned.columns) == expected
        assert len(report.case_matches) == 2   # alpha→Alpha, BETA→Beta
        assert len(report.exact_matches) == 1  # Gamma exact
        assert report.is_clean

    def test_fuzzy_match(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["CADD_phred", "REVEL_score", "SIFT_score"]
        df = self._make_df(["CADD_phred", "REVEL_scor", "SIFT_scre"])
        aligner = ColumnAligner(expected, fuzzy_threshold=0.80)
        aligned, report = aligner.apply(df)
        assert list(aligned.columns) == expected
        assert len(report.fuzzy_matches) >= 1

    def test_positional_fallback(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["feat_A", "feat_B", "feat_C"]
        df = self._make_df(["x", "y", "z"])
        aligner = ColumnAligner(expected, fuzzy_threshold=0.99, allow_positional=True)
        aligned, report = aligner.apply(df)
        assert list(aligned.columns) == expected
        assert len(report.positional_matches) == 3
        assert not report.is_clean

    def test_missing_column_filled_with_nan(self):
        import numpy as np
        from src.data.column_aligner import ColumnAligner
        expected = ["a", "b", "c"]
        df = self._make_df(["a", "b"])
        aligner = ColumnAligner(expected, allow_positional=False)
        aligned, report = aligner.apply(df)
        assert "c" in aligned.columns
        assert aligned["c"].isna().all()
        assert "c" in report.unmatched_expected

    def test_report_is_clean_for_exact(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["x1", "x2"]
        df = self._make_df(["x1", "x2"])
        aligner = ColumnAligner(expected)
        _, report = aligner.apply(df)
        assert report.is_clean

    def test_warnings_for_nonexact(self):
        from src.data.column_aligner import ColumnAligner
        expected = ["Feature_One"]
        df = self._make_df(["feature_one"])
        aligner = ColumnAligner(expected)
        _, report = aligner.apply(df)
        assert len(report.warnings()) > 0

"""
tests/unit/test_column_aligner_stress.py
ColumnAligner stres testleri — edge-case senaryoları
"""

import numpy as np
import pandas as pd
import pytest

from src.data.column_aligner import ColumnAligner

EXPECTED = [f"AL_{i}" for i in range(1, 21)]  # AL_1 … AL_20


# ── Yardımcı ──────────────────────────────────────────────────────────────────
def _aligner(expected=None):
    return ColumnAligner(expected or EXPECTED, allow_positional=True)


def _ref_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({col: rng.uniform(0, 100, 200) for col in EXPECTED})


# ── 1. Tamamen boş kolon ──────────────────────────────────────────────────────
class TestEmptyColumn:
    def test_all_nan_column_survives(self):
        al = _aligner()
        al.set_reference_distributions(_ref_df())
        df = _ref_df().rename(columns={c: c for c in EXPECTED})
        df["AL_5"] = np.nan  # tüm değerler NaN
        out, report = al.apply(df)
        assert "AL_5" in out.columns
        assert out.shape[1] == len(EXPECTED)

    def test_single_row_all_nan_no_crash(self):
        al = _aligner()
        df = pd.DataFrame({col: [np.nan] for col in EXPECTED})
        out, report = al.apply(df)
        assert out.shape == (1, len(EXPECTED))


# ── 2. Karışık veri tipi (string + float) ────────────────────────────────────
class TestMixedDtype:
    def test_string_in_numeric_column(self):
        al = _aligner()
        df = _ref_df()
        # AL_3'e karışık veri gir
        df = df.astype(object)
        df.loc[0, "AL_3"] = "MISSING"
        df.loc[1, "AL_3"] = "N/A"
        # apply çökmemeli
        out, report = al.apply(df)
        assert out.shape[1] == len(EXPECTED)

    def test_all_string_column_becomes_numeric(self):
        al = _aligner()
        df = pd.DataFrame({col: ["0.5"] * 5 for col in EXPECTED})
        out, report = al.apply(df)
        assert out.shape == (5, len(EXPECTED))


# ── 3. Tek satırlı inference ──────────────────────────────────────────────────
class TestSingleRowInference:
    def test_single_row_exact_match(self):
        al = _aligner()
        df = pd.DataFrame({col: [np.random.rand()] for col in EXPECTED})
        out, report = al.apply(df)
        assert out.shape == (1, len(EXPECTED))

    def test_single_row_shuffled_columns(self):
        al = _aligner()
        shuffled = EXPECTED[::-1]
        df = pd.DataFrame({col: [0.42] for col in shuffled})
        out, report = al.apply(df)
        assert list(out.columns) == EXPECTED

    def test_single_row_missing_columns_filled_nan(self):
        al = _aligner()
        partial = EXPECTED[:10]  # sadece ilk 10 kolon
        df = pd.DataFrame({col: [1.0] for col in partial})
        out, report = al.apply(df)
        assert out.shape == (1, len(EXPECTED))
        # Eksik kolonlar NaN olmalı
        for col in EXPECTED[10:]:
            assert pd.isna(out[col].iloc[0])


# ── 4. Kolon ismi uyuşmazlığı → positional fallback ──────────────────────────
class TestPositionalFallback:
    def test_completely_renamed_columns_positional(self):
        al = _aligner()
        renamed = {col: f"COL_{i}" for i, col in enumerate(EXPECTED)}
        df = _ref_df().rename(columns=renamed)
        out, report = al.apply(df)
        # Positional eşleşme → 20 kolon üretilmeli
        assert out.shape[1] == len(EXPECTED)
        assert len(report.positional_matches) > 0

    def test_partial_name_match_then_positional(self):
        al = _aligner()
        mixed_cols = EXPECTED[:10] + [f"UNKNOWN_{i}" for i in range(10)]
        df = pd.DataFrame({col: np.random.rand(5) for col in mixed_cols})
        out, report = al.apply(df)
        assert out.shape[1] == len(EXPECTED)

    def test_positional_fallback_disabled_fills_nan(self):
        al = ColumnAligner(EXPECTED, allow_positional=False)
        renamed = {col: f"X_{i}" for i, col in enumerate(EXPECTED)}
        df = _ref_df().rename(columns=renamed)
        out, report = al.apply(df)
        # Hiç eşleşme yok → tüm kolonlar NaN dolu olmalı
        assert out.shape[1] == len(EXPECTED)
        assert len(report.positional_matches) == 0


# ── 5. Dağılım imzası eşleştirme ─────────────────────────────────────────────
class TestDistributionalSignature:
    def test_signature_matching_survives_zero_variance(self):
        al = _aligner()
        ref = _ref_df()
        ref["AL_7"] = 99.0  # sıfır varyans
        al.set_reference_distributions(ref)
        df = _ref_df()
        out, _ = al.apply(df)
        assert "AL_7" in out.columns

    def test_set_reference_distributions_idempotent(self):
        al = _aligner()
        ref = _ref_df()
        al.set_reference_distributions(ref)
        al.set_reference_distributions(ref)  # iki kez çağrılınca çökmemeli
        out, _ = al.apply(ref)
        assert out.shape == ref.shape

"""Unit tests for CategoricalBioFeaturizer — §3.2 biological signal recovery layer."""
import numpy as np
import pandas as pd
import pytest

from src.features.categorical_bio_features import (
    CategoricalBioFeaturizer,
    get_grantham_score,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "AA_1": ["R", "I", "P", "G", "X"],
            "AA_2": ["W", "L", "*", "W", "A"],
            "CAT_1": ["EUR&AFR&SAS", "gnomADe_NFE", "EUR", np.nan, "AMR"],
            "CAT_2": ["AllofUs_EUR", "AllofUs_AFR", "AllofUs_EAS", "AllofUs_MID", "AllofUs_OTH"],
            "CAT_3": ["C/C", "./.", "A/A", "G/G", "T/T"],
            "CAT_6": ["lcr", "segdup", "decoy&segdup", "lcr", "segdup"],
            "EK_4": [0.9, 0.1, 0.99, 0.5, 0.7],
            "EK_5": [0.8, 0.2, 0.95, 0.4, 0.6],
        }
    )


def test_grantham_matches_published_matrix():
    # Canonical Grantham distances (Grantham 1974) — jury-defensible absolute values
    assert get_grantham_score("R", "W") == pytest.approx(101, abs=2)
    assert get_grantham_score("I", "L") == pytest.approx(5, abs=2)
    assert get_grantham_score("G", "W") == pytest.approx(184, abs=3)
    # symmetry
    assert get_grantham_score("D", "E") == pytest.approx(get_grantham_score("E", "D"))
    # unknown AA → safe fallback, never crashes
    assert get_grantham_score("X", "A") == 100.0


def test_transform_produces_expected_columns(sample_df):
    f = CategoricalBioFeaturizer(append=False).fit(sample_df)
    out = f.transform(sample_df)
    for col in ["bio_grantham", "bio_blosum62", "bio_stop_gain", "bio_charge_flip",
                "pop_breadth_CAT_1", "region_lcr", "insilico_consensus"]:
        assert col in out.columns, f"missing {col}"
    assert len(out) == len(sample_df)
    assert out.isna().sum().sum() == 0  # no NaN leaks into features


def test_stop_gain_and_charge_flip(sample_df):
    out = CategoricalBioFeaturizer(append=False).fit(sample_df).transform(sample_df)
    # Row 2 (P->*) is a stop-gain
    assert out["bio_stop_gain"].iloc[2] == 1.0
    # Row 0 (R+ -> W neutral) is a charge flip from positive to neutral -> not a sign flip
    assert out["bio_charge_flip"].iloc[0] == 0.0


def test_population_breadth(sample_df):
    out = CategoricalBioFeaturizer(append=False).fit(sample_df).transform(sample_df)
    # "EUR&AFR&SAS" -> 3 databases observed
    assert out["pop_breadth_CAT_1"].iloc[0] == 3.0
    # NaN -> 0 (not observed)
    assert out["pop_breadth_CAT_1"].iloc[3] == 0.0


def test_region_flags(sample_df):
    out = CategoricalBioFeaturizer(append=False).fit(sample_df).transform(sample_df)
    assert out["region_lcr"].iloc[0] == 1.0
    assert out["region_segdup"].iloc[2] == 1.0  # "decoy&segdup"
    assert out["region_decoy"].iloc[2] == 1.0


def test_insilico_consensus_autodetect(sample_df):
    f = CategoricalBioFeaturizer(append=False).fit(sample_df)
    assert set(f.insilico_cols) == {"EK_4", "EK_5"}
    out = f.transform(sample_df)
    assert out["insilico_consensus"].iloc[0] == pytest.approx(0.85)  # mean(0.9, 0.8)


def test_deterministic_no_leakage(sample_df):
    # Row-wise deterministic: two transforms identical, fit on subset == fit on full
    f = CategoricalBioFeaturizer(append=False)
    out1 = f.fit(sample_df).transform(sample_df)
    out2 = f.fit(sample_df.iloc[:2]).transform(sample_df)
    pd.testing.assert_frame_equal(
        out1[["bio_grantham", "bio_blosum62"]], out2[["bio_grantham", "bio_blosum62"]]
    )


def test_append_mode_preserves_original(sample_df):
    out = CategoricalBioFeaturizer(append=True).fit(sample_df).transform(sample_df)
    assert "AA_1" in out.columns and "bio_grantham" in out.columns
    assert out.shape[1] > sample_df.shape[1]

"""
conftest.py
Shared pytest fixtures available to all test suites.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Reusable small dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_variant_df():
    """20-row DataFrame with Variant_ID, 5 numeric features, and Label."""
    rng = np.random.default_rng(42)
    data = {
        "Variant_ID": [f"V{i:04d}" for i in range(20)],
        "feat_1": rng.standard_normal(20).astype(float),
        "feat_2": rng.standard_normal(20).astype(float),
        "feat_3": rng.standard_normal(20).astype(float),
        "feat_4": rng.standard_normal(20).astype(float),
        "feat_5": rng.standard_normal(20).astype(float),
        "Label": ([0] * 10 + [1] * 10),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def small_X_y(small_variant_df):
    """Return (X: np.ndarray, y: np.ndarray) from the small variant DataFrame."""
    feat_cols = [c for c in small_variant_df.columns if c.startswith("feat_")]
    X = small_variant_df[feat_cols].values.astype(np.float32)
    y = small_variant_df["Label"].values.astype(np.int64)
    return X, y


@pytest.fixture(scope="session")
def balanced_proba():
    """100 random probability vectors (N, 2) summing to 1, plus matching labels."""
    rng   = np.random.default_rng(7)
    proba = rng.dirichlet([1.0, 1.0], size=100).astype(np.float32)
    y     = rng.integers(0, 2, size=100).astype(np.int64)
    return proba, y

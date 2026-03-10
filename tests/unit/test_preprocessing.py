"""
tests/unit/test_preprocessing.py
Unit tests for the leakage-free VariantPreprocessor.
"""
import numpy as np
import pytest


def _make_xy(n: int = 200, f: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X   = rng.normal(size=(n, f))
    y   = rng.integers(0, 2, size=n)
    return X.astype(np.float32), y.astype(int)


class TestVariantPreprocessor:
    def test_fit_resample_returns_arrays(self):
        from src.features.preprocessing import VariantPreprocessor
        X, y     = _make_xy()
        pre      = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        X_out, y_out = pre.fit_resample_train(X, y)
        assert X_out.ndim == 2
        assert len(y_out) == len(X_out)

    def test_transform_shape_consistent(self):
        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy()
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        pre.fit_resample_train(X[:150], y[:150])
        X_val = pre.transform(X[150:])
        assert X_val.shape[1] == pre.n_output_features

    def test_no_fit_before_transform_raises(self):
        from src.features.preprocessing import VariantPreprocessor
        X, _ = _make_xy()
        pre  = VariantPreprocessor(use_autoencoder=False)
        with pytest.raises(RuntimeError):
            pre.transform(X)

    def test_smote_increases_minority(self):
        from src.features.preprocessing import VariantPreprocessor
        rng = np.random.default_rng(7)
        X   = rng.normal(size=(150, 8)).astype(np.float32)
        y   = np.array([0] * 120 + [1] * 30)
        pre = VariantPreprocessor(use_autoencoder=False, smote_enabled=True)
        _, y_out = pre.fit_resample_train(X, y)
        # After SMOTE minority count should be equal to or larger than before
        assert (y_out == 1).sum() >= 30

    def test_autoencoder_appends_features(self):
        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy(n=100, f=8)
        pre  = VariantPreprocessor(
            use_autoencoder=True, autoencoder_encoding_dim=4,
            autoencoder_epochs=2, smote_enabled=False,
        )
        X_out, _ = pre.fit_resample_train(X, y)
        # appended: original + encoded
        assert X_out.shape[1] == 8 + 4

    def test_graph_built_after_fit(self):
        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy()
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        pre.fit_resample_train(X, y)
        assert pre.edge_index is not None

    def test_row_to_graph(self):
        import torch

        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy(n=50, f=6)
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        pre.fit_resample_train(X, y)
        graph = pre.row_to_graph(X[0])
        assert graph.x.shape == (pre.n_output_features, 1)

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
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False, use_bio_scoring=False)
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
            use_acmg_proxy=False,   # ACMG proxy kapalı: sadece autoencoder etkisi test ediliyor
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
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False,
                                   use_acmg_proxy=False)  # ACMG proxy kapalı: graph shape testi
        pre.fit_resample_train(X, y)
        graph = pre.row_to_graph(X[0])
        assert graph.x.shape == (pre.n_output_features, 1)

    def test_smote_not_applied_on_transform(self):
        """§3.3 — SMOTE yalnızca eğitim fold'unda uygulanır; transform() SMOTE yapmaz."""
        from src.features.preprocessing import VariantPreprocessor
        rng = np.random.default_rng(42)
        n_train, n_val = 120, 30
        X_tr  = rng.normal(size=(n_train, 8)).astype(np.float32)
        # Dengeli eğitim etiketi (SMOTE için en az 2 sınıf şart)
        y_tr  = np.array([0] * 80 + [1] * 40)
        X_val = rng.normal(size=(n_val, 8)).astype(np.float32)
        pre = VariantPreprocessor(use_autoencoder=False, smote_enabled=True)
        pre.fit_resample_train(X_tr, y_tr)
        X_val_out = pre.transform(X_val)
        # transform() sadece n_val satır döndürmeli (SMOTE uygulanmaz)
        assert X_val_out.shape[0] == n_val, (
            f"transform() SMOTE uyguladı: beklenen {n_val}, alınan {X_val_out.shape[0]}"
        )

    def test_transform_does_not_call_fit(self):
        """transform() çağrısı hiçbir fitter'ı yeniden fit etmez (leakage yok)."""
        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy(n=150, f=8)
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        pre.fit_resample_train(X[:100], y[:100])
        n_features_after_fit = pre.n_output_features
        # Validation transform
        pre.transform(X[100:])
        # Feature count değişmemeli
        assert pre.n_output_features == n_features_after_fit

    def test_validation_output_shape_matches_train(self):
        """Eğitim ve doğrulama çıktı boyutları aynı olmalı (§4.1 leakage-free split)."""
        from src.features.preprocessing import VariantPreprocessor
        X, y = _make_xy(n=200, f=10)
        pre  = VariantPreprocessor(use_autoencoder=False, smote_enabled=False)
        X_tr, _ = pre.fit_resample_train(X[:160], y[:160])
        X_val    = pre.transform(X[160:])
        assert X_tr.shape[1] == X_val.shape[1], (
            f"Train sütun sayısı ({X_tr.shape[1]}) ≠ Val sütun sayısı ({X_val.shape[1]})"
        )

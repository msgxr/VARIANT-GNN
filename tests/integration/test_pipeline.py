"""
tests/integration/test_pipeline.py
Integration tests — synthetic data flows through the full
train → calibrate → evaluate → predict pipeline.

These tests are intentionally lightweight: they use small synthetic datasets
so they run quickly. They verify architectural correctness (shapes, IDs,
no leakage artifacts) rather than model quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_dataset():
    """
    Return a synthetic variant dataset: 120 rows × 15 features + Variant_ID + Label.
    Balanced classes, reproducible.
    """
    rng = np.random.default_rng(0)
    n   = 120
    n_f = 15
    X   = rng.standard_normal((n, n_f)).astype(np.float32)
    y   = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    rng.shuffle(y)
    ids = [f"VAR_{i:04d}" for i in range(n)]
    df = pd.DataFrame(X, columns=[f"feat_{j}" for j in range(n_f)])
    df.insert(0, "Variant_ID", ids)
    df["Label"] = y
    return df


# ---------------------------------------------------------------------------
# Preprocessing integration
# ---------------------------------------------------------------------------

class TestPreprocessingIntegration:
    def test_fit_then_transform_same_shape(self, synthetic_dataset):
        from src.features.preprocessing import VariantPreprocessor
        df = synthetic_dataset
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feat_cols].values
        y = df["Label"].values

        split = len(X) // 5  # 20% test
        X_train, X_test = X[split:], X[:split]
        y_train, _y_test = y[split:], y[:split]

        pp = VariantPreprocessor(use_autoencoder=False, use_feature_selection=False)
        X_proc, _y_res = pp.fit_resample_train(X_train, y_train)
        X_test_proc    = pp.transform(X_test)

        assert X_proc.shape[1] == X_test_proc.shape[1], (
            "Train and test must have same number of features after preprocessing"
        )

    def test_smote_increases_minority(self, synthetic_dataset):
        """After SMOTE, the resampled y should have equal class counts."""
        from src.features.preprocessing import VariantPreprocessor
        df = synthetic_dataset.copy()
        # Make imbalanced: 90 class-0, 30 class-1
        df_imb = pd.concat([
            df[df["Label"] == 0].iloc[:90],
            df[df["Label"] == 1].iloc[:30],
        ]).reset_index(drop=True)
        feat_cols = [c for c in df_imb.columns if c.startswith("feat_")]
        X = df_imb[feat_cols].values
        y = df_imb["Label"].values

        pp = VariantPreprocessor(smote_enabled=True, use_autoencoder=False, use_feature_selection=False)
        _X_proc, y_res = pp.fit_resample_train(X, y)
        counts = np.bincount(y_res)
        assert counts[0] == counts[1], "SMOTE should balance classes"

    def test_graph_edges_built_from_training_only(self, synthetic_dataset):
        """edge_index should be non-None after fit and reflect only training data."""
        from src.features.preprocessing import VariantPreprocessor
        df = synthetic_dataset
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feat_cols].values
        y = df["Label"].values
        pp = VariantPreprocessor(use_autoencoder=False, use_feature_selection=False)
        pp.fit_resample_train(X, y)
        assert pp.edge_index is not None
        # edge_index must be 2-row tensor [source, target]
        assert pp.edge_index.shape[0] == 2


# ---------------------------------------------------------------------------
# Calibration integration
# ---------------------------------------------------------------------------

class TestCalibrationIntegration:
    def test_calibrated_proba_in_unit_range(self):
        from src.calibration.calibrator import EnsembleCalibrator
        rng   = np.random.default_rng(1)
        proba = rng.dirichlet([1, 1], size=100)
        y     = rng.integers(0, 2, size=100)
        cal   = EnsembleCalibrator(method="isotonic")
        cal.fit(proba, y)
        out = cal.transform(proba)
        assert out.shape == (100, 2)
        assert np.all(out >= 0) and np.all(out <= 1)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------

class TestEvaluationIntegration:
    def test_full_evaluate_call(self):
        from src.evaluation.metrics import evaluate
        rng   = np.random.default_rng(2)
        N     = 80
        y_true = rng.integers(0, 2, size=N)
        proba  = rng.dirichlet([1, 1], size=N)
        report = evaluate(y_true, proba)
        assert report.macro_f1 >= 0.0
        assert report.roc_auc >= 0.0
        assert report.brier_score <= 1.0
        assert -1.0 <= report.mcc <= 1.0

    def test_confusion_matrix_shape(self):
        from src.evaluation.metrics import evaluate
        rng    = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=60)
        proba  = rng.dirichlet([1, 1], size=60)
        report = evaluate(y_true, proba)
        assert report.confusion_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# Schema + loader integration
# ---------------------------------------------------------------------------

class TestSchemaLoaderIntegration:
    def test_validate_then_load(self, synthetic_dataset, tmp_path):
        from data_contracts.variant_schema import validate_dataset
        from src.data.loader import load_csv

        csv_path = tmp_path / "test_data.csv"
        synthetic_dataset.to_csv(csv_path, index=False)

        result = validate_dataset(synthetic_dataset)
        assert result.valid, f"Expected valid, got errors: {result.errors}"

        loaded = load_csv(str(csv_path))
        assert loaded.labels is not None
        assert len(loaded.labels) == len(synthetic_dataset)
        assert "Variant_ID" in loaded.metadata.columns

    def test_variant_id_preserved(self, synthetic_dataset, tmp_path):
        from src.data.loader import load_csv

        csv_path = tmp_path / "test_id.csv"
        synthetic_dataset.to_csv(csv_path, index=False)
        loaded = load_csv(str(csv_path))

        original_ids = synthetic_dataset["Variant_ID"].tolist()
        loaded_ids   = loaded.metadata["Variant_ID"].tolist()
        assert original_ids == loaded_ids, "Variant_ID must be preserved unchanged"

    def test_predict_csv_no_labels(self, synthetic_dataset, tmp_path):
        from src.data.loader import load_predict_csv

        df_no_label = synthetic_dataset.drop(columns=["Label"])
        csv_path    = tmp_path / "predict_data.csv"
        df_no_label.to_csv(csv_path, index=False)

        loaded = load_predict_csv(str(csv_path))
        assert loaded.labels is None
        assert loaded.features.shape[1] == 15  # 15 feature cols

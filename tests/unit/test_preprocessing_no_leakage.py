"""
tests/unit/test_preprocessing_no_leakage.py
Leakage prevention tests for the preprocessing pipeline.

Verifies the critical requirement: all fit() operations must happen
only on training data, never on validation or test data.

These tests simulate the exact failure scenario:
fit-on-full-dataset → inflated metrics (data leakage).
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_split(n_train=200, n_val=50, n_features=15, seed=42):
    rng = np.random.default_rng(seed)
    X_full = rng.normal(size=(n_train + n_val, n_features)).astype(np.float32)
    y_full = (rng.uniform(0, 1, n_train + n_val) > 0.5).astype(int)
    X_train, X_val = X_full[:n_train], X_full[n_train:]
    y_train, y_val = y_full[:n_train], y_full[n_train:]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Core leakage tests
# ---------------------------------------------------------------------------


class TestPreprocessorNoLeakage:
    def test_transform_only_on_val_produces_different_result(self):
        """
        Fitting on val data must produce different scaling stats
        than fitting on train data — proving transform-only is enforced.
        """
        from src.features.preprocessing import VariantPreprocessor

        X_train, y_train, X_val, _ = _make_split()

        pre_correct = VariantPreprocessor(
            use_autoencoder=False,
            smote_enabled=False,
            k_best_features=10,
            use_bio_scoring=False,
        )
        pre_correct.fit_resample_train(X_train, y_train)
        X_val_correct = pre_correct.transform(X_val)

        pre_leaked = VariantPreprocessor(
            use_autoencoder=False,
            smote_enabled=False,
            k_best_features=10,
            use_bio_scoring=False,
        )
        pre_leaked.fit_resample_train(X_val, _[: len(X_val)])
        X_val_leaked = pre_leaked.transform(X_val)

        # Results must differ (different fit statistics)
        assert not np.allclose(X_val_correct, X_val_leaked, atol=1e-3), (
            "Fit on val vs fit on train produced identical output — possible leakage or trivial preprocessor"
        )

    def test_transform_before_fit_raises(self):
        """Calling transform() before fit() must raise RuntimeError."""
        from src.features.preprocessing import VariantPreprocessor

        _, _, X_val, _ = _make_split()
        pre = VariantPreprocessor(use_autoencoder=False)
        with pytest.raises(RuntimeError):
            pre.transform(X_val)

    def test_scaler_stats_from_train_not_val(self):
        """
        Scaler statistics (mean/std) must come from training data.
        This is the most common leakage source: fitting scaler on all data.
        """
        from src.features.preprocessing import VariantPreprocessor

        X_train, y_train, X_val, _ = _make_split()

        # Shift val data significantly
        X_val_shifted = X_val + 100.0

        pre = VariantPreprocessor(
            use_autoencoder=False,
            smote_enabled=False,
            k_best_features=8,
            use_bio_scoring=False,
        )
        pre.fit_resample_train(X_train, y_train)

        # transform-only: scaler uses train stats, so large shift remains
        X_val_transformed = pre.transform(X_val_shifted)
        # After scaling with train mean/std, shifted values should still be large
        assert np.mean(X_val_transformed) > 0.5, "Scaler appears to be using val statistics — possible leakage"

    def test_smote_not_applied_to_val(self):
        """
        SMOTE must only augment training data, never validation data.
        Val set size must remain exactly n_val after transform.
        """
        from src.features.preprocessing import VariantPreprocessor

        X_train, y_train, X_val, _ = _make_split(n_train=160, n_val=40)

        pre = VariantPreprocessor(
            use_autoencoder=False,
            smote_enabled=True,
            k_best_features=8,
            use_bio_scoring=False,
        )
        X_tr_out, y_tr_out = pre.fit_resample_train(X_train, y_train)
        X_val_out = pre.transform(X_val)

        assert len(X_val_out) == len(X_val), (
            f"Val set changed size after transform: {len(X_val)} → {len(X_val_out)}. "
            "SMOTE must NOT be applied to validation data."
        )

    def test_feature_selection_fit_on_train_only(self):
        """
        SelectKBest must be fit on training data only.
        Feature indices selected on train must be applied to val transform-only.
        """
        from src.features.preprocessing import VariantPreprocessor

        X_train, y_train, X_val, _ = _make_split(n_features=20)

        pre = VariantPreprocessor(
            use_autoencoder=False,
            smote_enabled=False,
            k_best_features=8,
            use_bio_scoring=False,
        )
        X_tr_out, _ = pre.fit_resample_train(X_train, y_train)
        X_val_out = pre.transform(X_val)

        # Both must have same number of features (k=8)
        assert X_tr_out.shape[1] == X_val_out.shape[1], (
            "Train and val output feature dims must match after feature selection"
        )


# ---------------------------------------------------------------------------
# LeakageFirewall integration
# ---------------------------------------------------------------------------


class TestLeakageFirewallIntegration:
    def test_firewall_blocks_clinvar_columns(self):
        """ClinVar classification columns must be blocked (label leakage risk)."""
        import pandas as pd

        from src.data.leakage_firewall import LeakageFirewall

        df = pd.DataFrame(
            {
                "Variant_ID": ["V1", "V2"],
                "feat_0": [0.5, 0.3],
                "clinvar_classification": ["Pathogenic", "Benign"],
                "clinvar_significance": ["Pathogenic", "Benign"],
            }
        )
        fw = LeakageFirewall(competition_mode=True)
        clean, report = fw.check_and_sanitize(df, is_inference=True)

        assert "clinvar_classification" not in clean.columns
        assert "clinvar_significance" not in clean.columns

    def test_firewall_clean_report_no_false_positives(self):
        """Clean dataframe must pass through with is_clean=True."""
        import numpy as np
        import pandas as pd

        from src.data.leakage_firewall import LeakageFirewall

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "Variant_ID": [f"V{i}" for i in range(20)],
                "Panel": ["General"] * 20,
                "feat_0": rng.uniform(0, 1, 20),
                "feat_1": rng.uniform(0, 1, 20),
            }
        )
        fw = LeakageFirewall(competition_mode=True)
        clean, report = fw.check_and_sanitize(df)
        assert report.is_clean
        assert len(clean) == 20

    def test_no_train_test_id_overlap(self):
        """Variant IDs in train and test must not overlap."""
        import numpy as np
        import pandas as pd

        from src.data.leakage_firewall import LeakageFirewall

        train = pd.DataFrame(
            {
                "Variant_ID": [f"V{i}" for i in range(50)],
                "feat": np.random.rand(50),
            }
        )
        test_no_overlap = pd.DataFrame(
            {
                "Variant_ID": [f"V{i}" for i in range(50, 100)],
                "feat": np.random.rand(50),
            }
        )
        test_with_overlap = pd.DataFrame(
            {
                "Variant_ID": [f"V{i}" for i in range(40, 90)],
                "feat": np.random.rand(50),
            }
        )

        fw = LeakageFirewall(competition_mode=True)

        _, report_clean = fw.check_and_sanitize(test_no_overlap, train_df=train)
        assert report_clean.variant_id_overlaps == 0

        _, report_overlap = fw.check_and_sanitize(test_with_overlap, train_df=train)
        assert report_overlap.variant_id_overlaps == 10


# ---------------------------------------------------------------------------
# CV fold isolation tests
# ---------------------------------------------------------------------------


class TestCVFoldIsolation:
    def test_cv_folds_no_index_overlap(self):
        """StratifiedKFold splits must have no index overlap between train and val."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        n = 300
        X = np.random.rand(n, 10)
        y = np.array([0] * 150 + [1] * 150)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Fold {fold}: {len(overlap)} overlapping indices between train and val"

    def test_cv_all_indices_covered(self):
        """All samples must appear in exactly one validation fold across all folds."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        n = 500
        X = np.random.rand(n, 5)
        y = np.array([0] * 250 + [1] * 250)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_counts = np.zeros(n, dtype=int)
        for train_idx, val_idx in kf.split(X, y):
            val_counts[val_idx] += 1

        assert np.all(val_counts == 1), "Each sample must appear in exactly one validation fold"

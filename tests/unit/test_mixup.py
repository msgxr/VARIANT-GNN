"""
tests/unit/test_mixup.py
Unit tests for Mixup data augmentation (src/training/mixup.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TestMixupData:
    """Input-level Mixup testleri."""

    def test_output_shapes(self):
        from src.training.mixup import mixup_data
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=0.2, seed=42)
        assert X_mix.shape == (32, 10)
        assert y_a.shape == (32,)
        assert y_b.shape == (32,)
        assert lam.shape == (32,)

    def test_alpha_zero_no_mixing(self):
        """α=0 → no mixing (λ=1 → original data unchanged)."""
        from src.training.mixup import mixup_data
        X = torch.randn(16, 5)
        y = torch.randint(0, 2, (16,))
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=0.0)
        assert torch.allclose(lam, torch.ones(16))
        assert torch.equal(X_mix, X)

    def test_lambda_in_range(self):
        from src.training.mixup import mixup_data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        _, _, _, lam = mixup_data(X, y, alpha=0.2, seed=42)
        assert (lam >= 0).all() and (lam <= 1).all()

    def test_same_class_only(self):
        from src.training.mixup import mixup_data
        X = torch.randn(20, 5)
        y = torch.tensor([0]*10 + [1]*10)
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=0.2, same_class_only=True, seed=42)
        # y_a and y_b should be same class for each pair
        assert torch.equal(y_a, y_b)


class TestMixupLoss:
    def test_scalar_output(self):
        from src.training.mixup import mixup_data, mixup_loss
        X = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=0.2, seed=42)

        model = torch.nn.Linear(10, 2)
        criterion = torch.nn.CrossEntropyLoss()
        logits = model(X_mix)
        loss = mixup_loss(criterion, logits, y_a, y_b, lam)
        assert loss.shape == ()
        assert loss.item() > 0


class TestTabularMixup:
    def test_with_noise(self):
        from src.training.mixup import tabular_mixup
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        X_mix, y_a, y_b, lam = tabular_mixup(X, y, alpha=0.2, feature_noise_std=0.05, seed=42)
        assert X_mix.shape == (32, 10)

    def test_without_noise_equals_standard(self):
        from src.training.mixup import tabular_mixup, mixup_data
        X = torch.randn(16, 5)
        y = torch.randint(0, 2, (16,))
        X_tab, _, _, _ = tabular_mixup(X, y, alpha=0.2, feature_noise_std=0.0, seed=42)
        X_std, _, _, _ = mixup_data(X, y, alpha=0.2, seed=42)
        assert torch.allclose(X_tab, X_std)


class TestMixupNumpy:
    def test_augments_samples(self):
        from src.training.mixup import mixup_numpy
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        X_aug, y_aug = mixup_numpy(X, y, alpha=0.2, seed=42)
        assert X_aug.shape[0] == 200  # original + synthetic
        assert y_aug.shape[0] == 200

    def test_custom_n_synthetic(self):
        from src.training.mixup import mixup_numpy
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        X_aug, y_aug = mixup_numpy(X, y, alpha=0.2, n_synthetic=30, seed=42)
        assert X_aug.shape[0] == 80

    def test_alpha_zero_returns_copy(self):
        from src.training.mixup import mixup_numpy
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        X_aug, y_aug = mixup_numpy(X, y, alpha=0.0, seed=42)
        assert X_aug.shape == X.shape
        np.testing.assert_array_equal(X_aug, X)

    def test_same_class_only(self):
        from src.training.mixup import mixup_numpy
        X = np.random.randn(40, 5)
        y = np.array([0]*20 + [1]*20)
        X_aug, y_aug = mixup_numpy(X, y, alpha=0.2, same_class_only=True, seed=42)
        assert X_aug.shape[0] > 40

    def test_labels_are_valid(self):
        from src.training.mixup import mixup_numpy
        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)
        _, y_aug = mixup_numpy(X, y, alpha=0.2, seed=42)
        assert set(np.unique(y_aug)).issubset({0, 1})

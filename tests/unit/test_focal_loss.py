# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_focal_loss.py
Unit tests for FocalLoss (src/training/focal_loss.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestFocalLoss:
    """FocalLoss modülü testleri."""

    def test_forward_shape(self):
        from src.training.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (), "Loss should be a scalar"
        assert loss.item() >= 0.0

    def test_gamma_zero_equals_ce(self):
        """γ=0 → Focal Loss == Cross-Entropy (without alpha)."""
        from src.training.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        fl = FocalLoss(gamma=0.0)(logits, targets)
        ce = torch.nn.CrossEntropyLoss()(logits, targets)
        assert abs(fl.item() - ce.item()) < 1e-5, "gamma=0 should equal CE"

    def test_higher_gamma_lower_easy_loss(self):
        """Higher gamma should down-weight easy examples → lower loss on confident predictions."""
        from src.training.focal_loss import FocalLoss

        # Create easy predictions (high confidence correct)
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [4.0, -4.0]], dtype=torch.float)
        targets = torch.tensor([0, 1, 0])

        loss_g0 = FocalLoss(gamma=0.0)(logits, targets).item()
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets).item()
        loss_g5 = FocalLoss(gamma=5.0)(logits, targets).item()
        assert loss_g2 < loss_g0, "gamma=2 should reduce easy example loss"
        assert loss_g5 < loss_g2, "gamma=5 should reduce even more"

    def test_alpha_weights(self):
        """Alpha class weights should affect loss."""
        from src.training.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(20, 2)
        targets = torch.randint(0, 2, (20,))

        alpha = torch.tensor([0.3, 0.7])
        loss_weighted = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)
        loss_unweighted = FocalLoss(gamma=2.0)(logits, targets)
        # They should be different (alpha changes the weighting)
        assert loss_weighted.item() != loss_unweighted.item()

    def test_reduction_none_shape(self):
        from src.training.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (10,), "reduction='none' should return per-sample loss"

    def test_reduction_sum(self):
        from src.training.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_from_labels_factory(self):
        """Factory method should compute balanced alpha weights."""
        from src.training.focal_loss import FocalLoss

        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # imbalanced: 3 vs 5
        fl = FocalLoss.from_labels(y, gamma=2.0, num_classes=2)
        assert fl.alpha is not None
        assert fl.alpha.shape == (2,)
        # Class 0 (minority) should have higher weight
        assert fl.alpha[0] > fl.alpha[1], "Minority class should have higher alpha"

    def test_gradient_flows(self):
        """Ensure gradients flow through FocalLoss."""
        from src.training.focal_loss import FocalLoss

        logits = torch.randn(8, 2, requires_grad=True)
        targets = torch.randint(0, 2, (8,))
        loss = FocalLoss(gamma=2.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None


"""
"""

"""
tests/unit/test_swa.py
Unit tests for Stochastic Weight Averaging (src/training/swa.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def _simple_model():
    """Create a simple model for SWA testing."""
    return nn.Sequential(
        nn.Linear(10, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


class TestSWABuffer:
    """SWABuffer testleri."""

    def test_should_collect_before_start(self):
        from src.training.swa import SWABuffer
        buf = SWABuffer(swa_start_fraction=0.75)
        assert not buf.should_collect(epoch=1, total_epochs=100)
        assert not buf.should_collect(epoch=74, total_epochs=100)

    def test_should_collect_after_start(self):
        from src.training.swa import SWABuffer
        buf = SWABuffer(swa_start_fraction=0.75)
        assert buf.should_collect(epoch=75, total_epochs=100)
        assert buf.should_collect(epoch=100, total_epochs=100)

    def test_push_stores_checkpoint(self):
        from src.training.swa import SWABuffer
        model = _simple_model()
        buf = SWABuffer(swa_start_fraction=0.75, max_checkpoints=5)
        # Before window — should not push
        assert not buf.push(epoch=10, total_epochs=100, model=model)
        assert len(buf) == 0
        # In window — should push
        assert buf.push(epoch=80, total_epochs=100, model=model)
        assert len(buf) == 1

    def test_max_checkpoints_eviction(self):
        from src.training.swa import SWABuffer
        model = _simple_model()
        buf = SWABuffer(swa_start_fraction=0.0, max_checkpoints=3)
        for ep in range(1, 10):
            buf.push(ep, 10, model)
        assert len(buf) == 3  # deque maxlen should enforce limit

    def test_apply_averages_weights(self):
        from src.training.swa import SWABuffer
        model = _simple_model()
        buf = SWABuffer(swa_start_fraction=0.0, max_checkpoints=10)

        # Push two different weight states
        model.apply(lambda m: (
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2
            else None
        ))
        buf.push(1, 2, model)

        model.apply(lambda m: (
            nn.init.constant_(m.weight, 3.0)
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2
            else None
        ))
        buf.push(2, 2, model)

        # Apply SWA — weights should be averaged
        buf.apply(model)
        # First linear layer weight should be ~2.0 (average of 1 and 3)
        first_linear = model[0]
        assert abs(first_linear.weight.data.mean().item() - 2.0) < 0.1

    def test_apply_with_single_checkpoint_returns_unchanged(self):
        from src.training.swa import SWABuffer
        model = _simple_model()
        buf = SWABuffer(swa_start_fraction=0.0)
        buf.push(1, 1, model)
        original_weight = model[0].weight.data.clone()
        buf.apply(model)  # Should log warning but not change
        assert torch.equal(model[0].weight.data, original_weight)

    def test_clear(self):
        from src.training.swa import SWABuffer
        buf = SWABuffer(swa_start_fraction=0.0)
        model = _simple_model()
        buf.push(1, 2, model)
        buf.push(2, 2, model)
        buf.clear()
        assert len(buf) == 0
        assert buf.n_collected == 0

    def test_invalid_fraction_raises(self):
        from src.training.swa import SWABuffer
        with pytest.raises(ValueError):
            SWABuffer(swa_start_fraction=-0.1)
        with pytest.raises(ValueError):
            SWABuffer(swa_start_fraction=1.0)


class TestCyclicSWAScheduler:
    def test_lr_oscillates(self):
        from src.training.swa import CyclicSWAScheduler
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CyclicSWAScheduler(optimizer, lr_min=1e-5, lr_max=5e-3, cycle_length=5)

        lrs = [scheduler.step(ep) for ep in range(10)]
        assert min(lrs) >= 1e-5 - 1e-8
        assert max(lrs) <= 5e-3 + 1e-8
        # LR should not be constant
        assert len(set(round(lr, 8) for lr in lrs)) > 1

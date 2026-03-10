"""
tests/smoke/test_smoke.py
Smoke tests — fast import checks and basic instantiation.
These should pass even without any trained models on disk.
"""
from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

REQUIRED_MODULES = [
    "src.config",
    "src.config.settings",
    "src.data.loader",
    "src.features.autoencoder",
    "src.features.preprocessing",
    "src.graph.builder",
    "src.models.gnn",
    "src.models.dnn",
    "src.models.ensemble",
    "src.training.trainer",
    "src.calibration.calibrator",
    "src.evaluation.metrics",
    "src.evaluation.plots",
    "src.inference.pipeline",
    "src.utils.logging_cfg",
    "src.utils.seeds",
    "src.utils.serialization",
    "data_contracts.variant_schema",
]


@pytest.mark.parametrize("module_path", REQUIRED_MODULES)
def test_module_importable(module_path):
    """Each core module must be importable without errors."""
    mod = importlib.import_module(module_path)
    assert mod is not None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestSmokeConfig:
    def test_load_settings_returns_object(self):
        from src.config import load_settings
        from src.config.settings import reset_settings
        reset_settings()
        cfg = load_settings()
        assert cfg is not None

    def test_settings_has_seed(self):
        from src.config import get_settings
        cfg = get_settings()
        assert hasattr(cfg, "seed")
        assert isinstance(cfg.seed, int)

    def test_ensemble_weights_length(self):
        from src.config import get_settings
        cfg = get_settings()
        assert len(cfg.ensemble.weights) == 3


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

class TestSmokeModels:
    def test_gnn_instantiates(self):
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=16, num_classes=2)
        assert model is not None

    def test_dnn_instantiates(self):
        from src.models.dnn import VariantDNN
        model = VariantDNN(input_dim=20, hidden_dim=32, num_classes=2)
        assert model is not None

    def test_ensemble_instantiates(self):
        from src.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[0.4, 0.4, 0.2])
        assert ens is not None

    def test_preprocessor_instantiates(self):
        from src.features.preprocessing import VariantPreprocessor
        pp = VariantPreprocessor()
        assert pp is not None

    def test_calibrator_instantiates(self):
        from src.calibration.calibrator import EnsembleCalibrator
        cal = EnsembleCalibrator(method="isotonic")
        assert cal is not None

    def test_trainer_instantiates(self):
        import torch

        from src.training.trainer import VariantTrainer
        trainer = VariantTrainer(device=torch.device("cpu"))
        assert trainer is not None


# ---------------------------------------------------------------------------
# Schema smoke
# ---------------------------------------------------------------------------

class TestSmokeSchema:
    def test_validate_minimal_dataframe(self):
        import numpy as np
        import pandas as pd

        from data_contracts.variant_schema import validate_dataset

        df = pd.DataFrame({
            "Variant_ID": ["V1", "V2", "V3", "V4"],
            "feat_a":     [0.1, 0.2, 0.3, 0.4],
            "feat_b":     [1.0, 2.0, 3.0, 4.0],
            "Label":      [0, 1, 0, 1],
        })
        result = validate_dataset(df)
        assert result.is_valid, f"Smoke schema failed: {result.errors}"


# ---------------------------------------------------------------------------
# Logging + seeds
# ---------------------------------------------------------------------------

class TestSmokeUtils:
    def test_setup_logging_does_not_crash(self):
        from src.utils.logging_cfg import setup_logging
        setup_logging(level="WARNING")

    def test_set_global_seed_does_not_crash(self):
        from src.utils.seeds import set_global_seed
        set_global_seed(42)

    def test_evaluation_metrics_importable(self):
        from src.evaluation.metrics import evaluate, expected_calibration_error, find_best_threshold
        assert callable(evaluate)
        assert callable(find_best_threshold)
        assert callable(expected_calibration_error)

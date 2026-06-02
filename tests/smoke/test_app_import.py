"""
tests/smoke/test_app_import.py
Compile-time and import-safety checks for app.py and related modules.
These must pass without any trained model artifacts on disk.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class TestAppCompile:
    def test_app_py_compiles(self):
        """app.py must compile without SyntaxError."""
        py_compile.compile(str(ROOT / "app.py"), doraise=True)

    def test_main_py_compiles(self):
        """main.py must compile without SyntaxError."""
        py_compile.compile(str(ROOT / "main.py"), doraise=True)


class TestGNNSettingsModelType:
    def test_gnn_settings_has_model_type(self):
        """GNNSettings must expose model_type to prevent AttributeError in trainer.py."""
        from src.config.settings import GNNSettings

        cfg = GNNSettings()
        assert hasattr(cfg, "model_type"), "GNNSettings missing model_type field"
        assert isinstance(cfg.model_type, str)
        assert cfg.model_type  # non-empty

    def test_get_settings_gnn_model_type(self):
        """Full config load must populate gnn.model_type."""
        from src.config.settings import load_settings, reset_settings

        reset_settings()
        cfg = load_settings()
        assert hasattr(cfg.gnn, "model_type")
        assert cfg.gnn.model_type


class TestAppImportPaths:
    """src.ui modules referenced by app.py must be importable."""

    @pytest.mark.parametrize(
        "module",
        [
            "src.ui.styles",
            "src.ui.header",
            "src.ui.sidebar",
            "src.ui.analytics",
            "src.ui.explainability",
            "src.ui.performance",
            "src.ui.clinvar",
            "src.ui.reporting",
            "src.ui.protein_viz",
        ],
    )
    def test_ui_module_importable(self, module):
        import importlib

        mod = importlib.import_module(module)
        assert mod is not None


class TestPipelineFallback:
    def test_inference_pipeline_init_without_artifacts(self):
        """InferencePipeline.__init__ must not raise even if model files are absent."""
        from src.api.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        assert pipeline is not None
        assert not pipeline._loaded

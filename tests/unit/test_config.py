"""
tests/unit/test_config.py
Unit tests for the configuration loader.
"""
import pytest
from pathlib import Path


class TestSettings:
    def test_defaults_loaded(self):
        from src.config import load_settings
        from src.config.settings import reset_settings
        reset_settings()
        cfg = load_settings()
        assert cfg.seed == 42
        assert len(cfg.ensemble.weights) == 3
        assert abs(sum(cfg.ensemble.weights) - 1.0) < 0.01

    def test_paths_are_path_objects(self):
        from src.config import load_settings
        cfg = load_settings()
        assert isinstance(cfg.paths.data_dir, Path)
        assert isinstance(cfg.paths.models_dir, Path)

    def test_label_mapping_keys_are_strings(self):
        from src.config import load_settings
        cfg = load_settings()
        for k in cfg.schema.label_mapping:
            assert isinstance(k, str)

    def test_xgb_as_dict(self):
        from src.config import load_settings
        cfg = load_settings()
        d   = cfg.xgb.as_dict()
        assert "objective" in d
        assert "n_estimators" in d

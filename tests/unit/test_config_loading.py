# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_config_loading.py
Config file loading and validation tests.

Ensures all competition YAML configs (default, pdr, psr, final, dev_quick)
are parseable, contain required keys, and have correct types.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path("configs")

REQUIRED_TOP_KEYS = {"seed", "ensemble", "training", "reproducibility"}

COMPETITION_CONFIGS = [
    "default.yaml",
    "pdr.yaml",
    "psr.yaml",
    "final.yaml",
    "dev_quick.yaml",
]


# ---------------------------------------------------------------------------
# YAML parse tests
# ---------------------------------------------------------------------------


class TestYAMLParsing:
    @pytest.mark.parametrize("config_name", COMPETITION_CONFIGS)
    def test_config_parseable(self, config_name):
        """All competition configs must be valid YAML."""
        path = CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")
        with open(path, encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        assert data is not None, f"{config_name} is empty"
        assert isinstance(data, dict), f"{config_name} must parse to dict"

    @pytest.mark.parametrize("config_name", COMPETITION_CONFIGS)
    def test_config_has_seed(self, config_name):
        """All configs must declare a seed."""
        path = CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")
        with open(path, encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        assert "seed" in data or "reproducibility" in data, (
            f"{config_name}: neither 'seed' nor 'reproducibility' key found"
        )

    def test_pdr_config_seed_is_42(self):
        """PDR config must use seed=42 for reproducibility."""
        path = CONFIGS_DIR / "pdr.yaml"
        if not path.exists():
            pytest.skip("pdr.yaml not found")
        with open(path, encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        seed = data.get("seed") or data.get("reproducibility", {}).get("seed")
        assert seed == 42, f"pdr.yaml seed={seed}, expected 42"

    def test_dev_quick_config_fewer_epochs(self):
        """dev_quick config must have fewer GNN epochs than default."""
        path_dq = CONFIGS_DIR / "dev_quick.yaml"
        path_df = CONFIGS_DIR / "default.yaml"
        if not path_dq.exists() or not path_df.exists():
            pytest.skip("dev_quick.yaml or default.yaml not found")
        with open(path_dq, encoding="utf-8") as f:
            dq = yaml.safe_load(f)
        with open(path_df, encoding="utf-8") as f:
            df = yaml.safe_load(f)
        # GNN epochs stored under gnn.epochs key
        dq_epochs = (dq.get("gnn") or {}).get("epochs", 999)
        df_epochs = (df.get("gnn") or {}).get("epochs", 50)
        assert dq_epochs <= df_epochs, f"dev_quick gnn.epochs ({dq_epochs}) not <= default ({df_epochs})"


# ---------------------------------------------------------------------------
# Settings module tests
# ---------------------------------------------------------------------------


class TestSettingsModule:
    def test_load_settings_returns_object(self):
        """load_settings() must return a settings object with seed."""
        from src.config import load_settings

        cfg = load_settings()
        assert hasattr(cfg, "seed")
        assert cfg.seed == 42

    def test_ensemble_weights_sum_to_one(self):
        """Ensemble weights must sum to 1.0 (probability simplex)."""
        from src.config import load_settings

        cfg = load_settings()
        weights = cfg.ensemble.weights
        assert abs(sum(weights) - 1.0) < 0.01, f"Ensemble weights sum={sum(weights):.4f}, expected 1.0"

    def test_ensemble_has_four_components(self):
        """Ensemble must have exactly 4 component weights (§VIII architecture)."""
        from src.config import load_settings

        cfg = load_settings()
        assert len(cfg.ensemble.weights) == 4, f"Expected 4 ensemble weights, got {len(cfg.ensemble.weights)}"

    def test_paths_are_path_objects(self):
        """Config paths must be Path objects (not raw strings)."""
        from src.config import load_settings

        cfg = load_settings()
        assert isinstance(cfg.paths.data_dir, Path)
        assert isinstance(cfg.paths.models_dir, Path)

    def test_label_mapping_binary(self):
        """Label mapping must produce exactly 2 classes: 0 and 1."""
        from src.config import load_settings

        cfg = load_settings()
        values = set(cfg.schema.label_mapping.values())
        assert values == {0, 1} or values == {"0", "1"} or values == {0.0, 1.0}, (
            f"Label mapping values {values} must be binary {{0,1}}"
        )


# ---------------------------------------------------------------------------
# Threshold file tests
# ---------------------------------------------------------------------------


class TestThresholdFiles:
    def test_threshold_json_exists(self):
        """models/threshold.json must exist after training."""
        path = Path("models/threshold.json")
        assert path.exists(), "models/threshold.json missing — run training first"

    def test_threshold_value_in_range(self):
        """Global threshold must be in (0, 1)."""
        import json

        path = Path("models/threshold.json")
        if not path.exists():
            pytest.skip("threshold.json not found")
        with open(path) as f:
            data = json.load(f)
        t = data.get("threshold", data.get("classification_threshold"))
        assert t is not None, "No threshold key in threshold.json"
        assert 0.0 < float(t) < 1.0, f"Threshold {t} not in (0,1)"

    def test_panel_thresholds_all_panels_present(self):
        """panel_thresholds.json must contain all four panels."""
        import json

        path = Path("models/panel_thresholds.json")
        if not path.exists():
            pytest.skip("panel_thresholds.json not found")
        with open(path) as f:
            data = json.load(f)
        for panel in ("General", "Hereditary_Cancer", "PAH", "CFTR"):
            assert panel in data, f"Panel {panel} missing from panel_thresholds.json"
            t = float(data[panel])
            assert 0.0 < t < 1.0, f"{panel} threshold {t} out of (0,1)"

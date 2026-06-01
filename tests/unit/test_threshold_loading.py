"""Regression tests for threshold loading (TEKNOFEST §7.5 reproducibility).

Guards against two bugs that caused the jury inference path to reproduce NONE of
the reported numbers:
  1. load_threshold read key 'classification_threshold' while the shipped
     models/threshold.json uses key 'threshold'  -> silent fallback to default.
  2. load_panel_thresholds expected a {"panel_thresholds": {...}} wrapper while
     the shipped models/panel_thresholds.json is flat -> returned {} -> a single
     global threshold was applied to all four panels.
"""
import json

import pytest

from src.utils.serialization import ModelStore


@pytest.fixture
def store(tmp_path):
    return ModelStore(model_dir=str(tmp_path))


def test_load_threshold_accepts_threshold_key(store, tmp_path):
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.59, "source": "calibration_set"}))
    assert store.load_threshold(default=0.40) == pytest.approx(0.59)


def test_load_threshold_accepts_classification_threshold_key(store, tmp_path):
    (tmp_path / "threshold.json").write_text(json.dumps({"classification_threshold": 0.33}))
    assert store.load_threshold(default=0.40) == pytest.approx(0.33)


def test_load_threshold_missing_file_returns_default(store):
    assert store.load_threshold(default=0.40) == pytest.approx(0.40)


def test_load_panel_thresholds_flat_file(store, tmp_path):
    flat = {"__global__": 0.59, "General": 0.59, "Hereditary_Cancer": 0.281, "PAH": 0.138, "CFTR": 0.108}
    (tmp_path / "panel_thresholds.json").write_text(json.dumps(flat))
    out = store.load_panel_thresholds()
    assert out["General"] == pytest.approx(0.59)
    assert out["PAH"] == pytest.approx(0.138)
    assert len(out) == 5


def test_load_panel_thresholds_wrapped_file(store, tmp_path):
    wrapped = {"panel_thresholds": {"General": 0.59, "CFTR": 0.108}}
    (tmp_path / "panel_thresholds.json").write_text(json.dumps(wrapped))
    out = store.load_panel_thresholds()
    assert out["General"] == pytest.approx(0.59)
    assert out["CFTR"] == pytest.approx(0.108)


def test_load_panel_thresholds_ignores_non_numeric(store, tmp_path):
    flat = {"General": 0.59, "source": "calibration_set", "metric": "F1_max"}
    (tmp_path / "panel_thresholds.json").write_text(json.dumps(flat))
    out = store.load_panel_thresholds()
    assert out == {"General": pytest.approx(0.59)}

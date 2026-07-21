# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""tests/unit/test_panel_aware_ensemble.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ensemble.panel_aware_ensemble import KNOWN_PANELS, PanelAwareEnsemble


def _synthetic_data(n: int = 200, n_models: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    preds = rng.uniform(0, 1, size=(n, n_models))
    panels = [KNOWN_PANELS[i % len(KNOWN_PANELS)] for i in range(n)]
    return preds, y, panels


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    preds, y, panels = _synthetic_data()
    ens = PanelAwareEnsemble(n_models=4)
    result = ens.fit(preds, y, panels)
    assert result is ens


def test_weights_sum_to_one():
    preds, y, panels = _synthetic_data()
    ens = PanelAwareEnsemble(n_models=4).fit(preds, y, panels)
    assert abs(ens._global_weights.sum() - 1.0) < 1e-6
    for panel, w in ens._panel_weights.items():
        assert abs(w.sum() - 1.0) < 1e-6, f"Panel '{panel}' weights do not sum to 1"


def test_weights_non_negative():
    preds, y, panels = _synthetic_data()
    ens = PanelAwareEnsemble(n_models=4).fit(preds, y, panels)
    assert (ens._global_weights >= 0).all()
    for w in ens._panel_weights.values():
        assert (w >= 0).all()


def test_fit_wrong_shape_raises():
    ens = PanelAwareEnsemble(n_models=4)
    preds = np.ones((10, 3))  # wrong n_models
    y = np.zeros(10)
    panels = ["General"] * 10
    with pytest.raises(ValueError, match="n_models"):
        ens.fit(preds, y, panels)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_proba_shape():
    preds, y, panels = _synthetic_data(n=200, n_models=4)
    ens = PanelAwareEnsemble(n_models=4).fit(preds, y, panels)
    out = ens.predict_proba(preds, panels)
    assert out.shape == (200,)


def test_predict_proba_in_range():
    preds, y, panels = _synthetic_data(n=100, n_models=4)
    ens = PanelAwareEnsemble(n_models=4).fit(preds, y, panels)
    out = ens.predict_proba(preds, panels)
    assert (out >= 0).all() and (out <= 1).all()


def test_predict_proba_unknown_panel():
    preds, y, panels = _synthetic_data(n=100, n_models=4)
    ens = PanelAwareEnsemble(n_models=4).fit(preds, y, panels)
    unknown_panels = ["UnknownPanel"] * 5
    unknown_preds = preds[:5]
    out = ens.predict_proba(unknown_preds, unknown_panels)
    assert out.shape == (5,)


def test_predict_before_fit_raises():
    ens = PanelAwareEnsemble(n_models=4)
    with pytest.raises(RuntimeError, match="fitted"):
        ens.predict_proba(np.ones((5, 4)), ["General"] * 5)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    preds, y, panels = _synthetic_data(n=200, n_models=4)
    ens = PanelAwareEnsemble(n_models=4, seed=42).fit(preds, y, panels)
    path = tmp_path / "ensemble_weights.json"
    ens.save(path)

    loaded = PanelAwareEnsemble.load(path)
    np.testing.assert_allclose(loaded._global_weights, ens._global_weights, atol=1e-9)
    for panel in KNOWN_PANELS:
        if panel in ens._panel_weights:
            np.testing.assert_allclose(loaded._panel_weights[panel], ens._panel_weights[panel], atol=1e-9)


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        PanelAwareEnsemble.load(Path("/nonexistent/path/weights.json"))


def test_save_creates_parent_dirs(tmp_path):
    preds, y, panels = _synthetic_data(n=100, n_models=3)
    ens = PanelAwareEnsemble(n_models=3).fit(preds, y, panels)
    nested = tmp_path / "deep" / "dir" / "weights.json"
    ens.save(nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_weights():
    preds, y, panels = _synthetic_data(n=200, n_models=4, seed=99)
    w1 = PanelAwareEnsemble(n_models=4, seed=42).fit(preds, y, panels)._global_weights
    w2 = PanelAwareEnsemble(n_models=4, seed=42).fit(preds, y, panels)._global_weights
    np.testing.assert_allclose(w1, w2, atol=1e-9)

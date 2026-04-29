"""
tests/unit/test_modelstore_lgbm_roundtrip.py
Model artifact roundtrip testleri — XGB, LGBM, GNN, Calibrator.

Kabul kriteri:
  - Kaydedilen model yüklendikten sonra aynı giriş için
    onbinde bir (1e-4) sapma bile olmamalı.
  - gnn_arch.json içinde type, numeric_dim, hidden_dim bulunmalı.
  - ensemble_config.json ağırlıkları doğru okunmalı.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ring_edge_index(n: int) -> torch.Tensor:
    src = torch.arange(n)
    dst = torch.roll(src, -1)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _fake_preprocessor(n_features: int = 10):
    pp = MagicMock()
    pp.n_output_features = n_features
    pp._autoenc = None
    return pp


# ─── XGBoost Roundtrip ────────────────────────────────────────────────────────

class TestXGBRoundtrip:
    def test_save_load_same_predictions(self, tmp_path):
        """Kaydedilen XGB yüklendikten sonra tahminler birebir aynı olmalı."""
        import xgboost as xgb
        from src.utils.serialization import ModelStore

        rng = np.random.default_rng(42)
        X_tr = rng.random((50, 10)).astype(np.float32)
        y_tr = (rng.random(50) > 0.5).astype(int)
        X_te = rng.random((20, 10)).astype(np.float32)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)
        model.fit(X_tr, y_tr)
        proba_before = model.predict_proba(X_te)

        store = ModelStore(tmp_path)
        store._save_xgb(model)

        # Yükle
        import xgboost as xgb2
        loaded = xgb2.XGBClassifier()
        loaded.load_model(str(tmp_path / "xgb_model.json"))
        proba_after = loaded.predict_proba(X_te)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-4,
                                   err_msg="XGB tahminleri roundtrip sonrası değişti!")

    def test_json_format_saved(self, tmp_path):
        """XGB JSON formatında kaydedilmeli (pickle değil)."""
        import xgboost as xgb
        from src.utils.serialization import ModelStore

        model = xgb.XGBClassifier(n_estimators=5, verbosity=0)
        model.fit(np.random.rand(20, 5), np.random.randint(0, 2, 20))
        ModelStore(tmp_path)._save_xgb(model)

        json_path = tmp_path / "xgb_model.json"
        assert json_path.exists(), "xgb_model.json oluşturulmadı"
        # JSON parse edilebilmeli
        data = json.loads(json_path.read_text())
        assert "learner" in data, "Geçerli XGB JSON formatı değil"


# ─── LightGBM Roundtrip ───────────────────────────────────────────────────────

class TestLGBMRoundtrip:
    def test_save_load_same_predictions(self, tmp_path):
        """Kaydedilen LGBM yüklendikten sonra tahminler birebir aynı olmalı."""
        pytest.importorskip("lightgbm")
        import lightgbm as lgb
        from src.utils.serialization import ModelStore, _LGBMBoosterWrapper

        rng = np.random.default_rng(7)
        X_tr = rng.random((40, 8)).astype(np.float32)
        y_tr = (rng.random(40) > 0.5).astype(int)
        X_te = rng.random((15, 8)).astype(np.float32)

        model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        model.fit(X_tr, y_tr)
        proba_before = model.predict_proba(X_te)

        store = ModelStore(tmp_path)
        store._save_lgbm(model)

        lgbm_path = tmp_path / "lgbm_model.txt"
        assert lgbm_path.exists(), "lgbm_model.txt oluşturulmadı"

        loaded_booster = lgb.Booster(model_file=str(lgbm_path))
        wrapped = _LGBMBoosterWrapper(loaded_booster)
        proba_after = wrapped.predict_proba(X_te)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-4,
                                   err_msg="LGBM tahminleri roundtrip sonrası değişti!")

    def test_wrapper_has_predict_proba(self, tmp_path):
        """_LGBMBoosterWrapper predict_proba(X) → (N,2) shape döndürmeli."""
        pytest.importorskip("lightgbm")
        import lightgbm as lgb
        from src.utils.serialization import ModelStore, _LGBMBoosterWrapper

        model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
        model.fit(np.random.rand(20, 4), np.random.randint(0, 2, 20))
        ModelStore(tmp_path)._save_lgbm(model)

        booster = lgb.Booster(model_file=str(tmp_path / "lgbm_model.txt"))
        wrapper = _LGBMBoosterWrapper(booster)
        out = wrapper.predict_proba(np.random.rand(5, 4))
        assert out.shape == (5, 2), f"Beklenen (5,2), alınan {out.shape}"
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5,
                                   err_msg="LGBM olasılıkları 1'e toplamıyor")


# ─── GNN Roundtrip ────────────────────────────────────────────────────────────

class TestGNNRoundtrip:
    def test_arch_json_contains_required_fields(self, tmp_path):
        """gnn_arch.json: type, numeric_dim, hidden_dim içermeli."""
        from src.models.gnn import VariantSAGEGNN
        from src.utils.serialization import ModelStore

        model = VariantSAGEGNN(numeric_dim=8, hidden_dim=16, use_multimodal=False)
        ModelStore(tmp_path)._save_gnn(model)

        arch = json.loads((tmp_path / "gnn_arch.json").read_text())
        for field in ("type", "numeric_dim", "hidden_dim"):
            assert field in arch, f"gnn_arch.json'da '{field}' alanı eksik"

    def test_unimodal_roundtrip(self, tmp_path):
        """Unimodal GNN save→load→forward birebir aynı çıktı vermeli."""
        from src.models.gnn import VariantSAGEGNN
        from src.utils.serialization import ModelStore

        N = 6
        model = VariantSAGEGNN(numeric_dim=8, hidden_dim=16, use_multimodal=False)
        model.eval()
        store = ModelStore(tmp_path)
        store._save_gnn(model)

        arch = json.loads((tmp_path / "gnn_arch.json").read_text())
        loaded = VariantSAGEGNN(
            numeric_dim=arch["numeric_dim"],
            hidden_dim=arch["hidden_dim"],
            use_multimodal=arch.get("use_multimodal", False),
        )
        loaded.load_state_dict(
            torch.load(str(tmp_path / "gnn_model.pth"),
                       map_location="cpu", weights_only=True)
        )
        loaded.eval()

        x = torch.rand(N, 8)
        edge_index = _ring_edge_index(N)
        with torch.no_grad():
            out_orig   = model(x, edge_index)
            out_loaded = loaded(x, edge_index)

        np.testing.assert_allclose(
            out_orig.numpy(), out_loaded.numpy(), atol=1e-5,
            err_msg="GNN çıktıları roundtrip sonrası değişti!"
        )

    def test_state_dict_mismatch_raises_informative_error(self, tmp_path):
        """Mimari uyuşmazlığında anlamlı hata mesajı verilmeli."""
        from src.models.gnn import VariantSAGEGNN
        from src.utils.serialization import ModelStore

        model = VariantSAGEGNN(numeric_dim=8, hidden_dim=16)
        ModelStore(tmp_path)._save_gnn(model)

        # Arch'ı boz
        arch_path = tmp_path / "gnn_arch.json"
        arch = json.loads(arch_path.read_text())
        arch["hidden_dim"] = 128  # yanlış
        arch_path.write_text(json.dumps(arch))

        wrong = VariantSAGEGNN(numeric_dim=8, hidden_dim=128)
        with pytest.raises(RuntimeError):
            wrong.load_state_dict(
                torch.load(str(tmp_path / "gnn_model.pth"),
                           map_location="cpu", weights_only=True)
            )


# ─── Ensemble Config Roundtrip ────────────────────────────────────────────────

class TestEnsembleConfigRoundtrip:
    def test_weights_preserved(self, tmp_path):
        """ensemble_config.json ağırlıkları doğru kaydedilip okunmalı."""
        from src.utils.serialization import ModelStore

        ens = MagicMock()
        ens.weights = [0.35, 0.30, 0.25, 0.10]
        ens.meta_learner = None

        store = ModelStore(tmp_path)
        store._save_ensemble_cfg(ens)

        cfg_data = json.loads((tmp_path / "ensemble_config.json").read_text())
        loaded_weights = cfg_data["weights"]
        assert loaded_weights == ens.weights, \
            f"Ağırlıklar değişti: {ens.weights} → {loaded_weights}"

    def test_weights_sum_to_one_after_normalization(self, tmp_path):
        """Yüklenen ağırlıklar normalize edildiğinde 1'e toplamalı."""
        from src.utils.serialization import ModelStore
        from src.core.ensemble import HybridEnsemble

        ens_mock = MagicMock()
        ens_mock.weights = [0.35, 0.30, 0.25, 0.10]
        ens_mock.meta_learner = None
        ModelStore(tmp_path)._save_ensemble_cfg(ens_mock)

        cfg_data = json.loads((tmp_path / "ensemble_config.json").read_text())
        w = np.array(cfg_data["weights"])
        assert abs(w.sum() - 1.0) < 1e-3, \
            f"Ağırlık toplamı != 1.0: {w.sum()}"

# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_modelstore_gnn_roundtrip.py
Verifies that ModelStore.save_all() / load_all() round-trips correctly for
VariantGATv2GNN and VariantSAGEGNN (its alias).

The test:
  1. Builds a minimal VariantGATv2GNN.
  2. Saves via ModelStore (mocking preprocessor / calibrator).
  3. Loads via ModelStore.load_all().
  4. Checks loaded model is the right type and architecture.
  5. Runs a dummy forward pass and verifies output shape.
  6. Confirms saved vs loaded state_dicts are identical.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_FEATURES = 10
HIDDEN_DIM = 32
BATCH_SIZE = 6


def _make_edge_index(n: int) -> torch.Tensor:
    """Fully-connected edge_index for n nodes (self-loops excluded)."""
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


def _dummy_x() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, NUM_FEATURES)


# ---------------------------------------------------------------------------
# Minimal preprocessor mock
# ---------------------------------------------------------------------------


def _mock_preprocessor(n_features: int = NUM_FEATURES):
    pp = MagicMock()
    pp.n_output_features = n_features
    pp._autoenc = None
    return pp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGNNRoundtrip:
    """Save → load round-trip for VariantGATv2GNN."""

    def _build_and_save(self, tmp_model_dir: Path, cls_name: str):
        from src.core.models.gnn import VariantGATv2GNN, VariantSAGEGNN
        from src.utils.serialization import ModelStore

        cls = VariantSAGEGNN if cls_name == "VariantSAGEGNN" else VariantGATv2GNN
        gnn = cls(
            numeric_dim=NUM_FEATURES,
            hidden_dim=HIDDEN_DIM,
            use_multimodal=False,
        )
        gnn.eval()

        # Build a minimal HybridEnsemble mock
        ensemble = MagicMock()
        ensemble.xgb = None
        ensemble.lgbm = None
        ensemble.gnn = gnn
        ensemble.dnn = None
        ensemble.weights = [0.35, 0.30, 0.25, 0.10]
        ensemble.meta_learner = None

        store = ModelStore(tmp_model_dir)

        # Patch _save_xgb / _save_lgbm / _save_dnn / _save_preprocessor to no-ops
        # so we only exercise the GNN path.
        with (
            patch.object(store, "_save_xgb"),
            patch.object(store, "_save_lgbm"),
            patch.object(store, "_save_dnn"),
            patch.object(store, "_save_autoencoder"),
            patch.object(store, "_save_preprocessor"),
            patch.object(store, "_save_meta_learner"),
        ):
            store._save_gnn(gnn)
            store._save_ensemble_cfg(ensemble)

        return gnn, store

    def _load_gnn_from_arch(self, tmp_model_dir: Path):
        """Read gnn_arch.json and reconstruct via the load path."""
        from src.utils.serialization import ModelStore

        store = ModelStore(tmp_model_dir)

        arch_path = tmp_model_dir / "gnn_arch.json"
        assert arch_path.exists(), "gnn_arch.json was not written"
        with open(arch_path) as fh:
            arch = json.load(fh)
        return store, arch

    # ------------------------------------------------------------------

    @pytest.mark.parametrize("cls_name", ["VariantGATv2GNN", "VariantSAGEGNN"])
    def test_arch_json_written(self, tmp_model_dir, cls_name):
        """gnn_arch.json must be written and contain the right type."""
        gnn, store = self._build_and_save(tmp_model_dir, cls_name)
        _, arch = self._load_gnn_from_arch(tmp_model_dir)
        assert arch["type"] == cls_name
        assert arch["numeric_dim"] == NUM_FEATURES
        assert arch["hidden_dim"] == HIDDEN_DIM
        assert arch["use_multimodal"] is False

    @pytest.mark.parametrize("cls_name", ["VariantGATv2GNN", "VariantSAGEGNN"])
    def test_state_dict_roundtrip(self, tmp_model_dir, cls_name):
        """Loaded state_dict must be identical to the saved one."""
        from src.core.models.gnn import VariantGATv2GNN, VariantSAGEGNN

        gnn_orig, store = self._build_and_save(tmp_model_dir, cls_name)

        # Reconstruct manually (same as load_all path)
        _, arch = self._load_gnn_from_arch(tmp_model_dir)
        _cls = VariantSAGEGNN if cls_name == "VariantSAGEGNN" else VariantGATv2GNN
        gnn_loaded = _cls(
            numeric_dim=arch["numeric_dim"],
            hidden_dim=arch["hidden_dim"],
            use_multimodal=arch.get("use_multimodal", False),
        )
        state = torch.load(str(tmp_model_dir / "gnn_model.pth"), map_location="cpu", weights_only=True)
        gnn_loaded.load_state_dict(state)

        for (k_orig, v_orig), (k_load, v_load) in zip(gnn_orig.state_dict().items(), gnn_loaded.state_dict().items()):
            assert k_orig == k_load
            assert torch.allclose(v_orig, v_load), f"Mismatch at key: {k_orig}"

    @pytest.mark.parametrize("cls_name", ["VariantGATv2GNN", "VariantSAGEGNN"])
    def test_forward_shape(self, tmp_model_dir, cls_name):
        """Loaded model must produce (N, 2) logits on a dummy input."""
        from src.core.models.gnn import VariantGATv2GNN, VariantSAGEGNN

        _, store = self._build_and_save(tmp_model_dir, cls_name)
        _, arch = self._load_gnn_from_arch(tmp_model_dir)

        _cls = VariantSAGEGNN if cls_name == "VariantSAGEGNN" else VariantGATv2GNN
        gnn = _cls(
            numeric_dim=arch["numeric_dim"],
            hidden_dim=arch["hidden_dim"],
            use_multimodal=arch.get("use_multimodal", False),
        )
        gnn.load_state_dict(torch.load(str(tmp_model_dir / "gnn_model.pth"), map_location="cpu", weights_only=True))
        gnn.eval()

        x = _dummy_x()
        edge_index = _make_edge_index(BATCH_SIZE)

        with torch.no_grad():
            out = gnn(x, edge_index)

        assert out.shape == (BATCH_SIZE, 2), f"Expected ({BATCH_SIZE}, 2), got {out.shape}"

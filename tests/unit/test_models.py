"""
tests/unit/test_models.py
Unit tests for GNN, DNN, and HybridEnsemble.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pyg_batch(n_nodes: int = 8, n_samples: int = 4, in_channels: int = 1):
    """Return a simple PyG Data batch suitable for FeatureGNN forward pass."""
    from torch_geometric.data import Batch, Data

    graphs = []
    for _ in range(n_samples):
        # fully-connected edges between n_nodes nodes
        row = torch.arange(n_nodes).unsqueeze(0).repeat(n_nodes, 1).flatten()
        col = torch.arange(n_nodes).unsqueeze(1).repeat(1, n_nodes).flatten()
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        x = torch.randn(n_nodes, in_channels)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# FeatureGNN
# ---------------------------------------------------------------------------

class TestFeatureGNN:
    def test_forward_shape_gcn(self):
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=16, num_classes=2, use_gat=False)
        batch = _make_pyg_batch(n_nodes=10, n_samples=4)
        out = model(batch)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_forward_shape_gat(self):
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=16, num_classes=2, use_gat=True)
        batch = _make_pyg_batch(n_nodes=10, n_samples=3)
        out = model(batch)
        assert out.shape == (3, 2)

    def test_output_not_probabilities(self):
        """Raw logits should not sum to 1 (softmax not applied in forward)."""
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=16, num_classes=2)
        batch = _make_pyg_batch(n_nodes=6, n_samples=2)
        out = model(batch)
        row_sums = out.sum(dim=1).detach().numpy()
        # logits should NOT sum to 1.0
        assert not np.allclose(row_sums, 1.0, atol=0.01), "Forward should return logits, not probabilities"

    def test_gradient_flows(self):
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=16, num_classes=2)
        batch = _make_pyg_batch(n_nodes=8, n_samples=4)
        out   = model(batch)
        loss  = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_dropout_train_vs_eval(self):
        from src.models.gnn import FeatureGNN
        model = FeatureGNN(in_channels=1, hidden_dim=32, num_classes=2, dropout=0.9)
        batch = _make_pyg_batch(n_nodes=8, n_samples=10)
        model.train()
        model(batch)  # exercise training mode
        model.eval()
        out_eval  = model(batch).detach()
        # Eval output should be deterministic across two runs
        out_eval2 = model(batch).detach()
        assert torch.allclose(out_eval, out_eval2), "Eval outputs should be deterministic"


# ---------------------------------------------------------------------------
# VariantDNN
# ---------------------------------------------------------------------------

class TestVariantDNN:
    def test_forward_shape(self):
        from src.models.dnn import VariantDNN
        model = VariantDNN(input_dim=20, hidden_dim=64, num_classes=2)
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 2)

    def test_different_input_dims(self):
        from src.models.dnn import VariantDNN
        for dim in [10, 50, 100]:
            model = VariantDNN(input_dim=dim, hidden_dim=32, num_classes=2)
            x = torch.randn(4, dim)
            out = model(x)
            assert out.shape == (4, 2), f"Failed for input_dim={dim}"

    def test_gradient_flows(self):
        from src.models.dnn import VariantDNN
        model = VariantDNN(input_dim=16, hidden_dim=32, num_classes=2)
        x    = torch.randn(6, 16)
        out  = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_norm_requires_batch_size_gt_1(self):
        from src.models.dnn import VariantDNN
        model = VariantDNN(input_dim=8, hidden_dim=16, num_classes=2)
        model.eval()          # eval mode: BN uses running stats → single sample OK
        x = torch.randn(1, 8)
        out = model(x)
        assert out.shape == (1, 2)


# ---------------------------------------------------------------------------
# HybridEnsemble
# ---------------------------------------------------------------------------

class TestHybridEnsemble:
    def _make_ensemble(self):
        from src.models.ensemble import HybridEnsemble
        return HybridEnsemble(weights=[0.4, 0.4, 0.2])

    def test_combine_with_all_models_none(self):
        """combine() with no models should still return shape (N, 2) from uniform fallback."""
        from src.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[0.4, 0.4, 0.2])
        # Only pass xgb_proba, leave others None → ensemble uses available models
        xgb_proba = np.array([[0.7, 0.3], [0.4, 0.6]])
        result = ens.combine(xgb_proba=xgb_proba, gnn_proba=None, dnn_proba=None)
        assert result.shape == (2, 2)
        # Rows should sum to 1
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_combine_three_models(self):
        from src.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[0.4, 0.4, 0.2])
        N = 10
        xgb_p = np.random.dirichlet([1, 1], size=N)
        gnn_p = np.random.dirichlet([1, 1], size=N)
        dnn_p = np.random.dirichlet([1, 1], size=N)
        result = ens.combine(xgb_proba=xgb_p, gnn_proba=gnn_p, dnn_proba=dnn_p)
        assert result.shape == (N, 2)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_returns_class_indices(self):
        from src.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[0.4, 0.4, 0.2])
        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        preds = ens.predict(proba)
        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(preds, expected)

    def test_label_map_covers_all_classes(self):
        from src.models.ensemble import HybridEnsemble
        assert 0 in HybridEnsemble.LABEL_MAP
        assert 1 in HybridEnsemble.LABEL_MAP
        assert HybridEnsemble.LABEL_MAP[0] == "Benign"
        assert HybridEnsemble.LABEL_MAP[1] == "Pathogenic"

    def test_weights_sum_to_one_after_normalise(self):
        from src.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[2.0, 2.0, 1.0])  # un-normalised
        # combine normalises weights internally
        xgb_p = np.array([[0.6, 0.4], [0.3, 0.7]])
        gnn_p = np.array([[0.5, 0.5], [0.4, 0.6]])
        dnn_p = np.array([[0.7, 0.3], [0.2, 0.8]])
        result = ens.combine(xgb_proba=xgb_p, gnn_proba=gnn_p, dnn_proba=dnn_p)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)

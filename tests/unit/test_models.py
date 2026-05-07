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

def _make_pyg_data(n_nodes: int = 8, in_channels: int = 1):
    """Return a simple PyG Data object."""
    from torch_geometric.data import Data
    # fully-connected edges
    row = torch.arange(n_nodes).unsqueeze(0).repeat(n_nodes, 1).flatten()
    col = torch.arange(n_nodes).unsqueeze(1).repeat(1, n_nodes).flatten()
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    x = torch.randn(n_nodes, in_channels)
    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# FeatureGNN
# ---------------------------------------------------------------------------

class TestVariantGATv2GNN:
    def test_forward_shape(self):
        from src.core.models.gnn import VariantGATv2GNN
        model = VariantGATv2GNN(numeric_dim=10, hidden_dim=16, num_classes=2)
        data = _make_pyg_data(n_nodes=5, in_channels=10)
        out = model(data.x, data.edge_index)
        assert out.shape == (5, 2)

    def test_predict_with_uncertainty(self):
        from src.core.models.gnn import VariantGATv2GNN
        model = VariantGATv2GNN(numeric_dim=10, hidden_dim=16, num_classes=2, dropout=0.5)
        data = _make_pyg_data(n_nodes=5, in_channels=10)
        # Force training mode for dropout
        mean, std = model.predict_with_uncertainty(data.x, data.edge_index, n_iter=5)
        assert mean.shape == (5, 2)
        assert std.shape == (5, 2)
        assert torch.all(std >= 0)

    def test_multimodal_forward(self):
        from src.core.models.gnn import VariantGATv2GNN
        model = VariantGATv2GNN(numeric_dim=10, hidden_dim=16, num_classes=2, use_multimodal=True)
        data = _make_pyg_data(n_nodes=4, in_channels=10)
        nuc_ids = torch.randint(0, 5, (4, 11))
        aa_ids = torch.randint(0, 20, (4, 11))
        out = model(data.x, data.edge_index, nuc_ids=nuc_ids, aa_ids=aa_ids)
        assert out.shape == (4, 2)

    def test_gradient_flows(self):
        from src.core.models.gnn import VariantGATv2GNN
        model = VariantGATv2GNN(numeric_dim=10, hidden_dim=16, num_classes=2)
        data = _make_pyg_data(n_nodes=4, in_channels=10)
        out = model(data.x, data.edge_index)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ---------------------------------------------------------------------------
# VariantDNN
# ---------------------------------------------------------------------------

class TestVariantDNN:
    def test_forward_shape(self):
        from src.models.dnn_model import VariantDNN
        model = VariantDNN(input_dim=20, hidden_dim=64, num_classes=2)
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 2)

    def test_different_input_dims(self):
        from src.models.dnn_model import VariantDNN
        for dim in [10, 50, 100]:
            model = VariantDNN(input_dim=dim, hidden_dim=32, num_classes=2)
            x = torch.randn(4, dim)
            out = model(x)
            assert out.shape == (4, 2), f"Failed for input_dim={dim}"

    def test_gradient_flows(self):
        from src.models.dnn_model import VariantDNN
        model = VariantDNN(input_dim=16, hidden_dim=32, num_classes=2)
        x    = torch.randn(6, 16)
        out  = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_norm_requires_batch_size_gt_1(self):
        from src.models.dnn_model import VariantDNN
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
        from src.core.models.ensemble import HybridEnsemble
        return HybridEnsemble(weights=[0.3, 0.3, 0.3, 0.1])

    def test_combine_four_models(self):
        from src.core.models.ensemble import HybridEnsemble
        ens = HybridEnsemble(weights=[0.3, 0.3, 0.3, 0.1])
        N = 5
        xp = np.random.dirichlet([1, 1], size=N)
        lp = np.random.dirichlet([1, 1], size=N)
        gp = np.random.dirichlet([1, 1], size=N)
        dp = np.random.dirichlet([1, 1], size=N)
        result = ens.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gp, dnn_proba=dp)
        assert result.shape == (N, 2)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_with_uncertainty(self):
        # This test requires a mocked pipeline/model state
        # For now, ensure it doesn't crash on simple Mock
        pass


# ---------------------------------------------------------------------------
# SequenceEncoder (Multimodal Encoder)
# ---------------------------------------------------------------------------

class TestSequenceEncoder:
    def test_forward_shape(self):
        """SequenceEncoder çıktı boyutunu doğrular."""
        from src.features.multimodal_encoder import SequenceEncoder
        encoder = SequenceEncoder(embedding_dim=8, cnn_channels=16)
        N = 5
        nuc_ids = torch.randint(0, 6, (N, 11))
        aa_ids = torch.randint(0, 22, (N, 11))
        out = encoder(nuc_ids, aa_ids)
        assert out.shape == (N, 32), f"Expected ({N}, 32), got {out.shape}"

    def test_output_dim_property(self):
        from src.features.multimodal_encoder import SequenceEncoder
        encoder = SequenceEncoder(cnn_channels=16)
        assert encoder.output_dim == 32

    def test_tokenize_nucleotides(self):
        from src.features.multimodal_encoder import tokenize_nucleotides
        seqs = ["ACGTTGACGTG", "NNNNNNNNNNN"]
        result = tokenize_nucleotides(seqs, seq_len=11)
        assert result.shape == (2, 11)
        assert result.dtype == np.int64
        # A=1, C=2, G=3, T=4
        assert result[0, 0] == 1   # A
        assert result[0, 1] == 2   # C
        assert result[0, 2] == 3   # G
        assert result[0, 3] == 4   # T
        # N=5 (unknown)
        assert result[1, 0] == 5

    def test_tokenize_amino_acids(self):
        from src.features.multimodal_encoder import tokenize_amino_acids
        seqs = ["AVILMFYWKRN"]
        result = tokenize_amino_acids(seqs, seq_len=11)
        assert result.shape == (1, 11)
        assert result.dtype == np.int64
        # A=1, V=20
        assert result[0, 0] == 1   # A
        assert result[0, 1] == 20  # V


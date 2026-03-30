from __future__ import annotations

import torch.nn as nn

from src.core.gnn import VariantGATv2GNN


class VariantSAGEGNN(VariantGATv2GNN):
    """Backward-compatible alias for historical checkpoints/tests."""


class FeatureGNN(nn.Module):
    """Compatibility wrapper with the legacy FeatureGNN constructor."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        use_gat: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_gat = use_gat
        self.model = VariantGATv2GNN(
            numeric_dim=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x, edge_index, *args, **kwargs):
        return self.model(x, edge_index, *args, **kwargs)


__all__ = ["VariantGATv2GNN", "VariantSAGEGNN", "FeatureGNN"]

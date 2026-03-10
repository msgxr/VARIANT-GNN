"""
src/models/gnn.py
Feature-Interaction Graph Neural Network for variant pathogenicity prediction.
Dynamic input dimension — no hardcoded feature count.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class FeatureGNN(nn.Module):
    """
    GNN that treats each feature column as a graph node.
    Edge structure is provided by CorrelationGraphBuilder (fit on training fold).

    Parameters
    ----------
    in_channels : Node feature dimensionality (1 for scalar feature values).
    hidden_dim  : Hidden layer width.
    num_classes : Output classes (2 for binary Benign/Pathogenic).
    use_gat     : Use Graph Attention (GATConv) instead of GCN.
    dropout     : Dropout rate in classifier head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 2,
        use_gat: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_gat = use_gat

        if use_gat:
            self.conv1 = GATConv(in_channels,     hidden_dim,     heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim,       hidden_dim * 2, heads=4, concat=False)
            self.conv3 = GATConv(hidden_dim * 2,   hidden_dim,     heads=4, concat=False)
        else:
            self.conv1 = GCNConv(in_channels,     hidden_dim)
            self.conv2 = GCNConv(hidden_dim,       hidden_dim * 2)
            self.conv3 = GCNConv(hidden_dim * 2,   hidden_dim)

        self.fc1     = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2     = nn.Linear(32, num_classes)

    # ------------------------------------------------------------------
    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.use_gat:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
        else:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = F.relu(self.conv3(x, edge_index, edge_weight))

        x      = global_mean_pool(x, batch)
        x      = F.relu(self.fc1(x))
        x      = self.dropout(x)
        logits = self.fc2(x)
        return logits

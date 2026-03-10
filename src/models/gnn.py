"""
src/models/gnn.py
Graph Neural Network models for variant pathogenicity prediction.

Models
------
FeatureGNN     : Original feature-interaction GNN (features as nodes).
                 Kept for backward-compatibility.
VariantSAGEGNN : TEKNOFEST 2026 — inductive, node-level classifier.
                 Each VARIANT is a node; SAGEConv + BatchNorm + Skip
                 connections + Dropout(0.3).  Optionally fuses nucleotide
                 and amino-acid sequence context (multimodal input).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm as PyGBatchNorm,
    GATConv,
    GCNConv,
    SAGEConv,
    global_mean_pool,
)


# ===========================================================================
# FeatureGNN — legacy (features-as-nodes, unchanged)
# ===========================================================================

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


# ===========================================================================
# _SAGEBlock — reusable SAGEConv + BN + Skip unit
# ===========================================================================

class _SAGEBlock(nn.Module):
    """
    Single GraphSAGE residual block:
        x' = Dropout( ReLU( BN( SAGEConv(x, edge_index) ) ) ) + skip(x)

    If ``in_channels != out_channels`` a linear projection is used for the
    skip connection so dimensions match before the residual add.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        self.conv    = SAGEConv(in_channels, out_channels)
        self.bn      = PyGBatchNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # Skip connection: project if dimensions differ
        self.skip: nn.Module = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out      = self.conv(x, edge_index)
        out      = self.bn(out)
        out      = F.relu(out)
        out      = self.dropout(out)
        return out + residual


# ===========================================================================
# VariantSAGEGNN — TEKNOFEST 2026 primary model
# ===========================================================================

class VariantSAGEGNN(nn.Module):
    """
    Inductive node-level classifier using GraphSAGE convolutions.

    Architecture
    ------------
    [Optional] SequenceEncoder (nuc + AA)  ──┐
                                              ├── concat ──► Linear input_proj
    Numeric features (N_feat) ───────────────┘

    input_proj ──► SAGEBlock(hidden) ──► SAGEBlock(hidden) ──► SAGEBlock(hidden)
               └──────── skip ──────────────────────────────────────────────────┘
                                                                ↓
                                                        Linear(hidden, num_classes)

    Skip connections are applied within each SAGEBlock (residual addition).
    BatchNorm + Dropout(0.3) after every SAGEConv prevent over-fitting on
    low-data panels (70-140 samples).

    Inductive learning: SAGEConv aggregates from sampled neighbourhoods, so
    the model generalises to novel genes unseen during training.

    Parameters
    ----------
    numeric_dim    : Dimension of numeric feature vector (post-preprocessing).
    hidden_dim     : Width of each hidden SAGEConv layer.
    num_classes    : 2 for Benign / Pathogenic.
    dropout        : Dropout probability (default 0.3).
    use_multimodal : If True, fuse SequenceEncoder output with numeric features.
    seq_enc_dim    : Output dimension of SequenceEncoder (ignored when
                     use_multimodal=False).
    """

    def __init__(
        self,
        numeric_dim:    int,
        hidden_dim:     int   = 64,
        num_classes:    int   = 2,
        dropout:        float = 0.3,
        use_multimodal: bool  = False,
        seq_enc_dim:    int   = 32,   # SequenceEncoder.output_dim
    ) -> None:
        super().__init__()

        self.use_multimodal = use_multimodal

        # Optional sequence encoder (created externally and passed in, or
        # initialised here with default parameters)
        if use_multimodal:
            from src.features.multimodal_encoder import SequenceEncoder
            # cnn_channels=16 → output_dim=32 by default
            self.seq_encoder: nn.Module = SequenceEncoder(cnn_channels=seq_enc_dim // 2)
            in_channels = numeric_dim + self.seq_encoder.output_dim
        else:
            self.seq_encoder = None
            in_channels = numeric_dim

        # Input projection to hidden space
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Three SAGEConv residual blocks
        self.block1 = _SAGEBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.block2 = _SAGEBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.block3 = _SAGEBlock(hidden_dim, hidden_dim, dropout=dropout)

        # Node-level classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:          torch.Tensor,               # [N, numeric_dim]
        edge_index: torch.Tensor,               # [2, E]
        nuc_ids:    torch.Tensor | None = None, # [N, nuc_seq_len]
        aa_ids:     torch.Tensor | None = None, # [N, aa_seq_len]
    ) -> torch.Tensor:
        """
        Returns logits of shape [N, num_classes] — one prediction per node
        (variant).  No global pooling is applied.
        """
        if self.use_multimodal and nuc_ids is not None and aa_ids is not None:
            seq_feat = self.seq_encoder(nuc_ids, aa_ids)   # [N, seq_enc_dim]
            x = torch.cat([x, seq_feat], dim=1)            # [N, numeric+seq]

        # Project to hidden dimension
        x = F.relu(self.input_proj(x))   # [N, hidden_dim]

        # GraphSAGE convolutions with skip connections
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)

        # Per-node classification logits
        return self.classifier(x)         # [N, num_classes]


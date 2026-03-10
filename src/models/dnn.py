"""
src/models/dnn.py
Deep Neural Network for tabular variant features.
Dynamic input dimension — no hardcoded feature count.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VariantDNN(nn.Module):
    """
    Feed-forward DNN for tabular variant feature matrices.

    Parameters
    ----------
    input_dim   : Number of input features (determined after preprocessing).
    hidden_dim  : Width of the first hidden layer; subsequent layers use hidden_dim // 2.
    num_classes : Output classes.
    dropout     : Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

"""
src/graph/builder.py
Pluggable graph construction layer.

Two strategies:
  - CorrelationGraphBuilder : Feature-interaction graph based on Pearson correlation.
  - KNNGraphBuilder         : k-nearest-neighbour sample graph (alternative).

Both implement the ``GraphBuilder`` protocol so they can be swapped in config.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class GraphBuilder(ABC):
    """Protocol for feature-interaction graph builders."""

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "GraphBuilder":
        """Learn graph topology from training data."""
        ...

    @abstractmethod
    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        """Convert a single processed feature vector to a PyG Data object."""
        ...

    @property
    @abstractmethod
    def edge_index(self) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def edge_attr(self) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# Correlation-based builder (default)
# ---------------------------------------------------------------------------


class CorrelationGraphBuilder(GraphBuilder):
    """
    Builds a feature-interaction graph where edges connect feature nodes
    whose absolute Pearson correlation exceeds ``corr_threshold``.

    This is the original VARIANT-GNN approach, now extracted into a
    modular class so it can be replaced with any other strategy.
    """

    def __init__(self, corr_threshold: float = 0.25) -> None:
        self.corr_threshold = corr_threshold
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr:  Optional[torch.Tensor] = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "CorrelationGraphBuilder":
        corr = np.corrcoef(X_train, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        n = X_train.shape[1]
        edges: List[List[int]] = []
        weights: List[float]   = []

        for i in range(n):
            for j in range(n):
                if i != j and abs(corr[i, j]) >= self.corr_threshold:
                    edges.append([i, j])
                    weights.append(float(abs(corr[i, j])))

        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self._edge_attr  = torch.tensor(weights, dtype=torch.float)
        else:
            self._edge_index = torch.empty((2, 0), dtype=torch.long)
            self._edge_attr  = torch.empty((0,), dtype=torch.float)

        self._n_features = n
        logger.info(
            "CorrelationGraph: %d nodes, %d directed edges (threshold=%.2f)",
            n, len(edges), self.corr_threshold,
        )
        return self

    # ------------------------------------------------------------------
    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        x_tensor = torch.tensor(x_row, dtype=torch.float).unsqueeze(1)  # [N, 1]
        y_tensor = torch.tensor([label], dtype=torch.long) if label is not None else None
        return Data(
            x=x_tensor,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            y=y_tensor,
        )

    # ------------------------------------------------------------------
    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError("CorrelationGraphBuilder has not been fitted yet.")
        return self._edge_index

    @property
    def edge_attr(self) -> torch.Tensor:
        if self._edge_attr is None:
            raise RuntimeError("CorrelationGraphBuilder has not been fitted yet.")
        return self._edge_attr


# ---------------------------------------------------------------------------
# KNN sample-level graph builder (alternative strategy)
# ---------------------------------------------------------------------------


class KNNGraphBuilder(GraphBuilder):
    """
    Builds a sample-connectivity graph using k-nearest-neighbours in feature space.
    Each sample becomes a node; samples close in feature space are connected.

    NOTE: In the current VARIANT-GNN architecture the GNN treats *features*
    as nodes, not samples.  This builder is provided as an architectural
    alternative for future work.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr:  Optional[torch.Tensor] = None

    def fit(self, X_train: np.ndarray) -> "KNNGraphBuilder":
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean")
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_train)

        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            for dist, j in zip(dists[1:], nbrs[1:]):
                edges.append((i, j))
                edges.append((j, i))
                weights.extend([dist, dist])

        self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self._edge_attr  = torch.tensor(weights, dtype=torch.float)
        logger.info("KNNGraph: %d samples, k=%d, %d edges", len(X_train), self.k, len(edges))
        return self

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        raise NotImplementedError(
            "KNNGraphBuilder creates a sample graph, not a per-row feature graph. "
            "Use CorrelationGraphBuilder for per-variant inference."
        )

    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError("KNNGraphBuilder not fitted.")
        return self._edge_index

    @property
    def edge_attr(self) -> torch.Tensor:
        if self._edge_attr is None:
            raise RuntimeError("KNNGraphBuilder not fitted.")
        return self._edge_attr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_graph_builder(strategy: str = "correlation", **kwargs) -> GraphBuilder:
    """
    Return a graph builder by name.

    Supported strategies:
        - ``"correlation"`` (default): feature-correlation graph.
        - ``"knn"``                  : sample k-NN graph.
    """
    strategy = strategy.lower()
    if strategy == "correlation":
        return CorrelationGraphBuilder(**kwargs)
    if strategy == "knn":
        return KNNGraphBuilder(**kwargs)
    raise ValueError(f"Unknown graph strategy: {strategy!r}")

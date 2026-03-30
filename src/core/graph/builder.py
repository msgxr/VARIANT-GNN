"""
src/graph/builder.py
Pluggable graph construction layer.

Strategies:
  - CorrelationGraphBuilder : Feature-interaction graph (features as nodes).
  - KNNGraphBuilder         : Legacy k-NN sample graph via sklearn.
  - SampleKNNGraphBuilder   : TEKNOFEST 2026 — coordinate-free, cosine-similarity
                             k-NN graph where each VARIANT is a node.
                             Uses torch_geometric.nn.knn_graph with cosine=True.

All builders implement the ``GraphBuilder`` protocol so they can be swapped.
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
        - ``"knn"``                  : legacy sample k-NN graph.
        - ``"sample_knn"``           : TEKNOFEST 2026 coordinate-free cosine kNN.
    """
    strategy = strategy.lower()
    if strategy == "correlation":
        return CorrelationGraphBuilder(**kwargs)
    if strategy == "knn":
        return KNNGraphBuilder(**kwargs)
    if strategy == "sample_knn":
        return SampleKNNGraphBuilder(**kwargs)
    raise ValueError(f"Unknown graph strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# SampleKNNGraphBuilder — TEKNOFEST 2026 primary builder
# ---------------------------------------------------------------------------


class SampleKNNGraphBuilder:
    """
    Coordinate-free, feature-similarity k-NN graph builder.

    Per TEKNOFEST 2026 spec, chromosome/position identifiers are hidden to
    prevent label leakage.  This builder connects variants purely based on
    their biochemical/evolutionary feature similarity using cosine distance.

    Each VARIANT becomes a node whose feature vector is the full numeric
    feature vector after preprocessing.  torch_geometric.nn.knn_graph is
    used to connect each node to its k=5 nearest neighbours in feature space.

    Typical usage
    -------------
    >>> builder = SampleKNNGraphBuilder(k=5)
    >>> data = builder.build(X_scaled, y)   # train graph
    >>> test_data = builder.build(X_test)   # inference — fully inductive
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    # ------------------------------------------------------------------
    def build(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Build a PyG ``Data`` object where every sample is a node.

        Parameters
        ----------
        X : [N_samples, N_features] scaled feature matrix.
        y : Optional integer label array of length N_samples.

        Returns
        -------
        PyG Data with:
          - ``x``          : [N, N_features] node feature matrix
          - ``edge_index`` : [2, E] kNN edges built with cosine similarity
          - ``edge_attr``  : [E] cosine similarities for each edge
          - ``y``          : [N] or None
        """
        N = X.shape[0]
        x_tensor = torch.tensor(X, dtype=torch.float)
        k_actual  = min(self.k, N - 1)

        edge_index, cos_sim = self._knn_cosine(x_tensor, k_actual)

        y_tensor: Optional[torch.Tensor] = None
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.long)

        logger.info(
            "SampleKNNGraph: %d nodes, k=%d, %d edges (cosine similarity)",
            N, k_actual, edge_index.shape[1],
        )

        return Data(
            x          = x_tensor,
            edge_index = edge_index,
            edge_attr  = cos_sim,
            y          = y_tensor,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _knn_cosine(
        x: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build directed k-NN edges using cosine similarity.

        Tries torch_geometric.nn.knn_graph (requires torch-cluster); falls
        back to a pure-PyTorch batched cosine matrix when that package is
        unavailable.

        Returns
        -------
        edge_index : [2, N*k] long tensor
        cos_sim    : [N*k] float tensor of cosine similarities
        """
        try:
            from torch_geometric.nn import knn_graph as _knn_graph
            edge_index = _knn_graph(x, k=k, cosine=True)
            src, dst   = edge_index
            x_norm     = torch.nn.functional.normalize(x, p=2, dim=1)
            cos_sim    = (x_norm[src] * x_norm[dst]).sum(dim=1).clamp(-1.0, 1.0)
            return edge_index, cos_sim
        except ImportError:
            pass  # fall through to pure-PyTorch implementation

        # Pure-PyTorch fallback (no torch-cluster dependency)
        # Compute full cosine similarity matrix and pick top-k per row
        x_norm  = torch.nn.functional.normalize(x, p=2, dim=1)
        sim_mat = x_norm @ x_norm.t()               # [N, N]
        # Exclude self-loops by setting diagonal to -2
        sim_mat.fill_diagonal_(-2.0)

        topk_sim, topk_idx = sim_mat.topk(k, dim=1)   # [N, k]
        N = x.shape[0]

        src = torch.arange(N, dtype=torch.long).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = topk_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)    # [2, N*k]
        cos_sim    = topk_sim.reshape(-1)

        return edge_index, cos_sim

    # Stub to satisfy GraphBuilder protocol if needed
    def fit(self, X_train: np.ndarray) -> "SampleKNNGraphBuilder":
        return self

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        raise NotImplementedError(
            "SampleKNNGraphBuilder works on the full sample matrix. "
            "Call build(X, y) instead."
        )

    @property
    def edge_index(self) -> torch.Tensor:
        raise NotImplementedError("Call build(X) to get the full graph.")

    @property
    def edge_attr(self) -> torch.Tensor:
        raise NotImplementedError("Call build(X) to get the full graph.")

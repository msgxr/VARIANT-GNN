"""src.models — model namespace.

Lazy imports to avoid circular dependency:
  src.core.ensemble imports src.models.dnn_model at module level.
  Eager imports here would create a circular chain at collection time.

Canonical locations:
  VariantDNN      → src.models.dnn_model
  HybridEnsemble  → src.core.models.ensemble
  VariantGATv2GNN → src.core.models.gnn
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "VariantDNN",
    "HybridEnsemble",
    "VariantGATv2GNN",
    "VariantSAGEGNN",
    "FeatureGNN",
]


def __getattr__(name: str) -> Any:
    if name == "VariantDNN":
        return import_module("src.models.dnn_model").VariantDNN
    if name == "HybridEnsemble":
        return import_module("src.core.models.ensemble").HybridEnsemble
    if name in {"VariantGATv2GNN", "VariantSAGEGNN", "FeatureGNN"}:
        mod = import_module("src.core.models.gnn")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

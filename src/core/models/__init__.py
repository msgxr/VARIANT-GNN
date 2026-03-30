"""Backward-compatible model namespace with lazy exports."""

from importlib import import_module

__all__ = [
    "VariantDNN",
    "HybridEnsemble",
    "VariantGATv2GNN",
    "VariantSAGEGNN",
    "FeatureGNN",
]


def __getattr__(name: str):
    if name == "VariantDNN":
        return import_module("src.core.models.dnn").VariantDNN
    if name == "HybridEnsemble":
        return import_module("src.core.models.ensemble").HybridEnsemble
    if name in {"VariantGATv2GNN", "VariantSAGEGNN", "FeatureGNN"}:
        mod = import_module("src.core.models.gnn")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

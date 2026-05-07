"""Legacy src.models namespace backed by src.core.models wrappers."""

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
        return import_module("src.models.dnn_model").VariantDNN
    if name == "HybridEnsemble":
        return import_module("src.models.ensemble").HybridEnsemble
    if name in {"VariantGATv2GNN", "VariantSAGEGNN", "FeatureGNN"}:
        mod = import_module("src.models.gnn")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

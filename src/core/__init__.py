"""Core package exports with lazy loading to avoid import cycles."""

from importlib import import_module

__all__ = ["VariantGATv2GNN", "VariantDNN", "HybridEnsemble"]


def __getattr__(name: str):
	if name == "VariantDNN":
		return import_module("src.models.dnn_model").VariantDNN
	if name == "VariantGATv2GNN":
		return import_module("src.core.gnn").VariantGATv2GNN
	if name == "HybridEnsemble":
		return import_module("src.core.ensemble").HybridEnsemble
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

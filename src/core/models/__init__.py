# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""Backward-compatible model namespace with lazy exports."""

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

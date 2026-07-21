# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""Core package exports with lazy loading to avoid import cycles."""

from importlib import import_module
from typing import Any

__all__ = ["VariantGATv2GNN", "VariantDNN", "HybridEnsemble"]


def __getattr__(name: str) -> Any:
    if name == "VariantDNN":
        return import_module("src.models.dnn_model").VariantDNN
    if name == "VariantGATv2GNN":
        return import_module("src.core.gnn").VariantGATv2GNN
    if name == "HybridEnsemble":
        return import_module("src.core.ensemble").HybridEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

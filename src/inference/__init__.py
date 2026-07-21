# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

from typing import Any

from .artifact_loader import ArtifactLoader
from .external_validation_runner import ExternalValidationRunner
from .prediction_schema import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    validate_prediction_frame,
)
from .triage import ALL_FLAGS, TriageEngine


def __getattr__(name: str) -> Any:
    if name == "InferencePipeline":
        from src.api.pipeline import InferencePipeline  # lazy — döngüsel import önlenir

        return InferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InferencePipeline",
    "ExternalValidationRunner",
    "ArtifactLoader",
    "PREDICTION_COLUMNS",
    "build_prediction_frame",
    "validate_prediction_frame",
    "TriageEngine",
    "ALL_FLAGS",
]

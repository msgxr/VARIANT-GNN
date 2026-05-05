from .external_validation_runner import ExternalValidationRunner
from .artifact_loader import ArtifactLoader
from .prediction_schema import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    validate_prediction_frame,
)
from .triage import TriageEngine, ALL_FLAGS


def __getattr__(name: str):
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

"""Backward-compatible re-export. The canonical implementation lives in src.api.pipeline."""

try:
    from src.api.pipeline import InferencePipeline
except ImportError as _e:
    raise ImportError(
        f"InferencePipeline yuklenemedi (src.api.pipeline): {_e}\nKontrol et: src/api/pipeline.py ve bagimliliklari."
    ) from _e

__all__ = ["InferencePipeline"]

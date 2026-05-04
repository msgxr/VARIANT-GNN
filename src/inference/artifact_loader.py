"""
src/inference/artifact_loader.py
Loads serialised model artifacts from disk for offline inference.
No training, no fitting, no network access — pure deserialization.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_REQUIRED_ARTIFACTS = (
    "preprocessor.pkl",
    "ensemble.pkl",
)


class ArtifactLoader:
    """
    Loads inference-time artifacts from a model directory.

    Parameters
    ----------
    model_dir : Directory containing serialised artifacts.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    # ------------------------------------------------------------------
    def _load_pickle(self, filename: str) -> Any:
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.debug("Loaded artifact: %s", path)
        return obj

    # ------------------------------------------------------------------
    def load_preprocessor(self) -> Any:
        """Load VariantPreprocessor artifact."""
        return self._load_pickle("preprocessor.pkl")

    def load_ensemble(self) -> Any:
        """Load HybridEnsemble artifact."""
        return self._load_pickle("ensemble.pkl")

    def load_calibrator(self) -> Optional[Any]:
        """Load calibrator if present; return None otherwise."""
        path = self.model_dir / "calibrator.pkl"
        if not path.exists():
            logger.warning("No calibrator artifact found at %s; skipping.", path)
            return None
        return self._load_pickle("calibrator.pkl")

    def load_threshold(self, default: float = 0.5) -> float:
        """Load the F1-optimal threshold saved during training."""
        path = self.model_dir / "threshold.json"
        if not path.exists():
            logger.warning("No threshold.json found; using default %.2f", default)
            return default
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data.get("threshold", default))

    def load_feature_names(self) -> Optional[list[str]]:
        """Load feature column names saved during training."""
        path = self.model_dir / "feature_names.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def load_model_version(self) -> str:
        """Load model version string from manifest."""
        path = self.model_dir / "manifest.json"
        if not path.exists():
            return "unknown"
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return str(data.get("model_version", "unknown"))

    # ------------------------------------------------------------------
    def validate_artifacts(self) -> None:
        """Raise FileNotFoundError if any required artifact is missing."""
        for filename in _REQUIRED_ARTIFACTS:
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact missing: {path}. "
                    "Ensure the model has been trained and saved before running inference."
                )
        logger.info("ArtifactLoader: all required artifacts present in %s", self.model_dir)

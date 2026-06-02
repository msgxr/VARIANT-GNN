"""
src/utils/artifact_manifest.py
Generates and loads models/final/manifest.json for reproducibility auditing.

The manifest records:
  - Model version
  - Git commit hash (if available)
  - Python version
  - Key package versions
  - Config hash
  - Dataset fingerprints
  - Artifact file checksums
  - Timestamp
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, cast

from src.utils.fingerprinting import file_fingerprint

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "torch_geometric",
    "xgboost",
    "lightgbm",
    "scipy",
)

_ARTIFACT_FILES = (
    "preprocessor.pkl",
    "ensemble.pkl",
    "calibrator.pkl",
    "threshold.json",
    "feature_names.json",
)


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(pkg.replace("-", "_").split(".")[0])
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def build_manifest(
    model_dir: Path,
    config: Optional[dict] = None,
    dataset_fingerprints: Optional[Dict[str, str]] = None,
    model_version: str = "1.0.0",
) -> dict:
    """
    Build a manifest dict for the trained model.

    Parameters
    ----------
    model_dir            : Directory where artifacts are stored.
    config               : Training config dict (will be hashed, not stored verbatim).
    dataset_fingerprints : {'train': sha256, 'test': sha256, ...}
    model_version        : Semantic version string.

    Returns
    -------
    manifest dict.
    """
    model_dir = Path(model_dir)
    from src.utils.fingerprinting import config_fingerprint

    artifact_checksums: Dict[str, str] = {}
    for fname in _ARTIFACT_FILES:
        fpath = model_dir / fname
        if fpath.exists():
            artifact_checksums[fname] = file_fingerprint(fpath)

    manifest = {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "config_hash": config_fingerprint(config) if config else "not_provided",
        "dataset_fingerprints": dataset_fingerprints or {},
        "artifact_checksums": artifact_checksums,
    }
    return manifest


def save_manifest(
    model_dir: Path,
    config: Optional[dict] = None,
    dataset_fingerprints: Optional[Dict[str, str]] = None,
    model_version: str = "1.0.0",
) -> Path:
    """
    Build and save manifest.json to model_dir.

    Returns
    -------
    Path to the saved manifest file.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        model_dir,
        config=config,
        dataset_fingerprints=dataset_fingerprints,
        model_version=model_version,
    )
    out_path = model_dir / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Manifest saved to %s", out_path)
    return out_path


def load_manifest(model_dir: Path) -> dict:
    """Load manifest.json from model_dir."""
    path = Path(model_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")
    with open(path, encoding="utf-8") as fh:
        return cast(dict, json.load(fh))

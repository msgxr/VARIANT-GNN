# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/utils/reproducibility_manifest.py
=======================================
TEKNOFEST 2026 §7.5 Reproducibility Manifest

Şartname §7.5 alıntı:
  "Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını ve
   beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."

Bu modül, bir eğitim koşusunun TÜM koşullarını imzalı bir manifest olarak
arşivler, böylece jüri:

  1. Aynı veri ile aynı kodu yeniden çalıştırdığında
  2. Aynı F1 sonucunu (epsilon dahilinde) elde eder
  3. Manifest hash'leri ile artefakt bütünlüğünü doğrulayabilir

Manifest İçeriği:
  - Yarışma metadata: tarih, takım, model versiyonu
  - Veri kaynak fingerprint: input CSV SHA256 + satır sayısı + label dağılımı
  - Çevre: Python, OS, paket versiyonları (donanım dışı)
  - Random state: tüm RNG seed'leri ve deterministik flag'ler
  - Eğitim hyperparameters: tam dump (Settings nesnesi)
  - Artifact hash'leri: model dosyaları SHA256
  - Metric snapshot: rapor edilen F1 + güven aralığı
  - Git commit + branch + dirty flag

Çıktı: ``models/reproducibility_manifest.json``  (insanlığa ve jüriye uygun)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA256 of a file. Returns None if path missing."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_info() -> Dict[str, Optional[str | bool]]:
    """Capture current git state (commit, branch, dirty)."""
    info: Dict[str, Optional[str | bool]] = {
        "commit": None,
        "branch": None,
        "is_dirty": None,
        "remote": None,
    }
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["is_dirty"] = bool(status)
        try:
            info["remote"] = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _package_versions() -> Dict[str, str]:
    """Capture key ML package versions for reproducibility."""
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "torch-geometric",
        "xgboost",
        "lightgbm",
        "pyyaml",
        "joblib",
        "matplotlib",
        "shap",
    ]
    # Resolve by PyPI *distribution* name (importlib.metadata) rather than by
    # __import__, which fails for dist-name != import-name packages such as
    # scikit-learn (import 'sklearn') — previously always 'not_installed'.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    versions: Dict[str, str] = {}
    for name in pkgs:
        try:
            versions[name] = _dist_version(name)
        except PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _data_fingerprint(csv_path: Optional[Path]) -> Dict[str, Any]:
    """
    Compute non-PII fingerprint of an input CSV.

    Captures: SHA256, row count, column count, label distribution (if Label
    column present). Does NOT capture row contents.
    """
    fp: Dict[str, Any] = {
        "path": str(csv_path) if csv_path else None,
        "sha256": None,
        "n_rows": None,
        "n_cols": None,
        "label_distribution": {},
    }
    if csv_path is None or not csv_path.exists():
        return fp

    fp["sha256"] = _sha256_file(csv_path)

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)
        fp["n_rows"] = int(len(df))
        fp["n_cols"] = int(df.shape[1])

        for label_col in ("Label", "label", "y", "target"):
            if label_col in df.columns:
                vc = df[label_col].astype(str).str.lower().value_counts()
                fp["label_distribution"] = {str(k): int(v) for k, v in vc.items()}
                break
    except Exception as exc:
        logger.debug("CSV fingerprint failed (%s).", exc)

    return fp


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReproducibilityManifest:
    """Full TEKNOFEST §7.5 jury-reproducibility manifest."""

    schema_version: str = "1.0"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    competition: str = "TEKNOFEST 2026 — Sağlıkta Yapay Zeka (Üniversite ve Üzeri)"
    pipeline: str = "VARIANT-GNN"
    pipeline_version: str = "2.0.0"

    # Random state
    seed: int = 42
    deterministic_torch: bool = True

    # Environment
    python_version: str = ""
    platform: str = ""
    os_environ_hash: str = ""  # hash of relevant env vars
    package_versions: Dict[str, str] = field(default_factory=dict)

    # Code state
    git: Dict[str, Optional[str | bool]] = field(default_factory=dict)

    # Data state
    train_data: Dict[str, Any] = field(default_factory=dict)
    test_data: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    settings_dump: Dict[str, Any] = field(default_factory=dict)

    # Artifact hashes
    artifact_hashes: Dict[str, Optional[str]] = field(default_factory=dict)

    # Metric snapshot
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Self-checksum
    manifest_sha256: Optional[str] = None

    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        d = asdict(self)
        # Don't include manifest_sha256 in serialization for hashing
        d.pop("manifest_sha256", None)
        return d

    def compute_self_hash(self) -> str:
        """Compute SHA256 of the canonical JSON representation."""
        d = self.as_dict()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return _sha256_string(canonical)

    def to_json(self, path: Path) -> None:
        """Write manifest to JSON, including self-hash for tamper detection."""
        self.manifest_sha256 = self.compute_self_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Reproducibility manifest → %s  [sha256=%s]", path, self.manifest_sha256[:12])

    @classmethod
    def from_json(cls, path: Path) -> "ReproducibilityManifest":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(**d)

    def verify(self) -> bool:
        """
        Verify manifest hasn't been tampered with: recompute hash and compare.

        Returns True iff stored ``manifest_sha256`` matches re-computed hash.
        """
        stored = self.manifest_sha256
        if stored is None:
            logger.warning("Manifest has no stored hash — cannot verify.")
            return False
        # Temporarily clear for re-computation
        self.manifest_sha256 = None
        actual = self.compute_self_hash()
        self.manifest_sha256 = stored
        match = stored == actual
        if not match:
            logger.warning(
                "MANIFEST INTEGRITY FAILURE: stored=%s  actual=%s",
                stored[:12],
                actual[:12],
            )
        return match


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ManifestBuilder:
    """
    High-level builder that fills a ReproducibilityManifest from runtime state.

    Usage
    -----
    ::

        builder = ManifestBuilder(seed=42)
        builder.with_data(train_csv="data/train.csv", test_csv="data/test.csv")
        builder.with_settings(get_settings())
        builder.with_artifacts(models_dir=Path("models"))
        builder.with_metrics({"binary_f1": 0.94, "panel_f1": {...}})
        builder.save(Path("models/reproducibility_manifest.json"))
    """

    def __init__(self, seed: int = 42) -> None:
        self.manifest = ReproducibilityManifest(
            seed=seed,
            python_version=platform.python_version(),
            platform=platform.platform(),
            git=_git_info(),
            package_versions=_package_versions(),
        )
        # Hash of relevant env vars (no full dump for privacy)
        relevant_env = {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        }
        self.manifest.os_environ_hash = _sha256_string(json.dumps(relevant_env, sort_keys=True))

    def with_data(
        self,
        train_csv: Optional[str | Path] = None,
        test_csv: Optional[str | Path] = None,
    ) -> "ManifestBuilder":
        if train_csv is not None:
            self.manifest.train_data = _data_fingerprint(Path(train_csv))
        if test_csv is not None:
            self.manifest.test_data = _data_fingerprint(Path(test_csv))
        return self

    def with_settings(self, settings_obj: Any) -> "ManifestBuilder":
        """Capture the full Settings dataclass as a nested dict."""
        try:
            from dataclasses import asdict as _asdict
            from dataclasses import is_dataclass

            if is_dataclass(settings_obj):
                self.manifest.settings_dump = _asdict(cast("Any", settings_obj))
            else:
                # Fallback: collect __dict__
                self.manifest.settings_dump = {k: str(v) for k, v in vars(settings_obj).items()}
        except Exception as exc:
            logger.debug("Settings dump failed (%s).", exc)
        return self

    def with_artifacts(self, models_dir: Path) -> "ManifestBuilder":
        """Hash each artefact file in ``models_dir``."""
        files = [
            "xgb_model.json",
            "lgbm_model.txt",
            "gnn_model.pth",
            "gnn_arch.json",
            "dnn_model.pth",
            "autoencoder.pth",
            "preprocessor.pkl",
            "calibrator.pkl",
            "meta_learner.pkl",
            "ensemble_config.json",
            "threshold.json",
            "panel_thresholds.json",
        ]
        for f in files:
            path = models_dir / f
            self.manifest.artifact_hashes[f] = _sha256_file(path)
        return self

    def with_metrics(self, metrics: Dict[str, Any]) -> "ManifestBuilder":
        self.manifest.metrics = dict(metrics)
        return self

    def with_git(self) -> "ManifestBuilder":
        """Refresh git info (in case it was stale)."""
        self.manifest.git = _git_info()
        return self

    def build(self) -> ReproducibilityManifest:
        return self.manifest

    def save(self, path: Path) -> ReproducibilityManifest:
        self.manifest.to_json(path)
        return self.manifest


# ---------------------------------------------------------------------------
# Verify-on-load helper for jüri use case
# ---------------------------------------------------------------------------


def verify_manifest_chain(
    manifest_path: Path,
    models_dir: Path,
    raise_on_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Full verification entry point for jury reproduction:

      1. Load manifest from disk.
      2. Verify manifest's self-hash (tamper detection).
      3. Re-hash artefact files; compare against stored hashes.
      4. Report git commit alignment (informational).

    Parameters
    ----------
    manifest_path     : Path to ``reproducibility_manifest.json``.
    models_dir        : Path to artefacts directory.
    raise_on_mismatch : Raise ``RuntimeError`` if any check fails.

    Returns
    -------
    dict with verification results:
      {
        "manifest_integrity": bool,
        "artifact_matches":   {filename: bool, ...},
        "all_match":          bool,
        "git_aligned":        bool|None,
        "git_current":        str|None,
        "git_manifest":       str|None,
      }
    """
    manifest = ReproducibilityManifest.from_json(manifest_path)
    integrity = manifest.verify()

    # Verify each artefact
    artifact_matches: Dict[str, bool] = {}
    for filename, expected_hash in manifest.artifact_hashes.items():
        path = models_dir / filename
        actual = _sha256_file(path) if path.exists() else None
        artifact_matches[filename] = actual == expected_hash

    # An empty artifact_hashes must NOT pass: all([]) is True, which would
    # report "verification PASSED" when ZERO artefacts were actually checked.
    no_artifacts_recorded = not artifact_matches
    all_match = integrity and bool(artifact_matches) and all(artifact_matches.values())

    # Git alignment (informational)
    current_git = _git_info()
    git_aligned: Optional[bool] = None
    if current_git["commit"] and manifest.git.get("commit"):
        git_aligned = current_git["commit"] == manifest.git["commit"]

    result = {
        "manifest_integrity": integrity,
        "artifact_matches": artifact_matches,
        "all_match": all_match,
        "no_artifacts_recorded": no_artifacts_recorded,
        "git_aligned": git_aligned,
        "git_current": current_git.get("commit"),
        "git_manifest": manifest.git.get("commit"),
    }

    if not all_match:
        msg = (
            f"Manifest verification FAILED: "
            f"integrity={integrity}, artefact mismatches="
            f"{[k for k, v in artifact_matches.items() if not v]}"
        )
        logger.error(msg)
        if raise_on_mismatch:
            raise RuntimeError(msg)
    else:
        logger.info("Manifest verification PASSED for %s", manifest_path)

    return result

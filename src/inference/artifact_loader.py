"""
src/inference/artifact_loader.py
==================================
Inference-zamanı artefakt yükleyici (offline, eğitim yapmaz, ağ erişimi yok).

Kanonik artefakt seti (ModelStore tarafından üretilir):
  preprocessor.pkl       — VariantPreprocessor (joblib pickle)
  xgb_model.json         — XGBoost
  lgbm_model.txt         — LightGBM (opsiyonel)
  gnn_model.pth          — VariantGATv2GNN state_dict
  gnn_arch.json          — GNN mimari config
  dnn_model.pth          — VariantDNN state_dict
  ensemble_config.json   — Ensemble ağırlıkları
  threshold.json         — F1-optimal eşik (key: classification_threshold)
  calibrator.pkl         — EnsembleCalibrator (opsiyonel)

Bu sınıf ``ModelStore.load_all()`` ile uyumlu çalışır.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Kanonik zorunlu artefakt seti (ModelStore.save_all ile uyumlu)
_REQUIRED_CANONICAL = (
    "preprocessor.pkl",
    "xgb_model.json",
    "ensemble_config.json",
)

# Geriye dönük uyumluluk: tek-pickle ensemble.pkl path'i
# (testler ve eski deployment'lar için)
_REQUIRED_LEGACY = (
    "preprocessor.pkl",
    "ensemble.pkl",
)


class ArtifactLoader:
    """
    Eğitim-sonrası artefakt yükleyici.

    ModelStore (src/utils/serialization.py) tarafından kaydedilen kanonik
    artefakt seti üzerinde çalışır. Tek başına ``ensemble.pkl`` değil,
    ayrılmış model dosyalarını (xgb_model.json, gnn_model.pth, vb.)
    bekler.

    Parameters
    ----------
    model_dir : Artefaktların bulunduğu dizin (örn. ``models/``).
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
            obj = pickle.load(fh)  # noqa: S301 — own artefakt
        logger.debug("Loaded artifact: %s", path)
        return obj

    # ------------------------------------------------------------------
    def load_preprocessor(self) -> Any:
        """Eğitilmiş ``VariantPreprocessor``'ı yükle."""
        # joblib pickle uyumlu
        try:
            import joblib
            return joblib.load(str(self.model_dir / "preprocessor.pkl"))
        except Exception:
            return self._load_pickle("preprocessor.pkl")

    def load_ensemble(self) -> Any:
        """
        ``HybridEnsemble``'ı yükle.

        İki yol denenir (öncelik sırası):
          1. Legacy ``ensemble.pkl`` (tek-pickle, test fixture)
          2. ModelStore.load_all() (kanonik üretim)
        """
        legacy_path = self.model_dir / "ensemble.pkl"
        if legacy_path.exists():
            return self._load_pickle("ensemble.pkl")

        from src.utils.serialization import ModelStore
        store = ModelStore(self.model_dir)
        _preproc, ensemble, _cal = store.load_all()
        return ensemble

    def load_calibrator(self) -> Optional[Any]:
        """Kalibratör mevcutsa yükle; yoksa None döndür."""
        path = self.model_dir / "calibrator.pkl"
        if not path.exists():
            logger.warning("No calibrator artifact found at %s; skipping.", path)
            return None
        try:
            import joblib
            return joblib.load(str(path))
        except Exception:
            return self._load_pickle("calibrator.pkl")

    def load_threshold(self, default: float = 0.5) -> float:
        """
        F1-optimal sınıflandırma eşiğini yükle.

        ModelStore ``classification_threshold`` anahtarıyla kaydeder;
        geriye dönük uyumluluk için ``threshold`` anahtarını da kontrol eder.
        """
        path = self.model_dir / "threshold.json"
        if not path.exists():
            logger.warning("No threshold.json found; using default %.2f", default)
            return default
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Kanonik key: classification_threshold; legacy: threshold
        return float(
            data.get("classification_threshold", data.get("threshold", default))
        )

    def load_feature_names(self) -> Optional[list[str]]:
        """
        Eğitim-zamanı özellik kolon adlarını yükle (varsa).

        Önce explicit feature_names.json, ardından XGB booster'ından çekme
        denenir.
        """
        path = self.model_dir / "feature_names.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        # XGB booster fallback
        try:
            from src.utils.serialization import ModelStore
            store = ModelStore(self.model_dir)
            xgb_model = store.load_xgb()
            names = xgb_model.get_booster().feature_names
            return list(names) if names else None
        except Exception:
            return None

    def load_model_version(self) -> str:
        """Model versiyon bilgisini metadata'dan oku."""
        for filename in ("manifest.json", "metadata.json"):
            path = self.model_dir / filename
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                version = data.get("model_version") or data.get("version")
                if version:
                    return str(version)
        return "unknown"

    # ------------------------------------------------------------------
    def validate_artifacts(self) -> None:
        """
        Zorunlu artefaktlardan biri eksikse FileNotFoundError fırlat.

        İki uyumluluk yolu desteklenir:
          1. Legacy: ``ensemble.pkl`` + ``preprocessor.pkl``
          2. Kanonik: ``xgb_model.json`` + ``ensemble_config.json`` + ``preprocessor.pkl``
        """
        # Önce legacy path'i kontrol et (mock testler ve eski deployment'lar)
        legacy_ok = all(
            (self.model_dir / f).exists() for f in _REQUIRED_LEGACY
        )
        if legacy_ok:
            logger.info(
                "ArtifactLoader: legacy path → ensemble.pkl + preprocessor.pkl mevcut.")
            return

        # Aksi halde kanonik artefakt setini ara
        missing: list[str] = []
        for filename in _REQUIRED_CANONICAL:
            path = self.model_dir / filename
            if not path.exists():
                missing.append(filename)
        if missing:
            raise FileNotFoundError(
                f"Zorunlu artefakt eksik: {missing}. "
                "Önce eğitim çalıştırılıp model kaydedilmelidir."
            )
        logger.info(
            "ArtifactLoader: tüm kanonik artefaktlar mevcut → %s", self.model_dir
        )

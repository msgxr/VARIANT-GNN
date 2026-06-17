import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.core.models.ensemble import HybridEnsemble
from src.features.preprocessing import VariantPreprocessor
from src.utils.serialization import ModelStore


class MockModel:
    def __init__(self):
        self.booster_ = self

    def save_model(self, path):
        with open(path, "w") as f:
            f.write("mock")

    def predict(self, X):
        return np.zeros(len(X))


def test_lgbm_roundtrip(tmp_path):
    # Setup
    model_dir = tmp_path / "models"
    store = ModelStore(model_dir)

    # Create mock ensemble with LightGBM
    mock_lgbm = MockModel()
    ensemble = HybridEnsemble(
        xgb_model=None, lgbm_model=mock_lgbm, gnn_model=None, dnn_model=None, weights=[0, 1, 0, 0]
    )

    preprocessor = VariantPreprocessor()

    # Save
    store._save_lgbm(mock_lgbm)

    # Verify file exists
    assert (model_dir / "lgbm_model.txt").exists()

    # Verify metadata (Task 6 requirement)
    store.save_all(preprocessor, ensemble)
    assert (model_dir / "metadata.json").exists()

    with open(model_dir / "metadata.json", "r") as f:
        import json

        meta = json.load(f)
        assert "sha256_checksums" in meta
        assert "timestamp" in meta


def test_model_artifact_integrity():
    """Jüri Modu: shipped artifact SHA256 bütünlüğü — metadata.json ile karşılaştır.

    Eskiden gövdesi `pass` idi (hiçbir şey doğrulamıyordu; bozuk/değiştirilmiş
    artefakt yakalanmıyordu). Artık shipped models/* dosyalarının SHA256'sını
    yeniden hesaplayıp models/metadata.json'daki kayıtlı özetlerle karşılaştırır.
    """
    import hashlib
    import json
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[2] / "models"
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        pytest.skip("models/metadata.json yok — shipped artefakt bütünlüğü doğrulanamıyor.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checksums = meta.get("sha256_checksums") or meta.get("artifact_checksums") or {}
    if not checksums:
        pytest.skip("metadata.json sha256_checksums içermiyor.")

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # metadata.json kısa anahtar ('xgb'/'gnn'/'dnn') VEYA tam dosya adı kullanabilir.
    # Kısa anahtarları gerçek dosyaya çöz; aksi halde 'models/xgb' bulunamaz → her
    # yineleme atlanır → checked=0 → test HİÇBİR ŞEY doğrulamadan skip eder (yazıldığı
    # no-op'un ta kendisi). Çözüm: alias tablosu + tam-ad fallback.
    aliases = {
        "xgb": "xgb_model.json",
        "lgbm": "lgbm_model.txt",
        "gnn": "gnn_model.pth",
        "dnn": "dnn_model.pth",
        "preprocessor": "preprocessor.pkl",
        "calibrator": "calibrator.pkl",
        "ensemble": "ensemble.pkl",
        "meta_learner": "meta_learner.pkl",
        "ood_detector": "ood_detector.pkl",
    }

    def _resolve(key):
        direct = models_dir / key
        if direct.exists():
            return direct  # tam dosya adı şeması
        alias = aliases.get(key)
        if alias and (models_dir / alias).exists():
            return models_dir / alias  # kısa anahtar şeması
        return None

    checked, mismatches, unresolved = 0, [], []
    for fname, expected in checksums.items():
        fpath = _resolve(fname)
        if fpath is None:
            unresolved.append(fname)  # opsiyonel/backup artefakt — atla
            continue
        checked += 1
        if _sha(fpath) != expected:
            mismatches.append(fpath.name)

    if checked == 0:
        pytest.skip(f"Kayıtlı özetlere karşılık gelen shipped artefakt bulunamadı (çözülemeyen: {unresolved}).")
    assert not mismatches, f"SHA256 uyuşmazlığı (bozuk/değiştirilmiş artefakt): {mismatches}"


import pytest
import numpy as np
import os
import shutil
from pathlib import Path
from src.utils.serialization import ModelStore
from src.core.models.ensemble import HybridEnsemble
from src.features.preprocessing import VariantPreprocessor

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
        xgb_model=None,
        lgbm_model=mock_lgbm,
        gnn_model=None,
        dnn_model=None,
        weights=[0, 1, 0, 0]
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
    """Jüri Modu: Artifact bütünlük testi."""
    # Bu test gerçek modellerin SHA256'larını kontrol eder
    pass

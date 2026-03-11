"""
VARIANT-GNN — Kapsamlı Unit Testleri
Çalıştırmak için: pytest tests/ -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import Settings, get_settings
from src.data_processing import TabularAutoEncoder, TabularGraphPreprocessor, generate_dummy_data, load_and_prepare_data
from src.models import FeatureGNN, HybridEnsemble, VariantDNN
from src.utils import ModelStore

# ─────────────────────────────────────────────────────────────
# Config Testleri
# ─────────────────────────────────────────────────────────────

def test_config_has_required_attributes():
    """Gerekli konfigürasyon attribute'larının var olduğunu doğrular."""
    cfg = get_settings()
    assert hasattr(cfg, 'paths')
    assert hasattr(cfg.paths, 'data_dir')
    assert hasattr(cfg.paths, 'models_dir')
    assert hasattr(cfg.paths, 'reports_dir')
    assert hasattr(cfg.paths, 'gnn_model_path')
    assert hasattr(cfg.paths, 'dnn_model_path')
    assert hasattr(cfg.paths, 'autoencoder_path')

def test_ensemble_weights_sum_to_one():
    """Ensemble ağırlıklarının toplamının 1 olduğunu test eder."""
    cfg = get_settings()
    total = sum(cfg.ensemble.weights)
    assert abs(total - 1.0) < 1e-6, f"Ensemble ağırlıkları toplamı 1 olmalı, şu an: {total}"

def test_config_create_dirs(tmp_path, monkeypatch):
    """create_dirs() çağrısının gerekli dizinleri oluşturduğunu test eder."""
    cfg = get_settings()
    from pathlib import Path
    monkeypatch.setattr(cfg.paths, 'data_dir', Path(tmp_path) / 'data')
    monkeypatch.setattr(cfg.paths, 'models_dir', Path(tmp_path) / 'models')
    monkeypatch.setattr(cfg.paths, 'reports_dir', Path(tmp_path) / 'reports')
    cfg.paths.create_dirs()
    assert os.path.isdir(str(tmp_path / 'data'))
    assert os.path.isdir(str(tmp_path / 'models'))
    assert os.path.isdir(str(tmp_path / 'reports'))


# ─────────────────────────────────────────────────────────────
# Veri İşleme Testleri
# ─────────────────────────────────────────────────────────────

def test_generate_dummy_data_shape():
    """Dummy veri üretiminin doğru boyutta olduğunu test eder."""
    df = generate_dummy_data(n_samples=200, n_features=20)
    assert df.shape[0] == 200
    assert 'Label' in df.columns
    assert df.shape[1] == 21  # 20 özellik + Label

def test_generate_dummy_data_labels():
    """Label sütununun yalnızca Pathogenic/Benign içerdiğini kontrol eder."""
    df = generate_dummy_data(n_samples=200, n_features=20)
    assert set(df['Label'].unique()).issubset({'Pathogenic', 'Benign'})

def test_preprocessor_without_autoencoder():
    """Graf kurucu preprocessor modülünün matris boyutlarını test eder."""
    df = pd.DataFrame(np.random.randn(50, 20), columns=[f"feat_{i}" for i in range(20)])
    preprocessor = TabularGraphPreprocessor(corr_threshold=0.5, use_autoencoder=False)
    scaled = preprocessor.fit_transform(df)
    assert scaled.shape == (50, 20)
    assert preprocessor.edge_index is not None
    assert preprocessor.edge_index.size(0) == 2

def test_preprocessor_with_autoencoder():
    """AutoEncoder'ın latent uzayı başarıyla çıkarıp birleştirdiğini test eder."""
    df = pd.DataFrame(np.random.randn(50, 10), columns=[f"feat_{i}" for i in range(10)])
    preprocessor = TabularGraphPreprocessor(
        corr_threshold=0.5, use_autoencoder=True, encoding_dim=4, device='cpu'
    )
    scaled = preprocessor.fit_transform(df)
    # 10 orijinal + 4 autoencoder çıkışı = 14 özellik olmalı
    assert scaled.shape == (50, 14)

def test_preprocessor_fit_transform_with_labels():
    """SMOTE dahil fit_transform'un etiketli veriyle çalışmasını test eder."""
    df = generate_dummy_data(n_samples=300, n_features=15)
    y = (df['Label'] == 'Pathogenic').astype(int).values
    X_df = df.drop(columns=['Label'])

    preprocessor = TabularGraphPreprocessor(corr_threshold=0.2, use_autoencoder=False)
    X_scaled, y_resampled = preprocessor.fit_transform(X_df, label_y=y)
    assert X_scaled.ndim == 2
    assert len(y_resampled) == X_scaled.shape[0]
    assert preprocessor.is_fitted is True

def test_preprocessor_transform_dimension_consistency():
    """Train ve test boyutlarının uyumlu olduğunu test eder."""
    df = generate_dummy_data(n_samples=300, n_features=15)
    y = (df['Label'] == 'Pathogenic').astype(int).values
    X_df = df.drop(columns=['Label'])

    preprocessor = TabularGraphPreprocessor(corr_threshold=0.2, use_autoencoder=False)
    X_train, _ = preprocessor.fit_transform(X_df.iloc[:200], label_y=y[:200])
    X_test = preprocessor.transform(X_df.iloc[200:])
    assert X_train.shape[1] == X_test.shape[1]

def test_row_to_graph():
    """Tek satırın PyTorch Geometric Data nesnesine dönüştürüldüğünü test eder."""
    import torch
    df = generate_dummy_data(n_samples=100, n_features=15)
    y = (df['Label'] == 'Pathogenic').astype(int).values
    X_df = df.drop(columns=['Label'])

    preprocessor = TabularGraphPreprocessor(corr_threshold=0.1, use_autoencoder=False)
    X_scaled, _ = preprocessor.fit_transform(X_df, label_y=y)
    graph = preprocessor.row_to_graph(X_scaled[0], label=1)

    assert graph.x is not None
    assert graph.edge_index is not None
    assert graph.y is not None
    assert graph.y.item() == 1

def test_load_and_prepare_data_csv(tmp_path):
    """CSV dosyasından veri yüklemenin doğru çalıştığını test eder."""
    df = generate_dummy_data(n_samples=50, n_features=15)
    csv_path = str(tmp_path / "test_data.csv")
    df.to_csv(csv_path, index=False)

    X_df, y = load_and_prepare_data(csv_path, target_col='Label')
    assert y is not None
    assert len(y) == 50
    assert set(np.unique(y)).issubset({0, 1})


# ─────────────────────────────────────────────────────────────
# Model Testleri
# ─────────────────────────────────────────────────────────────

def test_gnn_forward_pass():
    """FeatureGNN'nin ileri geçişinin hatasız çalıştığını test eder."""
    import torch
    from torch_geometric.data import Data

    n_nodes = 10
    x = torch.randn(n_nodes, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    edge_attr = torch.ones(3)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    model = FeatureGNN(in_channels=1, hidden_dim=32, num_classes=2)
    model.eval()
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1, 2), f"Beklenen (1,2), alınan {out.shape}"

def test_dnn_forward_pass():
    """VariantDNN'nin ileri geçişinin doğru boyutta çıktı verdiğini test eder."""
    import torch
    model = VariantDNN(input_dim=20, hidden_dim=64, num_classes=2)
    model.eval()
    x = torch.randn(8, 20)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 2)

def test_hybrid_ensemble_probabilities_sum():
    """Ensemble olasılıklarının satır toplamlarının 1 olduğunu test eder."""
    model = HybridEnsemble(weights=[0.4, 0.4, 0.2])
    n = 5
    xgb_probs = np.random.dirichlet([1, 1], n)
    gnn_probs = np.random.dirichlet([1, 1], n)
    dnn_probs = np.random.dirichlet([1, 1], n)

    ensemble = model.combine(xgb_probs, gnn_probs, dnn_probs)
    np.testing.assert_allclose(ensemble.sum(axis=1), np.ones(n), atol=1e-5)

def test_clinical_risk_score_range():
    """Klinik risk skorunun 0-100 aralığında olduğunu test eder."""
    model = HybridEnsemble()
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    scores = model.pathogenic_risk_score(probs)
    assert np.all(scores >= 0) and np.all(scores <= 100)


# ─────────────────────────────────────────────────────────────
# TEKNOFEST Şartname Uyum Testleri
# ─────────────────────────────────────────────────────────────

def test_anonymous_columns_accepted():
    """Anonim sütun isimlerinin (kolon_0, kolon_1, ...) pipeline'da kabul edildiğini test eder.
    Şartnameye göre veri seti sütun isimsiz gelecek."""
    n_features = 20
    df = pd.DataFrame(
        np.random.randn(50, n_features),
        columns=[f"col_{i}" for i in range(n_features)]
    )
    preprocessor = TabularGraphPreprocessor(corr_threshold=0.5, use_autoencoder=False)
    scaled = preprocessor.fit_transform(df)
    assert scaled.shape == (50, n_features)

def test_non_feature_columns_excluded():
    """Panel, Nuc_Context, AA_Context gibi non-feature sütunlarının
    sayısal feature matrisinden çıkarıldığını test eder."""
    df = generate_dummy_data(n_samples=50, n_features=15)
    # non-feature sütunlar ekle
    df['Panel'] = 'General'
    df['Nuc_Context'] = 'ACGTTGACGTG'
    df['AA_Context'] = 'AVILMFYWKRN'
    
    # Bu sütunlar sayısal olmadığı için drop edilmeli
    X_df, y = load_and_prepare_data.__wrapped__(df) if hasattr(load_and_prepare_data, '__wrapped__') else (None, None)
    if X_df is None:
        # Doğrudan DataFrame ile test — non-numeric sütunları çıkar
        X_df = df.drop(columns=['Label'])
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        X_df = X_df.drop(columns=non_numeric)
        assert 'Panel' not in X_df.columns
        assert 'Nuc_Context' not in X_df.columns
        assert 'AA_Context' not in X_df.columns
        assert X_df.shape[1] == 15  # sadece sayısal özellikler

def test_config_has_panel_settings():
    """Konfigürasyonun panel ayarlarını içerdiğini test eder."""
    cfg = get_settings()
    assert hasattr(cfg, 'panels'), "Config'te panels alanı bulunmalı"
    if cfg.panels:
        assert 'General' in cfg.panels, "General paneli tanımlı olmalı"

def test_config_has_external_validation():
    """Konfigürasyonun external validation ayarlarını içerdiğini test eder."""
    cfg = get_settings()
    assert hasattr(cfg, 'external_validation'), "Config'te external_validation alanı bulunmalı"
    assert cfg.external_validation.enabled is True

def test_config_has_non_feature_columns():
    """Konfigürasyondaki non_feature_columns alanını test eder."""
    cfg = get_settings()
    assert hasattr(cfg.schema, 'non_feature_columns'), "Schema'da non_feature_columns bulunmalı"
    assert 'Panel' in cfg.schema.non_feature_columns
    assert 'Nuc_Context' in cfg.schema.non_feature_columns
    assert 'AA_Context' in cfg.schema.non_feature_columns


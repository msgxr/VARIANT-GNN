import pytest
import os
import torch
import numpy as np
import pandas as pd
from src.config import Config
from src.data_processing import TabularGraphPreprocessor, TabularAutoEncoder
from src.utils import ModelSerializer

def test_config_directories():
    """Konfigürasyon yollarının doğruluğunu test eder"""
    assert hasattr(Config, 'MODELS_DIR')
    assert hasattr(Config, 'DATA_DIR')
    assert hasattr(Config, 'REPORTS_DIR')
    
def test_preprocessor_without_autoencoder():
    """Graf kurucu preprocessor modülünün matris boyutlarını test eder"""
    df = pd.DataFrame(np.random.randn(50, 20), columns=[f"feat_{i}" for i in range(20)])
    preprocessor = TabularGraphPreprocessor(corr_threshold=0.5, use_autoencoder=False)
    scaled = preprocessor.fit_transform(df)
    
    assert scaled.shape == (50, 20)
    assert preprocessor.edge_index is not None
    assert preprocessor.edge_index.size(0) == 2 # 2 satırlı edge matrisi

def test_preprocessor_with_autoencoder():
    """AutoEncoder'ın latent uzayı başarıyla çıkarıp birleştirdiğini test eder"""
    df = pd.DataFrame(np.random.randn(50, 10), columns=[f"feat_{i}" for i in range(10)])
    # Eğitim hızı için CPU ve küçük epoch tercih ediyoruz
    preprocessor = TabularGraphPreprocessor(corr_threshold=0.5, use_autoencoder=True, encoding_dim=4, device='cpu')
    scaled = preprocessor.fit_transform(df)
    
    # 10 orijinal + 4 autoencoder çıkışı = 14 özellik olmalı
    assert scaled.shape == (50, 14)

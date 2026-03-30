# Tip Güvenliği (Type Hinting) ve Veri Versiyonlama (DVC) - Görev Listesi

## 1. Tip Güvenliği (Type Hinting — Aşama 3)
- [x] `src/config/settings.py` tip tanımlamalarını ekle
- [x] `src/models/ensemble.py` tip tanımlamalarını ekle
- [x] `src/features/preprocessing.py` tip tanımlamalarını ekle
- [x] `src/inference/pipeline.py` tip tanımlamalarını ekle
- [x] `src/ui/*.py` (Yeni UI modülleri) tip tanımlamalarını ekle
- [x] `pytest tests/smoke` ile çalışma zamanını doğrula

## 2. Veri Versiyonlama (DVC — Aşama 4)
- [x] `pip install dvc` ile kurulumu yap
- [x] `dvc init` ile projeyi başlat
- [x] `data/pretrain_100k.csv` dosyasını DVC'ye ekle
- [x] `.gitignore` ve `.dvc` dosyalarını kontrol et
- [x] `dvc status` ile doğrula

## 3. Genel Kontrol ve Kapanış
- [x] Tüm testleri çalıştır (`pytest tests/`)
- [x] Kullanıcı için rehber döküman (walkthrough) oluştur

# VARIANT-GNN

**Grafik Sinir Ağları (GNN) ve Açıklanabilir Yapay Zeka ile Genomik Varyant Patojenite Tahmini**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![CI](https://github.com/msgxr/VARIANT-GNN/actions/workflows/ci.yml/badge.svg)](https://github.com/msgxr/VARIANT-GNN/actions)

> **Uyarı:** Bu sistem yalnızca araştırma amaçlıdır. Bağımsız bir uzman onayı olmadan klinik teşhis veya tedavi kararları için kullanılamaz. (TEKNOFEST 2026 - Sağlıkta Yapay Zeka)

---

## 🌟 Proje Özeti

VARIANT-GNN, insan genomundaki bir genetik varyantın **Patojenik (Hastalık Yapıcı)** mi yoksa **Benign (Zararsız)** mi olduğunu; XGBoost, Grafik Sinir Ağı (VariantSAGEGNN) ve Derin Sinir Ağı (DNN) algoritmalarını birleştirerek yüksek hassasiyetle tahmin eder.

**v3.0 sürümündeki Yeni Özellikler (TEKNOFEST 2026 Şartname Uyumlu):**
- **🧬 43 Öznitelikli Şartname Uyumlu Veri Seti:** Sekans, biyokimyasal, evrimsel, popülasyon ve in silico risk skorları
- **📊 4 Panel Desteği:** Genel, Herediter Kanser, PAH, CFTR — şartnameye uygun varyant sayıları
- **🕸️ VariantSAGEGNN:** GraphSAGE tabanlı indüktif model — skip connections, BatchNorm, WeightedBCELoss
- **🔬 Multimodal Encoder:** ±5 nükleotid/amino asit bağlamı için Embedding+CNN dual-branch encoder
- **✅ External Validation Modu:** Eğitilmiş modeli yeni veri ile test et, F1/AUC/Brier raporla
- **🔬 Gerçek Zamanlı ClinVar API:** NCBI E-utilities üzerinden klinik doğrulama
- **📄 PDF Klinik Rapor Üretici:** Türkçe açıklamalı, SHAP/GNN grafikleri içeren profesyonel rapor
- **💬 Klinik Karar Destek Asistanı:** Model sonuçlarını Türkçe biyolojik metinlere çevirir
- **📊 WeightedBCELoss:** Sınıf dengesizliği için dinamik ağırlıklı kayıp fonksiyonu (SAGE + DNN)

Detaylı güvenlik ve kullanım bilgileri için [SECURITY.md](SECURITY.md) ve [MODEL_CARD.md](MODEL_CARD.md) dosyalarını inceleyebilirsiniz.

---

## 🏗️ Mimari Yapı

```text
Girdi CSV Dosyası (Variant_ID korunarak)
       |
[Şema Doğrulama]     data_contracts/variant_schema.py
       |                (Panel, Nuc_Context, AA_Context ayrıştırılır)
       |
[Veri Ön İşleme]     YALNIZCA Train Fold'da eğitilir (Leakage yok)
    Eksik Veri Tamamlama (Median)
    Robust Ölçeklendirme (RobustScaler)
    AutoEncoder ile Boyut Gizleme (43 → 16 dim latent)
    Aşırı Örnekleme (SMOTE)
    Kosinüs k-NN Grafik Yapısı (Varyant-düğüm tabanlı)
       |
       +--------------------+--------------------+
       |                    |                    |
       ▼                    ▼                    ▼
    XGBoost          VariantSAGEGNN            DNN
     (0.40)      (GraphSAGE + Multimodal)    (0.20)
                       (0.40)
       |                    |                    |
       +--------------------+--------------------+
                            |
                   [Hibrit Ensemble] 
                            |
                [Isotonic Kalibrasyon]
                            |
          Çıktı: Öncelik Takibi, Risk Skoru, SHAP
```

---

## 📁 Proje Klasör Yapısı (v3.0)

```text
VARIANT-GNN/
 ├── configs/
 │   └── default.yaml              Tüm hiperparametrelerin merkezi (Single source of truth)
 ├── data_contracts/
 │   └── variant_schema.py         Pydantic v2 giriş şema denetleyicisi
 ├── src/
 │   ├── config/                   Ayar yükleyici (settings.py)
 │   ├── data/                     Güvenli CSV yükleyici
 │   ├── features/                 AutoEncoder, Preprocessing, Multimodal Encoder
 │   ├── graph/                    Grafik mimarisi oluşturucu
 │   ├── models/                   GNN, DNN ve Hibrit modeller
 │   ├── calibration/              İzotonik risk kalibratörü
 │   ├── evaluation/               Performans grafikleri (ROC, PR, Confusion Matrix)
 │   ├── inference/                Tahmin Pipeline'ı
 │   └── explainability/
 │       ├── clinvar_api.py        NCBI ClinVar Entegrasyonu
 │       ├── pdf_report.py         Türkçe PDF Rapor Üretici
 │       ├── clinical_insight.py   Türkçe Karar Destek Asistanı
 │       ├── graph_viz.py          GNN Grafı ve Isı Haritası (NetworkX)
 │       └── shap_lime_explainer   Yapay Zeka modülleri
 ├── tests/                        Tüm CI/CD test dosyaları
 ├── data/                         Şartname uyumlu varyant veri tabanı (43 öznitelik, 4 panel)
 ├── models/                       Eğitilmiş yapay zeka dosyaları
 ├── app.py                        Streamlit Arayüzü Uygulaması
 └── main.py                       Eğitim ve Tahmin Ana Çalıştırıcı (CLI)
```

---

## 🚀 Kurulum

```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# Sadece CPU PyTorch Kurulumu (Mac/Windows)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Diğer Gerekli Kütüphaneler
pip install -r requirements.txt
pip install fpdf2  # YENİ - PDF Analiz Çıktıları için 
```

---

## 💻 Kullanım Komutları

### Web Arayüzünü (Klinik Dashboard) Başlatma
TEKNOFEST jürisine sunulacak ana arayüz:
```bash
python3 -m streamlit run app.py
# Tarayıcıda http://localhost:8502 adresinde açılır.
```

### Modeli Sıfırdan Eğitme
```bash
python3 main.py --mode train --data_file data/train_variants.csv
```

### Panel Bazlı Eğitim
```bash
python3 main.py --mode train --data_file data/train_general.csv --panel General
```

### Kör Test (Jüri Verisiyle Tahmin Alma)
```bash
python3 main.py --mode predict --test_file data/test_variants_blind.csv
# Sonuçlar 'reports/predictions.csv' olarak kaydedilir.
```

### External Validation (Dış Geçerlilik Testi)
```bash
python3 main.py --mode external_val --test_file data/test_variants.csv
python3 main.py --mode external_val --test_file data/test_variants.csv --panel CFTR
# Sonuçlar 'reports/external_validation_report.json' olarak kaydedilir.
```

---

## 🛠️ Temel Karar Ayarları (`configs/default.yaml`)

```yaml
ensemble:
  weights: [0.4, 0.4, 0.2]      # Model Ağırlıkları: XGBoost, GNN, DNN
  optimize_weights: false

calibration:
  method: isotonic               # Modellerin karar güvenliğini kalibre eden metod
  
preprocessing:
  smote_enabled: true            # Sınıf Dengesizliği Çözümü
  use_autoencoder: true          # Yapay Zeka Özellik Çıkarımı
```

---

## 📈 Değerlendirme Metrikleri

| Metrik | Anlamı ve İşlevi |
|---|---|
| **Macro F1** | Dengelenmiş doğruluğu gösteren **birincil metrik** (şartname) |
| **ROC-AUC** | Modelin Patojenik ile Benign'i ayırma yeteneği |
| **Brier Skoru** | Modelin yüzdelik güven tahmininin kalibrasyon düzeyi |

---

## 📝 Lisans

MIT Lisansı ile lisanslanmıştır detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz. 
Bu proje TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması için **msgxr** takımı çatısı altında tasarlanmış ve geliştirilmiştir.

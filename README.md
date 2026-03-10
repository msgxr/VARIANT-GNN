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

VARIANT-GNN, insan genomundaki bir genetik varyantın **Patojenik (Hastalık Yapıcı)** mi yoksa **Benign (Zararsız)** mi olduğunu; XGBoost, Grafik Sinir Ağı (GNN) ve Derin Sinir Ağı (DNN) algoritmalarını birleştirerek yüksek hassasiyetle tahmin eder.

**v2.0 sürümündeki Yeni Özellikler (TEKNOFEST 2026 İçin Hazır):**
- **🔬 Gerçek Zamanlı ClinVar API:** NCBI E-utilities üzerinden gerçek varyantların klinik doğrulamalarını çeker (Arayüz entegreli).
- **📋 Varyant Önceliklendirme Sistemi:** Kritik derecede yüksek riskli varyantları "Önce İncele" tablosu ile renk kodlu şekilde sunar.
- **📄 PDF Klinik Rapor Üretici:** Matbu çıktıya hazır, Türkçe açıklamalı ve SHAP/GNN grafikleri içeren profesyonel hekim raporu basar.
- **💬 Klinik Karar Destek Asistanı:** Model sonuçlarını tıp uzmanları için anlamlı, Türkçe biyolojik metinlere (NLP) çevirir.
- **🕸️ GNN Etkileşim Grafı:** Yapay zekanın kararı alırken genetik özellikler arasında kurduğu bağlantıları görsel ağ (NetworkX) ile açıklar.
- **📊 10.000 Varyantlık Gelişmiş Veri Seti:** 5-K fold çapraz doğrulama ile **Makro F1: 0.9998** başarısına ulaşılmıştır.

Detaylı güvenlik ve kullanım bilgileri için [SECURITY.md](SECURITY.md) ve [MODEL_CARD.md](MODEL_CARD.md) dosyalarını inceleyebilirsiniz.

---

## 🏗️ Mimari Yapı

```text
Girdi CSV Dosyası (Variant_ID korunarak)
       |
[Şema Doğrulama]     data_contracts/variant_schema.py
       |
[Veri Ön İşleme]     YALNIZCA Train Fold'da eğitilir (Leakage yok)
    Eksik Veri Tamamlama (Median)
    Robust Ölçeklendirme (RobustScaler)
    AutoEncoder ile Boyut Gizleme (34 → 16 dim)
    Aşırı Örnekleme (SMOTE)
    Grafik Kenarları (Train-Korelasyon tabanlı)
       |
       +--------------------+--------------------+
       |                    |                    |
       ▼                    ▼                    ▼
    XGBoost                GNN                  DNN
     (0.40)               (0.40)               (0.20)
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

## 📁 Proje Klasör Yapısı (v2.0)

```text
VARIANT-GNN/
 ├── configs/
 │   └── default.yaml              Tüm hiperparametrelerin merkezi (Single source of truth)
 ├── data_contracts/
 │   └── variant_schema.py         Pydantic v2 giriş şema denetleyicisi
 ├── src/
 │   ├── config/                   Ayar yükleyici (settings.py)
 │   ├── data/                     Güvenli CSV yükleyici
 │   ├── features/                 AutoEncoder ve Preprocessing modülleri
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
 ├── data/                         10.000 satırlık varyant veri tabanı
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

### Modeli Sıfırdan Eğitme (10.000 Varyant)
```bash
python3 main.py --mode train --data_file data/train_variants.csv
```

### Kör Test (Jüri Verisiyle Tahmin Alma)
```bash
python3 main.py --mode predict --test_file data/test_variants_blind.csv
# Sonuçlar 'reports/predictions.csv' olarak kaydedilir.
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

| Metrik | Anlamı ve İşlevi | Skor |
|---|---|---|
| **Macro F1** | Dengelenmiş doğruluğu gösteren **birincil metrik** | 0.9998 |
| **ROC-AUC** | Modelin Patojenik ile Benign'i ayırma yeteneği | 1.0000 |
| **Brier Skoru** | Modelin yüzdelik güven tahmininin kalibrasyon düzeyi | 0.0000 |

---

## 📝 Lisans

MIT Lisansı ile lisanslanmıştır detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz. 
Bu proje TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması için **msgxr** takımı çatısı altında tasarlanmış ve geliştirilmiştir.

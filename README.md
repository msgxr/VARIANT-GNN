# 🧬 VARIANT-GNN

**Graph Neural Network ve Açıklanabilir Yapay Zeka ile Genetik Varyant Patojenite Tahmini**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-red)](https://www.teknofest.org/)

> ⚠️ **Uyarı:** Bu sistem yalnızca araştırma ve eğitim amaçlıdır. Klinik tanı veya tedavi kararı için kullanılamaz.

---

## 📌 Proje Hakkında

VARIANT-GNN, insan genomundaki genetik varyantların **patojenik (hastalık yapıcı)** mi yoksa **benign (zararsız)** mı olduğunu yüksek doğrulukla tahmin eden ve kararlarını açıklayabilen bir yapay zeka sistemidir.

**TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması** kapsamında geliştirilmiştir.

### Neden VARIANT-GNN?

Klasik makine öğrenmesi yöntemleri varyantları **bağımsız** olarak değerlendirir. Bu proje:

- Varyant özelliklerini bir **graf yapısı** içinde temsil eder
- **Graph Neural Network (GNN)** ile özellikler arası ilişkileri öğrenir
- Tahminlerini **SHAP, LIME ve GNNExplainer** ile açıklar
- Klinisyenlere **risk skoru + biyolojik faktör özeti** sunar

---

## 🏗️ Mimari

```
Varyant CSV Verisi
       │
       ▼
┌──────────────────────────────────────┐
│         Veri Ön İşleme              │
│  • Medyan Imputation                 │
│  • RobustScaler Normalizasyon        │
│  • SMOTE Veri Dengeleme             │
│  • Feature Selection (VT + MI)      │
│  • AutoEncoder Gizli Özellik        │
│  • Korelasyon Bazlı Graf Oluşturma  │
└─────────────┬────────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼         ▼          ▼
 XGBoost    GNN (GAT   DNN (MLP)
 Model 1   /GCN)       Model 3
            Model 2
    │         │          │
    └─────────┴──────────┘
              │
         Ensemble
    (Ağırlıklı Ortalama)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
 Tahmin              XAI Panel
 + Risk Skoru    (SHAP/LIME/GNN)
```

### Modeller

| Model | Teknoloji | Rol |
|-------|-----------|-----|
| **Model 1** | XGBoost | Tabular özellik öğrenimi |
| **Model 2** | GNN (GCNConv / GATConv) | Özellik etkileşim grafı |
| **Model 3** | Deep Neural Network | Derin özellik analizi |
| **Ensemble** | Ağırlıklı ortalama [0.4, 0.4, 0.2] | Final karar |

---

## 🔬 Özellikler

### 🧠 Açıklanabilir Yapay Zeka (XAI)
- **SHAP (Global)** — Tüm varyantlarda özellik önem haritası
- **SHAP Waterfall (Yerel)** — Tek varyant karar açıklaması
- **LIME** — Yerel lineer yaklaşım ile tahmin açıklaması
- **GNNExplainer** — GNN katmanlarında hangi bağlantıların kritik olduğunu açıklar

### 📊 Veri İşleme
- **SMOTE** ile sınıf dengesizliği giderme
- **AutoEncoder** ile gizli özellik sentezi
- **VarianceThreshold + SelectKBest (mutual_info)** feature selection
- **PCA** boyut indirgeme (opsiyonel)
- Korelasyon bazlı **Feature-Interaction Graph** oluşturma

### 📐 Model Değerlendirme
- **Stratified K-Fold Cross-Validation** (5-fold)
- Tüm fold'larda F1 Macro skoru raporlaması
- ROC-AUC ve Precision-Recall eğrileri
- Confusion matrix görselleştirmesi

---

## 🚀 Kurulum

```bash
# 1. Repoyu klonlayın
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

---

## 💻 Kullanım

### Eğitim

```bash
# Sentetik veri ile hızlı test
python main.py --mode train

# Gerçek veri seti ile eğitim
python main.py --mode train --data_file data/train.csv
```

### Diğer Modlar

```bash
# Hiperparametre optimizasyonu (Optuna)
python main.py --mode tune --data_file data/train.csv

# 5-Fold Cross-Validation
python main.py --mode crossval --data_file data/train.csv

# Kaydedilmiş model değerlendirmesi
python main.py --mode eval --data_file data/train.csv

# Blind tahmin üretimi (submission.csv)
python main.py --mode predict --test_file data/test.csv
```

### Web Arayüzü

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini açın.

---

## 🖥️ Web Arayüzü Özellikleri

- 📂 **CSV Yükleme** — Varyant özellik dosyasını sürükle-bırak
- 📊 **Özet Metrikler** — Patojenik/Benign sayıları, Risk skoru, Güven skoru
- 🔍 **Detaylı Sonuç Tablosu** — İndirilebilir CSV çıktısı
- 🧠 **XAI Paneli:**
  - 🌍 Global SHAP özet grafiği
  - 🔍 Seçili varyant için SHAP Waterfall
  - 🟡 LIME HTML + Bar grafiği
  - 🕸️ GNN Graf açıklanabilirliği (opsiyonel)

---

## 📁 Proje Yapısı

```
VARIANT-GNN/
├── main.py                  ← Ana eğitim/değerlendirme/tahmin scripti
├── app.py                   ← Streamlit web arayüzü
├── requirements.txt         ← Python bağımlılıkları
├── src/
│   ├── config.py            ← Merkezi konfigürasyon
│   ├── models.py            ← GNN, DNN, Ensemble model tanımları
│   ├── data_processing.py   ← Veri ön işleme + Graf oluşturucu
│   ├── train.py             ← Eğitim döngüleri + Cross-Validation
│   ├── evaluate.py          ← Metrik hesaplama + görselleştirme
│   ├── explain.py           ← SHAP, LIME, GNNExplainer modülü
│   ├── tune.py              ← Optuna hiperparametre optimizasyonu
│   └── utils.py             ← Model kayıt/yükleme yardımcıları
├── models/                  ← Eğitilmiş model dosyaları (.pth, .json)
├── reports/                 ← SHAP grafikleri, confusion matrix, ROC eğrisi
├── data/                    ← Veri seti klasörü (README içinde format var)
└── tests/
    └── test_pipeline.py     ← Temel pipeline testi
```

---

## 📊 Performans Metrikleri

Yarışma değerlendirme kriteri: **Macro F1-Score**

| Metrik | Ölçüm |
|--------|-------|
| F1 Score (Macro) | Ana metrik |
| Precision (Macro) | Yardımcı |
| Recall (Macro) | Yardımcı |
| ROC-AUC | Yardımcı |

---

## 🗃️ Veri Formatı

`data/train.csv` beklenen format:

```
Feature_1, Feature_2, ..., Feature_N, Label
0.85, 1.22, ..., 0.31, Pathogenic
0.10, 0.55, ..., 0.80, Benign
```

Desteklenen etiket değerleri: `Pathogenic`, `Likely Pathogenic`, `Benign`, `Likely Benign`, `0`, `1`

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| GNN Motoru | PyTorch Geometric (GCNConv / GATConv) |
| Gradient Boosting | XGBoost |
| Derin Öğrenme | PyTorch |
| XAI | SHAP, LIME, GNNExplainer |
| Feature Engineering | scikit-learn, imbalanced-learn |
| Web Arayüzü | Streamlit |
| Hiperparametre | Optuna |

---

## 📜 Lisans

Bu proje araştırma amaçlı geliştirilmiştir. Ticari kullanım ve klinik uygulama için uygun değildir.

---

<p align="center">
  Geliştirilen: TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması<br/>
  <strong>VARIANT-GNN Ekibi</strong>
</p>

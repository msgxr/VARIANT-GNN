# VARIANT-GNN: Graph Neural Network Tabanlı Genetik Varyant Patojenite Tahmini

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TEKNOFEST 2026](https://img.shields.io/badge/TEKNOFEST-2026%20Sağlıkta%20AI-red.svg)](https://teknofest.org)
[![Araştırma Prototipi](https://img.shields.io/badge/Durum-Araştırma%20Prototipi-orange.svg)](MODEL_CARD.md)

> ⚠️ **Araştırma Amaçlıdır — Klinik Kullanım Dışı.**  
> Bu sistem yalnızca araştırma ve yarışma demonstrasyonu amacıyla tasarlanmıştır. Tıbbi karar almada kullanılmamalıdır.

---

## 🧬 Proje Özeti

VARIANT-GNN, genomik varyantların **patojenik / benign** olarak sınıflandırılması için geliştirilmiş çok modelli (multi-modal) bir yapay zeka sistemidir.

**Temel bileşenler:**
- 🌳 **XGBoost** — Gradient boosted decision tree (tabular özellikler)
- 🕸️ **GNN (Graph Convolutional Network)** — Özellik-korelasyon grafı üzerinde mesaj geçişi
- 🧠 **DNN (Deep Neural Network)** — Derin tabular öğrenme
- 🔍 **XAI Modülü** — SHAP + LIME + GNNExplainer açıklamaları

**Giriş:** 34 genomik özellik (SIFT, CADD, REVEL, PolyPhen2, gnomAD AF, GERP++, ...)  
**Çıkış:** Pathogenic / Benign tahmini + 0-100 Risk Skoru + SHAP açıklaması

---

## 🏗️ Mimari

```
Giriş: 34 Genomik Özellik
          │
    Ön İşleme (RobustScaler + Imputer + SMOTE)
          │
    ┌─────┴──────┬───────────────┐
    ▼            ▼               ▼
 XGBoost    GNN (GCNConv)    DNN (MLP)
 [%40]        [%40]           [%20]
    │            │               │
    └─────┬──────┴───────────────┘
          ▼
   Ağırlıklı Ensemble
          │
          ▼
   Risk Skoru + SHAP/LIME Açıklama
```

---

## ⚡ Hızlı Başlangıç

### Windows (Tek Tıkla)
```bat
baslat.bat
```

### Manuel Kurulum
```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py --mode train
streamlit run app.py
```

### Docker
```bash
docker build -t variant-gnn .
docker run -p 8501:8501 variant-gnn
# http://localhost:8501 adresini açın
```

---

## 📋 Kullanım

| Komut | Açıklama |
|-------|---------|
| `python main.py --mode train` | Model eğitimi (sentetik veri) |
| `python main.py --mode train --data_file data/train_variants.csv` | Gerçek CSV ile eğitim |
| `python main.py --mode crossval` | Leakage-free 5-fold cross-validation |
| `python main.py --mode eval` | Kaydedilmiş modelleri değerlendirme |
| `python main.py --mode predict --test_file data/test_variants_blind.csv` | Yarışma submission tahmini |
| `python main.py --mode tune` | Optuna hiperparametre optimizasyonu |
| `streamlit run app.py` | Web arayüzü |
| `pytest tests/ -v` | Testleri çalıştır |

---

## 📊 Veri Formatı

Giriş CSV dosyası aşağıdaki 34 özelliği içermelidir (bkz. [`data_contracts/input_schema.json`](data_contracts/input_schema.json)):

| Kategori | Özellikler |
|----------|-----------|
| Fonksiyon Etki | SIFT_score, PolyPhen2_HDIV/HVAR_score, CADD_phred, REVEL_score, MutPred2_score, VEST4_score, PROVEAN_score, MutationTaster_score |
| Evrimsel Korunmuşluk | GERP_RS, PhyloP100way_vertebrate, phastCons100way_vertebrate, SiPhy_29way_logOdds |
| Popülasyon Frekansı | gnomAD_exomes_AF, gnomAD_exomes_AF_afr/eur/eas, ExAC_AF |
| Biyokimyasal | AA_polarity_change, AA_hydrophobicity_diff, AA_size_diff, Protein_impact_score, Secondary_structure_disruption |
| Varyant Tipi | Variant_type (0=missense,1=nonsense,2=frameshift,3=synonymous), In_critical_protein_domain, Splice_site_distance, Is_exonic, Exon_conservation_ratio |
| Klinik | ClinVar_review_status, ClinVar_submitter_count, OMIM_disease_gene |
| Meta Skorlar | MetaSVM_score, MetaLR_score, MCAP_score |

Opsiyonel: `Variant_ID` sütunu (tahmin çıktısında korunur)  
Eğitim için: `Label` sütunu → `Pathogenic` / `Benign` / `Likely Pathogenic` / `Likely Benign`

---

## 🧪 Test

```bash
pytest tests/ -v --tb=short
```

---

## ⚠️ Bilinen Sınırlamalar

| Sınırlama | Durum |
|-----------|-------|
| Gerçek klinik veri | ❌ Mevcut sürüm sentetik veri kullanıyor |
| ACMG 5-sınıf uyumluluk | ❌ Yalnızca binary P/B |
| Risk skoru kalibrasyonu | ⚠️ Ham prob×100 — kalibrasyonsuz |
| VUS (belirsiz sınıf) | ❌ Desteklenmiyor |

Detaylı sınırlamalar için: [MODEL_CARD.md](MODEL_CARD.md)

---

## 📁 Proje Yapısı

```
VARIANT-GNN/
├── src/              # Kaynak kod modülleri
├── data/             # Veri dosyaları (sentetik)
├── data_contracts/   # Giriş/çıkış şema tanımları
├── configs/          # Konfigürasyon dosyaları
├── models/           # Eğitilmiş model ağırlıkları (gitignore)
├── reports/          # Üretilen raporlar ve görseller
├── tests/            # Unit testler
├── .github/workflows/ # CI/CD pipeline
├── app.py            # Streamlit web arayüzü
├── main.py           # CLI giriş noktası
├── MODEL_CARD.md     # Model kartı
└── CITATION.cff      # Atıf dosyası
```

---

## 📄 Lisans

MIT License — Bkz. [LICENSE](LICENSE)

---

## 🔖 Atıf

```bibtex
@software{variant_gnn_2026,
  title  = {VARIANT-GNN: Graph Neural Network Based Variant Pathogenicity Prediction},
  author = {VARIANT-GNN Ekibi},
  year   = {2026},
  url    = {https://github.com/msgxr/VARIANT-GNN}
}
```

---

## ⚕️ Tıbbi Sorumluluk Reddi

Bu yazılım tıbbi cihaz, tanı aracı veya klinik karar destek sistemi değildir. Genetik varyantların klinik yorumu için akredite tıbbi genetik uzmanına başvurunuz.

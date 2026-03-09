# VARIANT-GNN: Graph Neural Network Tabanlı Genetik Varyant Patojenite Tahmini

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TEKNOFEST 2026](https://img.shields.io/badge/TEKNOFEST-2026%20Sağlıkta%20AI-red.svg)](https://teknofest.org)

> **Araştırma Prototipi — Klinik Kullanım Dışı.**  
> Bu sistem yalnızca araştırma ve geliştirme amacıyla tasarlanmıştır.  
> Tıbbi karar alma için kullanılmamalıdır.

---

## Proje Özeti

VARIANT-GNN, genomik varyantların patojenik/benign olarak sınıflandırılması için geliştirilmiş çok modelli (multi-modal) bir yapay zeka sistemidir. XGBoost, Graph Neural Network (GNN) ve Deep Neural Network (DNN) modellerini bir ensemble içinde birleştirerek SIFT, CADD, REVEL, gnomAD ve 31 ek özellik üzerinden tahmin yapar.

**Önemli Not:** Mevcut sürüm gerçekçi sentetik veri üzerinde çalışmaktadır. Klinik geçerlilik için gerçek ClinVar/gnomAD verisi ile doğrulama gereklidir.

---

## Mimari

```
Giriş: 34 Genomik Özellik (SIFT, CADD, REVEL, gnomAD AF, ...)
          │
          ▼
    Ön İşleme (RobustScaler + SimpleImputer + SMOTE)
          │
    ┌─────┴──────┬───────────────┐
    ▼            ▼               ▼
 XGBoost      GNN (GCN)       DNN (MLP)
 (Model 1)  (Model 2)       (Model 3)
    │            │               │
    └─────┬──────┴───────────────┘
          ▼
   Weighted Ensemble [0.4, 0.4, 0.2]
          │
          ▼
   Risk Skoru + SHAP/LIME Açıklama
```

---

## Kurulum

### Gereksinimler
- Python 3.10+
- CUDA 11.8+ (opsiyonel, GPU için)

### Hızlı Kurulum

```bash
# Repoyu klonla
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Windows Hızlı Başlatıcı
```bash
baslat.bat
```

---

## Kullanım

### Model Eğitimi
```bash
python main.py --mode train --data_file data/train_variants.csv
```

### Hiperparametre Optimizasyonu
```bash
python main.py --mode tune --data_file data/train_variants.csv
```

### Cross-Validation
```bash
python main.py --mode crossval --data_file data/train_variants.csv
```

### Tahmin (Kör Test)
```bash
python main.py --mode predict --test_file data/test_variants_blind.csv
```

### Web Arayüzü
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t variant-gnn .
docker run -p 8501:8501 variant-gnn
```

---

## Veri Formatı

Giriş CSV dosyası aşağıdaki 34 özelliği içermelidir:

| Kategori | Özellikler |
|----------|-----------|
| Fonksiyon Etki Skorları | SIFT_score, PolyPhen2_HDIV_score, PolyPhen2_HVAR_score, CADD_phred, REVEL_score, MutPred2_score, VEST4_score, PROVEAN_score, MutationTaster_score |
| Evrimsel Korunmuşluk | GERP_RS, PhyloP100way_vertebrate, phastCons100way_vertebrate, SiPhy_29way_logOdds |
| Popülasyon Frekansı | gnomAD_exomes_AF, gnomAD_exomes_AF_afr, gnomAD_exomes_AF_eur, gnomAD_exomes_AF_eas, ExAC_AF |
| Biyokimyasal | AA_polarity_change, AA_hydrophobicity_diff, AA_size_diff, Protein_impact_score, Secondary_structure_disruption |
| Varyant Tipi | Variant_type, In_critical_protein_domain, Splice_site_distance, Is_exonic, Exon_conservation_ratio |
| Klinik | ClinVar_review_status, ClinVar_submitter_count, OMIM_disease_gene |
| Meta Skorlar | MetaSVM_score, MetaLR_score, MCAP_score |

Etiket sütunu: `Label` → `Pathogenic` / `Benign`  
Opsiyonel ID sütunu: `Variant_ID`

---

## Test

```bash
# Tüm testler
pytest tests/ -v

# Sadece unit testler
pytest tests/unit/ -v
```

---

## Sınırlamalar

1. **Gerçek Klinik Veri:** Mevcut sürüm istatistiksel sentetik veri üzerinde eğitilmiştir.
2. **ACMG Uyumu:** Sistem yalnızca binary (P/B) sınıflandırma yapar; ACMG 5 sınıfını desteklemez.
3. **Kalibrasyon:** Risk skoru kalibre edilmemiştir; olasılık değerleri klinik karar için kullanılmamalıdır.
4. **VUS:** Belirsizlik sınıfı desteklenmemektedir.

---

## Lisans

MIT License — Bkz. [LICENSE](LICENSE)

---

## Atıf

```bibtex
@software{variant_gnn_2026,
  title  = {VARIANT-GNN: Graph Neural Network Based Variant Pathogenicity Prediction},
  author = {VARIANT-GNN Ekibi},
  year   = {2026},
  url    = {https://github.com/msgxr/VARIANT-GNN}
}
```

---

## Uyarı

Bu yazılım tıbbi cihaz, tanı aracı veya klinik karar destek sistemi değildir. Yanlış sınıflandırma hasta sağlığını etkileyebilir. Genetik varyantların klinik yorumu için akredite genetik uzmanına başvurunuz.

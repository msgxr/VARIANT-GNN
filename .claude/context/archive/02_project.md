---
Dosya: 02_project.md
Klasör: context/
---

# VARIANT-GNN — Proje Bağlamı

## Takım Yapısı — XYRA3

| Kişi | KYS Rolü | Proje İçi İşlevsel Rol |
|---|---|---|
| Şeyma Nur Çebi | Takım Kaptanı, İletişim Sorumlusu | ML/İstatistik Koordinasyonu, Model Geliştirme Ana Sorumluluğu |
| Muhammed Sina Gün | Takım Üyesi | Sistem Kurgusu, Teknik Tasarım, Deney Doğrulama, Raporlama, Yarışma Uyumu |
| Şahin Kara | Takım Üyesi | Biyoinformatik Uzmanı, Veri ve Etiket Kalitesi |
| Burak Küçükcengiz | Takım Üyesi | Yazılım Geliştirici, MLOps ve Uygulama Altyapısı |
| Pınar Karadayı Ataş | Danışman | Akademik ve Yöntemsel Rehberlik |

## Problem

Missense varyantlar için Patojenik/Benign ikili sınıflandırması. Model yalnızca yarışma komitesinden gelen anonim öznitelik vektörleri üzerinden öğrenir.

---

## Gerçek Teknik Mimari — Kod Doğrulandı (src/core/gnn.py)

### Ensemble

| Model | Ağırlık | Not |
|---|---|---|
| XGBoost | %30 | SHAP, eksik değerlere dayanıklı |
| LightGBM | %30 | Hız, regularizasyon |
| VariantGATv2GNN | %25 | GATv2Conv dinamik dikkat |
| DNN | %15 | BatchNorm + Dropout, 3 katman |
| Stacking Meta-Learner | — | Lojistik regresyon |

### GNN — VariantGATv2GNN (GATv2Conv)

- GATv2Conv: 4 kafa, dinamik dikkat (statik GAT problemi çözülmüş)
- 3 blok: LayerNorm + LeakyReLU + Dropout(0.3) + Skip connection
- hidden_dim=128 | MC Dropout: 30 forward pass
- Graf: Cosine k-NN (k=10, eşik=0.3)

**PSR-Kod Uyuşmazlığı:** PSR "VariantSAGEGNN/GraphSAGE" yazdı — kod GATv2Conv kullanıyor.
PDR'de: VariantGATv2GNN yazılmalı + PSR ile fark not edilmeli.

### Veri Pipeline (sızıntı-güvenli, tümü train fold'a fit)

1. Medyan Imputation → 2. RobustScaler → 3. SelectKBest(k=35)
4. AutoEncoder(43→16) → 5. SMOTE(yalnız train) → 6. Cosine k-NN Graf

Veri bölme: 65/15/20 — Stratified 5-Fold CV — random_state=42
Karar eşiği: **0.4357** (calibration_set optimize, duyarlılık öncelikli)

---

## Gerçek Yarışma Sonuçları (reports/cv_report.json — 2026-05-15)

**Genel:**
- CV Binary F1: 0.8347 ± 0.0114 | Test Binary F1: **0.8706**
- Test Macro F1: 0.6885 | MCC: 0.4063 | PR-AUC: 0.8843 | ROC-AUC: 0.7797

**Panel (test seti):**

| Panel Kodu | PDR Adı | Binary F1 | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| General | MASTER | 0.8675 | 0.4199 | 0.8778 | 0.7795 |
| Hereditary_Cancer | KANSER | 0.8515 | 0.5112 | 0.9095 | 0.8812 |
| PAH | PAH | 0.9051 | 0.1466 ⚠️ | 0.9395 | 0.6704 |
| CFTR | CFTR | 0.8750 | 0.2435 ⚠️ | 0.8394 | 0.6083 |

**⚠️ MCC Riski:** PAH(0.15) ve CFTR(0.24) MCC düşük — Benign sınıfında model başarısız.
Binary F1 yüksek ama MCC düşük = yüksek recall, düşük precision. PDR'de açıklanmalı.

**PSR vs Gerçek Karşılaştırması:**
PSR'de pilotta MCC=0.892 iddia edildi. Gerçekte 0.40. Bu büyük fark jüri sorusu olacak.
Açıklama: Pilot veri (ClinVar Expert Panel) daha temiz — yarışma verisi daha zorlu.

## PDR Güçlendirme Planı (PSR zayıf noktaları)

| §4.4 Açıklanabilirlik (3.33/5) | Bireysel SHAP + GNNExplainer subgraph + panel bazlı tablo |
| §4.5 Teknik Evrim (3.33/5) | Deney günlüğü tablosu + ablation: XGBoost-only vs ensemble |
| §5.1 Mimari Gerekçe (4/5) | 5 model × 4 panel karşılaştırma tablosu |
| MCC tutarsızlığı | Pilot vs gerçek farkı açıkla: veri zorluğu, eşik seçimi |

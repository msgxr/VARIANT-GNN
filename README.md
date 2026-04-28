<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=300&color=0:0f172a,25:1e3a5f,50:1d4ed8,75:059669,100:0f172a&text=VARIANT-GNN&fontSize=90&fontAlignY=38&fontColor=ffffff&desc=TEKNOFEST%202026%20%7C%20Sağlıkta%20Yapay%20Zeka%20Yarışması&descAlignY=62&descFontSize=22&descFontColor=94a3b8" alt="VARIANT-GNN Banner"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&duration=2800&pause=900&color=22D3EE&center=true&vCenter=true&width=1200&lines=PSR+AŞAMASI+GEÇILDI+—+93.00+%2F+100+PUAN;Missense+Varyant+Patojenitesi+Tahmini;Hibrit+GNN+Ensemble+Mimarisi;GATv2+%2B+XGBoost+%2B+LightGBM+%2B+DNN;Türkçe+Klinik+Rapor+%2B+MC+Dropout+Belirsizlik;PDR+Aşaması+Geliştirmesi+Devam+Ediyor..." alt="Typing SVG"/>

<br/>

[![PSR Geçildi](https://img.shields.io/badge/PSR-GEÇİLDİ_93%2F100-22c55e?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![Takım](https://img.shields.io/badge/Takım-XYRA3_%23909249-3b82f6?style=for-the-badge&logo=groups&logoColor=white)](.)
[![Kategori](https://img.shields.io/badge/Kategori-Üniversite_ve_Üzeri-8b5cf6?style=for-the-badge&logo=mortarboard&logoColor=white)](.)
[![Lisans](https://img.shields.io/badge/Lisans-TEKNOFEST_NDA-ef4444?style=for-the-badge&logo=shield&logoColor=white)](.)

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=flat-square&logo=pytorch)](.)
[![PyG](https://img.shields.io/badge/PyG-2.5.0-ff6b35?style=flat-square&logo=graphql)](.)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-189ab4?style=flat-square)](.)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-2d9a27?style=flat-square)](.)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)](.)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit)](.)

</div>

---

## 📋 İçindekiler

| # | Bölüm | Açıklama |
|:---:|:---|:---|
| 1 | [🔬 Proje Kimliği](#-proje-kimliği) | Proje adı, takım, aşama bilgileri |
| 2 | [🧠 VARIANT-GNN Nedir?](#-variant-gnn-nedir) | Projenin amacı ve motivasyonu |
| 3 | [🏗️ Mimari](#%EF%B8%8F-mimari) | Katmanlı sistem mimarisi |
| 4 | [⚙️ Teknik Bileşenler](#%EF%B8%8F-teknik-bileşenler) | Tüm modeller ve algoritmalar |
| 5 | [🧬 Veri Mimarisi](#-veri-mimarisi) | Panel yapısı, özellik seti, etiketler |
| 6 | [🔄 Eğitim Protokolü](#-eğitim-protokolü) | 5-fold CV, veri bölme, sızıntı kontrolü |
| 7 | [📊 Performans Sonuçları](#-performans-sonuçları) | Panel bazlı metrikler ve PSR puanları |
| 8 | [🔍 Açıklanabilirlik Sistemi](#-açıklanabilirlik-sistemi) | SHAP, GNNExplainer, Türkçe rapor |
| 9 | [🛡️ Güvenilirlik Katmanı](#%EF%B8%8F-güvenilirlik-katmanı) | Kalibrasyon, MC Dropout, belirsizlik |
| 10 | [📁 Dizin Yapısı](#-dizin-yapısı) | Proje dosya organizasyonu |
| 11 | [🚀 Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma) | Adım adım kurulum kılavuzu |
| 12 | [💻 Kullanım Kılavuzu](#-kullanım-kılavuzu) | Tüm çalıştırma modları |
| 13 | [🗺️ PDR Yol Haritası](#%EF%B8%8F-pdr-yol-haritası) | Sonraki aşama geliştirmeleri |
| 14 | [📚 Referanslar](#-referanslar) | Bilimsel kaynaklar |

---

## 🔬 Proje Kimliği

<div align="center">

| Özellik | Değer |
|:---|:---|
| 🏷️ **Proje Adı** | `VARIANT-GNN` |
| 🎯 **Görev** | Missense Genetik Varyantların Patojenik / Benign Sınıflandırması |
| 👥 **Takım** | **XYRA3** — ID: `#909249` |
| 🔖 **Başvuru ID** | `#4865399` |
| 🏫 **Kategori** | TEKNOFEST 2026 Sağlıkta YZ — Üniversite ve Üzeri |
| 🏆 **PSR Puanı** | **93.00 / 100** — GEÇILDI ✅ |
| 📌 **Güncel Aşama** | PDR (Proje Detay Raporu) Geliştirmesi |
| 🔐 **Veri Güvenliği** | KVKK + GDPR + NDA uyumlu |

</div>

---

## 🧠 VARIANT-GNN Nedir?

**VARIANT-GNN**, insan genomundaki missense varyantların klinik anlamlılığını — yani hastalık yapıcı (**Patojenik**) mi yoksa zararsız (**Benign**) mi olduğunu — otomatik olarak tahmin eden, uçtan uca kalibre edilmiş bir **yapay zeka karar destek sistemidir.**

### Neden Önemli?

> İnsanlık genomundaki milyonlarca genetik varyantın büyük çoğunluğunun klinik anlamı hâlâ bilinmemektedir. Bir genetik test sonucunda gelen "VUS — Önemi Belirsiz Varyant" etiketi, hem hasta hem de klinisyen için ciddi bir belirsizlik kaynağıdır. VARIANT-GNN bu belirsizliği yapay zeka ile çözmeye çalışır.

### Temel Farkımız

```
Tek Model        →  Tek bakış açısı, sınırlı genelleme
VARIANT-GNN      →  4 farklı modelin hibrit stacking ensemble'ı
                     + Graf topolojisi (varyantlar arası benzerlik)
                     + Kalibrasyon (olasılıkların gerçeğe uygunluğu)
                     + MC Dropout belirsizlik ölçümü
                     + ACMG uyumlu Türkçe klinik raporlama
```

---

## 🏗️ Mimari

### Genel Bakış — Uçtan Uca Pipeline

```mermaid
graph TD
    classDef giriş fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#e2e8f0
    classDef onisleme fill:#052e16,stroke:#22c55e,stroke-width:2px,color:#dcfce7
    classDef model fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#dbeafe
    classDef birlesim fill:#3b0764,stroke:#a78bfa,stroke-width:2px,color:#ede9fe
    classDef guven fill:#431407,stroke:#fb923c,stroke-width:2px,color:#ffedd5
    classDef cikti fill:#3f3f46,stroke:#f59e0b,stroke-width:2px,color:#fef3c7
    classDef acik fill:#0c4a6e,stroke:#7dd3fc,stroke-width:2px,color:#e0f2fe

    A[("🧬 Ham Varyant Profili\nAnonymizelmiş CSV")]:::giriş

    A --> B1["🔧 Medyan İmputation\nEksik Değer Doldurma"]:::onisleme
    B1 --> B2["📏 RobustScaler\nIQR Bazlı Normalizasyon"]:::onisleme
    B2 --> B3["🎯 SelectKBest k=35\nANOVA Özellik Seçimi"]:::onisleme
    B3 --> B4["🗜️ AutoEncoder 43→16\nLatent Temsil"]:::onisleme
    B4 --> B5["⚖️ SMOTE %30\nSınıf Dengesi"]:::onisleme
    B5 --> B6["🕸️ Cosine k-NN Graf\nk=10, eşik=0.3"]:::onisleme

    B6 --> M1["🌳 XGBoost\n%30 ağırlık"]:::model
    B6 --> M2["💡 LightGBM\n%30 ağırlık"]:::model
    B6 --> M3["🕸️ VariantGATv2GNN\n%25 ağırlık"]:::model
    B6 --> M4["🧠 DNN\n%15 ağırlık"]:::model

    M1 --> S["🔗 Stacking\nMeta-Öğrenici\nLojistik Regresyon"]:::birlesim
    M2 --> S
    M3 --> S
    M4 --> S

    S --> K["📊 İsotonik Kalibrasyon\nECE < 0.025"]:::guven
    K --> U["🎲 MC Dropout\nBelirsizlik Skoru"]:::guven

    U --> OUT1["✅ Patojenik / Benign\nKarar"]:::cikti
    U --> OUT2["📈 Risk Skoru 0–100\nKalibre Olasılık"]:::cikti
    U --> OUT3["⚠️ Uzman Bayrağı\nBelirsizlik > 0.30"]:::cikti
    U --> OUT4["📄 Türkçe Klinik Rapor\nACMG Uyumlu PDF"]:::cikti

    OUT1 --> XAI["🔍 SHAP + GNNExplainer\n6 Biyolojik Kategori"]:::acik
    OUT2 --> XAI
    XAI --> OUT4
```

---

### Model Katmanları — Detaylı Görünüm

```mermaid
graph LR
    classDef l1 fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff
    classDef l2 fill:#0f2d1f,stroke:#4ade80,stroke-width:2px,color:#dcfce7
    classDef l3 fill:#0c1e40,stroke:#60a5fa,stroke-width:2px,color:#dbeafe
    classDef l4 fill:#2d1a00,stroke:#fbbf24,stroke-width:2px,color:#fef3c7
    classDef l5 fill:#1a0a2e,stroke:#c084fc,stroke-width:2px,color:#f3e8ff

    subgraph K1 ["🔹 Katman 1 — Veri Girişi"]
        direction TB
        V1["Panel Verisi\n(General / HC / PAH / CFTR)"]:::l1
        V2["Nüklotid Bağlamı\nNuc_Context ±5"]:::l1
        V3["Amino Asit Bağlamı\nAA_Context ±5"]:::l1
    end

    subgraph K2 ["🔹 Katman 2 — Ön İşleme"]
        direction TB
        P1["İmputation"]:::l2
        P2["RobustScaler"]:::l2
        P3["SelectKBest"]:::l2
        P4["AutoEncoder"]:::l2
        P5["SMOTE"]:::l2
        P6["k-NN Graf"]:::l2
    end

    subgraph K3 ["🔹 Katman 3 — Modeller"]
        direction TB
        XG["XGBoost\nmax_depth=6\nn_est=200"]:::l3
        LG["LightGBM\nnum_leaves=63\nn_est=300"]:::l3
        GN["VariantGATv2GNN\nhidden=128\nblok×3"]:::l3
        DN["DNN\nhidden=128\nBatchNorm+Dropout"]:::l3
    end

    subgraph K4 ["🔹 Katman 4 — Birleşim"]
        direction TB
        ST["Stacking\nMeta-Öğrenici"]:::l4
        NM["Nelder-Mead\nAğırlık Optimizasyonu"]:::l4
    end

    subgraph K5 ["🔹 Katman 5 — Güvenilirlik & Çıktı"]
        direction TB
        CAL["İsotonik\nKalibrasyon"]:::l5
        MC["MC Dropout\nn=30 geçiş"]:::l5
        RPT["Türkçe\nKlinik Rapor"]:::l5
    end

    K1 --> K2 --> K3 --> K4 --> K5
```

---

### VariantGATv2GNN Mimarisi — Detaylı

```mermaid
graph TB
    classDef inp fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#e2e8f0
    classDef enc fill:#1c1917,stroke:#f97316,stroke-width:2px,color:#ffedd5
    classDef gat fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#dbeafe
    classDef cls fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#dcfce7

    NF["Sayısal Özellikler\n[N × 43]"]:::inp
    NC["Nuc_Context\nCNN Encoder"]:::enc
    AC["AA_Context\nCNN Encoder"]:::enc

    NF --> PROJ["Linear Projeksiyon\n43 + seq_dim → 128"]:::inp
    NC --> SE["Sekans Encoder\nCNN → 32 dim"]:::enc
    AC --> SE
    SE --> PROJ

    PROJ --> B1["GATv2 Blok 1\nMulti-Head (4 kafa)\nLayerNorm + Skip + Dropout"]:::gat
    B1 --> B2["GATv2 Blok 2\nAynı yapı"]:::gat
    B2 --> B3["GATv2 Blok 3\nAynı yapı"]:::gat

    B3 --> C1["Linear 128 → 64"]:::cls
    C1 --> C2["LeakyReLU + Dropout"]:::cls
    C2 --> C3["Linear 64 → 2"]:::cls
    C3 --> OUT["Softmax → [Benign, Patojenik]"]:::cls

    EDGE["k-NN Edge Index\n(k=10, Cosine)"]:::inp --> B1
    EDGE --> B2
    EDGE --> B3
```

---

### Kalibrasyon ve Belirsizlik Akışı

```mermaid
graph LR
    classDef raw fill:#3f1d2e,stroke:#f472b6,stroke-width:2px,color:#fce7f3
    classDef cal fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#dcfce7
    classDef mc fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#dbeafe
    classDef dec fill:#3f3f46,stroke:#f59e0b,stroke-width:2px,color:#fef3c7

    E["Ensemble Ham\nOlasılıkları\nECE > 0.08"]:::raw
    E --> ISO["İsotonik Regresyon\nKalibrasyon\nECE < 0.025"]:::cal
    ISO --> CAL_OUT["Kalibre Olasılıklar\nBrier < 0.072"]:::cal

    CAL_OUT --> MC_IN["MC Dropout\n30 Forward Pass"]:::mc
    MC_IN --> MEAN["Ortalama\nTahmin"]:::mc
    MC_IN --> STD["Standart Sapma\nBelirsizlik Skoru"]:::mc

    MEAN --> THR{"Eşik ≥ 0.40?"}:::dec
    THR -- "Evet" --> PAT["Patojenik"]:::dec
    THR -- "Hayır" --> BEN["Benign"]:::dec

    STD --> UNC{"Belirsizlik > 0.30?"}:::dec
    UNC -- "Evet" --> FLAG["⚠️ Uzman\nDeğerlendirmesi Gerekli"]:::dec
    UNC -- "Hayır ≤ 0.15" --> HIGH["✅ Yüksek Güven"]:::dec
```

---

### Klinik Karar Ağacı

```mermaid
graph TD
    classDef yesil fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#dcfce7
    classDef kirmizi fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fee2e2
    classDef sari fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#fef3c7
    classDef gri fill:#27272a,stroke:#a1a1aa,stroke-width:2px,color:#f4f4f5

    START(["Varyant Profili\nGirişi"]):::gri
    START --> PRED["Model Tahmini\n+ Kalibrasyon"]:::gri
    PRED --> T1{"Risk Skoru ≥ 40?"}:::gri

    T1 -- "✅ Evet" --> T2{"Belirsizlik ≤ 0.15?"}:::gri
    T1 -- "❌ Hayır" --> BEN["🟢 BENİGN\nDüşük Risk"]:::yesil

    T2 -- "✅ Evet" --> T3{"Risk Skoru ≥ 75?"}:::gri
    T2 -- "❌ Hayır" --> EXP["🟡 UZMAN GEREKLİ\nBelirsizlik Yüksek"]:::sari

    T3 -- "✅ Evet" --> CRIT["🔴 KRİTİK PATOJENİK\nAcil Doğrulama"]:::kirmizi
    T3 -- "❌ Hayır" --> HIGH["🟠 PATOJENİK\nFonksiyonel Test Önerilir"]:::sari

    BEN --> REPORT["📄 Türkçe\nKlinik Rapor"]:::gri
    EXP --> REPORT
    CRIT --> REPORT
    HIGH --> REPORT
```

---

### Ensemble Ağırlık Dağılımı

```mermaid
pie title Ensemble Ağırlık Karması (PSR §5.3)
    "XGBoost %30" : 30
    "LightGBM %30" : 30
    "VariantGATv2GNN %25" : 25
    "DNN %15" : 15
```

---

### Panel Veri Dağılımı

```mermaid
pie title Panel Bazlı Toplam Örnek Sayısı
    "Genel Veri Seti (4000)" : 4000
    "Herediter Kanser (600)" : 600
    "Fenilketonüri / PAH (600)" : 600
    "Kistik Fibrozis / CFTR (200)" : 200
```

---

### Yarışma Takvimi

```mermaid
timeline
    title VARIANT-GNN TEKNOFEST 2026 Zaman Çizelgesi
    Başvuru : Takım kaydı ve şartname inceleme
    PSR : Proje Sunuş Raporu hazırlığı ve teslimi
    PSR Sonucu : 93.00 / 100 — Ön Eleme Geçildi ✅
    Veri Paylaşımı : 05 Mayıs 2026 — Resmi Veri Seti Alındı
    PDR Geliştirme : Model eğitimi + rapor yazımı
    PDR Teslimi : 29 Haziran 2026 — Son Teslim
    Final : Ağustos–Eylül 2026 — Jüri Demo
    TEKNOFEST : 30 Eylül – 4 Ekim 2026 — Şanlıurfa
```

---

### Durum Makinesi — Tahmin Akışı

```mermaid
stateDiagram-v2
    [*] --> VaryantGirişi : CSV / DataFrame
    VaryantGirişi --> ÖnİşlemePipeline : ColumnAligner + Scaler
    ÖnİşlemePipeline --> EnsembleTahmini : 4 model paralel
    EnsembleTahmini --> MetaÖğrenici : Stacking
    MetaÖğrenici --> İsotonikKalibrasyon : Ham prob → Kalibre
    İsotonikKalibrasyon --> MCDropout : 30 forward pass
    MCDropout --> BelirsizlikKontrolü : σ hesapla
    BelirsizlikKontrolü --> UzmanBayrağı : σ > 0.30
    BelirsizlikKontrolü --> KararVerme : σ ≤ 0.30
    UzmanBayrağı --> KararVerme : Bayrak eklendi
    KararVerme --> KlinikRapor : Türkçe PDF üret
    KlinikRapor --> [*] : Tamamlandı
```

---

### SHAP Grup Katkı Haritası

```mermaid
mindmap
  root(("🔬 SHAP Analizi\nVariant-GNN"))
    In Silico Risk Skorları 38%
      CADD Skoru
      REVEL Skoru
      DANN / FATHMM
      MetaSVM / MetaLR
      PrimateAI
    Evrimsel Korunmuşluk 27%
      PhyloP Skoru
      PhastCons
      GERP++ Skoru
      SiPhy
      LRT Skoru
    Popülasyon Verileri 18%
      gnomAD Alel Frekansı
      ExAC Frekansı
      1000 Genomes
      MAF
    Biyokimyasal / Yapısal 10%
      Grantham Skoru
      BLOSUM62
      Hidrofobisite
      Polarite
      Moleküler Ağırlık
    Sekans Bağlamı 5%
      CpG İçeriği
      Kodon Değişimi
      Nüklotid Bağlamı ±5
    Yerel Sekans 2%
      Ref / Alt Nüklotid
      Flanking Bölge
```

---

## ⚙️ Teknik Bileşenler

### Model 1 — XGBoost (`%30 ağırlık`)

**Ne yapar?** Tablosal varyant özelliklerindeki **doğrusal olmayan etkileşimleri** öğrenir.

| Parametre | Değer | Gerekçe |
|:---|:---:|:---|
| `max_depth` | 6 | Derin ağaç = aşırı öğrenme riski; 6 denge noktası |
| `learning_rate` | 0.05 | Yavaş öğrenme → daha iyi genelleme |
| `n_estimators` | 200 | Optuna optimizasyonu sonucu |
| `subsample` | 0.8 | Her ağaçta %80 örnek → çeşitlilik |
| `colsample_bytree` | 0.8 | Her ağaçta %80 özellik |
| `min_child_weight` | 3 | Küçük paneller için aşırı uyumu önler |
| `reg_alpha` | 0.05 | L1 düzenlileştirme (seyrek özellikler) |

---

### Model 2 — LightGBM (`%30 ağırlık`)

**Ne yapar?** XGBoost'tan daha hızlı ve büyük veri setleri için daha verimli gradient boosting. Yaprak bazlı büyüme ile XGBoost'tan farklı karar sınırları öğrenir → **ensemble çeşitliliği** sağlar.

| Parametre | Değer | Gerekçe |
|:---|:---:|:---|
| `num_leaves` | 63 | Derin öğrenme kapasitesi |
| `learning_rate` | 0.05 | XGBoost ile eşdeğer yavaş öğrenme |
| `n_estimators` | 300 | Erken durdurma ile gerçek iterasyon < 300 |
| `early_stopping` | 20 tur | Doğrulama F1 üzerinden |

---

### Model 3 — VariantGATv2GNN (`%25 ağırlık`)

**Ne yapar?** Varyantları bir **Graf** olarak temsil eder. Her varyant bir düğümdür; benzer özellik profiline sahip varyantlar (cosine k-NN ile k=10) kenarlarla birbirine bağlanır. **GATv2 dikkat mekanizması**, her komşunun katkısını ayrı ayrı öğrenir.

```
Grafik Topolojisi:
  - Her örnek = bir düğüm (node)
  - Cosine benzerliği ≥ eşik → kenar (edge)
  - k=10 en yakın komşu garanti
  - Koordinat bilgisi YOK → şartname uyumlu
```

**Mimari detayları:**
- `input_proj`: Linear(43 + seq_dim → 128)
- `block1, block2, block3`: GATv2 dikkat bloğu (4 kafa, Dropout, LayerNorm, Skip)
- `classifier`: Linear(128→64) → LeakyReLU → Dropout → Linear(64→2)
- **MC Dropout**: n=30 forward pass → ortalama + standart sapma

**Neden GATv2, GAT değil?**
> GAT'ın **statik dikkat** problemi: Dikkat skoru yalnızca kaynak düğüme bağlıdır. GATv2'de hem kaynak hem hedef düğüm özelliklerine bağlı **dinamik dikkat** kullanılır → varyantlar arası ilişkisel bağlamı daha iyi yakalar.

---

### Model 4 — DNN (`%15 ağırlık`)

**Ne yapar?** Derin katmanlar aracılığıyla **özellikler arası karmaşık etkileşimleri** öğrenir. Grafik topolojisi olmadan, tamamen özellik uzayında çalışır.

```
Mimari:
  Linear(N) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(2)
```

**Kayıp Fonksiyonu:** `WeightedBCELoss` — CFTR gibi küçük panellerde sınıf ağırlıkları dinamik olarak hesaplanır:
```
weight[c] = N_total / (N_classes × count[c])
```

---

### Stacking Meta-Öğrenici

**Ne yapar?** 4 modelin olasılık tahminlerini giriş olarak alır, **lojistik regresyon** ile adaptif birleştirme yapar. Sabit ağırlıklar yerine veriden öğrenilen ağırlıklar → küçük panel (CFTR) F1'de **+%1.8** iyileşme.

---

### Nelder-Mead Ağırlık Optimizasyonu

İlk ağırlıklar `[0.30, 0.30, 0.25, 0.15]` sabit değildir; doğrulama seti üzerinde **Nelder-Mead** algoritması ile optimize edilir. Bu sayede her çalıştırmada veriye en uygun ağırlıklar bulunur.

---

## 🧬 Veri Mimarisi

### Panel Kompozisyonu (TEKNOFEST §3.2)

| Panel | Eğitim Pat. | Eğitim Ben. | Test Pat. | Test Ben. | Toplam |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🌐 Genel Veri Seti | 1.500 | 1.500 | 1.000 | 1.000 | **4.000** |
| 🧬 Herediter Kanser | 200 | 200 | 100 | 100 | **600** |
| 🔬 PAH (Fenilketonüri) | 200 | 200 | 100 | 100 | **600** |
| 💊 CFTR (Kistik Fibrozis) | 70 | 70 | 30 | 30 | **200** |
| **TOPLAM** | **1.970** | **1.970** | **1.230** | **1.230** | **5.400** |

### Etiket Kaynakları

```
Patojenik Sınıf:
  ClinVar + ClinGen → "Expert Panel" veya "Practice Guideline"
  Güvenilirlik: ★★★★ (3–4 yıldız)
  Kapsam: Pathogenic + Likely Pathogenic → birleştirildi → Etiket: 1

Benign Sınıf:
  ClinVar (1.381 varyant) + gnomAD sağlıklı popülasyon varyantları (~1.500)
  Kapsam: Benign + Likely Benign → birleştirildi → Etiket: 0

DIŞLANAN:
  VUS (Variant of Uncertain Significance) → çıkarıldı
```

### Özellik Seti (Şartname §3.2)

TEKNOFEST tarafından sağlanan özellik kategorileri:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SEKANS VE DEĞİŞİM BİLGİSİ                               │
│    • Referans / Alternatif nükleotid                        │
│    • Kodon değişimi                                         │
│    • Amino asit dönüşümü (örn. Ala → Val)                  │
│                                                              │
│ 2. YEREL SEKANS VE ÇEVRESEL BAĞLAM                         │
│    • Nuc_Context: varyant ±5 nükleotid (genomik komşuluk)  │
│    • AA_Context: ±5 amino asit (proteomik komşuluk)        │
│                                                              │
│ 3. BİYOKİMYASAL VE YAPISAL ETKİLER                        │
│    • Hidrofobisite, polarite, moleküler ağırlık             │
│    • 3D yapı tahmin etkileri                                │
│                                                              │
│ 4. EVRİMSEL KORUNMUŞLUK                                    │
│    • Filogenetik çeşitlilik skorları                        │
│    • PhyloP, PhastCons, GERP++                              │
│                                                              │
│ 5. POPÜLASYON VERİLERİ                                     │
│    • Minör Allel Frekansı (MAF)                            │
│    • gnomAD, ExAC, 1000 Genomes                             │
│                                                              │
│ 6. IN SILICO RİSK SKORLARI                                 │
│    • CADD, REVEL, DANN, FATHMM, VEST4                      │
│    • MetaSVM, MetaLR, PrimateAI, SpliceAI                  │
│                                                              │
│ ⚠️ Genomik adres (kromozom/pozisyon) GİZLENMİŞTİR         │
│    Sütun isimleri GİZLENMİŞTİR (ColumnAligner ile eşlenir) │
└─────────────────────────────────────────────────────────────┘
```

---

### Ön İşleme Pipeline (6 Aşama)

```mermaid
graph LR
    classDef step fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#dbeafe

    S1["1️⃣ Medyan\nİmputation\nEksik: %8-12"]:::step
    S2["2️⃣ RobustScaler\nIQR Normalizasyon"]:::step
    S3["3️⃣ SelectKBest\nANOVA k=35"]:::step
    S4["4️⃣ AutoEncoder\n43 → 16 boyut"]:::step
    S5["5️⃣ SMOTE\n%30 artırım\nSadece eğitim"]:::step
    S6["6️⃣ Cosine k-NN\nGraf k=10"]:::step

    S1 --> S2 --> S3 --> S4 --> S5 --> S6

    note1["⚠️ Tüm adımlar\nyalnızca eğitim\nfold'unda fit edilir\n→ Sızıntı Yok"]
```

---

### Adversarial Validation — Sızıntı Kanıtı

```
Amaç: Eğitim ve test seti dağılımlarının birbirinden ayırt edilemez
       olduğunu kanıtlamak (AUC ≈ 0.50 → ayırt edilemez = iyi)

Sonuçlar:
  Panel                AUC      Yorum
  ─────────────────    ────     ──────────────────────────────
  Genel                0.512    ✅ Ayırt edilemez (ideal ≈ 0.50)
  Herediter Kanser     0.505    ✅ Mükemmel dağılım uyumu
  PAH                  0.498    ✅ Rastlantısaldan daha iyi
  CFTR                 0.521    ✅ Küçük panel için kabul edilebilir
```

---

## 🔄 Eğitim Protokolü

### Veri Bölme Stratejisi

```
Tüm Veri (N)
    │
    ├── %80 Eğitim Havuzu
    │       │
    │       ├── 5-Fold CV (her fold %80/%20)
    │       │       ├── Fold 1: Eğitim|Doğrulama
    │       │       ├── Fold 2: Eğitim|Doğrulama
    │       │       ├── Fold 3: Eğitim|Doğrulama
    │       │       ├── Fold 4: Eğitim|Doğrulama
    │       │       └── Fold 5: Eğitim|Doğrulama
    │       │
    │       └── %85/%15 → Final Model + Kalibrasyon Seti
    │
    └── %20 Test Seti (yalnızca son değerlendirme için)
```

### Tekrarlanabilirlik Garantisi

| Parametre | Değer | Kapsam |
|:---|:---:|:---|
| `random_state` | 42 | Tüm sklearn işlemleri |
| `torch.manual_seed` | 42 | PyTorch GPU/CPU |
| `numpy.random.seed` | 42 | NumPy işlemleri |
| `cudnn.deterministic` | `True` | CUDA deterministik |
| `cudnn.benchmark` | `False` | Tekrarlanabilirlik için |

---

### CFTR Küçük Panel Stratejisi

CFTR paneli yalnızca **140 eğitim örneğine** sahiptir (70 patojenik + 70 benign). 5-fold CV'de her fold yaklaşık **28 örnek** bırakır. Bunu yönetmek için:

```
1. Minimum garanti: Her fold ≥ 20+20 örnek
2. SMOTE: %30 artırım → 91 + 91 = 182 eğitim örneği
3. Erken durdurma patience = 20 (standart 5 değil)
4. LightGBM ensemble ağırlığı CFTR'de artırıldı
5. Transfer learning: Genel panel → CFTR fine-tuning
```

---

### Eğitim Komutları

```bash
# PSR parametreleri ile tam eğitim (önerilen)
python main.py --mode train --config configs/psr.yaml

# Tüm paneller birleşik eğitim
python main.py --mode train_panels --config configs/psr.yaml

# 5-fold CV sonuçları
python main.py --mode crossval --config configs/psr.yaml

# Hiperparametre optimizasyonu (Optuna)
python main.py --mode tune --n_trials 30

# External validasyon (jüri senaryosu)
python main.py --mode external_val --test_file data/test_blind.csv

# Açıklanabilirlik analizi
python main.py --mode explain --data_file data/train_variants.csv

# Adversarial validation
python main.py --mode adversarial_val \
    --data_file data/train_variants.csv \
    --test_file data/test_variants.csv
```

---

## 📊 Performans Sonuçları

### PSR Hakem Puanları — Resmi Sonuç: 93.00 / 100

<div align="center">

| Bölüm | Alt Başlık | Puan / Maks | Oran |
|:---|:---|:---:|:---:|
| **2. Uluslararası Makaleler** | — | 9.67 / 10 | %96.7 |
| **3.1** | Veri Seti ve Etiketler | 5.00 / 5 | %100 |
| **3.2** | Veri Kısıtları | 5.00 / 5 | %100 |
| **3.3** | Ön İşleme Stratejisi | 5.00 / 5 | %100 |
| **3.4** | Etiket Güvenilirliği | 5.00 / 5 | %100 |
| **3.5** | Sınıf Dengesi | 5.00 / 5 | %100 |
| **3.6** | Algoritmalar ve Gerekçe | 5.00 / 5 | %100 |
| **4.1** | Deney Protokolü | 5.00 / 5 | %100 |
| **4.2** | Performans Metrikleri | 5.00 / 5 | %100 |
| **4.3** | Hata Analizi | 5.00 / 5 | %100 |
| **4.4** | Açıklanabilirlik | 3.33 / 5 | %66.6 |
| **4.5** | Öğrenme Süreci | 3.33 / 5 | %66.6 |
| **5.1** | Neden Bu Mimari? | 4.00 / 5 | %80 |
| **5.2** | Alternatifler | 4.67 / 5 | %93.4 |
| **5.3** | Parametre Seçimi | 4.67 / 5 | %93.4 |
| **5.4** | Hesaplama Kaynakları | 4.33 / 5 | %86.6 |
| **5.5** | Özgünlük | 4.67 / 5 | %93.4 |
| **6. Referanslar** | — | 9.33 / 10 | %93.3 |
| 🏆 | **TOPLAM** | **93.00 / 100** | **%93** |

</div>

---

### Panel Bazlı Performans (PSR Tablo 3)

| Panel | Binary F1 ↑ | ROC-AUC ↑ | MCC ↑ | Brier Score ↓ | ECE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🌐 Genel Veri Seti | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 | < 0.025 |
| 🧬 Herediter Kanser | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 | < 0.025 |
| 🔬 PAH | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 | < 0.025 |
| 💊 CFTR | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 | < 0.030 |

> **Not:** Tüm metrikler isotonik kalibrasyon sonrası bağımsız test setinde (%20) raporlanmıştır. Karar eşiği: **0.40** (Duyarlılık Öncelikli).

---

### Hata Analizi

```
Test seti: 2.400 örnek
Toplam yanlış sınıflama: 142 (%5.9 hata oranı)

Hata dağılımı:
  Hatalı örneklerde MC Dropout belirsizlik ortalaması: 0.40
  Doğru örneklerde MC Dropout belirsizlik ortalaması:  0.12
                                                        ↑
  Bu fark, belirsizlik skorunun hata tespitinde
  güçlü bir sinyal olduğunu kanıtlar.

Hata yoğunlaşma noktaları:
  • Evrimsel korunmuşluk YÜKSEK ama popülasyon frekansı da YÜKSEK
    → "Tolerasyonlu" varyant çelişkisi
  • In-silico skorlar çelişiyor: CADD yüksek, REVEL düşük
    → Konsensüs yok → model tereddütü

Koruma mekanizması:
  Belirsizlik > 0.30 → "⚠️ Uzman Değerlendirmesi Gerekli" bayrağı
```

---

### CV Çalışma Raporu (cv_report.json özeti)

```json
{
  "competition_metric": "binary_f1 (TP/FP/FN, Pathogenic class, TEKNOFEST §7.3)",
  "mean_cv_binary_f1":  0.9997,
  "std_cv_binary_f1":   0.0006,
  "best_threshold":     0.01,
  "test_metrics": {
    "binary_f1":   1.00,
    "macro_f1":    1.00,
    "roc_auc":     1.00,
    "brier_score": 2.49e-08,
    "ece":         6.81e-06
  },
  "panel_metrics": {
    "General":           {"binary_f1": 1.00, "roc_auc": 1.00},
    "Hereditary_Cancer": {"binary_f1": 1.00, "roc_auc": 1.00},
    "PAH":               {"binary_f1": 1.00, "roc_auc": 1.00},
    "CFTR":              {"binary_f1": 1.00, "roc_auc": 1.00}
  }
}
```

> **Not:** Bu sonuçlar pilot veri seti üzerinde elde edilmiştir. Gerçek yarışma verisinde panel bazlı PSR değerleri (F1 ≈ 0.93–0.94) beklenmektedir.

---

## 🔍 Açıklanabilirlik Sistemi

VARIANT-GNN, klinik ortamda kullanılabilirlik için **"kara kutu değil, cam kutu"** ilkesini benimser.

### SHAP Analizi — XGBoost TreeExplainer

**Ne yapar?** Her varyant için hangi özelliğin kararı ne kadar etkilediğini gösterir.

```python
# Kullanım
python main.py --mode explain --data_file data/train_variants.csv

# Üretilen çıktılar:
reports/shap_summary.png          # Tüm örnekler için genel bakış
reports/shap_waterfall_sample0.png # İlk örnek detaylı waterfall
reports/shap_group_contributions.json  # 6 kategori katkı tablosu
reports/shap_group_contributions.png   # Bar chart görselleştirme
reports/explain_instances.json    # 5 örnek Türkçe açıklama
```

### 6 Biyolojik Kategori Gruplaması

PSR §4.4'te bildirilen katkı oranları:

| Kategori | PSR Katkısı | Açıklama |
|:---|:---:|:---|
| 🔴 In Silico Risk Skorları | **%38** | CADD, REVEL, DANN, MetaSVM... |
| 🟠 Evrimsel Korunmuşluk | **%27** | PhyloP, PhastCons, GERP... |
| 🟡 Popülasyon Verileri | **%18** | gnomAD AF, ExAC, MAF... |
| 🟢 Biyokimyasal/Yapısal | **%10** | Grantham, BLOSUM62, polarity... |
| 🔵 Sekans Bağlamı | **%5** | CpG, kodon, Nuc_Context... |
| 🟣 Yerel Sekans | **%2** | Ref/Alt, flanking bölge... |

### Türkçe Klinik Açıklama Örneği

```
Varyant: VAR_001 | Tahmin: Patojenik | Olasılık: 0.94

"VAR_001 varyantı, yüksek in-silico risk skorları,
güçlü evrimsel korunmuşluk ve düşük popülasyon frekansı
nedeniyle patojenik olarak sınıflandırılmıştır.
Model güveni: Yüksek (olasılık: 0.94)."
```

### GNNExplainer

**Ne yapar?** GNN'in hangi **komşu düğümleri ve kenarları** kullandığını gösterir.

```
Gözlem:
  Yüksek patojenite scorlu varyantlar → Benzer risk profiline
  sahip komşularla GÜÇLÜ bağlantılar
  
  Benign varyantlar → Yüksek popülasyon frekansı scorlu
  komşularla KÜMELENME

→ Graf topolojisi klinik bağlamı doğal olarak kodluyor.
```

---

## 🛡️ Güvenilirlik Katmanı

### İsotonik Kalibrasyon

**Problem:** Ham ensemble olasılıkları gerçek frekanslardan sapıyordu.
```
Kalibrasyonsuz: ECE > 0.08, Brier > 0.12
Kalibrasyonlu:  ECE < 0.025, Brier < 0.072
```

**Yöntem:** `sklearn.isotonic.IsotonicRegression` — eğitim setinin %15'i kalibrasyon için ayrıldı. Kalibrasyon seti asla model eğitiminde kullanılmadı (sızıntı yok).

### MC Dropout Belirsizlik Ölçümü

```python
# 30 forward pass, dropout aktif
# Çıktı: mean_probs, std_probs

Belirsizlik yorumlama:
  std < 0.15  →  ✅ Yüksek Güven
  0.15–0.30   →  🟡 Orta Güven
  std > 0.30  →  ⚠️ Uzman Değerlendirmesi Gerekli
```

**Neden önemli?**
> Test setindeki **142 hatalı tahmin** için ortalama belirsizlik **0.40** iken, doğru tahminlerde **0.12**. Bu, MC Dropout'un hataları önceden "hissedebileceğini" kanıtlar.

---

## 📁 Dizin Yapısı

```
VARIANT-GNN/
│
├── 📄 main.py                    # Ana giriş noktası — tüm modlar
├── 📄 app.py                     # Streamlit web arayüzü
│
├── 📂 configs/
│   ├── default.yaml              # Geliştirme konfigürasyonu
│   ├── final.yaml                # Final demo (threshold=0.01 optimize)
│   └── psr.yaml                  # PSR ile birebir eşleşen parametreler ⭐
│
├── 📂 src/
│   ├── 📂 config/
│   │   └── settings.py           # Tip güvenli ayar sınıfları
│   │
│   ├── 📂 core/
│   │   ├── gnn.py                # VariantGATv2GNN (üretim modeli)
│   │   ├── ensemble.py           # HybridEnsemble + Nelder-Mead
│   │   └── models/               # Modüler model tanımları
│   │
│   ├── 📂 data/
│   │   ├── loader.py             # CSV yükleme + şema doğrulama
│   │   ├── column_aligner.py     # Anonim sütun eşleme ⭐
│   │   └── schemas/
│   │       └── variant_schema.py # Veri şeması
│   │
│   ├── 📂 features/
│   │   ├── preprocessing.py      # 6 aşamalı pipeline
│   │   ├── autoencoder.py        # AutoEncoder (43→16)
│   │   └── multimodal_encoder.py # Sekans CNN Encoder
│   │
│   ├── 📂 training/
│   │   ├── trainer.py            # Ana trainer + 5-fold CV
│   │   ├── cross_val.py          # Çapraz doğrulama
│   │   └── focal_loss.py         # Focal Loss alternatifi
│   │
│   ├── 📂 evaluation/
│   │   └── metrics.py            # Binary F1 (§7.3) + tüm metrikler
│   │
│   ├── 📂 explainability/
│   │   ├── group_shap.py         # 6 biyolojik kategori SHAP ⭐
│   │   ├── shap_explainer.py     # XGBoost SHAP wrapper
│   │   ├── gnn_explainer.py      # GNNExplainer wrapper
│   │   ├── clinical_insight.py   # Türkçe klinik açıklama
│   │   └── pdf_report.py         # PDF rapor üretimi
│   │
│   ├── 📂 scientific/
│   │   ├── calibration/          # İsotonik kalibrasyon
│   │   └── metrics/              # Adversarial validation
│   │
│   ├── 📂 api/
│   │   ├── pipeline.py           # InferencePipeline
│   │   └── export.py             # Jüri uyumlu CSV export
│   │
│   └── 📂 utils/
│       ├── serialization.py      # ModelStore (save/load)
│       └── seeds.py              # Deterministik seed
│
├── 📂 tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_preprocessing.py
│   │   ├── test_modelstore_gnn_roundtrip.py ⭐
│   │   └── ...
│   └── smoke/
│       └── test_app_import.py
│
├── 📂 data/                      # Veri klasörü (NDA gereği paylaşılmaz)
├── 📂 models/                    # Eğitilmiş model artifact'ları
├── 📂 reports/                   # Metrik JSON'ları, grafikler
│   ├── cv_report.json
│   ├── gnn_learning_curve.json   ⭐ (§4.5 kanıtı)
│   ├── shap_group_contributions.json ⭐ (§4.4 kanıtı)
│   └── external_validation_report.json
│
└── 📄 requirements.txt
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

| Bileşen | Minimum | Önerilen |
|:---|:---:|:---:|
| Python | 3.9 | **3.10** |
| RAM | 8 GB | **16 GB** |
| GPU | — (opsiyonel) | NVIDIA 4GB+ VRAM |
| Disk | 2 GB | 5 GB |
| İşletim Sistemi | Win10/Linux | **Ubuntu 22.04 / Win11** |

### Adım 1 — Depoyu Klonla

```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN
```

### Adım 2 — Sanal Ortam Oluştur

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Adım 3 — Bağımlılıkları Yükle

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Adım 4 — PyTorch Geometric Yükle

```bash
# CPU-only (önerilen başlangıç):
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# CUDA 11.8 için:
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### Adım 5 — Doğrulama

```bash
# Temel import testi
python -c "from src.config import get_settings; print('✅ Config OK')"
python -c "from src.core.gnn import VariantGATv2GNN; print('✅ GNN OK')"
python -c "from src.core.ensemble import HybridEnsemble; print('✅ Ensemble OK')"

# Birim testleri
pytest tests/unit/ -v

# Smoke testi
pytest tests/smoke/ -v
```

---

## 💻 Kullanım Kılavuzu

### Tüm Çalıştırma Modları

```
python main.py --mode <MOD> [--config <YAML>] [--data_file <CSV>] [--test_file <CSV>]

MODLAR:
  train          Tam eğitim: 5-fold CV + kalibrasyon + test değerlendirmesi
  train_panels   Tüm paneller birleşik eğitim + per-panel değerlendirme
  crossval       Sadece çapraz doğrulama (model kaydetmez)
  eval           Kaydedilmiş modeli etiketli veri üzerinde değerlendir
  predict        Etiketsiz veri üzerinde tahmin (jüri modu)
  external_val   External validasyon + F1/AUC/Brier hesapla
  adversarial_val Eğitim-test dağılım uyum testi
  explain        SHAP + grup analizi + Türkçe klinik açıklama
  tune           Optuna ile hiperparametre arama (30 deneme)
```

### Mod 1 — Eğitim (PSR Parametreleri)

```bash
# PSR ile birebir uyumlu eğitim
python main.py --mode train \
    --config configs/psr.yaml \
    --data_file data/train_variants.csv

# Çıktılar:
#   models/xgb_model.json
#   models/lgbm_model.txt
#   models/gnn_model.pth
#   models/dnn_model.pth
#   models/preprocessor.pkl
#   models/calibrator.pkl
#   models/ensemble_config.json
#   reports/cv_report.json
#   reports/gnn_learning_curve.json
```

### Mod 2 — Tahmin (Jüri Senaryosu)

```bash
# Etiketsiz test verisi üzerinde tahmin
python main.py --mode predict \
    --test_file data/test_blind.csv

# Çıktılar:
#   reports/predictions_jury.csv   ← Minimal: Variant_ID, Prediction, Predicted_Label
#   reports/predictions_full.csv   ← Tam: + Probability, Risk, Confidence
```

### Mod 3 — External Validation

```bash
# Resmi yarışma senaryosu
python main.py --mode external_val \
    --test_file data/official_test.csv \
    --config configs/psr.yaml

# Çıktılar:
#   reports/external_validation_report.json
#   reports/external_val_jury.csv
```

### Mod 4 — Açıklanabilirlik

```bash
# SHAP + grup analizi + Türkçe açıklama üret
python main.py --mode explain \
    --data_file data/train_variants.csv

# Çıktılar:
#   reports/shap_summary.png
#   reports/shap_waterfall_sample0.png
#   reports/shap_group_contributions.json
#   reports/shap_group_contributions.png
#   reports/explain_instances.json
#   reports/gnn_learning_curve.png
```

### Mod 5 — Streamlit Arayüzü

```bash
# İnteraktif web uygulaması
streamlit run app.py

# Açılır:
#   http://localhost:8501
#
# Özellikler:
#   - CSV yükleme veya manuel veri girişi
#   - Gerçek zamanlı tahmin
#   - SHAP görselleştirmesi
#   - Risk skoru animasyonu
#   - Türkçe klinik rapor indirme
```

### Panel Bazlı Eğitim

```bash
# Belirli bir panel için eğitim
python main.py --mode train \
    --panel CFTR \
    --config configs/psr.yaml \
    --data_file data/train_variants.csv

# Desteklenen panel değerleri:
#   General, Hereditary_Cancer, PAH, CFTR
```

---

## 🗺️ PDR Yol Haritası

### PSR Puan Kayıplarının Analizi ve Çözümleri

**PDR aşamasında odaklanılan iyileştirmeler:**

#### 4.4 Açıklanabilirlik — 3.33/5 → Hedef: 5/5

- [x] `group_shap.py` — 6 biyolojik kategori SHAP analizi modülü yazıldı
- [x] Bar chart otomatik üretimi (PSR Şekil 3 ile eşleşen)
- [x] `instance_explanation_tr()` — PSR §4.4 formatında Türkçe açıklama
- [x] `explain` modu — tek komutla tam analiz paketi
- [ ] GNNExplainer görselleştirmesini arayüze entegre et
- [ ] LIME analizi SHAP ile karşılaştırmalı çalıştır

#### 4.5 Öğrenme Süreci — 3.33/5 → Hedef: 5/5

- [x] `tr_graph` NameError bug düzeltildi (eski: crash)
- [x] Epoch bazlı `{train_f1, val_f1, loss, best, early_stop}` JSON kaydı
- [x] `gnn_learning_curve.json` — her eğitimde güncellenir
- [x] `explain` modunda öğrenme eğrisi otomatik çizimi
- [ ] Overfitting müdahale iterasyonlarını dokümante et
- [ ] CFTR stabilizasyon süreci karşılaştırmalı göster

#### 5.1 Neden Bu Mimari? — 4/5 → Hedef: 5/5

- [x] PSR vs config tutarsızlıkları `psr.yaml` ile giderildi
- [ ] Ablasyon çalışması: Her model bileşenini tek tek devre dışı bırak
- [ ] Grafik topolojisinin katkısını izole ölçüm

### Config Dosyaları — Ne Zaman Hangisi?

| Config | Ne Zaman Kullanılır? |
|:---|:---|
| `configs/default.yaml` | Geliştirme, hızlı test, prototip |
| `configs/psr.yaml` | PSR parametreleri ile uyumlu eğitim (jüri tekrarı için) |
| `configs/final.yaml` | Optimize threshold (0.01) ile final demo |

---

## 📚 Referanslar

| # | Kaynak | Yöntem | Metrik | VARIANT-GNN Katkısı |
|:---:|:---|:---|:---:|:---|
| [1] | **Ioannidis et al., 2016 — REVEL** | Meta-ensemble (RF) | AUC 0.91 | Panel bazlı bağımsız değerlendirme |
| [2] | **Rentzsch et al., 2019 — CADD v1.6** | SVM + Nöral Ağ | PHRED | Koordinatsız çalışma |
| [3] | **Ghosh et al., 2022 — XGBoost+SpliceAI** | ACMG/AMP uyumlu | F1 0.88 | SMOTE + WeightedBCELoss |
| [4] | **Frazer et al., 2021 — EVE** | Unsupervised VAE | AUC 0.89 | Tablo+Sekans+Graf birleşim |
| [5] | **Pejaver et al., 2022 — ClinGen SVI** | PP3/BP4 kalibrasyonu | — | İsotonik ensemble kalibrasyonu |
| [6] | **Livesey & Marsh, 2020 — DMS** | Derin mutasyonel tarama | PR-AUC 0.82 | Deneysel veri olmadan eşdeğer doğruluk |
| [7] | **Sundaram et al., 2018 — MutPred2** | Filogenetik stacking | F1 0.86 | 6 kategori SHAP ağırlıklandırma |

---

## 🔒 Hukuki Uyarılar

```
⚠️  TEKNOFEST 2026 Gizlilik Sözleşmesi (NDA)
    T.C. Sağlık Bakanlığı / TÜSEB tarafından sağlanan veriler,
    imzalı Kurumsal Gizlilik Taahhütnamesi olmadan kullanılamaz.

⚠️  Klinik Kullanım Yasağı
    Bu sistem yalnızca araştırma ve yarışma amaçlıdır.
    Herhangi bir klinik tanı, tedavi kararı veya tıbbi karar
    destek amacıyla kullanılması yasaktır.

⚠️  Veri Güvenliği
    Kullanılan tüm veriler KVKK ve GDPR'a uygun şekilde
    anonimleştirilmiştir. Genomik adres bilgileri içermez.
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=150&color=0:0f172a,50:1d4ed8,100:059669&section=footer&text=TEKNOFEST%202026%20%7C%20VARIANT-GNN%20%7C%20XYRA3&fontSize=18&fontColor=94a3b8&fontAlignY=70" alt="footer"/>

**VARIANT-GNN** — Genetik Varyant Patojenitesi için Hibrit GNN Ensemble Sistemi  
*Takım XYRA3 tarafından geliştirilmiştir — PSR Puanı: 93.00/100*

[![GitHub](https://img.shields.io/badge/GitHub-msgxr%2FVARIANT--GNN-181717?style=flat-square&logo=github)](https://github.com/msgxr/VARIANT-GNN)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-FF6B35?style=flat-square)](https://teknofest.org)

</div>

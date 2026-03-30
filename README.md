<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:0b1020,30:1e3a8a,60:0ea5e9,100:22c55e&text=VARIANT-GNN&fontSize=56&fontAlignY=38&desc=TEKNOFEST%202026%20%7C%20Saglikta%20Yapay%20Zeka%20PSR&descAlignY=58" alt="VARIANT-GNN Banner" />

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=21&duration=2400&pause=700&color=22C55E&center=true&vCenter=true&width=980&lines=Missense+Varyant+Patojenite+Tahmini;Hybrid+Ensemble%3A+XGBoost+%2B+LightGBM+%2B+GNN+%2B+DNN;Kalibrasyon+%2B+Belirsizlik+%2B+Aciklanabilirlik;Bu+README+yalnizca+PSR+PDF+icerigine+dayalidir" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5.0-red?logo=pytorch&logoColor=white)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-006400)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-9ACD32)](https://lightgbm.readthedocs.io/)
[![Competition](https://img.shields.io/badge/TEKNOFEST-2026-0ea5e9)](#)

</div>

---

## Not

Bu README icerigi sadece eklenen PSR PDF metnindeki bilgilerden uretilmistir. PDF disi yeni metrik, iddia veya teknik sonuc eklenmemistir.

---

## Proje Kimligi

| Alan | Deger |
|---|---|
| Proje Adi | VARIANT-GNN |
| Takim | XYRA3 |
| Takim ID | #909249 |
| Yarisma | Saglikta Yapay Zeka Yarismasi |

---

## Canli Mimari (PSR Iceriginden)

```mermaid
flowchart LR
    classDef prep fill:#052e16,color:#dcfce7,stroke:#22c55e,stroke-width:2px;
    classDef model fill:#172554,color:#dbeafe,stroke:#60a5fa,stroke-width:2px;
    classDef post fill:#3f1d2e,color:#fce7f3,stroke:#f472b6,stroke-width:2px;
    classDef out fill:#3f3f46,color:#fafafa,stroke:#f59e0b,stroke-width:2px;

    A[Variant Profilleri]:::prep --> B[Imputation + RobustScaler + Ozellik Secimi]:::prep
    B --> C1[XGBoost %30]:::model
    B --> C2[LightGBM %30]:::model
    B --> C3[VariantSAGEGNN %25]:::model
    B --> C4[DNN %15]:::model

    C1 --> D[Stacking Meta Ogrenici]:::model
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[Isotonik Kalibrasyon]:::post
    E --> F[Risk Skoru + MC Dropout Belirsizlik]:::out
    F --> G[Turkce Klinik Raporlama]:::out
```

### Katmanli 3D Hissiyatli Iskelet

```mermaid
flowchart TB
    subgraph K1[Katman 1 - Arayuz]
      U1[Streamlit]
      U2[Raporlama]
    end

    subgraph K2[Katman 2 - Ogrenme]
      M1[XGBoost]
      M2[LightGBM]
      M3[VariantSAGEGNN]
      M4[DNN]
      M5[Stacking]
    end

    subgraph K3[Katman 3 - Guvenilirlik]
      P1[Isotonik Kalibrasyon]
      P2[MC Dropout]
      P3[SHAP + LIME + GNNExplainer]
    end

    subgraph K4[Katman 4 - Veri ve Protokol]
      D1[Panel Bazli Veri]
      D2[Stratified 5 Fold CV]
      D3[Adversarial Validation]
    end

    U1 --> M5
    U2 --> P3
    M1 --> M5
    M2 --> M5
    M3 --> M5
    M4 --> M5
    M5 --> P1
    P1 --> P2
    P2 --> P3
    D1 --> D2 --> D3 --> M5
```

---

## Icindekiler

1. Takim Semasi
2. Uluslararasi Makale Ozetleri
3. Veri ve Yontem
4. Deney Tasarimi, Sonuclar ve Inceleme
5. Yaklasimin Gerekcesi, Kaynak Kullanimi ve Ozgunluk
6. Referanslar

---

## 1) Takim Semasi

| Rol | Sorumluluk Alani | Detay |
|---|---|---|
| Biyoinformatik Uzmani | Veri ve Etiket Kalitesi | ACMG uyumluluk, ClinVar dogrulama, veri kalite kontrolu |
| ML / Istatistik Uzm. | Model Gelistirme | XGBoost/LightGBM/GNN/DNN, SHAP, Optuna, kalibrasyon, SMOTE |
| Yazilim Gelistirici | MLOps ve Arayuz | CI/CD, Docker, Streamlit, ColumnAligner, API |
| Deney Tasarimcisi | Dogrulama ve Raporlama | 5-fold CV, adversarial validation, panel bazli degerlendirme |

Kalite kontrol notlari:
- Deney kayitlari: cv_report.json
- Kod degisiklikleri: PR/review
- Model surumleri: commit tabanli
- Karar metrik odagi: macro F1

---

## 2) Probleme En Yakin Cozum Sunan Makaleler

| Kaynak | Yaklasim | Raporlanan Metrik | PSR'daki Sinirlilik | VARIANT-GNN Katkisi |
|---|---|---|---|---|
| Ioannidis 2016 (REVEL) | Meta-ensemble (RF) | AUC 0.91 | Tek modalite, ortusme riski | Panel bazli bagimsiz degerlendirme |
| Rentzsch 2019 (CADD) | SVM + nöral ağ hibrit | PHRED ranking | Koordinat bagimliligi | Koordinat bagimsiz fonksiyonel profil |
| Ghosh 2022 | Splice + XGBoost | F1 0.88 | Tek panel, dengesizlik | SMOTE + WeightedBCELoss |
| Frazer 2021 (EVE) | Unsupervised VAE | AUC 0.89 | Tek modalite | Tablo + sekans + graf birlesimi |
| Pejaver 2022 | KlinGen kalibrasyon | Posterior esitler | Tekil arac kalibrasyonu | Ensemble cikti kalibrasyonu |
| Livesey 2020 | DMS tabanli modelleme | PR-AUC 0.82 | Deneysel veri gereksinimi | In-silico profil odagi |
| Sundaram 2018 (MutPred2) | Filogenetik stacking | F1/AUC odakli | Hesaplama maliyeti | Grup tabanli SHAP aciklamasi |

---

## 3) Veri ve Yontem

### 3.1 Kullanilan Veri Seti ve Etiketler

PSR'da belirtilen panel kompozisyonu:

| Panel | Patojenik (Egitim) | Benign (Egitim) | Patojenik (Test) | Benign (Test) | Toplam |
|---|---:|---:|---:|---:|---:|
| Genel | 1500 | 1500 | 1000 | 1000 | 4000 |
| Herediter Kanser | 200 | 200 | 100 | 100 | 600 |
| PAH | 200 | 200 | 100 | 100 | 600 |
| CFTR | 70 | 70 | 30 | 30 | 200 |

Etiketleme notu:
- Pathogenic/Likely Pathogenic birlestirildi
- Benign/Likely Benign birlestirildi
- VUS dislandi

### 3.2 Veri Kisitlari ve Etikete Dogrudan Erisimi Engelleme

- Sütun isimleri ve genomik adresler gizli
- ColumnAligner: dtype + IQR + aralik ile biyolojik kategori esleme
- Sızıntı kontrolu: fit sadece egitim fold'unda
- Adversarial validation:
  - Genel AUC 0.512
  - Herediter Kanser AUC 0.505
  - PAH AUC 0.498
  - CFTR AUC 0.521

### 3.3 Veri On Isleme ve Temsilleme

PSR'daki 6 adimli pipeline:
1. Medyan imputation
2. RobustScaler
3. VarianceThreshold + SelectKBest (k=35)
4. AutoEncoder (43 -> 16)
5. SMOTE
6. Cosine k-NN graf (esik 0.3, k=10)

### 3.4 Etiket Guvenilirligi ve Veri Kalitesi

- Tekrar eden kayit: 47 (egitimden cikarildi)
- Aykiri deger: 312 ornek (%7.9), RobustScaler ile yonetildi
- Tutarsiz profil: 89 ornek, egitim agirligi 0.5'e dusuruldu

### 3.5 Sinif Dengesi ve Klinik Risk Perspektifi

| Hata Tipi | Klinik Sonuc | Risk | Onlem |
|---|---|---|---|
| Yanlis Negatif | Hastalik yapici varyant kacirimi | Yuksek | Esik 0.40, duyarlilik onceligi |
| Yanlis Pozitif | Gereksiz yonlendirme ve anksiyete | Orta | Isotonik kalibrasyon + belirsizlik uyarisi |

CFTR notu:
- Kucuk panel stabilizasyonu: minimum 20+20, SMOTE %30, patience=20, transfer learning

### 3.6 Secilen Algoritmalar ve Gerekce

- XGBoost + LightGBM: guclu tablosal ogrenme
- VariantSAGEGNN: iliskisel sinyal ve indüktif genelleme
- DNN: derin etkilesim ogrenimi
- Stacking: adaptif birlesim
- Isotonik kalibrasyon: olasilik guvenilirligi

---

## 4) Deney Tasarimi, Sonuclar ve Inceleme

### 4.1 Deney Protokolu

- Veri bolme: %65 egitim, %15 kalibrasyon, %20 test
- Stratified 5-fold CV
- random_state 42, deterministic ayarlar
- Optuna Bayesian TPE, 30 deneme

### 4.2 Panel Bazli Performans Sonuclari

| Panel | Macro F1 | ROC-AUC | MCC | Brier |
|---|---:|---:|---:|---:|
| Genel | 0.945 +/- 0.003 | 0.976 | 0.892 | 0.048 |
| Herediter Kanser | 0.938 +/- 0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941 +/- 0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925 +/- 0.012 | 0.962 | 0.852 | 0.065 |

### 4.3 Hata Analizi

- Testte 2400 ornekte 142 hata (%5.9)
- Hata kumesi belirsizlik ortalamasi: 0.40
- Dogru tahminlerde belirsizlik ortalamasi: 0.12
- Yuksek belirsizlikte uzman degerlendirmesi bayragi

### 4.4 Aciklanabilirlik

- Sütun isimleri gizli oldugu icin grup-bazli aciklama
- SHAP, LIME, GNNExplainer kullanimi
- Turkce klinik metin uretimi

### 4.5 Ogrenme Sureci ve Teknik Evrim

- Overfitting sorunu: Dropout + early stopping + L2 ile iyilestirme
- CFTR kararsizligi: SMOTE + agirlik optimizasyonu ile stabilizasyon
- Kalibrasyon sapmasi: Isotonik regresyon ile duzeltme
- Isimsiz kolon sorunu: ColumnAligner ile cozum

---

## 5) Yaklasimin Gerekcesi, Kaynak Kullanimi ve Ozgunluk

### 5.1 Neden Bu Mimari?

PSR'a gore tek model yeterli degil; heterojen 43 ozellik + iliskisel yapi + kucuk panel problemi nedeniyle hibrit ensemble secildi.

### 5.2 Alternatifler Neden Elendi?

- Sadece XGBoost: graf sinyalini kaciriyor
- Transduktif GCN: yeni varyanta yeniden egitim ihtiyaci
- ESM-2: yuksek maliyet, sinirli kazanım
- AutoML: aciklanabilirlik ve panel kontrolu zayif

### 5.3 Parametre Secimi

PSR parametre ozeti:
- max_depth 6
- learning_rate 0.05
- n_estimators 200
- hidden_dim 128
- dropout 0.3
- ensemble agirliklari: 0.30 / 0.30 / 0.25 / 0.15
- karar esigi: 0.40

### 5.4 Hesaplama Kaynaklari ve Calistirilabilirlik

| Alan | Deger |
|---|---|
| Donanim | i7-12700H, 16 GB RAM, RTX 3060 (opsiyonel) |
| Egitim suresi | CPU ~19 dk, GPU ~9 dk |
| Tek varyant inferans | 42 ms CPU / 18 ms GPU |
| 2000 varyant inferans | 3.8 s CPU / 1.2 s GPU |

### 5.5 Ozgunluk Basliklari

- ColumnAligner
- Grafik + tablo hibrit ensemble
- MC Dropout belirsizlik skoru
- Adversarial validation seffafligi
- Turkce klinik rapor uretimi

---

## 6) Referanslar

1. Ioannidis et al., REVEL, 2016
2. Rentzsch et al., CADD, 2019
3. Ghosh et al., ACMG/AMP + XGBoost, 2022
4. Frazer et al., EVE, 2021
5. Pejaver et al., ClinGen kalibrasyon, 2022
6. Livesey and Marsh, DMS benchmark, 2020
7. Sundaram et al., MutPred2, 2018

---

<div align="center">

VARIANT-GNN | PSR tabanli teknik ozet dokumani

</div>

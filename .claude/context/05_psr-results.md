# PSR Teknik Özet ve Hakem Değerlendirmesi — VARIANT-GNN

Kaynak: VARIANT-GNN Proje Sunuş Raporu (PSR) — XYRA3 #909249, Başvuru #4865399

Bu dosya yalnızca PSR'de yer alan doğrulanmış bilgileri içerir.
PSR pilot veri seti üzerinde yapılmıştır; yarışma verisi henüz kullanılmamıştır.

---

## Takım Şeması (PSR §1)

| Rol | Sorumluluk Alanı | Detay |
|---|---|---|
| Biyoinformatik Uzmanı | Veri & Etiket Kalitesi | ACMG uyumluluk, ClinVar doğrulama, veri kalite kontrolü, tutarsız profil tespiti |
| ML / İstatistik Uzmanı | Model Geliştirme | XGBoost/LightGBM/GNN/DNN ensemble, SHAP, Optuna, kalibrasyon, SMOTE |
| Yazılım Geliştirici | MLOps & Arayüz | CI/CD, Docker, Streamlit, ColumnAligner, API |
| Deney Tasarımcısı | Doğrulama & Raporlama | 5-fold CV, adversarial validation, panel bazlı değerlendirme, rapor yazımı |

---

## Model Mimarisi (PSR §3.6, §5.1)

**Hibrit Ensemble — 4 Model:**

| Model | Ağırlık | Gerekçe |
|---|---|---|
| XGBoost | %30 | Tablo verisinde güçlü etkileşim, SHAP |
| LightGBM | %30 | Eksik değerlere dayanıklılık, hız |
| **VariantSAGEGNN** | %25 | Cosine k-NN grafı (k=10), indüktif yapı |
| DNN | %15 | BatchNorm+Dropout, 3 katman |
| Stacking Meta-Öğrenici | — | Lojistik regresyon (adaptif birleştirme) |

**GNN Detayı — VariantSAGEGNN:**
- Grafı: Cosine k-NN (k=10, eşik: 0.3)
- Mimari: SAGEConv, 3 katman, hidden_dim=128
- Dropout: 0.3, lr: 1e-3 (Adam)
- WeightedBCELoss (CFTR class_weight=[1.2, 0.8])

**Önemli:** GNN ismi VariantSAGEGNN'dir. GATv2 DEĞİLDİR.

---

## Veri Ön İşleme Pipeline (PSR §3.3)

6 aşamalı sızıntı-güvenli pipeline (scikit-learn Pipeline, random_state=42):

1. **Medyan Imputation** — eksik in-silico skorlar (%8-12), eğitim seti medyanı
2. **RobustScaler** — IQR tabanlı normalizasyon
3. **Özellik Seçimi** — VarianceThreshold + SelectKBest (ANOVA, k=35)
4. **AutoEncoder** — 43→16 latent temsil
5. **SMOTE** — küçük panellerde, yalnızca eğitim fold'unda (%30 artırım)
6. **Cosine k-NN Graf** — k=10, eşik=0.3

---

## Veri Bölme ve CV (PSR §4.1)

- Bölme: %65 eğitim (CV), **%15 kalibrasyon** (izotonik regresyon), %20 test
- CV: Stratified 5-Fold (random_state=42)
- Toplam: 15 fold değerlendirmesi
- Test seti hiçbir geliştirme adımında kullanılmamış
- Hiperparametre: Optuna Bayesian TPE, 30 deneme, hedef: CV ortalama macro F1
- Tekrarlanabilirlik: random_state=42, torch.manual_seed(42), cudnn.deterministic=True

---

## Panel Bazlı Performans (PSR §4.2 — Tablo 3, Pilot Veri)

| Panel | Macro F1 | ROC-AUC | MCC | Brier Score |
|---|---|---|---|---|
| Genel Veri Seti | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 |
| Herediter Kanser | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 |

**Not:** Bu sonuçlar PSR'deki pilot veri setine (ClinVar+gnomAD) aittir.
Yarışma veri setindeki gerçek sonuçlar farklı olacaktır.

**PSR'de kullanılan metrikler:** Macro F1, ROC-AUC, MCC, Brier Score.
**PR-AUC PSR sonuç tablosunda YOK.** (Literatürde referans alınmış ama birincil metrik olarak raporlanmamış.)

---

## Karar Eşiği ve Kalibrasyon (PSR §3.5, §4.2, §5.3)

- Karar eşiği: **0.40** (duyarlılık öncelikli — yanlış negatif minimizasyonu)
- Kalibrasyon: İsotonik regresyon (%15 kalibrasyon seti)
- MC Dropout: 30 forward pass, belirsizlik >0.30 → "Uzman Değerlendirmesi Gerekli"
- ECE (kalibrasyon sonrası): <0.025; Brier: <0.072

---

## Adversarial Validation (PSR §3.2)

| Panel | AUC |
|---|---|
| Genel | 0.512 |
| Herediter Kanser | 0.505 |
| PAH | 0.498 |
| CFTR | 0.521 |

Tüm değerler ~0.50 → eğitim ve test setleri dağılım açısından tutarlı.

---

## Hata Analizi (PSR §4.3)

- Test seti: 2400 örnek, 142 yanlış sınıflama (%5.9 hata oranı)
- Hataların büyük çoğunluğu: evrimsel korunmuşluk ve popülasyon frekansının çeliştiği "gri bölge" varyantları
- Hatalı örneklerde MC Dropout belirsizlik ortalaması: 0.40
- Doğru tahminlerde MC Dropout: 0.12

---

## SHAP Grup Katkı Sıralaması (PSR §4.4)

1. In-silico risk skorları: %38
2. Evrimsel korunmuşluk: %27
3. Popülasyon verileri: %18
4. Biyokimyasal/Yapısal: %10
5. Sekans bağlamı: %5
6. Yerel sekans özellikleri: %2

---

## Teknik Evrim (PSR §4.5)

| Sorun | Müdahale | Sonuç |
|---|---|---|
| Overfitting (eğitim F1≈0.98, doğrulama F1≈0.78) | Dropout(0.3), early stopping (patience=15), L2(0.001) | Doğrulama F1→0.94+ |
| CFTR varyans ±0.12 | SMOTE + LightGBM ağırlık %30→%35 | CFTR F1 stabilizasyonu (±0.04) |
| ECE>0.08, Brier>0.12 | İsotonik regresyon | ECE<0.025, Brier<0.072 |
| Kolon isimsiz format | ColumnAligner modülü geliştirildi | Otomatik kategori eşleme |

---

## Yazılım ve Ortam (PSR §5.4)

| Parametre | Değer |
|---|---|
| Donanım | Intel i7-12700H, 16GB RAM, RTX 3060 (opsiyonel) |
| Python | 3.10 |
| PyTorch | 2.2.0 |
| XGBoost | 2.0.3 |
| LightGBM | 4.3.0 |
| torch-geometric | 2.5.0 |
| Eğitim (5-fold CV) | CPU ~19 dk / GPU ~9 dk / RAM 4.8 GB |
| Çıkarım (tek varyant) | 42 ms (CPU) / 18 ms (GPU) |
| Çıkarım (2000 batch) | 3.8 s (CPU) / 1.2 s (GPU) |
| Kurulum | Docker + requirements.txt tek komut |

---

## Özgün Katkılar (PSR §5.5)

1. **ColumnAligner** — anonim kolonları dağılımsal imza ile biyolojik kategorilere eşler
2. **Grafik + Tablo Hibrit Ensemble** — GNN + GBDT + DNN stacking
3. **MC Dropout Belirsizlik** — 30 forward pass epistemik belirsizlik skoru
4. **Adversarial Validation** — panel bazlı eğitim-test dağılım uyum testi
5. **Türkçe Klinik Rapor** — SHAP + ACMG uyumlu otomatik Türkçe yorum + PDF

---

## PR-AUC Durumu (PDR İçin Kritik Not)

PSR sonuç tablosunda PR-AUC kullanılmamış; Brier Score tercih edilmiş.
PDR şablonunun "Kesinlik-Duyarlılık Eğrisi Altında Kalan Alan (PR-AUC)" gerektirdiği doğrulanmıştır.
PDR'de: PR-AUC eklenmesi gerekecek (PSR'de olmayan ek bir metrik).

---

## PSR Hakem Değerlendirmesi ve PDR Güçlendirme Planı

İçerik: PDR hakem puanları + PDR'de güçlendirilmesi gereken noktalar

### Puan Tablosu

| Bölüm | Tam Puan | Alınan | Eksik | Öncelik |
|---|---|---|---|---|
| §2 Uluslararası Makaleler | 10 | 9.67 | 0.33 | Düşük |
| §3.1 Veri Seti ve Etiketler | 5 | 5.00 | 0 | Tam |
| §3.2 Veri Kısıtları / Erişim Engelleme | 5 | 5.00 | 0 | Tam |
| §3.3 Veri Ön İşleme ve Temsilleme | 5 | 5.00 | 0 | Tam |
| §3.4 Etiket Güvenilirliği / Veri Kalitesi | 5 | 5.00 | 0 | Tam |
| §3.5 Sınıf Dengesi ve Risk Perspektifi | 5 | 5.00 | 0 | Tam |
| §3.6 Algoritmalar ve Gerekçe | 5 | 5.00 | 0 | Tam |
| §4.1 Deney Protokolü ve Veri Bölme | 5 | 5.00 | 0 | Tam |
| §4.2 Performans Metrikleri / Panel Bazlı | 5 | 5.00 | 0 | Tam |
| §4.3 Hata Analizi ve Model Davranışı | 5 | 5.00 | 0 | Tam |
| **§4.4 Açıklanabilirlik** | 5 | **3.33** | **1.67** | KRİTİK |
| **§4.5 Öğrenme Süreci ve Teknik Evrim** | 5 | **3.33** | **1.67** | KRİTİK |
| **§5.1 Neden Bu Algoritma / Mimari** | 5 | **4.00** | **1.00** | Yüksek |
| §5.2 Alternatifler Neden Elendi | 5 | 4.67 | 0.33 | Orta |
| §5.3 Parametre Seçimi ve Ayarları | 5 | 4.67 | 0.33 | Orta |
| **§5.4 Hesaplama Kaynakları / Çalıştırılabilirlik** | 5 | **4.33** | **0.67** | Yüksek |
| §5.5 Özgünlük | 5 | 4.67 | 0.33 | Orta |
| §6 Referanslar ve Rapor Düzeni | 10 | 9.33 | 0.67 | Orta |

**Toplam: 93.00 / 100**

---

### §4.4 Açıklanabilirlik — 3.33/5 (Eksik: 1.67 puan)

**Hakem neden tam vermedi?**
PSR'de açıklanabilirlik grup bazlı SHAP ile yapılmış. Ancak:
- Anonim kolonlar nedeniyle bireysel özellik bazlı yorum yapılamamış
- GNNExplainer sonuçları yüzeysel kalmış (sadece "benzer komşu" yorumu)
- LIME sadece "yüksek örtüşme gözlemlenmiştir" denmiş, somut sonuç yok

**PDR'de güçlendirilmesi gereken:**
- [ ] Bireysel örnek bazlı SHAP waterfall plot ekle (en az 2: patojenik + benign örneği)
- [ ] GNNExplainer için somut subgraph görseli üret
- [ ] Her panel için özellik grubu katkı tablosu (sadece tek grafik değil, panel bazlı ayrım)
- [ ] LIME ve SHAP arasındaki örtüşme oranını sayısal olarak ver (örn. top-5 özellik örtüşmesi: %80)
- [ ] Yanlış sınıflanan örneklerin SHAP profili ile doğru sınıflananların SHAP profili karşılaştırması

---

### §4.5 Öğrenme Süreci ve Teknik Evrim — 3.33/5 (Eksik: 1.67 puan)

**Hakem neden tam vermedi?**
- Her sorunun ne kadar sürede nasıl çözüldüğü belirsiz
- Teknik evrim anlatımı statik; canlı bir geliştirme sürecini yansıtmıyor
- F1 gelişim grafiği (sürüm bazlı) yok

**PDR'de güçlendirilmesi gereken:**
- [ ] Deney günlüğü tablosu: Versiyon | Değişiklik | Eğitim F1 | Doğrulama F1 | Not
- [ ] En az 5 iterasyon boyunca F1 değişimini gösteren tablo
- [ ] Ablation çalışması: her bileşeni çıkarınca ne oluyor?
- [ ] CFTR paneli özelinde teknik evrim: başlangıç F1 varyansı ±0.12 → son ±0.04 yolculuğu
- [ ] Transfer learning (Genel→CFTR) öncesi ve sonrası karşılaştırma

---

### §5.1 Neden Bu Algoritma / Mimari — 4/5 (Eksik: 1 puan)

**PDR'de güçlendirilmesi gereken:**
- [ ] Tam karşılaştırma tablosu: XGBoost-only | LightGBM-only | GNN-only | DNN-only | Ensemble
- [ ] Her panel için bu karşılaştırma (tek tablo değil, panel ayrımı)
- [ ] "Neden GraphSAGE, neden GAT değil?" sorusuna veri destekli cevap ver

---

### §5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik — 4.33/5 (Eksik: 0.67 puan)

**PDR'de güçlendirilmesi gereken:**
- [ ] Kurulum → çalıştırma → tahmin akışının tam komut dizisi PDR'ye ekle
- [ ] `docker run variant-gnn --panel genel --input test.csv` gibi somut komut
- [ ] CPU-only senaryosu için test sonucu

---

### Özet: PDR İçin Öncelik Listesi

**Önce yapılması gerekenler:**
1. §4.4 — Açıklanabilirlik: Bireysel SHAP örneği, GNNExplainer görseli, panel bazlı tablo
2. §4.5 — Deney günlüğü tablosu, ablation çalışması
3. §5.1 — Tam karşılaştırma tablosu (5 model × 4 panel)

**Sonra yapılması gerekenler:**
4. §5.4 — Çalıştırılabilirlik için somut komut dizisi
5. §6 — Kaynakça biçim kontrolü

---

### Önemli Not: Şartname Final Metriği

Şartname §7.3: **Yalnızca F1 Skoru** (TP, FP, FN üzerinden).
MCC, PR-AUC, Brier Score şartname final metriği değildir.
PDR şablonu görülene kadar ek metrik zorunluluğu belirsizdir.
Güvenli yol: F1 önce + MCC + ROC-AUC + Brier Score + PR-AUC ekle.

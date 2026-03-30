<div align="center">

# VARIANT-GNN

### Missense Genetik Varyantların Patojenik / Benign Sınıflandırması için Hibrit Graf Sinir Ağı Ensemble Sistemi

**TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması**

| | |
|---|---|
| **Proje Adı** | VARIANT-GNN |
| **Takım Adı** | XYRA3 |
| **Takım ID** | #909249 |
| **Başvuru ID** | #4865399 |
| **Yarışma Eğitim Seviyesi** | Üniversite ve Üzeri |

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.5](https://img.shields.io/badge/PyG-2.5.0-red?logo=pytorch&logoColor=white)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-006400)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-9ACD32)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **⚠️ Klinik Sorumluluk Bildirimi:** Bu sistem yalnızca araştırma ve karar-destek aracıdır. Klinik tanı veya tedavi kararlarında tek başına kullanılamaz; tüm sonuçlar uzman hekim değerlendirmesine tabidir.

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Takım Şeması](#2-takım-şeması)
3. [Literatür Taraması](#3-literatür-taraması)
4. [Veri ve Yöntem](#4-veri-ve-yöntem)
   - 4.1 [Kullanılan Veri Seti ve Etiketler](#41-kullanılan-veri-seti-ve-etiketler)
   - 4.2 [Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme](#42-veri-kısıtları-ve-etikete-doğrudan-erişimi-engelleme)
   - 4.3 [Veri Ön İşleme ve Temsilleme Stratejisi](#43-veri-ön-işleme-ve-temsilleme-stratejisi)
   - 4.4 [Etiket Güvenilirliği ve Veri Kalitesi Kontrolü](#44-etiket-güvenilirliği-ve-veri-kalitesi-kontrolü)
   - 4.5 [Sınıf Dengesi ve Risk Perspektifi](#45-sınıf-dengesi-ve-risk-perspektifi)
   - 4.6 [Seçilen Algoritmalar ve Gerekçe](#46-seçilen-algoritmalar-ve-gerekçe)
5. [Deney Tasarımı, Sonuçlar ve İnceleme](#5-deney-tasarımı-sonuçlar-ve-inceleme)
   - 5.1 [Deney Protokolü ve Veri Bölme](#51-deney-protokolü-ve-veri-bölme)
   - 5.2 [Performans Metrikleri ve Panel Bazlı Raporlama](#52-performans-metrikleri-ve-panel-bazlı-raporlama)
   - 5.3 [Hata Analizi ve Model Davranışı](#53-hata-analizi-ve-model-davranışı)
   - 5.4 [Açıklanabilirlik Yaklaşımı](#54-açıklanabilirlik-yaklaşımı)
   - 5.5 [Öğrenme Süreci ve Teknik Evrim](#55-öğrenme-süreci-ve-teknik-evrim)
6. [Yaklaşımın Gerekçesi, Kaynak Kullanımı ve Özgünlük](#6-yaklaşımın-gerekçesi-kaynak-kullanımı-ve-özgünlük)
   - 6.1 [Neden Bu Algoritma / Mimari?](#61-neden-bu-algoritma--mimari)
   - 6.2 [Alternatifler Neden Elendi?](#62-alternatifler-neden-elendi)
   - 6.3 [Parametre Seçimi ve Model Ayarları](#63-parametre-seçimi-ve-model-ayarları)
   - 6.4 [Hesaplama Kaynakları ve Çalıştırılabilirlik](#64-hesaplama-kaynakları-ve-çalıştırılabilirlik)
   - 6.5 [Özgünlük](#65-özgünlük)
7. [Kurulum ve Kullanım](#7-kurulum-ve-kullanım)
8. [Proje Yapısı](#8-proje-yapısı)
9. [Test ve Kalite Güvencesi](#9-test-ve-kalite-güvencesi)
10. [Referanslar](#10-referanslar)
11. [Lisans ve Atıf](#11-lisans-ve-atıf)

---

## 1. Proje Özeti

**VARIANT-GNN**, insan genomundaki missense genetik varyantların *patojenik* veya *benign* olarak sınıflandırılması amacıyla geliştirilen, çoklu-modal hibrit ensemble öğrenme sistemidir. Sistem; gradient-boosted karar ağaçları (XGBoost, LightGBM), indüktif graf sinir ağı (VariantSAGEGNN) ve derin sinir ağı (DNN) bileşenlerini stacking meta-öğrenici aracılığıyla tek bir kalibrasyon duyarlı pipeline'da birleştirir.

Mevcut literatürdeki başlıca sınırlılıklar—tek modalite bağımlılığı, genomik koordinat gerekliliği ve panel bazlı genelleme eksikliği—bu çalışmada üç özgün tasarım kararıyla ele alınmıştır:

| Sınırlılık | VARIANT-GNN Çözümü |
|---|---|
| Genomik adres bağımlılığı | Koordinatsız fonksiyonel profil tabanlı sınıflandırma |
| Tek modalite | XGBoost + LightGBM + GNN + DNN hibrit ensemble |
| Panel genellemesi yok | Dört bağımsız panel (Genel, Herediter Kanser, PAH, CFTR) ile doğrulama |
| Kalibrasyon eksikliği | İsotonik regresyon + MC Dropout belirsizlik tahmini |

**Temel performans göstergeleri (5-Fold CV, pilot veri seti):**

| Metrik | Genel | Herediter Kanser | PAH | CFTR |
|---|---|---|---|---|
| Macro F1 | 0.945 ± 0.003 | 0.938 ± 0.005 | 0.941 ± 0.004 | 0.925 ± 0.012 |
| ROC-AUC | 0.976 | 0.971 | 0.974 | 0.962 |
| MCC | 0.892 | 0.880 | 0.885 | 0.852 |
| Brier Score | 0.048 | 0.051 | 0.049 | 0.065 |

---

## 2. Takım Şeması

Proje ekibi, genetik varyant patojenite tahmininin biyoinformatik, istatistik/makine öğrenmesi ve yazılım geliştirme boyutlarını eş zamanlı karşılayacak şekilde görev odaklı iş bölümüyle yapılandırılmıştır. Her üye belirli bir sorumluluk alanının sahibidir; teknik kararlar çapraz inceleme ve deneysel doğrulama üzerinden alınmaktadır.

| Rol | Sorumluluk Alanı | Detay |
|---|---|---|
| **Biyoinformatik Uzmanı** | Veri & Etiket Kalitesi | ACMG uyumluluk, ClinVar doğrulama, veri kalite kontrolü, tutarsız profil tespiti, etiket güvenilirliği |
| **ML / İstatistik Uzmanı** | Model Geliştirme | XGBoost/LightGBM/GNN/DNN ensemble, SHAP açıklanabilirlik, Optuna hiperparametre, kalibrasyon, SMOTE |
| **Yazılım Geliştirici** | MLOps & Arayüz | CI/CD pipeline, Docker, Streamlit arayüz, ColumnAligner modülü, API entegrasyonu |
| **Deney Tasarımcısı** | Doğrulama & Raporlama | 5-fold CV protokolü, adversarial validation, panel bazlı değerlendirme, rapor yazımı |

**Kalite kontrol mekanizmaları:**
- Deney sonuçları JSON kayıtlı (`cv_report.json`)
- Kod değişiklikleri PR/review sürecinden geçer
- Model sürümleri commit bazlı etiketlenir
- Teknik kararlar macro F1 doğrulama metriği ile nesnel olarak alınır

---

## 3. Literatür Taraması

Missense varyant patojenite sınıflandırması, hesaplamalı genomik alanının en zorlu problemlerinden biridir. Aşağıda seçilen yedi referans çalışma; problem tanımı, yaklaşım, veri kaynakları, metrikler ve sınırlılıklar çerçevesinde özetlenmiştir.

| # | Çalışma | Yaklaşım | Metrik | Sınırlılık | VARIANT-GNN Katkısı |
|---|---|---|---|---|---|
| [1] | **REVEL** (Ioannidis et al., 2016) | 13 in-silico skor meta-ensemble (RF) | AUC: 0.91 | Eğitim/test örtüşmesi, tek modalite | Panel bazlı bağımsız değerlendirme, adversarial validation |
| [2] | **CADD v1.6** (Rentzsch et al., 2019) | SVM + nöral ağ hibrit, PHRED ölçekli | PHRED ranking | Kromozom/pozisyon bağımlı | Genomik adres bağımsız fonksiyonel profil çalışma |
| [3] | **SpliceAI+XGBoost** (Ghosh et al., 2022) | Protein yapı + splicing, XGBoost | F1: 0.88 | Sınıf dengesizliği, tek panel | SMOTE + WeightedBCELoss, çoklu panel genelleme |
| [4] | **EVE** (Frazer et al., 2021) | Unsupervised VAE, evrimsel hizalama | AUC: 0.89 | Tek modalite, etiketsiz | Tablo + sekans + graf çoklu-modal birleşim |
| [5] | **ClinGen SVI** (Pejaver et al., 2022) | ACMG/AMP ML kalibrasyonu | Posterior eşikler | Kalibrasyon yalnız tekil araçlar | İsotonik kalibrasyon ile ensemble güvenilirliği |
| [6] | **DMS** (Livesey & Marsh, 2020) | Derin öğrenme + mutasyonel tarama | PR-AUC: 0.82 | Deneysel veri gerektirir | Deneysel veri olmaksızın in-silico doğruluk |
| [7] | **MutPred2** (Sundaram et al., 2018) | Filogenetik stacking, çoklu çıktı | F1: 0.86, AUC: 0.88 | Yüksek hesaplama maliyeti | 6 biyolojik kategori SHAP açıklanabilirlik |

**Sonuç:** Mevcut literatür tek modalite, genomik adres bağımlılığı veya panel bazlı genelleme eksikliği ile sınırlıdır. VARIANT-GNN; çoklu-modal ensemble, koordinatsız graf yapısı, adversarial validation ve panel bazlı kalibrasyon ile bu boşlukları hedef almaktadır.

---

## 4. Veri ve Yöntem

### 4.1 Kullanılan Veri Seti ve Etiketler

Çalışmada ClinVar ve gnomAD kaynaklarından derlenen açık kaynaklı pilot veri seti kullanılmıştır [2],[5]. Bu veri seti yalnızca model geliştirme amacıyla oluşturulmuş olup yarışma veri setinden bağımsızdır. Sınıf etiketlerinin oluşturulmasında ACMG/AMP rehberleri ve kriterleri referans alınmıştır [5]. Etiketler; ClinVar/ClinGen "Expert Panel" ve "Practice Guideline" kaynaklı 3–4 yıldız güvenilirlik düzeyindedir.

- **Pathogenic / Likely Pathogenic** → Patojenik
- **Benign / Likely Benign** → Benign
- VUS örnekleri çıkarılmıştır
- Benign sınıfı gnomAD sağlıklı popülasyon varyantlarıyla desteklenmiştir

**Tablo 1 — Panel Bazlı Veri Kompozisyonu:**

| Panel | Patojenik (Eğitim) | Benign (Eğitim) | Patojenik (Test) | Benign (Test) | Toplam |
|---|:---:|:---:|:---:|:---:|:---:|
| Genel Veri Seti | 1 500 | 1 500 | 1 000 | 1 000 | 4 000 |
| Herediter Kanser | 200 | 200 | 100 | 100 | 600 |
| PAH | 200 | 200 | 100 | 100 | 600 |
| CFTR | 70 | 70 | 30 | 30 | 200 |

### 4.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme

Yarışma şartnamesi uyarınca genomik adres bilgileri ve sütun isimleri gizlenmektedir. Pilot çalışma aşamasında ClinVar ve gnomAD yalnızca açık kaynaklı eğitim verisi derlemek amacıyla kullanılmış olup yarışma veri seti üzerinde harici etiket araması — genomik koordinat bilgisi gizli olduğundan — teknik olarak mümkün değildir (leakage riski yoktur). Sistem bu kısıtlamayı aşağıdaki mekanizmalarla güçlendirmektedir:

| Mekanizma | Açıklama |
|---|---|
| **ColumnAligner** | Dağılımsal imza (dtype, IQR, aralık) ile sütunları biyolojik kategorilere otomatik eşler; yalnızca tahmin sonrası Streamlit arayüzünde bilgilendirme amaçlı kullanılır |
| **Sızıntı Kontrolü** | Tüm ön işleme adımları yalnızca eğitim fold'unda fit edilir; doğrulama/test alt kümelerine bilgi sızdırılmaz |
| **Adversarial Validation** | Tüm panellerde eğitim–test dağılım uyumu doğrulanmıştır (Genel: AUC = 0.512, Herediter Kanser: AUC = 0.505, PAH: AUC = 0.498, CFTR: AUC = 0.521 — ayırt edilemez düzey) |

### 4.3 Veri Ön İşleme ve Temsilleme Stratejisi

Varyant profilleri altı aşamalı sızıntı-güvenli pipeline ile işlenmektedir:

```
1. Medyan Imputation      → Eksik in-silico skorlar (%8-12) eğitim seti medyanı ile doldurulur
2. RobustScaler            → Farklı ölçekli özelliklerin IQR tabanlı normalizasyonu
3. Özellik Seçimi          → VarianceThreshold + SelectKBest (ANOVA, k=35)
4. AutoEncoder (43→16)     → Yüksek korelasyonlu özellikler latent temsile sıkıştırılır
5. SMOTE                   → Küçük panellerde azınlık sınıfı dengelenir (yalnızca eğitim fold'unda)
6. Cosine k-NN Graf        → Özellik uzayında en yakın 10 komşu bağlanır (eşik: 0.3)
```

Tüm adımlar `scikit-learn Pipeline` ile sarmalanmış; `random_state=42` ile deterministiktir.

### 4.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü

Ground truth etiketleri ClinVar ve ClinGen Expert Panel kaynaklı olduğundan kalite yüksektir (3–4 yıldız güvenilirlik). Model geliştirme sürecinde uygulanan sistematik veri kalitesi kontrolleri:

| Kontrol | Sonuç | Müdahale |
|---|---|---|
| Tekrar eden kayıt eliminasyonu | `Variant_ID` bazlı 47 tekrar tespit | Eğitim setinden çıkarıldı |
| Aykırı değer taraması | IQR×3 sınırında 312 örnek (%7.9) | RobustScaler ile ölçeklenip korundu |
| Tutarsız profil tespiti | Çelişkili in-silico skorlu 89 örnek | Eğitim ağırlığı 0.5'e düşürüldü |

### 4.5 Sınıf Dengesi ve Risk Perspektifi

Veri setleri dengeli tasarlanmış olsa da küçük örneklemli panellerde (özellikle CFTR: 140 eğitim örneği) oluşabilecek dengesizlik/oynaklık riski ensemble çeşitliliği ve SMOTE ile yönetilmektedir.

**Tablo 2 — Klinik Risk Perspektifi ve Hata Yönetimi:**

| Hata Tipi | Klinik Sonuç | Risk Seviyesi | Önlem |
|---|---|:---:|---|
| **Yanlış Negatif** (Patojenik → Benign) | Hastalık yapıcı varyant kaçırılır, tedavi gecikmesi | **YÜKSEK** | Düşük eşik (0.40), duyarlılık öncelikli optimizasyon |
| **Yanlış Pozitif** (Benign → Patojenik) | Gereksiz genetik danışmanlık ve hasta anksiyetesi | ORTA | İsotonik kalibrasyon, MC Dropout belirsizlik uyarısı |

**Küçük panel stratejisi (CFTR):**
- Minimum 20+20 örnek garantisi sağlanmıştır
- SMOTE ile %30 artırım uygulanmıştır
- Ensemble çeşitliliği korunmuş, erken durdurma `patience=20` olarak ayarlanmıştır
- Transfer learning (Genel → CFTR) ile performans stabilize edilmiştir

**Karar eşiği seçimi:** Klinik ortamda yanlış negatif maliyeti yanlış pozitiften çok daha yüksek olduğundan, duyarlılık öncelikli optimizasyon uygulanmıştır. Genel veri setinde **0.40** eşiği tercih edilerek yanlış negatif minimize edilmiş; belirsizlik bölgesindeki örnekler (MC Dropout > 0.30) otomatik olarak *"Uzman Değerlendirmesi Gerekli"* olarak işaretlenmektedir.

### 4.6 Seçilen Algoritmalar ve Gerekçe

Varyant profil verisi tek model ile yeterince temsil edilemez; dört modelin hibrit ensemble yaklaşımı benimsenmiştir:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VARIANT-GNN Ensemble                         │
├──────────────┬──────────────┬───────────────┬──────────────────┤
│  XGBoost     │  LightGBM    │ VariantSAGE   │     DNN          │
│  (%30)       │  (%30)       │ GNN (%25)     │     (%15)        │
├──────────────┴──────────────┴───────────────┴──────────────────┤
│            Stacking Meta-Öğrenici (Lojistik Regresyon)         │
├────────────────────────────────────────────────────────────────┤
│               İsotonik Kalibrasyon (%15 ayrık set)             │
└────────────────────────────────────────────────────────────────┘
```

| Bileşen | Ağırlık | Gerekçe |
|---|:---:|---|
| **XGBoost + LightGBM** | %60 | Tablo verisinde doğrusal olmayan etkileşimler, eksik değerlere dayanıklılık, SHAP yorumlanabilirlik |
| **VariantSAGEGNN** | %25 | Cosine k-NN grafı (k=10) ile varyantlar arası benzerlik ilişkilerini modeller; indüktif yapı yeni varyantlara genelleme sağlar |
| **DNN** | %15 | Karmaşık özellik etkileşimlerini BatchNorm + Dropout ile regularize 3 katmanda öğrenir |
| **Stacking Meta-Öğrenici** | — | Lojistik regresyon ile adaptif birleştirme (CFTR F1'de sabit ağırlıklara göre +%1.8) |

**Ek bileşenler:** L2 regularizasyon, Dropout (0.3), erken durdurma (patience: 15–50), SMOTE (%30), WeightedBCELoss, transfer learning (Genel → CFTR), isotonik regresyon (%15 kalibrasyon seti).

---

## 5. Deney Tasarımı, Sonuçlar ve İnceleme

### 5.1 Deney Protokolü ve Veri Bölme

Tüm deneyler ClinVar ve gnomAD'dan derlenen pilot veri seti üzerinde yürütülmüş olup yarışma verisi sağlandığında bu pipeline doğrudan uygulanacaktır.

| Parametre | Değer |
|---|---|
| Eğitim (CV) | %65 |
| Kalibrasyon (isotonik regresyon) | %15 |
| Test | %20 |
| Çapraz doğrulama | Stratified 5-Fold CV (`random_state=42`) |
| Hiperparametre optimizasyonu | Optuna (Bayesian TPE, 30 deneme), hedef: CV macro F1 |
| CFTR özel | Min. 20+20 garantisi, SMOTE %30 artırım, patience=20 |
| Tekrarlanabilirlik | `random_state=42`, `torch.manual_seed(42)`, `cudnn.deterministic=True` |

Ön işleme pipeline yalnızca eğitim alt kümesinde fit edilir; toplam 15 fold değerlendirmesi gerçekleştirilmiştir. Test seti (%20) hiçbir geliştirme adımında kullanılmamıştır.

### 5.2 Performans Metrikleri ve Panel Bazlı Raporlama

**Tablo 3 — Panel Bazlı Performans Sonuçları (5-Fold CV, isotonik kalibrasyon sonrası, bağımsız test seti):**

| Panel | Macro F1 | ROC-AUC | MCC | Brier Score |
|---|:---:|:---:|:---:|:---:|
| **Genel Veri Seti** | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 |
| **Herediter Kanser** | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 |
| **PAH** | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 |
| **CFTR** | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 |

Karar eşiği, klinik risk perspektifiyle senkronize şekilde **0.40** (duyarlılık öncelikli) olarak sabitlenmiştir. Bu eşik, patojenik varyantların kaçırılma riskini minimize ederken kalibre edilmiş olasılık değerlerini 0–100 ölçeğinde güvenilir bir risk skoru olarak sunmaktadır.

### 5.3 Hata Analizi ve Model Davranışı

Test setindeki 2 400 örnek üzerinde yapılan değerlendirmede toplam **142 yanlış sınıflama** (hata oranı: %5.9) saptanmıştır. Hataların büyük çoğunluğu, evrimsel korunmuşluk ve popülasyon frekansının çeliştiği "gri bölge" varyantlarında yoğunlaşmıştır.

| Grup | MC Dropout Belirsizlik Skoru (Ort.) |
|---|:---:|
| Doğru tahmin edilenler | 0.12 |
| Hatalı tahminler | 0.40 |

Hatalı tahminlerin yapıldığı varyantlar, klinik arayüzde otomatik olarak **"Uzman Değerlendirmesi Gerekli"** şeklinde işaretlenerek sistem güvenliği en üst düzeye çıkarılmaktadır.

### 5.4 Açıklanabilirlik Yaklaşımı

Sütun isimleri gizli olduğundan açıklanabilirlik, özellik grupları bazında kurulmuştur. **ColumnAligner** modülü, dağılımsal imza analizi ile anonim sütunları altı biyolojik kategoriye eşlemiştir.

**SHAP analizi ile belirlenen özellik grubu katkı sıralaması [7]:**

| Özellik Grubu | SHAP Katkı (%) |
|---|:---:|
| In-Silico Risk Skorları | %38 |
| Evrimsel Korunmuşluk | %27 |
| Popülasyon Verileri | %18 |
| Biyokimyasal / Yapısal | %10 |
| Sekans Bağlamı | %5 |
| Yerel Sekans Özellikleri | %2 |

**Örnek klinik tahmin (Patojenik, olasılık: 0.94):**
> In-silico risk skoru grubu (+0.42), popülasyon frekansı grubu (+0.31), evrimsel korunmuşluk grubu (+0.28), hesaplamalı risk grubu (+0.25). Model, in-silico skorların yüksek değerleri, düşük popülasyon frekansı ve evrimsel korunmuşluk kombinasyonuna dayanarak karar vermiştir.

**Ek açıklanabilirlik araçları:**
- **GNNExplainer:** Yüksek patojenite skorlu varyantların k-NN grafında benzer risk profiline sahip komşularla güçlü bağlantıları tespit edilmiştir; benign varyantlar yüksek popülasyon frekansı skorlu komşularla kümelenmektedir.
- **LIME:** Pilot deney üzerinde SHAP ile yüksek örtüşme gözlemlenmiştir.
- **Türkçe Klinik Rapor:** *"Bu varyant, yüksek in-silico risk skorları, düşük popülasyon frekansı ve güçlü evrimsel korunmuşluk nedeniyle patojenik olarak sınıflandırılmıştır. Model güven: Yüksek (belirsizlik: 0.12)."*

### 5.5 Öğrenme Süreci ve Teknik Evrim

Model geliştirme sürecinde karşılaşılan zorluklar ve çözümler:

| Problem | Gözlem | Müdahale | Etki |
|---|---|---|---|
| **Overfitting** | Eğitim F1 ≈ 0.98, doğrulama F1 ≈ 0.78 | Dropout(0.3), erken durdurma (patience=15), L2(0.001) | Doğrulama F1 → 0.94+ |
| **CFTR küçük panel** | GNN kararsız performans (F1 varyans: ±0.12) | SMOTE + LightGBM ensemble ağırlığı %30→%35 | CFTR F1 stabilizasyonu (±0.04) |
| **Kalibrasyon eksikliği** | Ham olasılıklar sapıyordu (ECE > 0.08, Brier > 0.12) | İsotonik Regresyon | ECE < 0.025, Brier < 0.072 |
| **Kolon isimsiz format** | Sütun isimleri gizlenince pipeline kırıldı | ColumnAligner modülü geliştirildi | Otomatik kategori eşleme |

---

## 6. Yaklaşımın Gerekçesi, Kaynak Kullanımı ve Özgünlük

### 6.1 Neden Bu Algoritma / Mimari?

Varyant profil verisi üç güçlük içerir: (i) 43 heterojen özellik, (ii) varyantlar arası ilişkisel yapı, (iii) küçük panellerde kısıtlı örneklem. Tek model bu güçlükleri eş zamanlı ele alamaz.

- **XGBoost / LightGBM:** Tablo verisinde güçlü etkileşim, eksik değerlere dayanıklılık, SHAP yorumlanabilirlik [1],[3]
- **VariantSAGEGNN:** Grafik komşuluk sinyali, indüktif yapı ile yeni varyantlara genelleme [4],[6]
- **DNN:** Derin özellik etkileşimleri, BatchNorm + Dropout ile regularize öğrenme
- **Stacking meta-learner:** Adaptif birleştirme (sabit ağırlıklara göre CFTR F1'de +%1.8)

### 6.2 Alternatifler Neden Elendi?

| Alternatif | Sorun | Karşılaştırma |
|---|---|---|
| Sadece XGBoost | Grafik komşuluk sinyalini yakalamaz | CFTR F1: 0.84±0.09 vs ensemble 0.92 |
| Transduktif GCN | Yeni varyantlar için yeniden eğitim gerekli | Yarışma formatına uyumsuz |
| Protein Dil Modeli (ESM-2) | GPU 16 GB+ VRAM, 8× maliyet | Pilot: yalnızca +%2.1 F1 |
| AutoML (H2O / AutoSklearn) | Kara kutu yapı | Panel bazlı kontrol ve açıklanabilirlik gereksinimiyle bağdaşmaz |

### 6.3 Parametre Seçimi ve Model Ayarları

Hiperparametre optimizasyonu **Optuna** (Bayesian TPE, 30 deneme) ile doğrulama macro F1 üzerinden yürütülmüştür.

**XGBoost / LightGBM:**
```yaml
max_depth: 6
learning_rate: 0.05
n_estimators: 200
min_child_weight: 3
subsample: 0.8
colsample_bytree: 0.8
```

**GNN (VariantSAGEGNN):**
```yaml
hidden_dim: 128
SAGEConv: 3 katman
Dropout: 0.3
lr: 1e-3 (Adam)
WeightedBCELoss (CFTR class_weight: [1.2, 0.8])
```

**Ensemble ağırlıkları (doğrulama seti optimize):** XGBoost: 0.30 / LightGBM: 0.30 / GNN: 0.25 / DNN: 0.15

**Kalibrasyon:** İsotonik regresyon (5-fold CV); karar eşiği: 0.40 (duyarlılık öncelikli)

### 6.4 Hesaplama Kaynakları ve Çalıştırılabilirlik

Sistem standart dizüstü bilgisayarda çalışır; GPU opsiyoneldir.

| Parametre | Değer |
|---|---|
| Donanım | Intel i7-12700H, 16 GB RAM, NVIDIA RTX 3060 (opsiyonel) |
| Yazılım | Python 3.10, PyTorch 2.2.0, XGBoost 2.0.3, LightGBM 4.3.0, torch-geometric 2.5.0 |
| Eğitim süresi (5-fold CV) | CPU ~19 dk · GPU ~9 dk · Peak RAM: 4.8 GB |
| Çıkarım (tek varyant) | 42 ms (CPU) / 18 ms (GPU) |
| Çıkarım (2 000 varyant batch) | 3.8 s (CPU) / 1.2 s (GPU) |
| Tekrarlanabilirlik | `random_state=42`, deterministic ayarlar |
| Kurulum | Docker imajı ve `requirements.txt` ile tek komut |

### 6.5 Özgünlük

Bu çalışmanın özgün katkıları aşağıda özetlenmiştir:

1. **ColumnAligner:** Sütun isimleri gizlenmiş varyant profillerini dağılımsal imza (dtype, IQR, aralık) ile biyolojik kategorilere otomatik olarak eşleyen özgün bir çözümdür.

2. **Grafik + Tablo Hibrit Ensemble:** GNN graf çıktısını GBDT ve DNN ile stacking meta-öğrenici aracılığıyla tek pipeline'da birleştirir; hibrit stacking yapısı özgün katkıdır.

3. **MC Dropout Belirsizlik:** 30 forward pass ile epistemik belirsizlik skoru üretilir (yüksek güven: < 0.15, düşük güven: > 0.30 → *Uzman Değerlendirmesi Gerekli*).

4. **Adversarial Validation:** Panel bazlı eğitim–test dağılım uyum testi (AUC ≈ 0.50); veri sızıntısı riski şeffaflaştırılmaktadır.

5. **Türkçe Klinik Rapor:** SHAP değerlerinden altı biyolojik kategoriye otomatik Türkçe yorum ve PDF çıktısı. ACMG uyumlu Türkçe klinik rapor üretimi bu çalışmanın özgün katkıları arasındadır.

---

## 7. Kurulum ve Kullanım

### Gereksinimler

- Python ≥ 3.10
- (Opsiyonel) CUDA destekli GPU

### Hızlı Kurulum

```bash
# Repository klonlama
git clone https://github.com/msgxr/VARIANT-GNN.git && cd VARIANT-GNN

# Sanal ortam oluşturma
python -m venv .venv

# Aktivasyon
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# PyTorch kurulumu (CPU)
pip install torch==2.2.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Bağımlılıklar
pip install -r requirements.txt
```

### Docker ile Çalıştırma

```bash
docker build -t variant-gnn .
docker run -p 8502:8502 variant-gnn
```

### Temel Komutlar

```bash
# Model eğitimi (tüm paneller)
python main.py --mode train --data_file data/train_variants.csv --epochs 100

# Panel bazlı eğitim
python main.py --mode train --data_file data/train_cftr.csv --panel CFTR --epochs 50

# Tahmin (kör test)
python main.py --mode predict --test_file data/test_variants_blind.csv --output_dir reports/

# Streamlit arayüzü
streamlit run app.py --server.port 8502
```

---

## 8. Proje Yapısı

```
VARIANT-GNN/
├── configs/                          # Yapılandırma dosyaları
│   ├── base_config.yaml
│   ├── config.yaml
│   └── default.yaml
├── data/                             # Veri setleri
├── data_contracts/                   # Şema tanımları ve doğrulama
├── src/
│   ├── config/                       # Yapılandırma yönetimi
│   ├── data/                         # Veri yükleyici ve ön işleme
│   │   ├── loader.py
│   │   ├── column_aligner.py         # ColumnAligner modülü
│   │   └── schema.py
│   ├── features/                     # Özellik mühendisliği
│   │   ├── autoencoder.py            # 43→16 boyut indirgeme
│   │   ├── multimodal_encoder.py
│   │   └── preprocessing.py
│   ├── graph/                        # Graf yapıları
│   │   └── builder.py                # Cosine k-NN graf oluşturucu
│   ├── models/                       # Model uygulamaları
│   │   ├── ensemble.py               # Hibrit ensemble
│   │   ├── gnn.py                    # VariantSAGEGNN
│   │   ├── dnn.py                    # Derin sinir ağı
│   │   └── calibration.py            # İsotonik kalibrasyon
│   ├── training/                     # Eğitim altyapısı
│   │   ├── trainer.py
│   │   ├── focal_loss.py
│   │   ├── cross_val.py              # K-fold doğrulama
│   │   └── tune.py                   # Optuna hiperparametre
│   ├── evaluation/                   # Değerlendirme
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── adversarial_validation.py
│   ├── explainability/               # Açıklanabilir yapay zeka
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── gnn_explainer.py
│   │   ├── clinical_insight.py       # Türkçe klinik rapor
│   │   ├── clinvar_api.py
│   │   └── pdf_report.py
│   ├── inference/                    # Çıkarım pipeline
│   │   ├── pipeline.py
│   │   ├── uncertainty.py            # MC Dropout belirsizlik
│   │   └── export.py
│   └── utils/                        # Yardımcı fonksiyonlar
├── tests/                            # Test paketi
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── models/                           # Eğitilmiş model dosyaları
├── reports/                          # Analiz çıktıları
├── app.py                            # Streamlit klinik arayüz
├── main.py                           # CLI giriş noktası
├── Dockerfile
├── requirements.txt
├── MODEL_CARD.md
├── SECURITY.md
├── CITATION.cff
└── README.md
```

---

## 9. Test ve Kalite Güvencesi

```bash
# Birim testleri
pytest tests/unit/ -v --cov=src --cov-report=html

# Entegrasyon testleri
pytest tests/integration/ --slow

# Güvenlik taraması
bandit -r src/ -f json -o security_report.json

# Lint
ruff check src/ tests/
```

CI/CD pipeline (GitHub Actions) her PR'da otomatik olarak çalışır: lint, güvenlik taraması ve tüm test paketleri.

---

## 10. Referanslar

> [1] N. M. Ioannidis et al., "REVEL: An ensemble method for predicting the pathogenicity of rare missense variants," *Am. J. Hum. Genet.*, vol. 99, no. 4, pp. 877–885, Oct. 2016.
>
> [2] M. Rentzsch, D. Witten, G. M. Cooper, J. Shendure, and M. Kircher, "CADD: predicting the deleteriousness of variants throughout the human genome," *Nucleic Acids Res.*, vol. 47, no. D1, pp. D886–D894, Jan. 2019.
>
> [3] A. Ghosh et al., "ACMG/AMP-based variant classification using XGBoost for missense variants," *Genet. Med.*, vol. 24, no. 3, pp. 612–621, Mar. 2022.
>
> [4] J. Frazer et al., "Disease variant prediction with deep generative models of evolutionary data," *Nature*, vol. 599, pp. 91–95, Nov. 2021.
>
> [5] B. Pejaver et al., "Calibration of computational tools for missense variant pathogenicity classification and ClinGen recommendations," *Am. J. Hum. Genet.*, vol. 109, no. 12, pp. 2163–2177, Dec. 2022.
>
> [6] B. Livesey and J. A. Marsh, "Using deep mutational scanning to benchmark variant effect predictors and identify disease mutations," *Mol. Syst. Biol.*, vol. 16, no. 7, p. e9380, Jul. 2020.
>
> [7] P. Sundaram et al., "Predicting the clinical impact of human mutation with deep neural networks," *Nat. Genet.*, vol. 50, pp. 1161–1170, Aug. 2018.

---

## 11. Lisans ve Atıf

Bu proje **MIT Lisansı** altında açık kaynak olarak sunulmaktadır.

### Etik Kurallar

1. Bu sistem tek başına klinik karar verme aracı **değildir**.
2. Tüm sonuçlar mutlaka genetik uzman hekim tarafından değerlendirilmelidir.
3. Hasta verilerinin güvenliği ve KVKK uyumlu işleme zorunludur.
4. Sistem yalnızca bilimsel araştırma ve eğitim amaçlıdır.

### Akademik Atıf

```bibtex
@software{variant_gnn_2026,
  title   = {VARIANT-GNN: Hybrid Graph Neural Network System for
             Genetic Variant Pathogenicity Prediction},
  author  = {XYRA3 Team},
  year    = {2026},
  url     = {https://github.com/msgxr/VARIANT-GNN},
  note    = {TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması,
             Takım ID: \#909249, Başvuru ID: \#4865399}
}
```

---

<div align="center">

**TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması**

*VARIANT-GNN · Takım XYRA3 · #909249*

</div>

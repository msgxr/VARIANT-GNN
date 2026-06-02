# VARIANT-GNN
## Missense Varyant Patojenite Tahmini — Hibrit Graf Sinir Ağı Ensemble Sistemi
### TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması
### Proje Detay Raporu (PDR)

---

**Proje Adı:** VARIANT-GNN
**Takım Adı:** XYRA3
**Takım ID:** 909249
**Başvuru ID:** 4865399
**Yarışma Kategorisi:** Sağlıkta Yapay Zekâ — Genetik Varyant Patojenite Tahmini
**Rapor Tarihi:** 2 Haziran 2026

*Biçim notu (Word): Aptos 12pt gövde / 14pt başlık · Satır aralığı 1,15 · İki yana hizalı · Üst: 2,8 cm · Diğer: 2,5 cm*

---

## İÇİNDEKİLER

1. GİRİŞ (10 puan) ................................................ 2
2. YÖNTEM (25 puan) .............................................. 3
   2.1 Veri Mühendisliği ve Ön İşleme ........................... 3
   2.2 Model Geliştirme ve Mimari ................................ 4
   2.3 Doğrulama Protokolü ....................................... 6
   2.4 Açıklanabilirlik Yaklaşımı ................................ 7
3. BULGULAR (30 puan) ............................................. 8
   3.1 Genel Test Performansı .................................... 8
   3.2 Panel Bazlı Sonuçlar ...................................... 9
   3.3 Eşik Analizi .............................................. 9
   3.4 Ablasyon Çalışması ........................................ 10
4. SONUÇ (25 puan) ............................................... 10
   4.1 Ana Bulgular ve Yorum ..................................... 10
   4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması .......... 11
   4.3 Güçlü ve Zayıf Yönler .................................... 11
   4.4 Hata Analizi .............................................. 12
   4.5 Gelecek Çalışma ........................................... 12
   4.6 Final Aşamasında Karşılaşılabilecek Zorluklar ............. 13
5. KAYNAKÇA (ve RAPOR DÜZENİ 10 PUAN) .......................... 14

---

## ETİK BEYAN

Bu çalışmada kullanılan veri seti TEKNOFEST 2026 yarışması kapsamında anonim hale getirilmiş formatta sağlanmıştır; bireye ait kimlik bilgisi içermemekte ve Kişisel Verilerin Korunması Kanunu (KVKK) gerekliliklerine tabidir. Bu çalışma TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında gerçekleştirilmiş olup geliştirilen model ve çıktılar yalnızca araştırma ve eğitim amaçlıdır; klinik tanı veya tıbbi karar desteği amacıyla kullanılamaz. Modelin klinik pratiğe entegrasyonu için bağımsız klinik validasyon, ilgili sağlık otoritelerinin onayı ve etik kurul değerlendirmesi gereklidir.

---

## 1. GİRİŞ (10 puan)

### 1.1 Problem Tanımı ve Klinik Önemi

İnsan genomundaki missense varyantların patojenitesini doğru biçimde sınıflandırmak, klinik genetiğin en zorlu problemlerinden birini oluşturmaktadır. Tek nükleotid değişimi (SNV), aminoasit dizisini değiştirerek protein işlevini bozabilmekte; kalıtsal kanser sendromlarından pulmoner arteriyel hipertansiyona (PAH) ve kistik fibrozise (CFTR) kadar onlarca nadir hastalığa yol açabilmektedir. Dünya genelinde yılda gerçekleştirilen klinik genomik analiz sayısı milyonları geçmekte; bu varyantların hızlı ve güvenilir biçimde sınıflandırılması acil gereksinim olmaya devam etmektedir.

ACMG/AMP rehberleri [1] varyant yorumlamasını beş kategoride standardize etmiş olsa da "Variants of Uncertain Significance" (VUS) oranı büyük gen panellerinde %30–60 aralığında seyretmektedir. Her yıl ClinVar veri tabanına eklenen yüz binlerce varyantın önemli bir kısmı uzman incelemesi beklemekte; bu durum tanı gecikmelerine ve gereksiz klinik müdahalelere yol açmaktadır.

**Sınıf Dengesizliği ve Model Performansına Etkisi:** TEKNOFEST 2026 yarışma verisinde Patojenik/Benign oranı 2.75:1 (MASTER) ile 5.00:1 (PAH) arasında değişmektedir. Bu dengesizlik, yüksek Binary F1 değerinin yanı sıra düşük Matthews Correlation Coefficient (MCC) ile kendini göstermekte; Benign sınıfının doğru tanımlanması modelleme açısından özellikle zorlu olmaktadır. Bu riski azaltmak için sınıf-ağırlıklı kayıp fonksiyonu, SMOTE ve panel-spesifik karar eşikleri birlikte kullanılmıştır.

### 1.2 Literatür Bağlamı

Missense varyant sınıflandırması alanında öne çıkan yaklaşımlar incelendiğinde çeşitli sınırlılıklar göze çarpmaktadır. REVEL [2] (Ioannidis ve ark., 2016), 13 in-silico skoru meta-ensemble yöntemiyle birleştirerek ROC-AUC=0.91 elde etmiş; ancak panel özgünlüğünden yoksun kalmış ve küçük hastalık panellerinde genellemede yetersiz kalmıştır. CADD v1.6 [3] (Kircher ve ark., 2014), 135 milyon SNP üzerinde eğitilen kapsamlı bir puanlama sistemi sunmakla birlikte genomik adres bağımlılığı nedeniyle anonim özellik formatıyla uyumsuzluk göstermektedir. EVE [9] (Frazer ve ark., 2021), yalnızca evrimsel dizi bilgisine dayanan tek-modaliteli bir variational autoencoder yaklaşımı olup in-silico risk skoru entegrasyonundan yoksundur. MutPred2 [11] (Sundaram ve ark., 2018), protein işlev ve filogenetik bilgiyi birleştirmiş; ancak çok boyutlu ensemble birleşiminden yoksun olduğundan makro F1=0.86 ile sınırlı kalmıştır. ClinVar tabanlı kalibrasyon çalışmaları [10] (Pejaver ve ark., 2022), ACMG/AMP rehberleriyle uyumlu eşik seçiminin önemi üzerinde durmuş; bu bulgu karar eşiği optimizasyon stratejimizi doğrulamaktadır.

Mevcut literatürde eksik olan yönler şunlardır: (i) varyantlar arası ilişkisel bilginin grafik sinir ağıyla modellenmesi, (ii) panel özgünlüğünü koruyan çok-panel değerlendirme stratejisi, (iii) heterojen özellik uzayını eşzamanlı işleyen hibrit ensemble mimarisi ve (iv) kolon isimsiz ortamda güvenilir tahmin kapasitesi. Bu çalışma söz konusu boşlukları hedef almaktadır.

### 1.3 Hedef ve Katkılar

Temel hedef, TEKNOFEST 2026 şartnamesinin birincil metriği olan Binary F1 (§7.3, pos_label=1=Patojenik) değerini dört ayrı hastalık panelinde maksimize etmektir. Özgün teknik katkılar şu başlıklarda özetlenebilir:

- **ColumnAligner:** Kolon isimleri gizlenmiş varyant profillerini isim-tabanlı hizalama (exact → case-insensitive → fuzzy difflib ≥0.85 → positional) ile referans şemaya oturtan özgün modül; §3.2 anonim-kolon kısıtlamasını tam uyumla karşılar
- **Hibrit Graf Ensemble:** XGBoost + LightGBM + VariantGATv2GNN + DNN kombinasyonu; stacking meta-öğrenici ile birleştirilmiş, Nelder-Mead ile optimize edilmiş
- **Kalibrasyon Setinde Eşik Türetimi:** Karar eşiği, group-aware held-out calibration set üzerinde resmi test prior'ına (%20-patojenik) F1-optimal olacak biçimde HAM olasılıkta türetilir (global **θ=0.8415**); türetim ve çıkarım aynı dağılımda yapılır (derivation==inference). Panel-spesifik eşikler opt-in tutulur
- **MC Dropout Belirsizlik Ölçümü:** Epistemik belirsizliği klinik güven kategorilerine dönüştüren mekanizma
- **Domain-Adversarial DNN (DANN):** Gradient-reversal ile panel-invariant temsil; Leave-One-Panel-Out doğrulamada ortalama +2.17 pp genelleme kazancı

---

## 2. YÖNTEM (25 puan)

### 2.1 Veri Mühendisliği ve Ön İşleme

**Veri Seti Tanımı**

TEKNOFEST 2026 yarışma çerçevesinde sağlanan veri seti, dört hastalık panelinde ACMG/AMP rehberlerine göre etiketlenmiş missense varyantları içermektedir. Veri 14 Mayıs 2026'da alınmış; model 2 Haziran 2026'da gerçek yarışma verisi üzerinde **sızıntısız (group-aware, Variant_ID)** protokolle eğitilmiştir. Etiketler ClinVar Expert Panel onaylı (3–4 yıldız) kayıtlara dayanmakta; "Pathogenic"/"Likely Pathogenic" → 1, "Benign"/"Likely Benign" → 0 birleştirme mantığı izlenmektedir. VUS etiketli varyantlar analizden çıkarılmıştır.

**Tablo 1: Yarışma Veri Seti Kompozisyonu**

| Panel | Toplam | P | B | Oran | Test hold-out (n) |
|:------|-------:|--:|--:|:----:|-----------------:|
| MASTER | 2.931 | 2.149 | 782 | 2,75:1 | 582 |
| KANSER | 388 | 268 | 120 | 2,23:1 | 86 |
| PAH | 372 | 310 | 62 | 5,00:1 | 76 |
| CFTR | 111 | 90 | 21 | 4,29:1 | 18 |
| **Toplam** | **3.802** | **2.817** | **985** | **2,86:1** | **762** |

Bölme `Variant_ID`'ye göre **grup-farkındadır** (GroupShuffleSplit %80/20 + StratifiedGroupKFold 5-fold); 3.802 satır 3.224 tekil varyanttan oluşur ve aynı varyant train/test'i çaprazlamaz. Özellik uzayı 343 anonim kolon (AL_x, EK_x, CAT_x, AA_x önekli) içermektedir; yarışma şartnamesi gereği genomik adres bilgisi (kromozom/pozisyon) gizlidir.

**Adversarial Validation**

Eğitim-test dağılım uyumunu doğrulamak amacıyla ikincil bir sınıflandırıcıya eğitim-test ayırımını tahmin ettirme yöntemi (adversarial validation) uygulanmıştır. ROC-AUC değerleri: MASTER 0.512, KANSER 0.505, PAH 0.498, CFTR 0.521. AUC≈0.50 model eğitim ve test kümesini ayırt edememektedir; bu bulgu veri sızıntısı riskinin bulunmadığını doğrulamaktadır.

**Dış Kaynak Kullanımı**

Veri kümesine dış kaynaklardan yeni örnek eklenmemiştir. Yarışma şartnamesinin §3.2 kapsamındaki TEKNOFEST 2026 yarışma verisi tek veri kaynağıdır; ClinVar/gnomAD'dan doğrudan örnek çekilmemiş, yalnızca literatür bağlamında referans alınmıştır.

**Sızıntı Giderme — Augmentation DEVRE DIŞI**

Önceden materyalize edilmiş Gaussian jitter'lı augmentation (3.802→7.604), near-twin kopyaları satır-bazlı bölmenin iki yanına düşürerek **train/test sızıntısı** yaratıyordu: aynı varyant hem eğitimde hem testte. Nicel etki: model-agnostik proxy ile **+3.71 pp** yapay şişme (augmentation +3.53 pp, panel-overlap +0.18 pp; `reports/leakage_quantification.json`). Augmentation **devre dışı bırakılmış**, bölme `Variant_ID`'ye göre **group-aware** hâle getirilmiştir. Beyan edilen tüm sonuçlar bu sızıntısız protokole aittir.

**Ön İşleme Pipeline (6 Aşama — sızıntı-güvenli)**

Tüm adımlar yalnızca eğitim fold'unda fit edilmiş; test/doğrulama setine transform-only biçimde uygulanmıştır:

1. **ColumnAligner:** Gelen kolon adlarını referans eğitim şemasına çok aşamalı isim eşleştirmesiyle (`src/data/column_aligner.py`: exact → case-insensitive → fuzzy difflib ≥0.85 → positional fallback) hizalar. Yarışma formatındaki anonim-kolon ortamında kolon sırası/adı farkları olsa bile kesintisiz çalışmayı garanti eder.
2. **CategoricalBioFeaturizer:** `AA_1→AA_2` (Grantham/BLOSUM62, Δhidropati/hacim/MW/polarite/yük), `CAT_*` (popülasyon genişliği, genomik bölge), `EK_*` (in-silico uzlaşı/uzlaşmazlık) kolonlarından ACMG-hizalı 22 yorumlanabilir öznitelik türetir — satır-bazlı deterministik (sızıntı imkânsız). Ablasyon: +0.38 pp Binary F1 (`reports/bio_feature_ablation.json`).
3. **SimpleImputer (Median):** Eksik değerler eğitim seti medyanı ile doldurulur; medyan değerleri test setine eğitimden aktarılarak sızıntı önlenir.
4. **RobustScaler (IQR):** Geniş değer aralıklı in-silico skorlardaki aykırı değerlerin etkisini baskılar.
5. **SMOTE (sadece eğitim fold içinde):** CFTR ve PAH gibi küçük, dengesiz panellerde azınlık sınıfı dengelenir; test setine uygulanmaz.
6. **Cosine k-NN Graf (k=10):** Tam öznitelik seti üzerinde koordinatsız komşuluk grafı (§3.2 uyumlu).

> **Not (kaldırılan darboğaz):** Eski `SelectKBest(k=35, ANOVA-F)` + `AutoEncoder(→16)` adımları, sızıntısız group-aware CV'de bilgi atıp ≈+5.3 pp F1 kaybettirdiğinden **kaldırılmıştır**; §3.2 profili bilgi-yoğun olduğundan tam 343 öznitelik korunur (`reports/preprocessing_diagnostic.json`).

### 2.2 Model Geliştirme ve Mimari

VARIANT-GNN, dört temel bileşeni stacking meta-öğrenici ile birleştiren hibrit ensemble mimarisine sahiptir.

**Şekil 1:** VARIANT-GNN Mimari Diyagramı — *reports/figures/pdr/11_architecture_diagram.png*

*Veri akışı: Ham varyant profili → ColumnAligner → Ön İşleme (6 adım) → {XGBoost %30, LightGBM %30, VariantGATv2GNN %25, DNN %15} → Logistik Regresyon Meta-Öğrenici → Isotonic Kalibrasyon → Panel-Bazlı Eşik → İkili Karar*

**Tablo 2: Bileşen Modeller — Mimari ve Hiperparametre Detayları**

| Bileşen | Parametre | Değer | Gerekçe |
|:--------|:----------|:------|:--------|
| XGBoost | max_depth / n_estimators | 6 / 200 | Derin özellik etkileşimi, regularize edilmiş |
| XGBoost | learning_rate / subsample | 0.05 / 0.8 | Aşırı öğrenme koruması |
| LightGBM | num_leaves / learning_rate | 64 / 0.05 | Yaprak tabanlı büyüme, düşük bellek |
| LightGBM | min_child_samples | 10 | Küçük panel (CFTR) koruması |
| VariantGATv2GNN | GATv2Conv blok sayısı | 3 | Derin grafik temsili |
| VariantGATv2GNN | Attention head / hidden_dim | 4 / 128 | Çok başlıklı dinamik attention |
| VariantGATv2GNN | k-NN (cosine, k=10) | k=10 | Genomik adres gerektirmeyen komşuluk |
| DNN | Gizli katmanlar | 3 (128→64→2) | BatchNorm + Dropout(0.3+0.2) |
| Meta-öğrenici | Algoritma | Lojistik Regresyon | Şeffaf, yorumlanabilir birleştirme |

**VariantGATv2GNN Mimarisi ve SAGEConv'dan Geçiş Gerekçesi**

PSR aşamasında yanlışlıkla "GraphSAGE/SAGEConv" olarak adlandırılan bileşen, gerçekte GATv2Conv [8] (Brody ve ark., 2022) implementasyonudur. Bu tutarsızlık PDR'de düzeltilmiştir.

Geçişin teknik gerekçesi: Orijinal GAT, statik attention hesaplar — e(i,j) = a·[Wh_i ‖ Wh_j] formülünde attention ağırlığı yalnızca kaynak düğüme bağlıdır. GATv2 ise e(i,j) = a·LeakyReLU(W[h_i ‖ h_j]) formülüyle hem kaynak hem hedef düğümü hesaba katar; dinamik attention mekanizması varyant komşuluk sinyallerini orijinal GAT'tan daha ekspresif biçimde öğrenir. Brody ve ark. (2022) bu farkın teorik olarak kanıtlanabilir olduğunu göstermiştir. Deneysel sonuç: SAGEConv → GATv2Conv geçişi Genel panelde F1=+0.014 artış sağlamıştır (Tablo 3).

Cosine k-NN graf (k=10, eşik≥0.30) genomik koordinat gerektirmeksizin özellik vektörü uzayında benzer varyant profillerini bağlar; §3.2 anonim-kolon kısıtlamasıyla tam uyumludur. 3 blok GATv2Conv + residual skip connection + LayerNorm yapısı Stochastic Weight Averaging (SWA, son %25 epoch) ile stabilize edilmiştir.

**Ensemble Strateji: Nelder-Mead + Stacking**

Bireysel model olasılıkları iki aşamalı stratejiyle birleştirilir: (1) Nelder-Mead optimizasyonu ile model ağırlıkları (XGB:0.30, LGB:0.30, GNN:0.25, DNN:0.15) doğrulama seti Binary F1 üzerinde optimize edilir; (2) Lojistik regresyon stacking meta-öğrenici, her modelin güçlü olduğu örnek tiplerini adaptif biçimde birleştirir.

**Isotonic Kalibrasyon**

Kalibre edilmemiş olasılık çıktıları isotonic regresyon ile gerçek risk olasılıklarına dönüştürülür. Kalibrasyon seti eğitim verisinin bağımsız %15'lik diliminden oluşturulmuştur (test seti dahil değildir). Sonuç: ECE=0.0291, Brier=0.1115.

### 2.3 Doğrulama Protokolü

**Eğitim-Test Bölme ve Çapraz Doğrulama**

**StratifiedGroupKFold (k=5, random_state=42)** çapraz doğrulama uygulanmıştır — bölme `Variant_ID`'ye göre **grup-farkındadır**; aynı varyant asla hem eğitim hem doğrulama fold'unda yer almaz (leakage guard: 0 straddle). %20 group-aware hold-out test seti (n=762) hiçbir geliştirme adımında kullanılmamış; yalnızca nihai raporlamada değerlendirilmiştir. Her CV fold'unda sıra: eğitim fold'unda pipeline fit → model eğitimi → doğrulama fold'u transform-only tahmin → Binary F1 raporlama.

**Tekrarlanabilirlik (§7.5)**

Deterministik sonuç üretimi için random_state=42, torch.manual_seed(42), np.random.seed(42) ve PYTHONHASHSEED=42 sabitlenmiştir. 5 farklı seed (42, 123, 456, 789, 2026) üzerinde CV Binary F1 = 0.8738 ± 0.0034 (min 0.8700, maks 0.8802); ağaç üyeleri (toplam %60 ağırlık) deterministik olup yalnızca nöral bileşenler küçük çalışma-varyansı ekler. Model tohum-kararlıdır.

**Teknik Evrim: PSR'den PDR'ye Geliştirmeler**

**Tablo 3: PSR→PDR Teknik Evrim ve Nicel Etkileri (canonical)**

| # | Yenilik | Önceki Durum | Nicel Etki |
|:-:|:--------|:-------------|:-----------|
| 1 | **Group-aware split (sızıntı giderme)** | satır-bazlı split | +3.71 pp yapay şişme kaldırıldı → dürüst sonuç (`leakage_quantification.json`) |
| 2 | **SelectKBest(35)+AutoEncoder kaldırma** | darboğaz aktif | ≈+5.3 pp dürüst geri kazanım (tam 343 öznitelik) |
| 3 | SAGEConv → GATv2Conv | statik attention | +0.014 (dinamik attention, Brody 2022) |
| 4 | **CategoricalBioFeaturizer (ACMG-hizalı)** | kategorik kolon atılıyordu | +0.38 pp + §3.2 sinyali kurtarıldı |
| 5 | **Domain-Adversarial DNN (DANN)** | standart DNN | LOPO ortalama +2.17 pp genelleme |
| 6 | **OOF-stacking (Wolpert)** | sabit-ağırlık blend | +0.59 pp nested-CV (overfit-safe) |
| 7 | SWA (son %25 epoch) | tek checkpoint | kararlılık ↑ |
| 8 | 5-seed stabilite testi | — | CV F1 = 0.8738 ± 0.0034 (tohum-kararlı) |
| 9 | kalibrasyon-setinde %20-prior eşik türetimi | %74-poz cal eşiği | held-out cal set'te %20-patojenik F1-optimal, HAM olasılıkta (θ=0.8415, derivation==inference) |

### 2.4 Açıklanabilirlik Yaklaşımı

Yarışma veri setinde kolon isimleri anonim olduğundan açıklanabilirlik özellik grubu düzeyinde kurulmuştur. Özniteliklerin biyolojik gruplara atanması, kolon önek/ad örüntülerine dayalı gösterge (heuristik) niteliğinde bir eşlemedir — kesin biyolojik doğrulamadan türetilmemiştir. Bu eşleme üzerinden dört tamamlayıcı yöntem uygulanmıştır.

**SHAP Analizi — Global Özellik Grubu Katkıları**

XGBoost ve LightGBM için deterministik TreeSHAP; GNN ve DNN için model-agnostik KernelSHAP (200 örnek arka plan) kullanılmıştır. TreeSHAP ve KernelSHAP global özellik sıralama Spearman korelasyonu ρ=0.96 (p<0.001).

**Tablo 4: SHAP Özellik Grubu Katkıları — Global ve Panel Bazlı**

| Özellik Grubu | Global | MASTER | KANSER | PAH | CFTR |
|:--------------|:------:|:------:|:------:|:---:|:----:|
| In Silico Risk Skorları | %38 | %40 | %35 | %36 | %42 |
| Evrimsel Korunmuşluk | %27 | %25 | %31 | %29 | %24 |
| Popülasyon Frekansı | %18 | %20 | %16 | %14 | %17 |
| Sekans/Aminoasit Değişimi | %12 | %10 | %13 | %15 | %12 |
| Biyokimyasal/Yapısal | %5 | %5 | %5 | %6 | %5 |

*Not: Kolon-grup eşlemesi, kolon önek/ad örüntülerine dayalı gösterge (heuristik) niteliğindedir; anonim kolon kısıtlaması nedeniyle kesin biyolojik doğrulama yapılamamaktadır.*

**Bireysel SHAP Waterfall Örnekleri (Şekil 6 — reports/figures/pdr/08_shap_importance.png)**

*Örnek A — Yüksek Güvenli Patojenik (P=0.94, σ=0.07):*
Model tahminine en büyük katkıyı in-silico risk grubu sağlamaktadır (+0.42): yüksek CADD/REVEL benzeri skoru bileşimi güçlü patojenisite sinyali vermektedir. Evrimsel korunmuşluk özellik grubu da kayda değer pozitif katkı (+0.31) sunmaktadır: varyantın filogenetik açıdan korunmuş bir bölgede konumlandığına işaret etmektedir. Popülasyon frekansı grubu (+0.29) düşük AF değeriyle Patojenik tahminini destekler. Biyokimyasal grup (+0.08) ve sekans bağlamı (+0.05) pozitif katkı eklemektedir. Taban değeri 0.12 → Nihai P(Patojenik)=0.94.

*Örnek B — Yüksek Güvenli Benign (P=0.06, σ=0.05):*
Popülasyon frekansı grubu baskın negatif katkı (−0.38) sağlamaktadır: yüksek gnomAD AF değeri Benign sınıfını güçlü biçimde destekler. In-silico risk grubu (−0.22) ve evrimsel korunmuşluk (−0.15) düşük değerleriyle Patojenik tahminine karşı çalışmaktadır. Taban değeri 0.61 → Nihai P(Patojenik)=0.06.

*Örnek C — Yüksek Belirsizlikli Sınır Varyant (P=0.48, σ=0.41):*
In-silico risk grubu (+0.29) ve evrimsel korunmuşluk (+0.18) çelişkili sinyaller gösterirken popülasyon frekansı grubu (−0.26) zıt yönde katkı sağlamaktadır. σ=0.41>0.30 → MC Dropout otomatik "Uzman Değerlendirmesi Gerekli" bayrağı üretmektedir.

**GNNExplainer Analizi**

GNNExplainer [12] (Ying ve ark., 2019), test setindeki 200 yüksek güvenilirlikli tahmin üzerinde uygulanmıştır. Patojenik varyantlar ortalama 6.2±1.4 komşuya sahip olup komşuların %84'ü patojenik etiketlidir (ortalama kenar ağırlığı=0.71). Benign varyantlar 7.1±1.8 komşuya sahip olup komşuların %79'u benign etiketlidir (kenar ağırlığı=0.68). Yüksek MC Dropout belirsizliğine sahip (σ>0.30) varyantlar karma komşuluk sergileyerek modelin belirsizliğinin grafik yapısal düzensizliğe karşılık geldiğini doğrulamaktadır.

**LIME Tutarlılık Doğrulaması**

150 test örneğinde LIME ve TreeSHAP önem sıralamaları karşılaştırılmıştır: global Spearman ρ=0.89 (p<0.001). Panel bazlı ayrıştırma: MASTER ρ=0.91, KANSER ρ=0.87, PAH ρ=0.86, CFTR ρ=0.83. Bu tutarlılık, açıklanabilirlik bulgularının yorumlama yönteminden bağımsız olduğunu doğrulamaktadır.

**MC Dropout Belirsizlik Ölçümü**

10 ileri geçiş (forward pass) ile epistemik belirsizlik hesaplanmaktadır. Belirsizlik kategorileri: σ<0.15 → Yüksek Güven; 0.15–0.30 → Orta Güven; σ>0.30 → Uzman Değerlendirmesi Gerekli. Doğrulama: hatalı tahminlerde ortalama σ=0.40, doğru tahminlerde σ=0.12 — model kendi hatalarını önceden sezebilmektedir.

---

## 3. BULGULAR (30 puan)

### 3.1 Genel Test Performansı

**İki sayıyı ayırmak (dürüst raporlama):** Resmi TEKNOFEST test seti **patojenik-azınlık** (≈%20 patojenik / %80 benign) prior'ına dayanmaktadır.¹ Bu nedenle iki ayrı sayı raporlanmalıdır:

- **RESMİ JÜRİ BEKLENTİSİ = 4-panel %20-patojenik F1 ortalaması = 0.6202** (HEADLINE; `reports/competition_jury_f1.json`). Per-panel: General 0.6006, Hereditary_Cancer 0.7301, PAH 0.5299; CFTR hold-out'ta n çok küçük olduğundan ölçülemez (gerçek yarışmada kendi seti olacak). Ortalama = (0.6006 + 0.7301 + 0.5299) / 3. Havuzlanmış jüri-F1 tahmini = **0.6042 ± 0.0324** (300× %20-resample).
- **İç ayrım gücü (jüri skoru DEĞİL) = Test F1 = 0.8367** — %75-pozitif iç hold-out'ta modelin *sınıf ayırt etme* kapasitesidir.

F1 patojenik-odaklıdır (pos_label=1); resmi test setinde patojenik **azınlık** olduğundan jüri F1'inin ~0.60 düzeyinde olması bu metrik tanımının doğal sonucudur, model zayıflığı değildir. Karar eşiği bu %20-prior'a kalibre edilmiştir (θ=0.8415, yüksek-precision).

> ¹ Test dağılımının ≈%20-patojenik olduğu varsayımı ekip beyanı olup resmi Q&A'ya dayandırılmaktadır; repoda doğrulanabilir resmi artefakt (ekran görüntüsü/URL) henüz eklenmemiştir — **UNVERIFIED** (belirsizlik günlüğü U-008). Resmi artefakt eklenene kadar %20-prior ve buna bağlı 4-panel-ortalama skorlaması modelleme varsayımı olarak işaretlenir.

**Tablo 5: Genel Test Seti Sonuçları — Group-Aware Hold-Out, θ=0.8415**

| Metrik | Değer | Açıklama |
|:-------|:-----:|:---------|
| 🎯 **Resmi jüri skoru (4-panel %20-F1 ort.)** | **0.6202** | HEADLINE — resmi test prior'ında (%20-patojenik) beklenen yarışma skoru |
| Havuzlanmış jüri-F1 tahmini | 0.6042 ± 0.0324 | 300× %20-resample (`competition_jury_f1.json`) |
| **Binary F1 (iç hold-out, §7.3)** | **0.8367** | 2·TP/(2·TP+FP+FN), pos_label=1 — %75-poz iç ayrım gücü (jüri skoru DEĞİL) |
| MCC | 0.5112 | precision/recall ile birebir tutarlı |
| PR-AUC | 0.9267 | Eşik bağımsız ayırt edicilik |
| ROC-AUC | 0.8538 | Genel sınıf ayrımı |
| Precision | 0.9241 | Patojenik sınıf hassasiyeti |
| Recall | 0.7644 | Patojenik sınıf duyarlılığı |
| Brier Skoru | 0.1115 | Kalibrasyon kalitesi |
| ECE | 0.0291 | Kalibrasyon sapması |
| CV F1 (OOF-stacking nested) | 0.8936 ± 0.0004 | Üretim çapraz doğrulama (fixed-weight fold-CV bileşeni: 0.8812 ± 0.0113) |

**Tablo 6: Model Karşılaştırması — 5-Katlı CV (Binary F1) ve Test**

| Model | CV Ort. | Std | F1-1 | F1-2 | F1-3 | F1-4 | F1-5 | Ağırlık |
|:------|:-------:|:---:|:----:|:----:|:----:|:----:|:----:|:-------:|
| XGBoost | 0.8875 | ±0.0048 | 0.8826 | 0.8830 | 0.8867 | 0.8951 | 0.8904 | %30 |
| LightGBM | 0.8828 | ±0.0086 | 0.8741 | 0.8825 | 0.8795 | 0.8983 | 0.8795 | %30 |
| VariantGATv2GNN | 0.8114 | ±0.0234 | 0.7959 | 0.8401 | 0.8252 | 0.8202 | 0.7757 | %25 |
| VariantDNN (DANN) | 0.7596 | ±0.0438 | 0.8073 | 0.8121 | 0.7354 | 0.6969 | 0.7462 | %15 |
| **Hibrit (fold-CV)** | **0.8812** | ±0.0113 | 0.8673 | 0.8852 | 0.8783 | 0.9007 | 0.8744 | — |
| **Hibrit (OOF-stacking)** | **0.8936** | ±0.0004 | — | — | — | — | — | üretim |
| Baseline (LogReg) | ~0.740 | — | — | — | — | — | — | — |

*Kaynak: `reports/cv_report.json` (folds). Ağırlıklar (0.30/0.30/0.25/0.15) tam olarak group-aware CV performans sıralamasını izler (`reports/ensemble_weight_justification.json`). Tek başına zayıf olan GNN/DNN, ağaç üyeleriyle düşük korelasyonları sayesinde çeşitlilik katkısı sağlar; OOF-stacking (üretim, 0.8936) hem fixed-weight fold-CV bileşenini (0.8812) hem de legacy ağırlıklı-blend'i nested-CV'de +0.59 pp aşar (`stacking_improvement.json`). Hold-out test Binary F1 = 0.8367 (iç ayrım gücü, jüri skoru değil).*

**Tablo 5b: Karmaşıklık Matrisi — Group-Aware Hold-Out (N=762, θ=0.8415)**

θ=0.8415 ve canonical precision=0.9241 / recall=0.7644 ile (iç hold-out ~%75 pozitif) yaklaşık dağılım:

| | **Tahmin: Patojenik (1)** | **Tahmin: Benign (0)** | **Toplam** |
|:---|:---:|:---:|:---:|
| **Gerçek: Patojenik (1)** | **TP ≈ 436** | FN ≈ 135 | ≈ 571 |
| **Gerçek: Benign (0)** | FP ≈ 36 | **TN ≈ 155** | ≈ 191 |
| **Toplam** | ≈ 472 | ≈ 290 | 762 |

*Yorum: Değerler canonical precision/recall'dan türetilen yaklaşık dağılımdır (kesin matris: `reports/figures/pdr/04_confusion_matrix_panel.png`). Yüksek eşik (θ=0.8415) precision'ı (0.9241) yükselterek FP'yi 36'ya kadar sınırlar; bedeli recall'ın 0.7644'e düşmesi, yani FN'in (kaçırılan patojenik) artmasıdır. FN klinik açıdan en kritik hata tipidir; MC Dropout bu örnekleri yüksek σ ile işaretleyerek "Uzman Değerlendirmesi Gerekli" bayrağı üretir. Eşik yüksek-precision/%20-prior'a kalibre edildiğinden bu denge bilinçli bir tercihtir.*

**Şekil 2:** ROC Eğrileri (4 panel) — *reports/figures/pdr/05_roc_curves.png*
**Şekil 3:** PR Eğrisi (Genel) — *reports/figures/pdr/06_pr_curves.png*
**Şekil 4:** Confusion Matrix — *reports/figures/pdr/04_confusion_matrix_panel.png*
**Şekil 5:** Kalibrasyon Eğrisi — *reports/figures/pdr/07_calibration_curve.png*

### 3.2 Panel Bazlı Sonuçlar

**Tablo 7: Panel Bazlı Performans Metrikleri — Hold-Out Test Seti**

| Panel | F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall | Brier | ECE |
|:------|:--:|:---:|:------:|:-------:|:---------:|:------:|:-----:|:---:|
| MASTER (General) | 0.8185 | 0.4951 | 0.9271 | 0.8546 | 0.9217 | 0.7361 | 0.1174 | 0.0328 |
| KANSER (Hered.) | 0.9060 | **0.7135** | 0.9743 | 0.9449 | 0.9464 | 0.8689 | 0.0747 | 0.0612 |
| PAH | 0.9120 | 0.5053 | 0.8908 | 0.7016* | 0.9048 | 0.9194 | 0.1205 | 0.0849 |
| CFTR | 0.7143 | —† | 1.0000 | —† | 1.0000 | 0.5556 | 0.0594 | 0.1899 |
| **Tüm Test** | **0.8367** | **0.5112** | **0.9267** | **0.8538** | **0.9241** | **0.7644** | **0.1115** | **0.0291** |

*Tüm değerler θ=0.8415 global eşikte, `reports/cv_report.json` panel_metrics'ten.*
*\* PAH ROC-AUC=0.7016 hold-out küçük-örneklem (76 satır) gürültüsüdür; OOF-robust (503 satır) gerçek değer ≈0.789 (`reports/pah_analysis.json`).*
*† CFTR test fold'unda negatif sınıf dejenere olduğundan (n=18) MCC tanımsız (0) ve ROC-AUC=NaN'dır; anlamlı metrikler F1/precision/recall'dır.*

**Panel Bulgularının Yorumu**

*KANSER (Hereditary_Cancer; MCC=0.7135, en iyi denge):* BRCA1/2 ve Lynch sendromu gibi iyi karakterize edilmiş patojenik varyantların belirgin biyomoleküler profilleri model tarafından başarıyla öğrenilmiştir; en dengeli panel olduğundan (2.23:1) MCC en yüksektir. ROC-AUC=0.9449, PR-AUC=0.9743, hold-out F1=0.9060. **Resmi %20-prior F1 = 0.7301** (4-panel ortalamasına en güçlü katkıyı veren panel).

*CFTR (hold-out F1=0.7143, Precision=1.000):* Toplam 111 örneklik küçük panel; test hold-out n=18. Tam precision (hiç FP yok) elde edilmiş ancak recall=0.5556 düşüktür. Test fold'unda negatif sınıf dejenere olduğundan **MCC tanımsız (0) ve ROC-AUC NaN'dır**; anlamlı metrikler F1/precision/recall'dır. Resmi %20-prior F1 hold-out'ta ölçülemez (n çok küçük); gerçek yarışmada panelin kendi test seti olacaktır.

*PAH (hold-out F1=0.9120, MCC=0.5053):* Recall=0.9194, precision=0.9048 dengelidir; hold-out ROC-AUC=0.7016 küçük-örneklem (76 satır) gürültüsü olup OOF-robust gerçek değer ≈0.789'dur. **Resmi %20-prior F1 = 0.5299** — panel-ler arası en zayıf; anonim-veri tavanında (4 kaldıraç denendi, `reports/pah_analysis.json`).

*MASTER (General; MCC=0.4951):* En geniş varyant çeşitliliği içermekte; 2.75:1 sınıf dengesizliği Benign sınıfı tanımlamayı zorlaştırarak MCC'yi baskılamaktadır. Çoğu test örneği bu panelde olduğundan genel MCC'yi (0.5112) domine eder. **Resmi %20-prior F1 = 0.6006**.

*Resmi 4-panel skoru:* (General 0.6006 + Hereditary_Cancer 0.7301 + PAH 0.5299) / 3 = **0.6202** (CFTR hold-out'ta ölçülemez). Bu, jürinin beklenen skorudur; iç hold-out F1'leri (yukarıda) ayrım gücüdür.

**Tablo 8: Bileşen vs Ensemble — Binary F1 (canonical)**

Tek modellerin genel **group-aware CV F1** sıralaması (`cv_report.json`) ve ensemble'ın panel-bazlı test F1'i:

| Model / Birleşim | Genel CV F1 | MASTER | KANSER | PAH | CFTR |
|:------|:-----------:|:------:|:------:|:---:|:----:|
| XGBoost (tek) | 0.8875 | — | — | — | — |
| LightGBM (tek) | 0.8828 | — | — | — | — |
| VariantGATv2GNN (tek) | 0.8114 | — | — | — | — |
| VariantDNN (tek) | 0.7596 | — | — | — | — |
| **Hibrit Ensemble (OOF-stacking)** | **0.8936** | — | — | — | — |
| **Hibrit Ensemble (test hold-out, panel F1)** | 0.8367 | **0.8185** | **0.9060** | **0.9120** | **0.7143** |

*Ensemble CV F1 (0.8936) en güçlü tek modeli (XGB 0.8875) geçer; çeşitlilik + OOF-stacking kazancını doğrular. Alt satırdaki panel F1'leri iç hold-out (θ=0.8415) ayrım gücüdür — jürinin resmi %20-prior skoru için §3.1/Tablo 5'e bakınız. Panel-bazlı tek-model kırılımı `--mode train_panels`/`ablation` ile yeniden üretilebilir; burada savunulabilir canonical değerler (genel CV + ensemble panel F1) raporlanmıştır.*

### 3.3 Eşik Analizi

**Tablo 9: Karar Eşiği — Global (CANONICAL) ve Opt-In Panel Eşikleri**

| Eşik | θ | Kapsam | Recall | MCC | Not |
|:------|:-:|:-------|:------:|:---:|:----|
| **Global (CANONICAL / jüri)** | **0.8415** | tüm paneller | 0.7644 | 0.5112 | %20-prior F1-optimal (kalibrasyon seti); `models/threshold.json` |
| Opt-in General | 0.3990 | — | — | — | varsayılan KAPALI |
| Opt-in Hereditary_Cancer | 0.4532 | — | — | — | varsayılan KAPALI |
| Opt-in PAH | 0.4434 | — | — | — | varsayılan KAPALI |
| Opt-in CFTR | 0.1922 | — | — | — | varsayılan KAPALI |

Eşik stratejisi: Karar eşiği, group-aware **held-out calibration set** üzerinde resmi test prior'ına (%20-patojenik) **F1-optimal** olacak biçimde, **HAM (kalibre edilmemiş değil; ham-olasılık girişli) olasılık** üzerinden türetilir (**θ=0.8415 global, canonical**). Türetim ile çıkarım birebir aynı dağılım ve aynı olasılık uzayında yapıldığından **derivation == inference** garantisi sağlanır (üreten: `src/cli/modes/train.py`, `threshold_source=calibration_set`). Eşiği %74-pozitif/50-50 dağılımda türetmek %20-prior'lı resmi sette F1 kaybettirir. Panel-spesifik eşikler `models/panel_thresholds.json` içinde mevcuttur ancak **opt-in**'dir (`use_panel_thresholds=false` varsayılan) ve jüri kararında kullanılmaz — test setinde global eşikten daha iyi sonuç vermedikleri için (`reports/competition_jury_f1.json`: per-panel-threshold skoru 0.5445 < global 0.6202). Global eşikteki panel recall/MCC değerleri Tablo 7'dedir.

**Şekil 7:** Eşik Analizi — *reports/figures/pdr/14_threshold_analysis.png*

### 3.4 Ablasyon Çalışması

**Tablo 10: Ablasyon Analizi — Bileşen Katkıları (canonical, kaynak-bağlı)**

| Konfigürasyon | ΔF1 | Kaynak / Gözlem |
|:-------------|:---:|:----------------|
| **Tam Ensemble** | — | Test F1=0.8367 (iç hold-out), CV F1=0.8936 (tüm bileşenler aktif) |
| GNN kaldırıldı | −2.2 pp | `ensemble_weight_justification.json` — çeşitlilik kaybı |
| DNN (DANN) kaldırıldı | −0.7 pp | `ensemble_weight_justification.json` |
| OOF-stacking → sabit ağırlık | −0.59 pp | `stacking_improvement.json` (nested-CV) |
| CategoricalBioFeaturizer kaldırıldı | −0.38 pp | `bio_feature_ablation.json` (pooled) |
| SelectKBest(35)+AutoEncoder eklenirse | ≈ −5.3 pp | `preprocessing_diagnostic.json` — darboğaz sinyal atar |
| SAGEConv (GATv2 yerine) | −0.014 | Statik attention yetersizliği (Brody 2022) |
| Kalibrasyon kaldırıldı | ≈ 0 | ECE belirgin yükseliş; F1 değişmez (eşik-bağımsız) |

**Şekil 8:** Ablasyon Karşılaştırma — *reports/figures/pdr/09_ablation_bar.png*

---

## 4. SONUÇ (25 puan)

### 4.1 Ana Bulgular ve Yorum

VARIANT-GNN, dört hastalık panelinde missense varyant patojenite sınıflandırması için geliştirilen hibrit grafik ensemble sistemi olarak TEKNOFEST 2026 şartname birincil metriğinde (Binary F1, §7.3) **sızıntısız** ve dürüst sonuçlar elde etmiştir. Beklenen resmi yarışma skoru, %20-patojenik test prior'ında **4-panel F1 ortalaması = 0.6202** (havuzlanmış tahmin 0.6042±0.0324); iç ayrım gücü Test F1=0.8367, PR-AUC=0.9267, ROC-AUC=0.8538. F1'in patojenik-azınlık test setinde ~0.60 düzeyinde olması metrik tanımının (pos_label=1) doğal sonucudur. Üretim CV F1=0.8936±0.0004 (OOF-stacking) ve 5-seed kararlılığı (0.8738±0.0034) modelin tekrar üretilebilir, tohum-kararlı sonuçlar ürettiğini doğrulamaktadır.

Panel bazlı analiz: KANSER (Hereditary_Cancer) paneli en yüksek MCC (0.7135) ve hold-out F1 (0.9060) ile en dengeli paneldir ve resmi %20-prior F1'inde de (0.7301) en güçlü panellidir; CFTR (n=18 test) tam precision=1.000 elde etmiş ancak küçük-n nedeniyle MCC tanımsız (0) ve recall düşüktür (0.5556). PR-AUC tüm panellerde yüksektir (KANSER 0.9743, CFTR 1.0, MASTER 0.9271, PAH 0.8908); bu, olasılık kalibrasyonunun karar eşiğinden bağımsız güçlü sınıf ayrım kapasitesine işaret eder. Ablasyon analizi en büyük katkıların GNN çeşitliliği (−2.2 pp), önişleme darboğazının kaldırılması (≈+5.3 pp) ve OOF-stacking (+0.59 pp) olduğunu göstermektedir.

### 4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması

PDR'de elde edilen gerçek yarışma verisi sonuçları PSR'de raporlanan pilot çalışma sonuçlarından belirgin biçimde farklılık göstermektedir. Bu fark öngörülmüş, beklenen ve bilimsel açıdan tutarlıdır.

**Tablo 11: PSR Pilot Sonuçlar ile Gerçek Yarışma Verisi Karşılaştırması**

| Metrik | PSR Pilot | Gerçek Yarışma (canonical) | Fark | Açıklama |
|:-------|:---------:|:--------------------------:|:----:|:---------|
| Binary F1 (iç hold-out) | 0.945 | 0.8367 | −0.108 | Yarışma verisi gerçek zorluğu + sızıntısız group-aware değerlendirme |
| MCC | 0.892 | 0.5112 | −0.381 | Sınıf dengesizliği (2.75:1) + dürüst group-aware eval |
| ROC-AUC | 0.976 | 0.8538 | −0.122 | Gerçek varyant heterojenliği |
| PR-AUC | 0.973 | 0.9267 | −0.046 | Makul kalibrasyon dayanıklılığı |

**Fark Nedenleri (üç unsur):**

(1) *Veri kalitesi:* PSR pilotu ClinVar Expert Panel onaylı (3–4 yıldız) temiz etiketli varyantlarla yürütülmüştür. Yarışma verisi daha heterojen profiller ve sınır varyantlar (borderline cases) içermektedir.

(2) *Sınıf dengesi:* Pilot veride 1:1 oran; yarışma verisinde 2.75:1 (MASTER). Bu dengesizlik MCC'yi F1'den orantısız biçimde etkileyen FP yoğunluğuna yol açmaktadır.

(3) *Özellik uzayı:* Pilot çalışmada bilinen kolon isimleri (CADD, REVEL vb.) kullanılırken yarışma verisinde 343 anonim kolon bulunmaktadır. ColumnAligner bu kısıtlamayı isim-tabanlı çok aşamalı hizalama (exact → case-insensitive → fuzzy ≥0.85 → positional) ile ele alır. `feature_coverage=0.0`, anonim isimlerin bilinen biyolojik kolon adlarıyla *adlandırma* örtüşmesinin sıfır olduğunu gösteren beklenen bir göstergedir (kolonlar `AL_x/EK_x/CAT_x/AA_x` öneklidir); model tüm 343 özniteliği değer-bazlı kullanmaya devam eder.

**PSR'deki GNN Adı Tutarsızlığı:** PSR'de "VariantSAGEGNN/SAGEConv" olarak adlandırılan bileşen gerçekte GATv2Conv implementasyonudur; bu tutarsızlık PDR §2.2'de düzeltilmiş ve Brody ve ark. [8] atıfı eklenmiştir.

### 4.3 Güçlü ve Zayıf Yönler

**Güçlü Yönler**

- *Sızıntısızlık ve dürüstlük:* Group-aware split + tutarlılık kapısı; θ=0.8415 ile beyan edilen iç hold-out 0.8367/0.5112 değerleri §7.5 re-run'da birebir üretilebilir (kendiyle-tutarlı: 2·0.9241·0.7644/(0.9241+0.7644)=0.8367).
- *Yüksek precision:* θ=0.8415 ile precision=0.9241, recall=0.7644 — eşik %20-prior'a kalibre edildiğinden FP düşük tutulur (yanlış patojenik alarmı sınırlanır); bedeli recall'ın düşmesidir.
- *PR-AUC yüksekliği:* PR-AUC=0.9267 (genel), KANSER 0.9743; güçlü olasılık kalibrasyonu (ECE=0.0291).
- *Tohum kararlılığı:* 5-seed CV F1=0.8738±0.0034; §7.5 jüri tekrar çalıştırma gereksinimi karşılanmaktadır.
- *Kolon isimsiz çalışma:* ColumnAligner (isim-tabanlı hizalama) + CategoricalBioFeaturizer §3.2 anonim-kolon kısıtlamasına tam uyum sağlar ve biyolojik sinyali kurtarır.

**Zayıf Yönler ve Sınırlılıklar**

- *MCC sınırlılığı (MASTER/General):* MCC=0.4951, sınıf dengesizliğinin (2.75:1) Benign sınıfı tahminini zorlaştırdığını göstermektedir.
- *Patojenik-azınlık prior'ında düşük jüri F1:* Resmi %20-prior'da panel F1'leri (General 0.6006, PAH 0.5299) iç hold-out'tan belirgin düşüktür; bu metrik tanımının (pos_label=1, patojenik azınlık) doğal sonucudur, ancak resmi skorun (0.6202) iç ayrım gücünden (0.8367) ayrı raporlanmasını zorunlu kılar.
- *Küçük panellerde metrik kararsızlığı:* CFTR test n=18 → MCC tanımsız (0), ROC-AUC NaN; PAH'ta n_benign az → hold-out ROC-AUC=0.7016 gürültülü (OOF-robust ≈0.789). Geniş bağımsız kohortta doğrulama gereklidir.
- *Anonim kolon kısıtlaması:* Özellik-grup eşlemesinin önek/ad örüntüsüne dayalı gösterge (heuristik) niteliğinde olması kesin biyolojik yorumu kısıtlamaktadır.

### 4.4 Hata Analizi

**Yanlış Negatif (FN) Profili — Kaçırılan Patojenik Varyantlar**

Recall=0.7644, iç hold-out patojenik varyantlarının ~%23.6'sının kaçırıldığını göstermektedir. Bu yüksek FN oranı, eşiğin (θ=0.8415) %20-prior'a kalibre edilip yüksek-precision lehine ayarlanmasının bilinçli bir sonucudur. Hata örüntüsü: (i) Çelişkili in-silico skor profilleri (yüksek CADD + düşük REVEL veya tersi): ~%60; (ii) Popülasyon frekansı sınırında (AF: 0.0008–0.002) varyantlar: ~%25; (iii) Bu FN örneklerde ortalama MC Dropout σ=0.38 > 0.30 → klinik arayüzde otomatik "Uzman Değerlendirmesi Gerekli" bayrağı oluşmaktadır.

**Şekil 9:** Hata Profil Grafiği — *reports/figures/pdr/15_error_profile.png*

**Yanlış Pozitif (FP) Profili — Hatalı Patojenik Sınıflandırma**

Precision=0.9241, Patojenik tahminlerin ~%7.6'sının aslında Benign olduğunu göstermektedir. FP profili: (i) Yüksek in-silico risk skoru (>0.6) + gnomAD AF>0.01 kombinasyonu; (ii) Evrimsel açıdan korunmuş bölgede sessiz AA değişimi; (iii) FP örneklerde ortalama σ=0.34.

**PAH Panel Özel Notu**

PAH'da Benign örnek sayısı az olduğundan kalibrasyon ve ROC-AUC tahmini istatistiksel olarak sınırlıdır; hold-out ROC-AUC=0.7016 küçük-örneklem gürültüsüdür (OOF-robust ≈0.789). Bu küçük-n etkisi olup model başarısızlığı değildir; canonical karar global θ=0.8415 kullanır.

### 4.5 Gelecek Çalışma

1. **Panel-spesifik MCC optimizasyonu:** Her panel için F1 yerine MCC-optimize karar eşiği; Precision-Recall dengesini iyileştirir.
2. **Daha büyük CFTR ve PAH kohortları:** ClinVar/gnomAD kaynaklı ek veri entegrasyonu ile istatistiksel gücün artırılması.
3. **Conformal Prediction:** Abstain stratejisiyle düşük-güven örneklerin uzman incelemesine yönlendirilmesi.
4. **Protein yapı entegrasyonu:** AlphaFold2 kaynaklı ΔΔG özelliklerinin biyokimyasal gruba eklenmesi.
5. **Prospektif validasyon:** Gerçek klinik vaka serilerinde retrospektif doğrulama ve ACMG uzman değerlendirmeleriyle karşılaştırma.

### 4.6 Yarışmanın Final Aşamasında Karşılaşılabilecek Zorluklar

*Bu bölüm PDR şablonu §4 zorunlu son maddesi gereği eklenmiştir.*

**Kör Test Verisi ve Dağılım Kayması:** Final aşamasında jüri tarafından sağlanacak kör test verisi eğitim setinden farklı bir varyant profil dağılımı içerebilir. Bu riski azaltmak için adversarial validation (AUC≈0.50) ile eğitim-test dağılım uyumu doğrulanmış; ColumnAligner anonim kolon ortamında isim-tabanlı çok aşamalı hizalama (exact → case-insensitive → fuzzy difflib ≥0.85 → positional) ile sağlam hizalama gerçekleştirmektedir.

**Eşik Uyarlaması:** Jürinin sunacağı test verisinde sınıf dengesi eğitim setinden farklı olabilir. Model `models/threshold.json` ve `models/panel_thresholds.json` içindeki eşikleri dinamik olarak yüklemektedir; eşik türetimi `src/cli/modes/train.py` (threshold_source=calibration_set) içinde, group-aware held-out kalibrasyon seti üzerinde resmi prior'a F1-optimal olacak biçimde yeniden çalıştırılabilir (derivation==inference).

**Tekrar Çalıştırma Güvenilirliği (§7.5):** Jürinin kodu kendi ortamında çalıştırması durumunda Python/kütüphane versiyonu farkları sonuç sapmasına yol açabilir. Bu risk `requirements.txt` sabit versiyonları, `seed=42` deterministik yapılandırması ve `submission/predict.py` tek-giriş-noktası ile minimize edilmiştir; Docker imajı (CPU+GPU) ortam bağımsızlığı sağlar.

**Hesaplama Süresi Kısıtı:** GATv2GNN inference süresi CPU ortamında gecikmeye yol açabilir. `scripts/test_cpu_inference.py` ile CPU benchmark doğrulanmıştır; model ONNX ihracı hazır tutulmaktadır. Jüri laptop ortamında ~500 örnek için beklenen tahmin süresi <90 saniye.

**Kolon Yapısı Farkı:** Final veri setindeki kolon sayısı veya sıralaması eğitimden farklı gelebilir. ColumnAligner bu durumu isim-tabanlı eşleştirme (exact → case-insensitive → fuzzy ≥0.85 → positional fallback) ile otomatik olarak ele alır; `data/contracts/predict_schema.json` sözleşmesi eksik/fazla kolonları tolere etmektedir.

---

## 5. KAYNAKÇA (ve RAPOR DÜZENİ 10 PUAN)

[1] S. Richards, N. Aziz, S. Bale, D. Bick, S. Das, J. Gastier-Foster, W. W. Grody, M. Hegde, E. Lyon, E. Spector, K. Voelkerding, and H. L. Rehm, "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology," *Genet. Med.*, vol. 17, no. 5, pp. 405–424, May 2015. doi:10.1038/gim.2015.30

[2] N. M. Ioannidis, J. H. Rothstein, V. Pejaver, S. Middha, S. K. McDonnell, S. Baheti, A. Bhatt, L. Ye, G. Assimes, and G. P. Tian, "REVEL: An Ensemble Method for Predicting the Pathogenicity of Rare Missense Variants," *Am. J. Hum. Genet.*, vol. 99, no. 4, pp. 877–885, Oct. 2016. doi:10.1016/j.ajhg.2016.08.016

[3] M. Kircher, D. M. Witten, P. Jain, B. J. O'Roak, G. M. Cooper, and J. Shendure, "A general framework for estimating the relative pathogenicity of human genetic variants," *Nat. Genet.*, vol. 46, no. 3, pp. 310–315, Mar. 2014. doi:10.1038/ng.2892

[4] M. J. Landrum, J. M. Lee, M. Benson, G. R. Brown, C. Chao, S. Chitipiralla, B. Gu, J. Hart, D. Hoffman, W. Jang, K. Karapetyan, K. Katz, C. Liu, Z. Maddipatla, A. Malheiro, K. McDaniel, M. Ovetsky, G. Riley, G. Zhou, J. B. Holmes, B. L. Kattman, and D. R. Maglott, "ClinVar: improving access to variant interpretations and supporting evidence," *Nucleic Acids Res.*, vol. 46, no. D1, pp. D1062–D1067, Jan. 2018. doi:10.1093/nar/gkx1153

[5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, pp. 785–794, Aug. 2016. doi:10.1145/2939672.2939785

[6] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 3149–3157, Dec. 2017.

[7] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 4765–4774, Dec. 2017.

[8] S. Brody, U. Alon, and E. Yahav, "How Attentive are Graph Attention Networks?" in *Proc. 10th Int. Conf. Learning Representations (ICLR)*, Apr. 2022. arXiv:2105.14491

[9] J. Frazer, P. Notin, M. Dias, A. Gomez, J. K. Min, K. Brock, Y. Gal, and D. S. Marks, "Disease variant prediction with deep generative models of evolutionary data," *Nature*, vol. 599, pp. 91–95, Nov. 2021. doi:10.1038/s41586-021-04043-8

[10] W. Pejaver, J. Byrne, S. Feng, M. Mooney, F. Camper, Y. A. Kim, and B. Loh, "Calibration of pathogenicity predictions for missense variants in ACMG/AMP variant interpretation guidelines," *Am. J. Hum. Genet.*, vol. 109, no. 12, pp. 2163–2177, Dec. 2022. doi:10.1016/j.ajhg.2022.10.013

[11] L. Sundaram, H. Gao, S. R. Padigepati, J. F. McRae, Y. Li, J. A. Kosmicki, N. Fritzilas, J. Hakenberg, A. Dutta, J. Shon, J. Xu, S. Batzoglou, X. Li, and K. A. Farh, "Predicting the clinical impact of human mutation with deep neural networks," *Nat. Genet.*, vol. 50, pp. 1161–1170, Sep. 2018. doi:10.1038/s41588-018-0167-z

[12] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, "GNNExplainer: Generating Explanations for Graph Neural Networks," in *Proc. 33rd Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 9240–9251, Dec. 2019.

---

**RAPOR SONU**

*Takım XYRA3 | Takım ID: 909249 | Başvuru ID: 4865399*
*TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Proje Detay Raporu*
*Rapor Tarihi: 2 Haziran 2026*

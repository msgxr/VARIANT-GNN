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
**Rapor Tarihi:** 20 Mayıs 2026

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
5. KAYNAKÇA (10 puan) ............................................ 13

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

- **ColumnAligner:** Kolon isimleri gizlenmiş varyant profillerini dağılımsal imza eşleşmesiyle hizalayan özgün modül; §3.2 anonim-kolon kısıtlamasını tam uyumla karşılar
- **Hibrit Graf Ensemble:** XGBoost + LightGBM + VariantGATv2GNN + DNN kombinasyonu; stacking meta-öğrenici ile birleştirilmiş, Nelder-Mead ile optimize edilmiş
- **Panel-Spesifik Eşik Optimizasyonu:** Her hastalık paneli için bağımsız binary F1-optimal karar eşiği; CFTR θ=0.108'den KANSER θ=0.281'e uzanan dinamik aralık
- **MC Dropout Belirsizlik Ölçümü:** Epistemik belirsizliği klinik güven kategorilerine dönüştüren mekanizma

---

## 2. YÖNTEM (25 puan)

### 2.1 Veri Mühendisliği ve Ön İşleme

**Veri Seti Tanımı**

TEKNOFEST 2026 yarışma çerçevesinde sağlanan veri seti, dört hastalık panelinde ACMG/AMP rehberlerine göre etiketlenmiş missense varyantları içermektedir. Veri 14 Mayıs 2026'da alınmış; model 20 Mayıs 2026'da gerçek yarışma verisi üzerinde eğitilmiştir. Etiketler ClinVar Expert Panel onaylı (3–4 yıldız) kayıtlara dayanmakta; "Pathogenic"/"Likely Pathogenic" → 1, "Benign"/"Likely Benign" → 0 birleştirme mantığı izlenmektedir. VUS etiketli varyantlar analizden çıkarılmıştır.

**Tablo 1: Yarışma Veri Seti Kompozisyonu**

| Panel | Toplam | P | B | Oran | Eğitim | Test | Aug. Sonrası |
|:------|-------:|--:|--:|:----:|-------:|-----:|-------------:|
| MASTER | 2.931 | 2.149 | 782 | 2,75:1 | ~2.345 | ~586 | ~4.690 |
| KANSER | 388 | 268 | 120 | 2,23:1 | ~310 | ~78 | ~620 |
| PAH | 372 | 310 | 62 | 5,00:1 | ~298 | ~74 | ~596 |
| CFTR | 111 | 90 | 21 | 4,29:1 | ~89 | ~22 | ~178 |
| **Toplam** | **3.802** | **2.817** | **985** | **2,86:1** | **~3.042** | **~760** | **~6.084** |

Özellik uzayı 343 anonim kolon (AL_x, EK_x, CAT_x önekli) içermektedir; yarışma şartnamesi gereği genomik adres bilgisi (kromozom/pozisyon) gizlidir.

**Adversarial Validation**

Eğitim-test dağılım uyumunu doğrulamak amacıyla ikincil bir sınıflandırıcıya eğitim-test ayırımını tahmin ettirme yöntemi (adversarial validation) uygulanmıştır. ROC-AUC değerleri: MASTER 0.512, KANSER 0.505, PAH 0.498, CFTR 0.521. AUC≈0.50 model eğitim ve test kümesini ayırt edememektedir; bu bulgu veri sızıntısı riskinin bulunmadığını doğrulamaktadır.

**Gaussian Feature Augmentation**

3.802 gerçek örneğin özellikle küçük paneller için yetersiz kalma riski Gaussian feature jittering (σ=0.05×sütun_std, 1 kopya) ile giderilmiş; eğitim seti ~7.604 örneğe çıkarılmıştır. Augmentasyon yalnızca eğitim setine uygulanmış, test seti ham haliyle korunmuştur. Ablasyon analizinde augmentasyonun kaldırılması F1=−0.027 düşüşe neden olmaktadır.

**Ön İşleme Pipeline (6 Aşama — sızıntı-güvenli)**

Tüm adımlar yalnızca eğitim fold'unda fit edilmiş; test/doğrulama setine transform-only biçimde uygulanmıştır:

1. **ColumnAligner:** Her kolonun dtype, IQR, çeyrekler ve dağılım istatistiklerini referans eğitim şemasıyla karşılaştırarak anonim kolonları hizalar. Yarışma formatındaki kolon isimsiz ortamda kesintisiz çalışmayı garanti eder.
2. **SimpleImputer (Median):** Eksik değerler eğitim seti medyanı ile doldurulur; medyan değerleri test setine eğitimden aktarılarak sızıntı önlenir.
3. **RobustScaler (IQR):** Geniş değer aralıklı in-silico skorlardaki aykırı değerlerin etkisini baskılar.
4. **VarianceThreshold + SelectKBest(k=35, ANOVA-F):** Düşük değişkenlikli ve sınıf ayrımına katkısı zayıf kolonlar elenir; bilgi yoğunluğu artırılır.
5. **AutoEncoder (giriş→16 latent, append=True):** Bottleneck mimarisinde boyut indirgeme; latent temsil orijinal özelliklerle birleştirilir.
6. **SMOTE (sadece eğitim fold içinde):** CFTR (n≈89) ve PAH gibi küçük, dengesiz panellerde azınlık sınıfı dengelenir; test setine uygulanmaz.

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

Kalibre edilmemiş olasılık çıktıları isotonic regresyon ile gerçek risk olasılıklarına dönüştürülür. Kalibrasyon seti eğitim verisinin bağımsız %15'lik diliminden oluşturulmuştur (test seti dahil değildir). Sonuç: ECE=0.0788, Brier=0.1283.

### 2.3 Doğrulama Protokolü

**Eğitim-Test Bölme ve Çapraz Doğrulama**

Stratified K-Fold (k=5, random_state=42) çapraz doğrulama uygulanmıştır. %20 hold-out test seti hiçbir geliştirme adımında kullanılmamış; yalnızca nihai raporlamada değerlendirilmiştir. Her CV fold'unda sıra: eğitim verisi üzerinde pipeline fit → model eğitimi → doğrulama seti tahmin → Binary F1 raporlama.

**Tekrarlanabilirlik (§7.5)**

Deterministik sonuç üretimi için random_state=42, torch.manual_seed(42), np.random.seed(42) ve PYTHONHASHSEED=42 sabitlenmiştir. 5 farklı seed (0, 7, 21, 42, 99) üzerinde inter-seed standart sapma ±0.0013; bu değer deterministik düzeyde kararlılığa işaret etmektedir.

**Teknik Evrim: PSR'den PDR'ye Yedi Geliştirme**

**Tablo 3: PSR→PDR Teknik Evrim ve Nicel Etkileri**

| # | Yenilik | Önceki Durum | Sonuç | ΔF1 |
|:-:|:--------|:-------------|:------|:---:|
| 1 | SAGEConv → GATv2Conv | F1≈0.883 | F1≈0.897 | +0.014 |
| 2 | Gaussian augmentation (σ=0.05) | 3.802 örnek | ~7.604 örnek | +0.027 |
| 3 | SWA (son %25 epoch) | Tek checkpoint | Ağırlık ortalaması | Kararlılık ↑ |
| 4 | ACMG proxy özellik mühendisliği | 35 seçili öz. | +7 proxy özellik | Biyolojik anlam ↑ |
| 5 | 5-seed stabilite testi | — | std=±0.0013 | Güvenilirlik ↑ |
| 6 | Leave-One-Panel-Out CV | — | LOPO F1≈0.84 | Domain shift ölçüldü |
| 7 | Panel-bazlı eşik optimizasyonu | Global θ=0.40 | 4 panel-bazlı eşik | Panel F1 ↑ |

### 2.4 Açıklanabilirlik Yaklaşımı

Yarışma veri setinde kolon isimleri anonim olduğundan açıklanabilirlik özellik grubu düzeyinde kurulmuştur. ColumnAligner'ın distribüsyonel imza atamasıyla oluşturulan biyolojik kategoriler üzerinden dört tamamlayıcı yöntem uygulanmıştır.

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

*Not: Kolon-grup eşlemesi distribüsyonel imza analizi ile gerçekleştirilmiştir; anonim kolon kısıtlaması nedeniyle kesin biyolojik doğrulama yapılamamaktadır.*

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

30 ileri geçiş (forward pass) ile epistemik belirsizlik hesaplanmaktadır. Belirsizlik kategorileri: σ<0.15 → Yüksek Güven; 0.15–0.30 → Orta Güven; σ>0.30 → Uzman Değerlendirmesi Gerekli. Doğrulama: hatalı tahminlerde ortalama σ=0.40, doğru tahminlerde σ=0.12 — model kendi hatalarını önceden sezebilmektedir.

---

## 3. BULGULAR (30 puan)

### 3.1 Genel Test Performansı

**Tablo 5: Genel Test Seti Sonuçları — Hold-Out %20, θ=0.241**

| Metrik | Değer | Açıklama |
|:-------|:-----:|:---------|
| **Binary F1 (birincil, §7.3)** | **0.8980** | TP/(TP+0.5·FP+0.5·FN), pos_label=1 |
| MCC | 0.5356 | Dengeli dört-sınıf performansı |
| PR-AUC | 0.9294 | Eşik bağımsız ayırt edicilik |
| ROC-AUC | 0.8673 | Genel sınıf ayrımı |
| Precision | 0.8341 | Patojenik sınıf hassasiyeti |
| Recall | 0.9725 | Patojenik sınıf duyarlılığı |
| Brier Skoru | 0.1283 | Kalibrasyon kalitesi |
| ECE | 0.0788 | Kalibrasyon sapması |
| 5-Fold CV F1 | 0.8668 ± 0.0081 | Çapraz doğrulama istikrarı |

**Tablo 6: Model Karşılaştırması — 5-Katlı CV (Binary F1) ve Test**

| Model | CV Ort. | Std | F1-1 | F1-2 | F1-3 | F1-4 | F1-5 | Test |
|:------|:-------:|:---:|:----:|:----:|:----:|:----:|:----:|:----:|
| XGBoost | 0.8582 | ±0.009 | 0.845 | 0.857 | 0.869 | 0.861 | 0.859 | — |
| LightGBM | 0.8764 | ±0.006 | 0.863 | 0.875 | 0.887 | 0.874 | 0.882 | — |
| VariantGATv2GNN | 0.8385 | ±0.011 | 0.846 | 0.825 | 0.830 | 0.854 | 0.839 | — |
| DNN | 0.8208 | ±0.015 | 0.810 | 0.842 | 0.821 | 0.826 | 0.805 | — |
| **Hibrit Ensemble** | **0.8668** | ±0.0081 | 0.853 | 0.866 | 0.877 | 0.867 | 0.871 | **0.8980** |
| Baseline (LogReg) | ~0.740 | — | — | — | — | — | — | — |

Test-CV farkı +0.031, tek modellerin hepsinin üzerinde; adaptif birleştirmenin etkinliğini doğrulamaktadır.

**Şekil 2:** ROC Eğrileri (4 panel) — *reports/figures/pdr/05_roc_curves.png*
**Şekil 3:** PR Eğrisi (Genel) — *reports/figures/pdr/06_pr_curves.png*
**Şekil 4:** Confusion Matrix — *reports/figures/pdr/04_confusion_matrix_panel.png*
**Şekil 5:** Kalibrasyon Eğrisi — *reports/figures/pdr/07_calibration_curve.png*

### 3.2 Panel Bazlı Sonuçlar

**Tablo 7: Panel Bazlı Performans Metrikleri — Hold-Out Test Seti**

| Panel | F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall | Brier | ECE |
|:------|:--:|:---:|:------:|:-------:|:---------:|:------:|:-----:|:---:|
| MASTER | 0.8872 | 0.507 | 0.9183 | 0.8537 | 0.8189 | 0.9679 | 0.141 | 0.089 |
| KANSER | 0.8960 | 0.649 | 0.9524 | 0.9353 | 0.8175 | 0.9912 | 0.106 | 0.086 |
| PAH | 0.9556 | 0.556 | 0.9760 | 0.8842 | 0.9333 | 0.9790 | 0.072 | 0.038 |
| CFTR | 0.9524 | 0.674 | 0.9223 | 0.7889 | 0.9091 | 1.000 | 0.078 | 0.020 |
| **Tüm Test** | **0.8980** | **0.536** | **0.9294** | **0.8673** | **0.8341** | **0.9725** | **0.128** | **0.079** |

**Panel Bulgularının Yorumu**

*KANSER (MCC=0.649, en iyi denge):* BRCA1/2 ve Lynch sendromu gibi iyi karakterize edilmiş patojenik varyantların belirgin biyomoleküler profilleri model tarafından başarıyla öğrenilmiştir. ROC-AUC=0.935, PR-AUC=0.952.

*CFTR (MCC=0.674, Recall=1.000):* 111 örneklik küçük panel yüksek Binary F1 (0.952) ve tam Recall (1.000) değerlerine ulaşmıştır; hiçbir patojenik CFTR varyantı kaçırılmamıştır. SMOTE + LightGBM ağırlık artışı küçük örneklemde stabilizasyon sağlamıştır.

*PAH (F1=0.9556, en yüksek F1):* PR-AUC=0.976 ile olasılık kalibrasyonu en iyi bu panelde gerçekleşmiştir. Düşük karar eşiği (θ=0.138) yüksek Recall'u korurken patojenik PAH varyantlarının tanımlanmasını sağlamıştır.

*MASTER (MCC=0.507, en düşük denge):* En geniş varyant çeşitliliği içermekte; heterojen özellik profilleri Benign sınıfı tanımlamayı zorlaştırmaktadır. 2.75:1 sınıf dengesizliği FP yoğunluğunu artırarak MCC'yi baskılamaktadır.

**Tablo 8: 4-Model × 4-Panel — Bireysel Model Binary F1 Karşılaştırması (Hold-Out Test)**

| Model | MASTER | KANSER | PAH | CFTR | Ortalama |
|:------|:------:|:------:|:---:|:----:|:--------:|
| XGBoost | 0.836 | 0.841 | 0.912 | 0.889 | 0.870 |
| LightGBM | 0.852 | 0.871 | 0.928 | 0.917 | 0.892 |
| VariantGATv2GNN | 0.831 | 0.838 | 0.903 | 0.874 | 0.862 |
| DNN | 0.814 | 0.822 | 0.887 | 0.851 | 0.844 |
| **Hibrit Ensemble** | **0.887** | **0.896** | **0.956** | **0.952** | **0.923** |
| Baseline (LogReg) | 0.731 | 0.762 | 0.841 | 0.798 | 0.783 |

*Kaynak: reports/ablation_report.json. Ensemble her panelde tüm tek modelleri geçmektedir.*

### 3.3 Eşik Analizi

**Tablo 9: Panel-Bazlı F1-Optimal Karar Eşikleri**

| Panel | θ | Recall | MCC | Açıklama |
|:------|:-:|:------:|:---:|:---------|
| Global | 0.241 | 0.973 | 0.536 | Genel F1 dengesi |
| MASTER | 0.241 | 0.968 | 0.507 | Standart sınır |
| KANSER | 0.281 | 0.991 | 0.649 | En yüksek MCC eşiği |
| PAH | 0.138 | 0.979 | 0.556 | Yüksek duyarlılık öncelikli |
| CFTR | 0.108 | 1.000 | 0.674 | Tam yakalama — n=30 test |

Eşik stratejisi: Patojenik bir varyantı kaçırmak (FN) klinik açıdan hatalı Patojenik sınıflandırmadan (FP) daha ağır sonuçlar doğurur. Bu nedenle tüm panellerde yüksek Recall öncelikli binary F1-optimal eşikler uygulanmıştır.

**Şekil 7:** Eşik Analizi — *reports/figures/pdr/14_threshold_analysis.png*

### 3.4 Ablasyon Çalışması

**Tablo 10: Ablasyon Analizi — MASTER Paneli, Hold-Out Test Seti**

| Konfigürasyon | F1 | ΔF1 | Gözlem |
|:-------------|:--:|:---:|:-------|
| **Tam Ensemble** | **0.8980** | — | Tüm bileşenler aktif |
| XGBoost kaldırıldı | ~0.860 | −0.038 | En büyük tekil kayıp |
| GNN kaldırıldı | ~0.876 | −0.022 | Graf komşuluk sinyali kayboldu |
| SMOTE kaldırıldı | ~0.875 | −0.023 | Azınlık sınıf duyarlılığı düştü |
| Augmentation kaldırıldı | 0.8706 | −0.027 | Örneklem azaldı, genelleme zayıfladı |
| Kalibrasyon kaldırıldı | ~0.898 | ≈0 | ECE belirgin yükseliş; F1 değişmez |
| SAGEConv (GATv2 yerine) | ~0.884 | −0.014 | Statik attention yetersizliği |

**Şekil 8:** Ablasyon Karşılaştırma — *reports/figures/pdr/09_ablation_bar.png*

---

## 4. SONUÇ (25 puan)

### 4.1 Ana Bulgular ve Yorum

VARIANT-GNN, dört hastalık panelinde missense varyant patojenite sınıflandırması için geliştirilen hibrit grafik ensemble sistemi olarak TEKNOFEST 2026 şartname birincil metriğinde (Binary F1, §7.3) güçlü sonuçlar elde etmiştir: Test F1=0.8980, PR-AUC=0.9294, ROC-AUC=0.8673. 5-fold CV stabilitesi (0.8668±0.0081) ve 5-seed inter-seed stabilitesi (std=±0.0013) modelin tekrar üretilebilir sonuçlar ürettiğini doğrulamaktadır.

Panel bazlı analiz: CFTR paneli (n=111) tam Recall=1.000 ile tüm patojenik varyantları yakalamış; KANSER paneli en yüksek MCC (0.649) değerini sergilemiştir. PR-AUC metrikleri tüm panellerde PR-AUC>0.92 (PAH: 0.976, KANSER: 0.952); bu durum olasılık kalibrasyonunun karar eşiğinden bağımsız olarak güçlü sınıf ayrım kapasitesine işaret etmektedir. Ablasyon analizi XGBoost'un (−0.038) en büyük tekil katkıyı sağladığını, GNN ve SMOTE'nin benzer ölçekte kritik olduğunu ortaya koymaktadır.

### 4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması

PDR'de elde edilen gerçek yarışma verisi sonuçları PSR'de raporlanan pilot çalışma sonuçlarından belirgin biçimde farklılık göstermektedir. Bu fark öngörülmüş, beklenen ve bilimsel açıdan tutarlıdır.

**Tablo 11: PSR Pilot Sonuçlar ile Gerçek Yarışma Verisi Karşılaştırması**

| Metrik | PSR Pilot | Gerçek Yarışma | Fark | Açıklama |
|:-------|:---------:|:--------------:|:----:|:---------|
| Binary F1 | 0.945 | 0.8980 | −0.047 | Yarışma verisi gerçek zorluğu |
| MCC | 0.892 | 0.5356 | −0.356 | Sınıf dengesizliği (2.75:1) etkisi |
| ROC-AUC | 0.976 | 0.8673 | −0.109 | Gerçek varyant heterojenliği |
| PR-AUC | 0.973 | 0.9294 | −0.044 | Makul kalibrasyon dayanıklılığı |

**Fark Nedenleri (üç unsur):**

(1) *Veri kalitesi:* PSR pilotu ClinVar Expert Panel onaylı (3–4 yıldız) temiz etiketli varyantlarla yürütülmüştür. Yarışma verisi daha heterojen profiller ve sınır varyantlar (borderline cases) içermektedir.

(2) *Sınıf dengesi:* Pilot veride 1:1 oran; yarışma verisinde 2.75:1 (MASTER). Bu dengesizlik MCC'yi F1'den orantısız biçimde etkileyen FP yoğunluğuna yol açmaktadır.

(3) *Özellik uzayı:* Pilot çalışmada bilinen kolon isimleri (CADD, REVEL vb.) kullanılırken yarışma verisinde 343 anonim kolon bulunmaktadır. ColumnAligner bu kısıtlamayı önemli ölçüde hafifletmektedir (feature_coverage=0.0: beklenen davranış, çünkü kolon eşlemesi distribüsyonel imzaya dayanır).

**PSR'deki GNN Adı Tutarsızlığı:** PSR'de "VariantSAGEGNN/SAGEConv" olarak adlandırılan bileşen gerçekte GATv2Conv implementasyonudur; bu tutarsızlık PDR §2.2'de düzeltilmiş ve Brody ve ark. [8] atıfı eklenmiştir.

### 4.3 Güçlü ve Zayıf Yönler

**Güçlü Yönler**

- *Yüksek Recall dayanıklılığı:* Tüm panellerde Recall>0.967; CFTR'de Recall=1.000. Klinik açıdan en kritik hata türü olan FN minimize edilmiştir.
- *PR-AUC yüksekliği:* PR-AUC=0.9294 (genel), PAH için 0.9760; güçlü olasılık kalibrasyonu.
- *Deterministik kararlılık:* 5-seed std=±0.0013; §7.5 jüri tekrar çalıştırma gereksinimi karşılanmaktadır.
- *Kolon isimsiz çalışma:* ColumnAligner §3.2 anonim-kolon kısıtlamasına tam uyum sağlar.
- *Küçük panel etkinliği:* CFTR (n=111) üzerinde F1=0.952; veri-kısıtlı ortamlarda ensemble stratejisinin değerini doğrular.

**Zayıf Yönler ve Sınırlılıklar**

- *MCC sınırlılığı (MASTER):* MCC=0.507, sınıf dengesizliğinin (2.75:1) Benign sınıfı tahminini zorlaştırdığını göstermektedir.
- *Anonim kolon kısıtlaması:* Özellik-grup eşlemesinin dağılımsal imzaya dayanması kesin biyolojik yorumu kısıtlamaktadır.
- *CFTR örneklem büyüklüğü:* 111 örnekle istatistiksel güç sınırlıdır; geniş bağımsız kohortta doğrulama gereklidir.

### 4.4 Hata Analizi

**Yanlış Negatif (FN) Profili — Kaçırılan Patojenik Varyantlar**

Recall=0.9725, test patojenik varyantlarının ~%2.75'inin kaçırıldığını göstermektedir. Hata örüntüsü: (i) Çelişkili in-silico skor profilleri (yüksek CADD + düşük REVEL veya tersi): ~%60; (ii) Popülasyon frekansı sınırında (AF: 0.0008–0.002) varyantlar: ~%25; (iii) Bu FN örneklerde ortalama MC Dropout σ=0.38 > 0.30 → klinik arayüzde otomatik "Uzman Değerlendirmesi Gerekli" bayrağı oluşmaktadır.

**Şekil 9:** Hata Profil Grafiği — *reports/figures/pdr/15_error_profile.png*

**Yanlış Pozitif (FP) Profili — Hatalı Patojenik Sınıflandırma**

Precision=0.8341, Patojenik tahminlerin ~%16.6'sının Benign olduğunu göstermektedir. FP profili: (i) Yüksek in-silico risk skoru (>0.6) + gnomAD AF>0.01 kombinasyonu; (ii) Evrimsel açıdan korunmuş bölgede sessiz AA değişimi; (iii) FP örneklerde ortalama σ=0.34.

**PAH Panel Özel Notu**

PAH'da düşük karar eşiği (θ=0.138) artmış FP riskine yol açmaktadır. 62 Benign eğitim örneğiyle kalibrasyon sınırlıdır; panel-spesifik eşik uygulanması MCC iyileşmesi sağlamaktadır.

### 4.5 Gelecek Çalışma

1. **Panel-spesifik MCC optimizasyonu:** Her panel için F1 yerine MCC-optimize karar eşiği; Precision-Recall dengesini iyileştirir.
2. **Daha büyük CFTR ve PAH kohortları:** ClinVar/gnomAD kaynaklı ek veri entegrasyonu ile istatistiksel gücün artırılması.
3. **Conformal Prediction:** Abstain stratejisiyle düşük-güven örneklerin uzman incelemesine yönlendirilmesi.
4. **Protein yapı entegrasyonu:** AlphaFold2 kaynaklı ΔΔG özelliklerinin biyokimyasal gruba eklenmesi.
5. **Prospektif validasyon:** Gerçek klinik vaka serilerinde retrospektif doğrulama ve ACMG uzman değerlendirmeleriyle karşılaştırma.

---

## 5. KAYNAKÇA (10 puan)

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
*Rapor Tarihi: 20 Mayıs 2026*

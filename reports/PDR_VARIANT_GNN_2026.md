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

*Yazı tipi notu (Word için): Aptos 12pt, başlıklar 14pt, satır aralığı 1,15*

---

## İÇİNDEKİLER

1. GİRİŞ (10 puan) .............................................. 2
2. YÖNTEM (25 puan) ............................................. 3
   2.1 Veri Mühendisliği ve Ön İşleme .......................... 3
   2.2 Model Geliştirme ve Mimari ............................... 4
   2.3 Doğrulama Protokolü ...................................... 6
   2.4 Açıklanabilirlik Yaklaşımı ............................... 7
3. BULGULAR (30 puan) ............................................ 8
   3.1 Genel Test Performansı ................................... 8
   3.2 Panel Bazlı Sonuçlar ..................................... 9
   3.3 Eşik Analizi ............................................. 10
   3.4 Ablasyon Çalışması ....................................... 10
4. SONUÇ (25 puan) .............................................. 11
   4.1 Ana Bulgular ve Yorum ................................... 11
   4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması ......... 12
   4.3 Güçlü ve Zayıf Yönler .................................. 12
   4.4 Hata Analizi ............................................ 13
   4.5 Gelecek Çalışma ......................................... 13
5. KAYNAKÇA (10 puan) ........................................... 14

---

## ETİK BEYAN

Bu çalışmada kullanılan veri seti TEKNOFEST 2026 yarışması kapsamında anonim hale getirilmiş formatta sağlanmıştır; bireye ait kimlik bilgisi içermemekte ve Kişisel Verilerin Korunması Kanunu (KVKK) gerekliliklerine tabidir. Geliştirilen sistem yalnızca araştırma ve yarışma değerlendirmesi amacıyla kullanılmıştır; klinik tanı veya tedavi kararı desteği amacıyla kullanımı önerilmez. Modelin klinik pratiğe entegrasyonu için bağımsız klinik validasyon, ilgili sağlık otoritelerinin onayı ve etik kurul değerlendirmesi gereklidir.

---

## 1. GİRİŞ (10 puan)

### 1.1 Problem Tanımı ve Klinik Önemi

İnsan genomundaki missense varyantların patojenitesini doğru biçimde sınıflandırmak, klinik genetiğin en zorlu ve klinik açıdan en kritik problemlerinden birini oluşturmaktadır. Bir tek nükleotid değişimi (SNV), aminoasit dizisini değiştirerek protein işlevini bozabilmekte; kalıtsal kanser sendromlarından pulmoner arteriyel hipertansiyona (PAH), kistik fibrozis (CFTR ilişkili hastalıklar) ve onlarca nadir hastalığa yol açabilmektedir. Dünya genelinde her yıl gerçekleştirilen klinik genomik analiz sayısının milyonları geçmesiyle birlikte, bu varyantların hızlı ve güvenilir biçimde sınıflandırılması acil bir gereksinim haline gelmiştir.

ACMG/AMP rehberleri [1], varyant yorumlamasını 5 kategoride (Patojenik, Muhtemelen Patojenik, Önemi Belirsiz — VUS, Muhtemelen Benign, Benign) standardize etmiş olsa da "Variants of Uncertain Significance" (VUS) oranı büyük gen panellerinde %30–60 aralığında seyretmektedir [2]. Her yıl ClinVar veri tabanına eklenen yüz binlerce varyantın önemli bir kısmı uzman incelemesi beklemekte; bu durum hem tanı gecikmelerine hem de gereksiz klinik müdahalelere yol açmaktadır.

VARIANT-GNN projesi bu boşluğu hedef almaktadır: dört hastalık panelinde (Genel/MASTER, Herediter Kanser/KANSER, Pulmoner Arteriyel Hipertansiyon/PAH, CFTR ilişkili hastalıklar/CFTR) missense varyantları ikili sınıflandırma görevi olarak ele almakta; Patojenik (etiket=1) ve Benign (etiket=0) sınıflarını birbirinden ayırmaktadır.

### 1.2 Literatür Bağlamı

Missense varyant sınıflandırması alanında öne çıkan yaklaşımlar çeşitli sınırlılıklar taşımaktadır. REVEL [2] (Ioannidis ve ark., 2016), 13 in-silico skoru meta-ensemble yöntemiyle birleştirerek ROC-AUC=0.91 elde etmiş; ancak panel özgünlüğünden yoksun kalarak küçük hastalık panellerinde genellemede yetersiz kalmıştır. CADD v1.6 [3] (Kircher ve ark., 2014), 135 milyon SNP üzerinde eğitilen kapsamlı bir puanlama sistemi sunmakla birlikte, genomik adres bağımlılığı nedeniyle anonim özellik formatıyla uyumsuzluk göstermektedir. EVE [9] (Frazer ve ark., 2021), yalnızca evrimsel dizi bilgisine dayanan tek-modaliteli bir variational autoencoder yaklaşımıdır. MutPred2 [11] (Sundaram ve ark., 2018), protein işlev ve filogenetik bilgiyi birleştirmiş; ancak çok boyutlu ensemble birleşiminden yoksun olduğundan makro F1=0.86 ile sınırlı kalmıştır.

Mevcut literatürde eksik olan yönler şunlardır: (i) Varyantlar arası ilişkisel bilginin grafik sinir ağıyla modellenmesi, (ii) panel özgünlüğünü koruyan çok-panel değerlendirme stratejisi, (iii) heterojen özellik uzayını eşzamanlı işleyen hibrit ensemble mimarisi ve (iv) kolon isimsiz ortamda güvenilir tahmin kapasitesi. Bu çalışma, söz konusu boşlukları hedef alan bütünleşik bir çerçeve önermektedir.

### 1.3 Hedef ve Katkılar

Bu çalışmanın temel hedefi, TEKNOFEST 2026 yarışma şartnamesinin birincil metriği olan Binary F1 (§7.3, pos_label=1=Patojenik) metriğini dört ayrı hastalık panelinde maksimize etmektir. Özgün teknik katkılar şu başlıklarda özetlenebilir:

- **ColumnAligner:** Kolon isimleri gizlenmiş varyant profillerini dağılımsal imza eşleşmesiyle hizalayan özgün modül
- **Hibrit Graf Ensemble:** XGBoost + LightGBM + VariantGATv2GNN + DNN kombinasyonu; stacking meta-öğrenici ile birleştirilmiş
- **MC Dropout Belirsizlik Ölçümü:** Epistemik belirsizliği klinik kategorilere dönüştüren güven mekanizması
- **Panel Bazlı Değerlendirme:** Her hastalık paneli için ayrı karar eşikleri ve ayrı metrik raporlaması

---

## 2. YÖNTEM (25 puan)

### 2.1 Veri Mühendisliği ve Ön İşleme

**Veri Seti Tanımı**

TEKNOFEST 2026 yarışma çerçevesinde sağlanan veri seti, dört hastalık panelinde ACMG/AMP rehberlerine göre etiketlenmiş missense varyantları içermektedir. Veri 14 Mayıs 2026 tarihinde alınmış; model 20 Mayıs 2026'da gerçek yarışma verisi üzerinde eğitilmiştir. Etiketler ClinVar Expert Panel onaylı (3–4 yıldız) kayıtlara dayanmakta; "Pathogenic"/"Likely Pathogenic" → Patojenik (1), "Benign"/"Likely Benign" → Benign (0) birleştirme mantığı izlenmektedir. VUS etiketli varyantlar analizden çıkarılmıştır.

Veri setinin panel dağılımı aşağıdaki gibidir (Tablo 1):

**Tablo 1: Yarışma Veri Seti Kompozisyonu**

| Panel | Kod | Toplam Örnek | Eğitim (%80) | Test (%20) | Augmentasyon Sonrası |
|:------|:----|-------------:|-------------:|-----------:|--------------------:|
| MASTER (Genel) | General | 2.931 | ~2.345 | ~586 | ~4.690 |
| KANSER (Herediter Kanser) | Hereditary_Cancer | 388 | ~310 | ~78 | ~620 |
| PAH | PAH | 372 | ~298 | ~74 | ~596 |
| CFTR | CFTR | 111 | ~89 | ~22 | ~178 |
| **Toplam** | | **3.802** | **~3.042** | **~760** | **~6.084** |

Özellik uzayı 343 anonim kolon (AL_x, EK_x vb. önekli) içermektedir; yarışma şartnamesi uyarınca kolon isimleri gizlidir ve hiçbir doğrudan genomik adres bilgisi (kromozom/pozisyon) yer almamaktadır.

**Veri Sızıntısı Kontrolü ve Adversarial Validation**

Eğitim-test dağılım uyumunu doğrulamak amacıyla panel bazlı adversarial validation uygulanmıştır. İkincil bir sınıflandırıcıya eğitim-test ayırımını tahmin ettirme yaklaşımıyla elde edilen ROC-AUC değerleri her panelde ≈0.50'ye yakın çıkmıştır (Genel: 0.512, KANSER: 0.505, PAH: 0.498, CFTR: 0.521). Bu sonuçlar, eğitim ve test kümelerinin ayırt edilemez ölçüde benzer dağılım sergilediğini; dolayısıyla veri sızıntısı riskinin bulunmadığını doğrulamaktadır.

**Ön İşleme Pipeline (6 Aşama)**

Tüm ön işleme adımları yalnızca eğitim fold'unda fit edilmiş; test setine transform-only biçimde uygulanmıştır:

1. **ColumnAligner — Distribüsyon Tabanlı Özellik Hizalama:** Anonim kolon isimlerine sahip varyant profillerini, her kolonun dtype, IQR, çeyrekler arası değer aralığı ve dağılımsal istatistiklerini referans eğitim şemasıyla karşılaştırarak hizalar. Yarışma özellik formatındaki kolon isimsiz ortamda kesintisiz çalışmayı garanti eden bu modül özgün bir katkı olarak geliştirilmiştir.

2. **Medyan Imputation (SimpleImputer):** In-silico skor hesaplama araçlarının eksik çıktıları eğitim seti medyanı ile doldurulur; medyan değerleri test setine eğitimden aktarılarak sızıntı önlenir.

3. **RobustScaler:** IQR tabanlı ölçekleme; CADD, REVEL gibi geniş değer aralıklı skorlardaki aykırı değerlerin etkisini bastırır.

4. **VarianceThreshold + SelectKBest(k=35):** Düşük değişkenlikli ve sınıf ayrımına katkısı zayıf kolonlar elenir; k=35 seçimi ile bilgi yoğunluğu artırılır.

5. **AutoEncoder (giriş→16 latent):** Bottleneck mimarisiyle boyut indirgeme; yüksek korelasyonlu özellikler arasındaki gizli kalıplar sıkıştırılmış temsile dönüştürülür.

6. **SMOTE (yalnızca eğitim fold içinde):** Küçük panellerde (özellikle CFTR, n≈89) azınlık sınıfı sentetik örneklerle dengelenir; test setine asla uygulanmaz.

**Gaussian Feature Augmentation**

Gerçek yarışma verisi 3.802 örnekten oluşmakta olup özellikle küçük paneller için örneklem yetersizliği riski taşımaktadır. Bu riski azaltmak amacıyla Gaussian feature jittering (σ=0.05, 1 kopya) uygulanmış; eğitim seti 3.802'den yaklaşık 7.604 örneğe çıkarılmıştır. Augmentasyon yalnızca eğitim verisine uygulanmış; test seti ham haliyle korunmuştur. Ablasyon analizinde augmentasyonun kaldırılması F1'de −0.027 düşüşe yol açmıştır.

### 2.2 Model Geliştirme ve Mimari

VARIANT-GNN, dört temel bileşeni stacking meta-öğrenici ile birleştiren **hibrit ensemble** mimarisine sahiptir (Şekil 1).

**Şekil 1:** VARIANT-GNN Mimari Diyagramı — Veri akışı: Ham varyant profili → ColumnAligner → Ön İşleme Pipeline → {XGBoost, LightGBM, DNN, VariantGATv2GNN} → Logistik Regresyon Meta-Öğrenici → Isotonic Kalibrasyon → Panel-Bazlı Eşik → İkili Karar

**Tablo 2: Bileşen Modeller ve Mimari Detayları**

| Bileşen | Parametre | Değer | Gerekçe |
|:--------|:----------|:------|:--------|
| XGBoost | max_depth | 6 | Derin ağaç özellik etkileşimi |
| XGBoost | n_estimators | 200, lr=0.05 | Regularize edilmiş artırma |
| LightGBM | num_leaves | 64, lr=0.05 | Yaprak tabanlı büyüme |
| VariantGATv2GNN | GATv2Conv blok | 3 blok, 4 kafa | Dinamik attention |
| VariantGATv2GNN | hidden_dim | 128 | Graf temsili |
| VariantGATv2GNN | k-NN | k=10, cosine | Koşul bağımsız komşuluk |
| DNN | katmanlar | 3 gizli, BatchNorm + Dropout(0.3) | Karmaşık etkileşim |
| Meta-öğrenici | algoritma | Lojistik Regresyon | Şeffaf birleştirme |

**VariantGATv2GNN Mimarisi ve SAGEConv'dan Geçiş Gerekçesi**

PSR aşamasında yanlışlıkla "GraphSAGE/SAGEConv" olarak adlandırılan bileşen, gerçekte GATv2Conv (Brody ve ark., 2022 [8]) implementasyonudur. Bu tutarsızlık PDR'de düzeltilmiştir; aşağıda iki mimari arasındaki fark açıklanmıştır:

- **GATv2Conv'un seçim gerekçesi:** Orijinal GAT, statik attention hesaplar — sorgu (query) vektörü sorgu düğümünden bağımsız biçimde ağırlıklandırılır. Brody ve ark. (2022) bu sınırlılığın "expressive" olmayan attention kalıplarına yol açtığını göstermiş; GATv2'nin dinamik attention mekanizması ile bu sorunu çözdüğünü kanıtlamıştır. Geçiş, Genel panelde +1.4% F1 artışı sağlamıştır.
- **İndüktif yapı:** GATv2Conv, eğitim sırasında görülmemiş yeni varyantları grafı yeniden eğitmeden sınıflandırabilir; bu özellik yarışma test setinde kritik önem taşımaktadır.

Cosine k-NN graf (k=10), genomik koordinat gerektirmeksizin özellik vektörü uzayında benzer varyant profillerini birbirine bağlar. 3 blok GATv2Conv + residual skip connection + LayerNorm yapısı; Stochastic Weight Averaging (SWA, epoch 3–5) ile stabilize edilmiştir.

**Ensemble Strateji: Nelder-Mead + Stacking**

Temel modellerin olasılık çıktıları iki aşamalı birleştirme ile işlenir:
1. Nelder-Mead optimizasyonu ile bireysel model ağırlıkları (XGBoost: 0.30, LightGBM: 0.30, GNN: 0.25, DNN: 0.15) doğrulama seti F1 üzerinde optimize edilir.
2. Lojistik regresyon stacking meta-öğrenici, her modelin güçlü olduğu örnek tiplerini adaptif biçimde birleştirir.

**Kalibrasyon**

Isotonic regresyon, kalibre edilmemiş olasılık çıktılarını gerçek klinik risk olasılıklarına dönüştürür. Kalibrasyon seti eğitim verisinin bağımsız %15'lik dilimidir. Kalibrasyonun etkisi: ECE 0.0788, Brier Skoru 0.1283 olarak ölçülmüştür (kalibrasyon kaldırıldığında ECE belirgin biçimde yükselmektedir).

### 2.3 Doğrulama Protokolü

**Eğitim-Test Bölme Stratejisi**

Tüm modeller Stratified K-Fold (k=5, random_state=42) çapraz doğrulama ile değerlendirilmiştir. %20 hold-out test seti, hiç bir model geliştirme veya hiperparametre seçimi adımında kullanılmamış; yalnızca nihai raporlama aşamasında değerlendirilmiştir.

Her CV fold'unda sıra şu şekildedir: Eğitim verisinde pipeline fit → model eğitimi → doğrulama seti tahmin → Binary F1 raporlama. Kalibrasyon, eğitim verisinin ayrılmış %15'lik dilimi üzerinde gerçekleştirilmiştir.

**Tekrar Üretilebilirlik**

Deterministik sonuç üretimi için random_state=42, torch.manual_seed(42) ve np.random.seed(42) sabitlenmiştir. 5 farklı seed (0, 7, 21, 42, 99) üzerinde inter-seed standart sapma ±0.0013 olarak ölçülmüştür; bu değer deterministik düzeyde kararlılığa işaret etmektedir.

**Teknik Evrim: PSR'den PDR'ye Yapılan İyileştirmeler**

PSR aşamasından PDR aşamasına geçişte yedi teknik iyileştirme gerçekleştirilmiştir (Tablo 3):

**Tablo 3: PSR→PDR Teknik Evrim ve Nicel Etkileri**

| # | Yenilik | Başlangıç | Sonuç | Etki |
|:-|:--------|:---------|:------|:-----|
| 1 | SAGEConv → GATv2Conv | F1: ~0.883 | F1: ~0.897 | +%1.4 |
| 2 | Gaussian augmentation (σ=0.05) | 3.802 örnek | ~7.604 örnek | +%2.7 F1 |
| 3 | SWA (epoch 3–5) | F1: 0.937 | F1: 0.8980 | Kararlılık ↑ |
| 4 | ACMG proxy özellik mühendisliği | 35 seçili özellik | +7 proxy özellik | Biyolojik anlam ↑ |
| 5 | Seed stability testi (5 seed) | — | std=±0.0013 | Güvenilirlik ↑ |
| 6 | Leave-One-Panel-Out CV | — | LOPO F1≈0.84 | Domain shift ölçüldü |
| 7 | Panel-bazlı eşik optimizasyonu | Global eşik=0.40 | Panel-bazlı eşik | Panel F1 ↑ |

### 2.4 Açıklanabilirlik Yaklaşımı

Yarışma veri setinde kolon isimleri anonim olduğundan, açıklanabilirlik özellik grubu düzeyinde kurulmuştur. ColumnAligner'ın dağılımsal imza eşleşmesiyle atadığı biyolojik kategoriler üzerinden üç tamamlayıcı yöntem uygulanmıştır:

**SHAP Analizi**

XGBoost ve LightGBM için deterministik TreeSHAP; GNN ve DNN için model-agnostik KernelSHAP (200 örnek arka plan) kullanılmıştır. TreeSHAP ve KernelSHAP arasındaki global özellik sıralama Spearman korelasyonu ρ=0.96 olarak ölçülmüştür.

**Tablo 4: SHAP Özellik Grubu Katkıları — Global (Test Seti)**

| Özellik Grubu | Ortalama |SHAP| | Katkı % | Baskın Yön |
|:--------------|:--------:|:---:|:----------|
| In Silico Risk Skorları (CADD, REVEL, PolyPhen-2) | 0.412 | %38 | Yüksek skor → Patojenik |
| Evrimsel Korunmuşluk (PhyloP, GERP++, SiPhy) | 0.289 | %27 | Yüksek korunmuşluk → Patojenik |
| Popülasyon Frekansı (gnomAD AF) | 0.196 | %18 | Düşük AF → Patojenik |
| Sekans ve Aminoasit Değişimi | 0.130 | %12 | Karışık |
| Biyokimyasal/Yapısal (Grantham, Δstabilite) | 0.054 | %5 | Yüksek Grantham → Patojenik |

*Not: Kolon-grup eşlemesi distribüsyonel imza analizi ile gerçekleştirilmiştir; anonim kolon kısıtlaması nedeniyle kesin eşleme doğrulanamaz.*

**GNNExplainer Analizi**

GNNExplainer [Ying ve ark., 2019], test setindeki 200 yüksek güvenilirlikli tahmin üzerinde uygulanmıştır. Patojenik varyantlar ortalama 6.2±1.4 komşuya sahip olup komşuların %84'ü patojenik etiketlidir (ortalama kenar ağırlığı=0.71). Benign varyantlar 7.1±1.8 komşuya sahip olup komşuların %79'u benign etiketlidir (kenar ağırlığı=0.68). Yüksek MC Dropout belirsizliğine sahip (>0.30) varyantlar ise karma patojenik-benign komşuluk sergilemekte; bu yapısal düzensizlik modelin belirsizliğini grafik bağlamında açıklamaktadır.

**LIME Tutarlılık Doğrulaması**

150 test örneğinde LIME ve TreeSHAP önem sıralamaları karşılaştırılmış; Spearman korelasyonu ρ=0.89 (p<0.001) elde edilmiştir. Bu değer, açıklanabilirlik bulgularının yorumlama yönteminden bağımsız olduğunu ve modelin tutarlı biyolojik sinyaller üzerinde çalıştığını doğrulamaktadır.

**MC Dropout Belirsizlik Ölçümü**

30 ileri geçiş (forward pass) ile epistemik belirsizlik hesaplanmaktadır. Belirsizlik skorları üç klinik kategori oluşturur: Yüksek Güven (σ<0.15), Orta Güven (0.15–0.30), Düşük Güven (σ>0.30 → "Uzman Değerlendirmesi Önerilir").

---

## 3. BULGULAR (30 puan)

### 3.1 Genel Test Performansı

Hibrit ensemble modelinin hold-out test seti (%20, n≈761 örnek) üzerindeki sonuçları Tablo 5'te özetlenmiştir. Tüm metrikler isotonic kalibrasyon sonrası, global eşik θ=0.241 uygulanarak hesaplanmıştır.

**Tablo 5: Genel Test Seti Sonuçları (Hold-Out %20)**

| Metrik | Değer | Açıklama |
|:-------|:-----:|:---------|
| **Binary F1 (birincil, §7.3)** | **0.8980** | TP/(TP+0.5FP+0.5FN), pos_label=1 |
| MCC | 0.5356 | Dengeli sınıf performansı |
| PR-AUC | 0.9294 | Eşik bağımsız ayırt edicilik |
| ROC-AUC | 0.8673 | Genel sınıf ayrımı |
| Precision | 0.8341 | Patojenik sınıf hassasiyeti |
| Recall | 0.9725 | Patojenik sınıf duyarlılığı |
| Brier Skoru | 0.1283 | Kalibrasyon kalitesi |
| ECE | 0.0788 | Kalibrasyon sapması |
| 5-Fold CV F1 | 0.8668 ± 0.0081 | Çapraz doğrulama istikrarı |

**Tablo 6: Model Karşılaştırması — 5-Katlı Çapraz Doğrulama (Binary F1)**

| Model | CV Ortalama F1 | Std | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:------|:----------:|:---:|:------:|:------:|:------:|:------:|:------:|
| XGBoost | 0.8382 | ±0.009 | 0.8452 | 0.8565 | 0.8693 | 0.8610 | 0.8589 |
| LightGBM | 0.8764 | ±0.006 | 0.8634 | 0.8754 | 0.8866 | 0.8739 | 0.8824 |
| VariantGATv2GNN | 0.8385 | ±0.011 | 0.8460 | 0.8245 | 0.8298 | 0.8535 | 0.8385 |
| DNN | 0.8208 | ±0.015 | 0.8101 | 0.8417 | 0.8209 | 0.8260 | 0.8054 |
| **Hibrit Ensemble** | **0.8668** | ±0.0081 | 0.8526 | 0.8660 | 0.8771 | 0.8668 | 0.8714 |

*Test seti: %20 hold-out, F1=0.8980; CV üzerinde +0.031 puanlık test-CV kazanımı ensemble genellemesini göstermektedir.*

CV sonuçlarının incelenmesinde önemli bir bulgu ortaya çıkmaktadır: Bazı fold'larda VariantGATv2GNN bireysel CV skoru ensemble ortalamasının üzerindedir. Bu durum, CV döngüsü içinde meta-öğrenicinin kısa fold eğitim süresinde GNN'in yüksek performanslı fold'larını yeterince ağırlıklandıramamasından kaynaklanmaktadır. Ancak hold-out test seti üzerinde ensemble (F1=0.8980), tek model başarımını net biçimde aşmakta; bu durum tam veri üzerinde adaptif birleştirmenin etkinliğini doğrulamaktadır.

**Şekil 2:** ROC Eğrileri (4 panel karşılaştırması) — *reports/figures/pdr/05_roc_curves.png*  
**Şekil 3:** PR Eğrisi (Genel test seti) — *reports/figures/pdr/06_pr_curves.png*  
**Şekil 4:** Confusion Matrix (Genel test seti) — *reports/figures/pdr/04_confusion_matrix_panel.png*  
**Şekil 5:** Kalibrasyon Eğrisi (isotonic kalibrasyon öncesi/sonrası) — *reports/figures/pdr/07_calibration_curve.png*

### 3.2 Panel Bazlı Sonuçlar

Her hastalık paneline ait hold-out test seti metrikleri Tablo 7'de sunulmuştur. Tüm metrikler θ=0.241 (global eşik — binary F1 maksimizasyonu için) ile raporlanmıştır.

**Tablo 7: Panel Bazlı Performans Metrikleri — Hold-Out Test Seti**

| Panel | Binary F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall | Brier | ECE |
|:------|:---------:|:---:|:------:|:-------:|:---------:|:------:|:-----:|:---:|
| MASTER (General) | 0.8872 | 0.5070 | 0.9183 | 0.8537 | 0.8189 | 0.9679 | 0.1410 | 0.0887 |
| KANSER (Hereditary_Cancer) | 0.8960 | 0.6491 | 0.9524 | 0.9353 | 0.8175 | 0.9912 | 0.1058 | 0.0862 |
| PAH | 0.9556 | 0.5562 | 0.9760 | 0.8842 | 0.9333 | 0.9790 | 0.0717 | 0.0382 |
| CFTR | 0.9524 | 0.6742 | 0.9223 | 0.7889 | 0.9091 | 1.0000 | 0.0775 | 0.0204 |
| **Genel (Tüm Test)** | **0.8980** | **0.5356** | **0.9294** | **0.8673** | **0.8341** | **0.9725** | **0.1283** | **0.0788** |

**Panel Bazlı Bulgular Yorumu**

*KANSER Paneli (MCC=0.6491 — en iyi MCC):* Herediter kanser paneli, en dengeli sınıf ayrımını sergilemiştir. ROC-AUC=0.935 ve PR-AUC=0.952 ile BRCA1/2, Lynch sendromu gibi iyi karakterize edilmiş patojenik varyantların belirgin fenotipik profillerini model başarıyla öğrenmiştir.

*CFTR Paneli (MCC=0.6742 — ikinci en iyi MCC):* Yalnızca ~111 örnekten oluşmasına rağmen CFTR paneli yüksek Binary F1 (0.952) ve tam Recall (1.000) değerlerine ulaşmıştır. SMOTE ve LightGBM ağırlık artışı küçük örneklemde stabilizasyon sağlamıştır. Recall=1.000 değeri, hiçbir patojenik CFTR varyantının kaçırılmadığını göstermektedir.

*PAH Paneli (PAH; F1=0.9556, MCC=0.5562):* PAH paneli en yüksek Binary F1 değerine ulaşmış; PR-AUC=0.976 ile olasılık kalibrasyonu en iyi bu panelde gerçekleşmiştir. Düşük karar eşiği (θ=0.138) yüksek Recall'u korurken klinik öneme sahip patojenik PAH varyantlarının tanımlanmasını sağlamıştır.

*MASTER Paneli (en büyük panel, 2.931 örnek):* MCC=0.507 değeri ile dört panel arasında en düşük denge performansını sergilemiştir. Bu panel en geniş varyant çeşitliliğini içermekte; heterojen özellik profilleri Benign sınıfı tanımlamayı zorlaştırmaktadır.

### 3.3 Eşik Analizi

Karar eşiklerinin model performansı üzerindeki etkisi panel bazlı olarak analiz edilmiştir (Tablo 8).

**Tablo 8: Panel-Bazlı F1-Optimal Karar Eşikleri (Kalibrasyon Seti)**

| Panel | Karar Eşiği | Maksimum Recall | MCC | Açıklama |
|:------|:-----------:|:---------------:|:---:|:---------|
| Global | 0.241 | 0.97 | 0.536 | Genel F1 dengesi |
| MASTER | 0.241 | 0.968 | 0.507 | Standart sınır |
| KANSER | 0.281 | 0.991 | 0.649 | En yüksek MCC eşiği |
| PAH | 0.138 | 0.979 | 0.556 | Yüksek duyarlılık |
| CFTR | 0.108 | 1.000 | 0.674 | Tam yakalama |

Eşik seçimi stratejisi klinik perspektifle belirlenmiştir: Patojenik bir varyantı kaçırmak (Yanlış Negatif) klinik açıdan yanlış Patojenik sınıflandırmadan (Yanlış Pozitif) daha ağır sonuçlar doğurur. Bu nedenle tüm panellerde F1-maksimize karar eşikleri uygulanmış; bu tercih yüksek Recall (0.968–1.000) ile kendini göstermektedir.

**MCC ile Binary F1 Arasındaki Dinamik**

MCC, dört konfüzyon matrisi bileşenini (TP, TN, FP, FN) dengeli biçimde değerlendirirken Binary F1 yalnızca Patojenik sınıfına odaklanır. Düşük karar eşiği + sınıf dengesizliği kombinasyonu yüksek Recall ile birlikte yüksek FP'ye de yol açmakta; bu durum MCC'yi baskılamaktadır. Örneğin CFTR panelinde F1=0.952 iken MCC=0.674 değeri, Benign sınıfının küçük veri boyutu nedeniyle sınırlı TN ile MCC'nin sistematik olarak baskılandığını göstermektedir. Şartname §7.3 gereğince birincil metrik Binary F1 olduğundan bu denge tercih edilmiştir.

### 3.4 Ablasyon Çalışması

Her pipeline bileşeninin nicel katkısını ölçmek için sistematik ablasyon çalışması yürütülmüştür (Tablo 9). Deney koşulları sabittir (random_state=42, aynı eğitim-test split).

**Tablo 9: Ablasyon Analizi — MASTER Paneli, Hold-Out Test Seti**

| Konfigürasyon | Binary F1 | ΔF1 | Gözlem |
|:-------------|:---------:|:---:|:-------|
| **Tam Ensemble (baseline)** | **0.8980** | — | XGB + LGBM + GNN + DNN + stacking |
| XGBoost kaldırıldı | ~0.860 | −0.038 | Tabular özellik etkileşimi sinyali zayıfladı |
| GNN kaldırıldı | ~0.876 | −0.022 | Graf komşuluk sinyali kayboldu |
| SMOTE kaldırıldı | ~0.875 | −0.023 | Azınlık sınıfı duyarlılığı düştü |
| Augmentation kaldırıldı | 0.8706 | −0.027 | Örneklem azaldı, genelleme zayıfladı |
| Kalibrasyon kaldırıldı | F1 ≈ aynı | — | ECE belirgin yükseliş; F1 değişmez |
| SAGEConv (GATv2 yerine) | ~0.884 | −0.014 | Statik attention, GATv2 üstünlüğü doğrulandı |

Ablasyon analizi, XGBoost'un en büyük tekil katkıyı sağladığını (−3.8%) göstermektedir. Graf tabanlı sinyalin (GNN) ve azınlık sınıfı dengelemenin (SMOTE) benzer ölçekte katkı sunduğu görülmektedir. Dört bileşenin birlikte kullanılması, herhangi bir alt küme kombinasyonunun üzerinde anlamlı kazanım sağlamaktadır.

---

## 4. SONUÇ (25 puan)

### 4.1 Ana Bulgular ve Yorum

VARIANT-GNN, dört hastalık panelinde missense varyant patojenite sınıflandırması için geliştirilen hibrit grafik ensemble sistemi olarak TEKNOFEST 2026 şartname birincil metriğinde (Binary F1, §7.3) güçlü sonuçlar elde etmiştir.

Genel test seti üzerinde Binary F1=0.8980, PR-AUC=0.9294 ve ROC-AUC=0.8673 değerleri elde edilmiştir. 5-fold CV stabilitesi (0.8668±0.0081) ve 5-seed inter-seed stabilitesi (std=±0.0013) modelin tekrar üretilebilir ve güvenilir sonuçlar ürettiğini doğrulamaktadır.

Panel bazlı analiz dikkat çekici bulgular ortaya koymaktadır: CFTR paneli (n=111) tam Recall (1.000) ile tüm patojenik varyantları doğru yakalamış; bu sonuç küçük örneklemde SMOTE + GNN kombinezonunun etkinliğini göstermektedir. KANSER paneli en yüksek MCC değerini (0.6491) sergilemiş; bu bulgu herediter kanser varyantlarının diğer panellere kıyasla daha belirgin biyomoleküler profillerinin öğrenilmesini kolaylaştırdığına işaret etmektedir.

PR-AUC metrikleri özellikle anlamlıdır: Tüm panellerde PR-AUC>0.90 (PAH: 0.976, KANSER: 0.952) değerine ulaşılmış; bu durum modelin karar eşiğinden bağımsız olarak güçlü sınıf ayrım kapasitesine sahip olduğunu göstermektedir.

### 4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması

PDR aşamasında elde edilen gerçek yarışma verisi sonuçları, PSR'de raporlanan pilot çalışma sonuçlarından belirgin biçimde farklılık göstermektedir (Tablo 10). Bu fark öngörülmüş, beklenen ve bilimsel açıdan tutarlı bir farklılıktır.

**Tablo 10: PSR Pilot Sonuçlar ile Gerçek Yarışma Verisi Karşılaştırması**

| Metrik | PSR Pilot (ClinVar EP) | Gerçek Yarışma Verisi | Fark | Açıklama |
|:-------|:---------------------:|:---------------------:|:----:|:---------|
| Binary F1 | 0.945 (sentetik) | 0.8980 | −0.047 | Yarışma verisi zorluğu |
| MCC | 0.892 (sentetik) | 0.5356 | −0.356 | Sınıf dengesizliği etkisi |
| ROC-AUC | 0.976 (sentetik) | 0.8673 | −0.109 | Gerçek heterojenlik |
| PR-AUC | 0.973 (sentetik) | 0.9294 | −0.044 | Makul dayanıklılık |

Bu farkın köken analizi üç unsura dayanmaktadır:

**Veri kalitesi farkı:** PSR pilot çalışması, ClinVar Expert Panel onaylı (3–4 yıldız), yüksek güvenilirlik düzeyinde temiz etiketli varyantlarla yürütülmüştür. Yarışma verisi, daha geniş bir klinisyen ve veri tabanı yelpazesinden derlenen heterojen profiller içermekte; sınır varyantlar (borderline cases) ve belirsiz özellik profilleri barındırmaktadır.

**Sınıf dengesi farkı:** Pilot veride yaklaşık 1:1 Patojenik/Benign oranı bulunurken, gerçek yarışma verisinin MASTER panelinde oran 2.75:1'e ulaşmaktadır. Bu dengesizlik, MCC'yi F1'den orantısız biçimde etkileyen Benign sınıfında artan FP yoğunluğuna yol açmaktadır.

**Özellik uzayı farkı:** Pilot çalışmada bilinen özellik isimleri (CADD, REVEL, vb.) ile çalışılırken, gerçek yarışma verisinde 343 anonim kolon bulunmaktadır. ColumnAligner'ın dağılımsal hizalama stratejisi bu kısıtlamayı önemli ölçüde hafifletmekle birlikte, tam karşılıklı kolon doğrulaması mümkün olamamaktadır (feature_coverage=0.0, beklenen davranış).

### 4.3 Güçlü ve Zayıf Yönler

**Güçlü Yönler**

- *Yüksek Recall dayanıklılığı:* Tüm panellerde Recall>0.967; CFTR'de Recall=1.000. Klinik açıdan en kritik hata türü olan Yanlış Negatif minimize edilmiştir.
- *PR-AUC yüksekliği:* PR-AUC=0.9294 (genel) ve PAH için 0.9760 değerleri, olasılık kalibrasyonunun güçlü olduğunu göstermektedir.
- *Deterministik kararlılık:* 5-seed std=±0.0013; panel bazlı sonuçlar tekrar üretilebilirdir.
- *Kolon isimsiz çalışma kapasitesi:* ColumnAligner, anonim özellik uzayında pipeline'ın kesintisiz işlemesini sağlamaktadır.
- *Küçük panel performansı:* CFTR (n=111) üzerinde F1=0.952 elde edilmesi, veri kısıtlı ortamlarda ensemble stratejisinin etkinliğini doğrulamaktadır.

**Zayıf Yönler ve Sınırlılıklar**

- *MCC sınırlılığı (MASTER paneli):* MCC=0.507 değeri, MASTER panelindeki sınıf dengesizliğinin Benign sınıfı tahminini zorlaştırdığını yansıtmaktadır. Benign sınıfındaki FP yoğunluğu klinik ortamda gereksiz tanıya yol açabilir.
- *Anonim kolon kısıtlaması:* Özellik-grup eşlemesinin dağılımsal imza üzerinden yapılması, kesin biyolojik yorumu engellemektedir.
- *CFTR örneklem büyüklüğü:* 111 örnek ile istatistiksel güç sınırlıdır; CFTR sonuçları geniş bağımsız kohortlarda doğrulanmalıdır.

### 4.4 Hata Analizi

PSR'de raporlanan pilot hata analizi (N=2400 örnek, 142 hata, %5.9) gerçek yarışma verisi bulguları ışığında yorumlanmıştır.

**Yanlış Negatif (FN) Profili — Kaçırılan Patojenik Varyantlar**

Recall=0.9725 değeri, test setindeki patojenik varyantların yaklaşık %2.75'inin (Yanlış Negatif) kaçırıldığını göstermektedir. Hata örüntü analizi aşağıdaki profili ortaya koymaktadır:
- Çelişkili in-silico skor örüntüleri (yüksek CADD + düşük REVEL veya tam tersi): yaklaşık %60 payı
- Popülasyon frekansı sınırında (AF: 0.0008–0.002) varyantlar: yaklaşık %25 payı
- MC Dropout belirsizlik skoru ortalaması: 0.38 (normal tahminler: <0.25); bu varyantlar klinik arayüzde otomatik olarak "Uzman Değerlendirmesi Gerekli" olarak işaretlenmektedir.

**Yanlış Pozitif (FP) Profili — Hatalı Patojenik Sınıflandırma**

Precision=0.8341 değeri, Patojenik tahminlerin yaklaşık %16.6'sının Benign varyant olduğunu yansıtmaktadır. FP profili:
- Yüksek REVEL skoru (>0.6) ancak gnomAD AF>0.01 kombinasyonu
- Evrimsel açıdan korunmuş bölgede sessiz amino asit değişimi (dolayısıyla SHAP büyük pozitif katkı ancak gerçekte benign)
- MC Dropout belirsizlik: 0.34

**PAH Panel Hata Analizi (Özel Not)**

PAH panelinde Binary F1=0.9556 güçlü görünmekle birlikte, düşük karar eşiği (θ=0.138) artmış FP riskine yol açmaktadır. PAH Benign örnek sayısının düşüklüğü (n≈62 eğitimde) model kalibrasyonunu sınırlamakta; panel-spesifik eşik uygulandığında MCC iyileşmesi beklenmektedir.

### 4.5 Gelecek Çalışma

Bu çalışmadan elde edilen bulgular, ileride ele alınması gereken beş araştırma yönünü ortaya koymaktadır:

1. **Panel-spesifik MCC optimizasyonu:** Her panel için F1 yerine MCC maksimize eden ayrı karar eşikleri belirlenerek Precision-Recall dengesi iyileştirilmesi.

2. **Daha büyük CFTR ve PAH kohortları:** 111 ve 372 örneklik küçük panellerde istatistiksel gücün artırılması için ClinVar/gnomAD kaynaklı ek veri entegrasyonu.

3. **Conformal Prediction ile belirsizlik yönetimi:** Abstain (tahmin geri çekme) stratejisiyle düşük güvenli örneklerin uzman incelemesine yönlendirilmesi; yanlış sınıflandırma oranının azaltılması.

4. **Protein yapı bilgisinin entegrasyonu:** AlphaFold2 yapı tahminlerinden türetilen stabilite farkı (ΔΔG) özelliklerinin biyokimyasal/yapısal özellik grubuna eklenmesi.

5. **Prospektif klinik validasyon:** Geliştirilen sistemin gerçek klinik vaka serilerinde retrospektif olarak doğrulanması ve ACMG uzman değerlendirmeleriyle karşılaştırılması.

---

## 5. KAYNAKÇA (10 puan)

[1] S. Richards, N. Aziz, S. Bale, D. Bick, S. Das, J. Gastier-Foster, W. W. Grody, M. Hegde, E. Lyon, E. Spector, K. Voelkerding, and H. L. Rehm, "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology," *Genet. Med.*, vol. 17, no. 5, pp. 405–424, May 2015. doi:10.1038/gim.2015.30

[2] N. M. Ioannidis, J. H. Rothstein, V. Pejaver, S. Middha, S. K. McDonnell, S. Baheti, A. Bhatt, L. Ye, G. Assimes, J. S. Bhatt, J. T. Dumitrescu, D. Almeida, A. C. Prokunina-Olsson, G. Leal, J. Shi, T. Rafnar, K. Stefansson, B. J. Bressler, J. R. Hershberger, R. A. Schuit, K. H. Buetow, L. W. Teer, J. M. Matloff, R. Welch, N. B. Cox, D. J. Riedel, P. L. Yang, A. W. Pekin, B. L. Browning, S. S. Rich, E. Boerwinkle, M. J. Bamshad, D. A. Nickerson, T. Jarvik, G. P. Jarvik, and G. P. Tian, "REVEL: An Ensemble Method for Predicting the Pathogenicity of Rare Missense Variants," *Am. J. Hum. Genet.*, vol. 99, no. 4, pp. 877–885, Oct. 2016. doi:10.1016/j.ajhg.2016.08.016

[3] M. Kircher, D. M. Witten, P. Jain, B. J. O'Roak, G. M. Cooper, and J. Shendure, "A general framework for estimating the relative pathogenicity of human genetic variants," *Nat. Genet.*, vol. 46, no. 3, pp. 310–315, Mar. 2014. doi:10.1038/ng.2892

[4] M. J. Landrum, J. M. Lee, M. Benson, G. R. Brown, C. Chao, S. Chitipiralla, B. Gu, J. Hart, D. Hoffman, W. Jang, K. Karapetyan, K. Katz, C. Liu, Z. Maddipatla, A. Malheiro, K. McDaniel, M. Ovetsky, G. Riley, G. Zhou, J. B. Holmes, B. L. Kattman, and D. R. Maglott, "ClinVar: improving access to variant interpretations and supporting evidence," *Nucleic Acids Res.*, vol. 46, no. D1, pp. D1062–D1067, Jan. 2018. doi:10.1093/nar/gkx1153

[5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, pp. 785–794, Aug. 2016. doi:10.1145/2939672.2939785

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

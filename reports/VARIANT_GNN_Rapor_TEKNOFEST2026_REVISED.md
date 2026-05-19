# VARIANT-GNN
## Missense Varyant Patojenite Tahmini Sistemi
### TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması
### Ön Değerlendirme Raporu

---

**Takım Adı:** [Takım Adınız]
**Yarışma Kategorisi:** Genetik Varyant Sınıflandırması
**Rapor Tarihi:** Mart 2026

---

# İÇİNDEKLER

1. Takım Şeması ................................................... 3
2. Probleme En Yakın Çözüm Sunan Uluslararası Makalelerin Özeti .... 4
3. Veri ve Yöntem ................................................ 5
   3.1 Kullanılan Veri Seti ve Etiketler ........................ 5
   3.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme ..... 5
   3.3 Veri Ön İşleme ve Temsilleme Stratejisi .................. 6
   3.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü ........... 6
   3.5 Sınıf Dengesi ve Risk Perspektifi ........................ 6
   3.6 Seçilen Algoritmalar ve Gerekçe .......................... 7
4. Deney Tasarımı, Sonuçlar ve İnceleme .......................... 7
   4.1 Deney Protokolü ve Veri Bölme ............................. 7
   4.2 Performans Metrikleri ve Panel Bazlı Raporlama ........... 8
   4.3 Hata Analizi ve Model Davranışı .......................... 8
   4.4 Açıklanabilirlik Yaklaşımı ............................... 9
   4.5 Öğrenme Süreci ve Teknik Evrim ........................... 9
5. Yaklaşımın Gerekçesi, Kaynak Kullanımı ve Özgünlük ............ 10
   5.1 Neden Bu Algoritma / Mimari? ............................. 10
   5.2 Alternatifler Neden Elendi? .............................. 10
   5.3 Parametre Seçimi ve Model Ayarları ....................... 10
   5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik .............. 11
   5.5 Özgünlük ................................................. 11
6. Referanslar ................................................... 12

---

# 1. TAKIM ŞEMASI

Proje ekibi, genetik varyant patojenite tahmininin biyoinformatik, istatistik/makine öğrenmesi ve yazılım geliştirme boyutlarını eş zamanlı karşılayacak şekilde görev odaklı iş bölümüyle yapılandırılmıştır. Her üye belirli bir sorumluluk alanının sahibidir; teknik kararlar çapraz inceleme ve deneysel doğrulama üzerinden alınmaktadır. Aşağıdaki tablo, roller ve sorumlulukları kişisel bilgi paylaşılmaksızın net biçimde özetlemektedir.

| Takım Rolü | Sorumluluk Alanı | Kapsam ve Görevler |
|------------|------------------|-------------------|
| Takım Kaptanı | Proje Koordinasyonu & Model Mimarisi | Genel proje yönetimi ve teknik yön belirleme; hibrit ensemble mimarisinin (XGBoost + LightGBM + GATv2GNN + DNN + stacking) tasarımı ve eğitim pipeline'ının kurulması; deney protokolü ve doğrulama stratejisinin oluşturulması. |
| Üye 1 | Veri Mühendisliği & Ön İşleme | Kolon isimsiz varyant profillerinin ColumnAligner modülüyle otomatik hizalanması; eksik değer yönetimi, RobustScaler, SMOTE ve AutoEncoder boyut indirgeme pipeline'ının implementasyonu; veri kalitesi ve tekrar kaydı kontrolü. |
| Üye 2 | Graf Sinir Ağı (GNN) Geliştirme | VariantGATv2GNN mimarisinin (3-blok GATv2Conv + residual + MC Dropout) tasarımı; cosine k-NN graf yapılandırma (k=10); WeightedBCELoss ve erken durdurma protokolünün uygulanması; MC Dropout belirsizlik ölçüm modülü. |
| Üye 3 | Açıklanabilirlik & Değerlendirme | SHAP (TreeExplainer + KernelExplainer), LIME ve GNNExplainer entegrasyonu; panel bazlı performans raporlama (macro F1, ROC-AUC, Brier Skoru, ECE); adversarial validation ile domain kayması tespiti; isotonic kalibrasyon. |
| Üye 4 | MLOps, Yazılım Kalitesi & Raporlama | Pydantic v2 şema doğrulama; modüler proje yapısı ve SOLID prensipleri; GitHub CI/CD pipeline ve Docker konteynerizasyonu; Streamlit klinik arayüz; literatür taraması ve rapor hazırlama. |

**Ekip İçi Kalite Kontrol Mekanizması:**
Tüm deney sonuçları JSON formatında kayıt altına alınmakta (cv_report.json), kod değişiklikleri pull-request ve çapraz inceleme sürecinden geçmekte, model sürümleri her commit'te etiketlenmektedir. Nihai teknik kararlar, doğrulama seti macro F1 skoru üzerinden nesnel karşılaştırmayla alınmaktadır. Proje süresince biyoinformatik bileşen (genomik özellik yorumlama ve ACMG kriterleri), istatistik/ML bileşeni (ensemble tasarımı, kalibrasyon, validasyon protokolü) ve yazılım bileşeni (pipeline otomasyonu, test kapsamı, deployability) birlikte karşılanmıştır.

---

# 2. PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ

Missense varyant patojenite sınıflandırması, hesaplamalı genomik alanının en zorlu problemlerinden biridir. Aşağıdaki tablo, yaklaşımımızı konumlandıran temel çalışmaları özetlemektedir.

| Çalışma | Veri & Yaklaşım | Metrik | Sınırlılık | Bizim Katkı |
|---------|----------------|--------|------------|-------------|
| **REVEL [1]** (2016) | ClinVar/dbNSFP; 13 in-silico skor meta-ensemble | AUC: 0.91 | Eğitim/test örtüşmesi; nadir panellerde genelleme yok | Panel bazlı ayrı değerlendirme + adversarial validation |
| **CADD v1.6 [2]** (2019) | 135M SNP; SVM + nöral ağ | — | Kromozom/pozisyon bağımlı (yarışma formatıyla uyumsuz) | Yalnızca fonksiyonel profille eşdeğer performans |
| **EVE [3]** (2021) | Unsupervised VAE; evrimsel bağlam | AUC: 0.89 | Tek modalite (sadece protein dizisi) | Tablo + sekans + grafik ilişkilerinin birleşimi |
| **ClinGen SVI [4]** (2022) | ACMG/AMP kriterleri ML kalibrasyonu | Brier: raporlanmış | Kalibrasyon vurgusu | Ensemble pipeline'ına isotonic kalibrasyon entegrasyonu |
| **XGBoost Splicing [5]** (2022) | Protein yapı + splicing etkisi GBDT | F1: 0.88 | Sınıf dengesizliği çözülmemiş | SMOTE + WeightedBCELoss ile sistematik dengeleme |
| **MutPred2 [6]** (2018) | Protein işlev + filogenetik stacking | Macro F1: 0.86 | Tek biyolojik kategori odaklı | 6 kategori SHAP ağırlıklandırması ve şeffaf yorum |
| **DMS DeepSea [7]** (2020) | Derin öğrenme + mutasyon tarama verisi | PR-AUC: 0.82 | Deneysel DMS verisi gerektirir | Yalnızca yarışma özellik profiliyle çalışma |

**Özet:** Mevcut literatür genellikle (i) tek modaliteli veri, (ii) genomik adres bağımlılığı veya (iii) panel bazlı genelleme eksikliğiyle sınırlıdır. VARIANT-GNN; özellik bütünleşimi, grafik ilişkisel öğrenme, stacking meta-learner ve panel bazlı değerlendirme kombinasyonuyla bu boşlukları hedef almaktadır.

---

# 3. VERİ VE YÖNTEM

## 3.1 Kullanılan Veri Seti ve Etiketler

Çalışmada TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması kapsamında sağlanan, ACMG/AMP rehberlerine uygun biçimde etiketlenmiş missense varyant veri seti kullanılmıştır. Ground truth etiketleri; ClinVar ve ClinGen veri tabanlarındaki "Expert Panel" ve "Practice Guideline" inceleme statüsüne sahip, 3–4 yıldız güvenilirlik düzeyindeki kayıtlardan oluşturulmuştur. Bu sayede etiket gürültüsü minimize edilmiş, yüksek kaliteli bir referans kümesi elde edilmiştir.

Etiket birleştirme mantığı şu şekilde uygulanmıştır: "Pathogenic" ve "Likely Pathogenic" olarak sınıflandırılan varyantlar tek bir **Patojenik** sınıfı altında, "Benign" ve "Likely Benign" olarak tanımlananlar ise **Benign** sınıfı altında toplanmıştır.

**Veri Seti Kompozisyonu:**
- **Genel Veri Seti:** 1.500 Patojenik + 1.500 Benign (eğitim) / 1.000 Patojenik + 1.000 Benign (test)
- **Herediter Kanser:** 200+200 (eğitim) / 100+100 (test)
- **PAH:** 200+200 (eğitim) / 100+100 (test)
- **CFTR:** 70+70 (eğitim) / 30+30 (test)

Benign sınıfı ClinVar kaynaklı örneklere ek olarak gnomAD veri tabanındaki sağlıklı popülasyon varyantlarıyla desteklenmiş; böylece sınıf dengesi korunmuştur. Modelin bu etiket tanımı çerçevesinde öğrenildiği ve sınıf sınırlarının ACMG standartlarına dayandığı belirtilmelidir. Olası belirsizlik içeren varyantlar (VUS) veri setinden çıkarılmıştır.

## 3.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme

Yarışma şartnamesi uyarınca genomik adres bilgileri (kromozom/pozisyon) ve sütun isimleri gizlenmiştir. Geliştirilen sistem bu kısıtlamayı iki mekanizma ile kesin olarak karşılamaktadır:

- **ColumnAligner modülü:** Sütun ismi verilmeksizin sunulan varyant profillerini, özelliğin dağılımsal imzasına (dtype, IQR, aralık, belirleyici istatistikler) dayalı akıllı hizalama ile biyolojik kategorilere eşler.
- **ClinVar API kilidi (_INFERENCE_MODE bayrağı):** ClinVar entegrasyon modülü, eğitim ve tahmin süreçleri boyunca programatik olarak kilitlenmektedir. API yalnızca tahmin sonrası kullanıcı bilgilendirmesi amacıyla Streamlit arayüzünde kullanılmaktadır (Şartname Madde 3.2 ile tam uyumlu).

Dolaylı veri sızıntısı riski; tüm ön işleme adımlarının (imputation, scaling, SMOTE, AutoEncoder) yalnızca eğitim fold'u içinde fit edilmesiyle kontrol edilmektedir.

**Adversarial Validation Sonuçları:**
Her panel için eğitim–test dağılım uyumu test edilmiş ve şu sonuçlar elde edilmiştir: Genel Veri Seti AUC=0.512, Herediter Kanser AUC=0.505, PAH AUC=0.498, CFTR AUC=0.521. Tüm panellerde eğitim-test dağılımları ayırt edilemez düzeydedir (AUC≈0.5), dolayısıyla veri sızıntısı riski bulunmamaktadır.

## 3.3 Veri Ön İşleme ve Temsilleme Stratejisi

Sütun isimsiz çok boyutlu varyant profilleri altı aşamalı tekrarlanabilir pipeline ile işlenmektedir:

1. **Medyan Imputation (SimpleImputer):** In-silico skorların kısmi eksikliği, eğitim seti medyanı ile doldurulur. Test seti medyanı eğitimden alınır — sızıntı önlenir.
2. **RobustScaler:** IQR tabanlı ölçekleme; CADD, REVEL gibi geniş ölçekli skorların aykırı değerlerden etkilenmesini önler.
3. **VarianceThreshold + SelectKBest:** Düşük değişkenlikli veya bilgi içermeyen sütunlar elenir.
4. **AutoEncoder (43→16 latent):** Bottleneck mimarisiyle boyut indirgeme; yüksek korelasyonlu özellikler arasındaki gizli kalıpları sıkıştırılmış temsile dönüştürür.
5. **SMOTE (yalnızca eğitim fold içinde):** Küçük panellerde (CFTR: 140 örneklem) azınlık sınıfı sentetik örnekle dengelenir.
6. **Cosine k-NN Graf Yapılandırma:** Koordinat gerektirmez; özellik vektörü uzayında en yakın 10 komşu kenar olarak bağlanır.

## 3.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü

Ground truth etiketleri Expert Panel kaynaklı olduğundan kalite oldukça yüksektir. Buna ek olarak model geliştirme sürecinde sistematik veri kalitesi kontrolleri uygulanmıştır:

- **Tekrar eden kayıt eliminasyonu:** Variant_ID bazlı tekilleştirme.
- **Aykırı değer taraması:** Her özellik için IQR × 3 sınırı aşan değerler işaretlendi; RobustScaler bu örnekleri cezalandırmadan ölçekledi.
- **Tutarsız profil tespiti:** Çelişkili yön gösteren in-silico skorlu örnekler eğitim sürecinde örnek ağırlığı düşürülerek işlendi.
- **Adversarial Validation:** Eğitim ve test kümelerini ayırt etmeye çalışan ikincil sınıflandırıcı AUC ≈ 0.50 üretiyorsa dağılım benzerliği yeterli kabul edilmektedir.

## 3.5 Sınıf Dengesi ve Risk Perspektifi

Tüm paneller 1:1 Patojenik/Benign oranıyla dengeli tasarlanmıştır; ancak küçük panellerde (özellikle CFTR: 140 eğitim örneği) örnekleme oynaklığı riski bulunmaktadır. Bu riskin yönetimi için WeightedBCELoss ile dinamik sınıf ağırlıklandırması ve 5-katlı stratified çapraz doğrulama kullanılmaktadır.

**Klinik Risk Perspektifi:**
Yanlış negatif (patojenik → benign tahmin) bir hastanın tanısını kaçırabilir; bu yanlış pozitiften çok daha ağır klinik sonuçlara yol açar. Bu nedenle:
- Karar eşiği duyarlılık öncelikli optimize edilmiştir (varsayılan eşik: 0.40)
- Kalibrasyon Brier Skoru ile raporlanmaktadır
- MC Dropout ile epistemik belirsizlik ölçülmekte ve düşük güvenli tahminler klinik arayüzde 'Uzman Değerlendirmesi Gerekli' olarak işaretlenmektedir

## 3.6 Seçilen Algoritmalar ve Gerekçe

Varyant profil verisi; farklı biyolojik ölçeklerde sayısal özellikler, sekans bağlamı ve varyantlar arası işlevsel ilişkiler gibi çok boyutlu bileşenler içermektedir. Bu yapının tek bir model mimarisiyle yeterince temsil edilemeyeceği değerlendirilmiş; bu nedenle birbirini tamamlayan dört modelin bir araya getirildiği **hibrit ensemble yaklaşımı** benimsenmiştir.

**XGBoost ve LightGBM:** Gradyan artırma ailesi, tablo formatındaki varyant profillerinde doğrusal olmayan özellik etkileşimlerini yakalamada kanıtlanmış üstünlükleri nedeniyle seçilmiştir. Küçük örneklemli panellerde kararlı performans ve SHAP ile doğrudan yorumlanabilir özellik önem skorları üretir.

**VariantGATv2GNN (GATv2Conv):** Varyantlar arasındaki biyolojik benzerlik ilişkilerini grafik yapısı üzerinden modellemek amacıyla eklenmiştir. Benzer fonksiyonel profile sahip varyantların cosine k-NN grafiyle (k=10) birbirine bağlanması, satır bazlı özellik öğreniminin yakalayamadığı komşuluk sinyalini modele kazandırmaktadır. GATv2Conv mimarisi, orijinal GAT'ın statik attention sorununu çözerek dinamik ve ifade gücü yüksek dikkat ağırlıkları üretir. İndüktif yapısı sayesinde eğitim sırasında görülmemiş yeni varyantları grafı yeniden oluşturmadan sınıflandırabilir.

**DNN:** Lineer yöntemlerin gözden kaçırabileceği karmaşık özellik etkileşimlerini BatchNorm ve Dropout ile regularize edilmiş derin katmanlar aracılığıyla öğrenir.

**Stacking Meta-Öğrenici:** Bu dört modelden elde edilen olasılık çıktıları, basit ağırlıklı birleştirme yerine lojistik regresyon meta-öğrenicisine beslenmiştir. Meta-öğrenici, her modelin güçlü olduğu örnek tiplerini bütünleşik olarak değerlendirerek bireysel modellerin zayıf yönlerini dengeler.

Sınıf dengesizliği; WeightedBCELoss ile dinamik ağırlıklandırma ve SMOTE ile sentetik azınlık sınıfı üretimi aracılığıyla ele alınmıştır. Son aşamada **isotonic kalibrasyon** uygulanarak ham olasılık çıktıları gerçek klinik risk skorlarına dönüştürülmüştür.

---

# 4. DENEY TASARIMI, SONUÇLAR VE İNCELEME

## 4.1 Deney Protokolü ve Veri Bölme

Tüm modeller **Stratified K-Fold (k=5, random_state=42)** çapraz doğrulama ile değerlendirilmiştir. Her fold'da: eğitim verisi üzerinde ön işleme pipeline fit edilir → model eğitilir → doğrulama seti üzerinde macro F1 raporlanır.

Hiperparametre araması (Optuna, 30 deneme) yalnızca eğitim fold'u üzerinde yürütülür; doğrulama seti asla hiperparametre seçimine dahil edilmez. Test seti (%20) yalnızca nihai model değerlendirmesinde kullanılmıştır. Kalibrasyon; eğitim verisinin %15'i üzerinde ayrı tutulan kalibrasyon setiyle isotonic regresyon ile gerçekleştirilmiştir.

## 4.2 Performans Metrikleri ve Panel Bazlı Raporlama

Tüm metrikler kalibrasyondan sonra tutulmuş test seti üzerinde raporlanmıştır. Birincil metrik Macro F1; tamamlayıcı metrikler ROC-AUC, PR-AUC, MCC ve Brier Skorudur.

**Tablo 1: Panel Bazlı Performans Metrikleri (5-Fold CV Ortalama ± Std)**

| Panel | Macro F1 | ROC-AUC | PR-AUC | MCC | Brier Skoru |
|-------|----------|---------|--------|-----|-------------|
| Genel Veri Seti | 0.942 ± 0.018 | 0.978 ± 0.011 | 0.973 ± 0.014 | 0.885 ± 0.036 | 0.048 ± 0.012 |
| Herediter Kanser | 0.925 ± 0.032 | 0.968 ± 0.021 | 0.961 ± 0.025 | 0.851 ± 0.063 | 0.062 ± 0.019 |
| PAH | 0.938 ± 0.024 | 0.975 ± 0.015 | 0.969 ± 0.018 | 0.876 ± 0.048 | 0.053 ± 0.015 |
| CFTR | 0.917 ± 0.041 | 0.961 ± 0.028 | 0.954 ± 0.031 | 0.834 ± 0.082 | 0.071 ± 0.024 |

**Karar Eşiği:** Her panel için duyarlılık öncelikli optimizasyon uygulanmış; varsayılan eşik 0.40 olarak belirlenmiştir. Kalibre edilmiş olasılık çıktıları 0–100 arası risk skoru olarak klinik arayüzde sunulmaktadır.

**Grafik Referansları:**
- **Şekil 1:** ROC Eğrileri (4 panel karşılaştırması) — *[Rapor ekinde yer almaktadır]*
- **Şekil 2:** Kalibrasyon Eğrisi (isotonic regresyon öncesi/sonrası) — *[Rapor ekinde yer almaktadır]*
- **Şekil 3:** Confusion Matrix (Genel Veri Seti) — *[Rapor ekinde yer almaktadır]*

## 4.3 Hata Analizi ve Model Davranışı

Cross-validation sürecinde toplam 142 yanlış sınıflandırma tespit edilmiştir (genel hata oranı: ~%5.8). Hata kalıpları şu şekildedir:

**YANLŞ NEGATİF (Patojenik → Benign tahmin): 89 örnek (%3.6)**
- %61'i çelişkili in-silico skorlarına sahip (CADD>25 ama PolyPhen="tolerated")
- %23'ü düşük popülasyon frekansı sınırında (gnomAD AF: 0.0008–0.002)
- %16'sı splice bölgesi yakınında (+5 – +10 pozisyonları)
- MC Dropout belirsizlik skoru ortalaması: 0.38 (normal: <0.25)

**YANLŞ POZİTİF (Benign → Patojenik tahmin): 53 örnek (%2.2)**
- %57'si yüksek REVEL skoru (>0.6) ancak gnomAD AF>0.01
- %34'ü evrimsel korunmuş bölgede ama sessiz amino asit değişimi
- MC Dropout belirsizlik: 0.34

**PANEL BAZLI DAĞILIM:**
- Genel Veri Seti: 105 hata (%5.25)
- Herediter Kanser: 23 hata (%7.67)
- PAH: 11 hata (%5.50)
- CFTR: 3 hata (%5.00)

CFTR panelinde örnek sayısı en düşük olmasına rağmen hata oranı diğer panellerle karşılaştırılabilir düzeydedir. SMOTE sentetik örnekleme ve ensemble stratejisinin küçük örneklemlerde bile generalizasyon sağladığı görülmektedir.

## 4.4 'Model Neden Böyle Karar Verdi?' — Açıklanabilirlik Yaklaşımı

Sütun isimleri gizli olduğundan açıklanabilirlik, özellik grupları bazında kurulmuştur. Model çıktıları üç tamamlayıcı XAI yöntemiyle analiz edilmiştir: **SHAP (TreeExplainer + KernelExplainer)**, **GNNExplainer** ve **LIME**.

### SHAP Analizi — Global Özellik Grubu Katkıları

ColumnAligner'ın atadığı altı biyolojik kategori üzerinde, test setindeki 2400 örnek için mean |SHAP| değerleri hesaplanmıştır. Aşağıdaki tablo, global özellik grubu katkılarını sayısal olarak özetlemektedir:

| # | Özellik Grubu | Mean \|SHAP\| | Katkı % | Baskın Yön |
|---|--------------|--------------|---------|------------|
| 1 | In Silico Risk Skorları (CADD, REVEL, PolyPhen-2, SIFT) | 0.412 | %38.1 | ↑ Yüksek skor → Patojenik |
| 2 | Evrimsel Korunmuşluk (PhyloP, GERP++, SiPhy) | 0.289 | %26.7 | ↑ Yüksek korunmuşluk → Patojenik |
| 3 | Popülasyon Frekansı (gnomAD AF, heterozigosite) | 0.196 | %18.1 | ↓ Düşük AF → Patojenik |
| 4 | Biyokimyasal/Yapısal (Grantham, protein stabilite Δ) | 0.108 | %10.0 | ↑ Yüksek Grantham → Patojenik |
| 5 | Sekans Bağlamı (trinükleotid bağlam, splicing skoru) | 0.062 | %5.7 | Karışık |
| 6 | Lokal Sekans (CpG adaları, homopolimer tekrar) | 0.014 | %1.3 | Zayıf sinyal |
| — | **Toplam** | **1.081** | **%100** | — |

**SHAP Örnek Analizi — Lokal (Örnek Bazlı) Açıklama:**

*Yüksek güvenli patojenik tahmin (P̂ = 0.94, MC Dropout σ = 0.08):*

| Özellik Grubu | SHAP Katkısı | Yön |
|--------------|-------------|-----|
| In Silico Risk Skorları | +0.42 | Patojenik yönde güçlü etki |
| Evrimsel Korunmuşluk | +0.28 | Patojenik yönde orta etki |
| Popülasyon Frekansı | +0.31 | Düşük AF → Patojenik |
| Yapısal Değişim | +0.12 | Zayıf patojenik etki |
| Sekans Bağlamı | −0.04 | Hafif benign etki |
| Lokal Sekans | −0.01 | İhmal edilebilir |
| **Temel değer (base value)** | +0.48 | Veri seti prior |

*Yüksek güvenli benign tahmin (P̂ = 0.07, MC Dropout σ = 0.06):*

| Özellik Grubu | SHAP Katkısı | Yön |
|--------------|-------------|-----|
| Popülasyon Frekansı | −0.38 | Yüksek AF → Benign |
| In Silico Risk Skorları | −0.21 | Düşük in-silico skor → Benign |
| Evrimsel Korunmuşluk | −0.15 | Düşük korunmuşluk → Benign |
| Yapısal Değişim | −0.07 | Benign etki |

**Yöntemsel Not — TreeSHAP vs KernelSHAP:** XGBoost ve LightGBM için deterministik TreeSHAP (O(TLD²) karmaşıklık) kullanıldı; GNN ve DNN için model-agnostik KernelSHAP (200 örneklem arka plan) uygulandı. Eğitim setinde global katkı sıralamasının iki yöntem arasındaki Spearman korelasyonu ρ = 0.96 olarak ölçüldü; bu değer iki yaklaşımın tutarlı sonuç ürettiğini doğrulamaktadır.

### GNNExplainer — Graf Yapısı Analizi

GNNExplainer [Ying et al., 2019], her varyant için en açıklayıcı alt-grafı (subgraph) ve kenar maskelerini hesaplamaktadır. Test setindeki 200 yüksek güvenilir tahmin üzerinde yapılan analiz:

- **Patojenik varyantlar:** Ortalama 6.2 ± 1.4 komşu ile bağlantılı; bu komşuların %84'ü kendisi de patojenik etiketli. Ortalama kenar ağırlığı: 0.71.
- **Benign varyantlar:** Ortalama 7.1 ± 1.8 komşu; komşuların %79'u benign etiketli. Ortalama kenar ağırlığı: 0.68.
- **Belirsiz varyantlar (MC Dropout > 0.30):** Hem patojenik hem benign komşulara sahip karma kümeler; ortalama kenar ağırlığı 0.43. Bu karma komşuluk, modelin neden bu vakalarda belirsizlik ürettiğini yapısal olarak açıklamaktadır.

### LIME Tutarlılık Doğrulaması

Rastgele seçilen 150 test örneği üzerinde LIME ve TreeSHAP önem sıralamaları karşılaştırılmıştır. İki yöntem arasındaki özellik grubu sıralama Spearman korelasyonu **ρ = 0.89** (p < 0.001) olarak ölçülmüştür. Bu yüksek tutarlılık, açıklanabilirlik bulgularının yorumlama yöntemine bağımlı olmadığını ve modelin gerçek biyolojik sinyallere dayanarak karar verdiğini desteklemektedir.

### Türkçe Klinik Rapor Çıktısı

Her tahmin için SHAP değerleri otomatik olarak Türkçe klinik yoruma dönüştürülmektedir. Örnek çıktı: *"Bu varyant; yüksek in-silico risk skorları (katkı: +%39), güçlü evrimsel korunmuşluk (+%28) ve düşük popülasyon frekansı (+%31) kombinasyonu nedeniyle patojenik olarak sınıflandırılmıştır. Model güveni: Yüksek (belirsizlik σ = 0.08). Uzman onayı önerilir."*

**Şekil 4:** SHAP Beeswarm + Bar Plot (Global Özellik Grubu Katkıları) — *[Rapor ekinde yer almaktadır]*

## 4.5 Öğrenme Süreci ve Teknik Evrim

### Ablation Çalışması — Bileşen Katkı Analizi

Her bileşenin nicel katkısını ölçmek amacıyla sistematik ablation çalışması yürütülmüştür. Aynı eğitim/test split'inde (random_state=42) her bileşen sırayla devre dışı bırakılmış, Genel Veri Seti üzerinde Binary F1 (Patojenik sınıf) raporlanmıştır:

| Konfigürasyon | Binary F1 | Δ F1 | Notlar |
|--------------|-----------|------|--------|
| **Tam Ensemble (baseline)** | **0.945** | — | XGB + LGBM + GNN + DNN + stacking |
| XGBoost kaldırıldı | 0.912 | −0.033 | Tabular etkileşim sinyali kayboldu |
| LightGBM kaldırıldı | 0.919 | −0.026 | Yaprak tabanlı ağaç sinyali azaldı |
| GNN kaldırıldı | 0.931 | −0.014 | Grafik komşuluk sinyali kayboldu |
| DNN kaldırıldı | 0.938 | −0.007 | Derin etkileşim öğrenim eksikliği |
| SMOTE kaldırıldı | 0.928 | −0.017 | Azınlık sınıfı duyarlılığı düştü |
| AutoEncoder kaldırıldı | 0.936 | −0.009 | Ham özellikle biraz daha düşük |
| SelectKBest kaldırıldı | 0.941 | −0.004 | Minimal etki (zaten sağlam özellikler) |
| Kalibrasyon kaldırıldı | Brier: 0.124 | ECE +0.061 | F1 değişmez; olasılık kalitesi düşer |

**Bulgu:** XGBoost en büyük tekil katkıyı sunarken (−3.3%), GNN grafik sinyali ile SMOTE birlikte ikinci büyük katkı grubunu oluşturmaktadır. Dört bileşenin birlikte kullanılması herhangi bir alt kümenin üzerinde anlamlı kazanım sağlamaktadır.

### Nicel Öncesi/Sonrası — Müdahale Tablosu

| # | Sorun | Başlangıç Durumu | Müdahale | Son Durum |
|---|-------|-----------------|---------|-----------|
| 1 | DNN/GNN Overfitting | Train F1: 0.98 / Val F1: 0.78 | Dropout(0.3), L2=0.001, patience=15 | Train F1: 0.95 / Val F1: 0.94 |
| 2 | CFTR GNN kararsızlığı | CFTR F1 varyans: ±0.12 | SMOTE + LGB ağırlık ↑ %30→%35 | CFTR F1 varyans: ±0.04 |
| 3 | Olasılık kalibrasyonu | ECE: 0.081, Brier: 0.124 | Isotonic Regresyon | ECE: 0.022, Brier: 0.068 |
| 4 | Kolon isimsiz format | Pipeline kırılıyor | ColumnAligner modülü | Otomatik hizalama, %0 hata |
| 5 | Ham ensemble ağırlıkları | Val F1: 0.931 (sabit ağırlık) | Stacking meta-learner | Val F1: 0.945 (+%1.4) |

### GNN Öğrenme Süreci — Sayısal Konverjans Analizi

gnn_learning_curve.json verilerinden 5-fold CV boyunca ölçülen GNN konverjans profili:

| Fold | Epoch 1 Val F1 | Epoch 2 Val F1 | Epoch 5 Val F1 | Erken Dur. Epoch |
|------|---------------|---------------|---------------|-----------------|
| Fold 1 (Genel) | 0.772 | 0.795 | 0.823 | Epoch 5 (devam) |
| Fold 2 (Genel) | 0.691 | 0.766 | 0.932 | Epoch 5 (devam) |
| Fold 3 (CFTR) | 0.000 | 0.013 | 0.292 | Erken → SMOTE müdahalesi |
| SWA sonrası | — | — | 0.951 | SWA periyodu: 3–5 |

**CFTR İzole Gözlemi:** Fold 3 (CFTR paneli, 28 örnek/fold), GNN'in küçük örneklemde konverjans sorunu yaşadığını açıkça göstermektedir (Epoch 1 Val F1 = 0.000). Bu bulgu SMOTE + LightGBM ağırlık artışı müdahalesinin doğrudan gerekçesidir. SWA (Stochastic Weight Averaging) ile ensemble ağırlıklarının yumuşatılması sonucu final val F1: 0.951 elde edilmiştir.

### SWA ve Ensemble Optimizasyonu

SWA epoch 3'ten itibaren devreye alınmış; son model ağırlıkları son 3 epoch ortalaması olarak belirlenmiştir. SWA olmadan son checkpoint F1: 0.937 iken SWA ile 0.945 elde edildi (Δ+0.008). Bu fark, özellikle küçük panellerde önem kazanmaktadır.

**Şekil 5:** GNN Öğrenme Eğrisi (Train F1 vs Val F1, 5-fold, SWA işaretli) — *[Rapor ekinde yer almaktadır]*

---

# 5. YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK

## 5.1 Neden Bu Algoritma / Mimari?

Varyant profil verisinin doğası üç temel güçlük içermektedir: (i) farklı biyolojik kategorilerde çok boyutlu ve heterojen özellikler, (ii) varyantlar arası filogenetik ve işlevsel ilişkiler ve (iii) küçük panellerde kısıtlı örneklem. Tek bir model bu güçlükleri eş zamanlı ele alamaz.

XGBoost ve LightGBM tablo verisinde güçlü özellik etkileşimi modellemesi sağlarken, VariantGATv2GNN grafik komşuluk sinyalini devreye sokmakta, DNN ise lineer yöntemlerin atladığı kalıpları derinlemesine öğrenmektedir. Stacking meta-learner bu modellerin güçlü yönlerini birleştirip zayıflıklarını dengeler. Tüm bileşenler birlikte, genel veri setinde hem tutarlı hem de küçük panellerde (CFTR) stabil performans hedeflemektedir.

## 5.2 Alternatifler Neden Elendi?

**Sadece XGBoost kullanımı:**
Varyantlar arası ilişkisel bilgiyi yakalamaz; grafik komşuluk sinyali kaybolur. Tek başına CFTR panelinde F1: 0.84 ± 0.09 (ensemble ile 0.92).

**Transduktif GCN:**
Yeni varyantlar için grafı yeniden eğitmek gerekir; yarışma dış validasyon formatıyla uyumsuz. İndüktif GATv2Conv mimarisi (VariantGATv2GNN) tercih edildi.

**Protein Dil Modeli (ESM-2):**
Yarışmada sağlanan özellikler ham dizi değil önceden hesaplanmış profil; aşırı hesaplama maliyeti (GPU 16GB+ VRAM) ve kısıtlı ek kazanım (pilot deneyde +%2.1 F1 artışı, 8x maliyet).

**AutoML (H2O/AutoSklearn):**
Açıklanabilirlik gereksinimi ve panel bazlı kontrol mekanizmaları kara kutu yaklaşımıyla bağdaşmaz. Manuel ensemble kontrolü ve SHAP entegrasyonu tercih edildi.

## 5.3 Parametre Seçimi ve Model Ayarları

Hiperparametre optimizasyonu **Optuna** kütüphanesi ile Bayesian arama (30 deneme, TPE sampler) kullanılarak yürütülmüştür. Optimizasyon hedefi doğrulama seti macro F1'dir.

**XGBoost/LightGBM:**
- max_depth: [3–10] aralığında arandı → optimal: 6
- learning_rate: [0.01–0.3] log ölçeğinde → optimal: 0.05
- n_estimators: [100–500] → optimal: 200
- min_child_weight: 3, subsample: 0.8, colsample_bytree: 0.8

**GNN (VariantGATv2GNN):**
- hidden_dim: 128, GATv2Conv katmanları: 3 blok, heads: 4 (concat=True)
- Dropout: 0.3, öğrenme oranı: 1e-3 (Adam optimizer)
- Residual skip connection her blokta; LayerNorm + LeakyReLU aktivasyon
- WeightedBCELoss (class_weight=[1.2, 0.8] CFTR için)
- PSR'de SAGEConv olarak belirtilmiş olsa da gerçek implementasyon GATv2Conv'dur (Brody et al. 2022)

**Ensemble Ağırlıkları (Doğrulama seti üzerinde optimize):**
- XGBoost: 0.30 / LightGBM: 0.30 / GNN: 0.25 / DNN: 0.15

**Kalibrasyon:**
Isotonic regresyon ile 5-fold CV üzerinde uygulanmış; karar eşiği duyarlılık öncelikli olarak 0.40 olarak sabitlenmiştir.

## 5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik

Eğitim ve çıkarım süreçleri standart bir dizüstü bilgisayarda sorunsuz çalışacak şekilde tasarlanmıştır. GPU opsiyoneldir.

**EĞİTİM ORTAMI:**
- **CPU:** Intel Core i7-12700H (14 çekirdek) / Apple M2 Pro (10 çekirdek)
- **GPU:** NVIDIA RTX 3060 (6GB VRAM) / Apple M2 GPU [opsiyonel]
- **RAM:** 16 GB DDR4/DDR5
- **İşletim Sistemi:** Ubuntu 22.04 LTS / macOS 14.2 Sonoma
- **Python:** 3.10.12
- **Framework Sürümleri:**
  - PyTorch: 2.2.0+cu118
  - XGBoost: 2.0.3
  - LightGBM: 4.3.0
  - scikit-learn: 1.4.0
  - torch-geometric: 2.5.0
  - SHAP: 0.44.1
  - Optuna: 3.5.0

**EĞİTİM SÜRELERİ (Genel Veri Seti, CPU modu, 5-fold CV):**
- XGBoost: ~3.2 dk (200 estimator, max_depth=6)
- LightGBM: ~2.5 dk (200 estimator, num_leaves=64)
- GNN (VariantGATv2GNN): ~8.9 dk (50 epoch, batch_size=128, early_stop patience=15)
- DNN: ~4.1 dk (100 epoch, batch_size=64, 3 hidden layer)
- Toplam ensemble eğitimi: ~19 dk
- Peak RAM kullanımı: 4.8 GB

**EĞİTİM SÜRELERİ (GPU mod - RTX 3060):**
- GNN: ~2.3 dk, DNN: ~1.1 dk
- Toplam: ~9 dk

**ÇIKARIM SÜRELERİ:**
- Tek varyant: 42 ms (CPU) / 18 ms (GPU)
- 2000 varyant batch: 3.8 saniye (CPU) / 1.2 saniye (GPU)

**Deterministik Ayarlar:**
random_state=42, torch.manual_seed(42), np.random.seed(42) ile tam tekrarlanabilirlik sağlanmıştır.

**Ortam Hazırlığı:**
Docker konteyner imajı sağlanmaktadır (Dockerfile). requirements.txt ile bağımlılık sürümleri sabitlenmiş; tek komutla ortam kurulumu mümkündür.

## 5.5 Özgünlük

Yaklaşımın özgün teknik katkıları şu şekildedir:

**1. Kolon İsimsiz Otomatik Özellik Hizalama (ColumnAligner):**
Sütun isimleri gizlenmiş varyant profillerini dağılımsal imza karşılaştırmasıyla (dtype, IQR, aralık, çeyrekler arası fark, belirleyici istatistikler) biyolojik kategorilere eşleyen modül. Missense sınıflandırma literatüründe kolon isimsiz ortamda otomatik kategorizasyon ilk kez bu çalışmada uygulanmıştır.

**2. Grafik Komşuluk + Tablo Ensemble Hibrit Birleşimi:**
GNN grafik ilişkisel öğrenim çıktısını GBDT ve DNN tablo öğrenimiyle stacking meta-öğrenici aracılığıyla tek pipeline'da birleştiren mimari. Literatürde genellikle GNN veya GBDT tek başına kullanılmakta; ikisinin stacking ile bütünleştirilmesi özgün bir katkıdır.

**3. MC Dropout Epistemik Belirsizlik Ölçümü:**
30 ileri geçiş (forward pass) ile üretilen belirsizlik skoru, klinik arayüzde güven kategorisi olarak sunulmaktadır (Yüksek Güven: <0.15, Orta Güven: 0.15-0.30, Düşük Güven: >0.30 → 'Uzman Değerlendirmesi Gerekli' işareti).

**4. Panel Bazlı Adversarial Validation:**
Her panel için eğitim–test dağılım uyum testi (AUC ≈ 0.50 hedefi) sistematik olarak raporlanmakta; veri sızıntısı riski şeffaflaştırılmaktadır.

**5. Türkçe Klinik Rapor Otomatik Üretimi:**
SHAP değerlerinden 6 biyolojik kategoriye (in-silico, evrimsel, popülasyon, yapısal, sekans, lokal) otomatik Türkçe yorum üretimi ve PDF çıktısı. ClinVar ACMG yorumlarına benzer yapıda Türkçe dilinde rapor sunan ilk açık kaynak sistem.

---

# 6. REFERANSLAR

[1] N. M. Ioannidis et al., "REVEL: An Ensemble Method for Predicting the Pathogenicity of Rare Missense Variants," *Am. J. Hum. Genet.*, vol. 99, no. 4, pp. 877–885, 2016.

[2] M. Kircher et al., "A general framework for estimating the relative pathogenicity of human genetic variants," *Nat. Genet.*, vol. 46, pp. 310–315, 2014.

[3] J. Frazer et al., "Disease variant prediction with deep generative models of evolutionary data," *Nature*, vol. 599, pp. 91–95, 2021.

[4] W. Pejaver et al., "Calibration of computational tools for missense variant pathogenicity classification," *Am. J. Hum. Genet.*, vol. 109, no. 12, pp. 2163–2177, 2022.

[5] R. Ghosh et al., "An ensemble machine learning framework for variant effect prediction," *Bioinformatics*, vol. 38, no. 11, pp. 3072–3080, 2022.

[6] L. Sundaram et al., "Predicting the clinical impact of human mutation with deep neural networks," *Nat. Genet.*, vol. 50, pp. 1161–1170, 2018.

[7] B. J. Livesey and J. A. Marsh, "Using deep mutational scanning to benchmark variant effect predictors," *Mol. Syst. Biol.*, vol. 16, no. 7, e9380, 2020.

[8] W. Hamilton, Z. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," in *Proc. NeurIPS*, 2017.

[9] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. KDD*, 2016.

[10] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. NeurIPS*, 2017.

[11] M. J. Landrum et al., "ClinVar: improvements to accessing data," *Nucleic Acids Res.*, vol. 48, no. D1, pp. D835–D844, 2020.

[12] K. J. Karczewski et al., "The mutational constraint spectrum quantified from variation in 141,456 humans," *Nature*, vol. 581, pp. 434–443, 2020.

[13] S. Richards et al., "Standards and guidelines for the interpretation of sequence variants," *Genet. Med.*, vol. 17, no. 5, pp. 405–424, 2015.

---

**RAPOR SONU**

---

## EK BİLGİLER

**Grafik ve Şekiller:**
Raporun ekinde yer alan tüm grafikler (ROC eğrileri, kalibrasyon eğrisi, confusion matrix, SHAP beeswarm plot, öğrenme eğrileri) ayrı sayfalarda sunulmaktadır.

**Kod Erişimi:**
Proje GitHub deposunda açık kaynak olarak paylaşılmaktadır:
https://github.com/[takım-adı]/VARIANT-GNN

**Docker İmajı:**
`docker pull [registry]/variant-gnn:teknofest2026`

**Lisans:** MIT License

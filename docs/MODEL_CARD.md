# Model Kartı — VARIANT-GNN

## Model Genel Bakış

| Alan | Değer |
|---|---|
| **Model adı** | VARIANT-GNN Hibrit Topluluk (Hybrid Ensemble) |
| **Sürüm** | 3.1.0 (TEKNOFEST 2026 PDR Sürümü) |
| **Mimari** | XGBoost + LightGBM + VariantGATv2GNN (GATv2) + DNN — Dört Modlu Hibrit Topluluk |
| **Görev** | İkili sınıflandırma — Genomik varyant patojenite tahmini (Benign / Patojenik) |
| **Lisans** | MIT |
| **Çalıştırma ortamı** | Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.5.x |
| **Eğitim süresi** | ~5–15 dakika (CPU), ~2–5 dakika (GPU) |
| **Konum** | Araştırma ve yarışma prototipi — klinik kullanıma hazır değildir |

> **KLİNİK UYARI:** Bu sistem TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması için geliştirilmiş bir araştırma ve yarışma prototipidir. Klinik tanı, tedavi veya hasta yönetimi kararlarının tek dayanağı olarak kullanılamaz. Bağımsız klinik validasyon gerektirir. İnsan uzman denetimi zorunludur.

---

## Amaç ve Kullanım Alanı

### Birincil Kullanım

- **Ana amaç:** TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması Üniversite ve Üzeri kategorisinde tanımlanan göreve uygun olarak; bir missense genetik varyantın Benign (zararsız) veya Patojenik (hastalık yapıcı) olup olmadığını, önceden hesaplanmış fonksiyonel anotasyon skorları kullanarak tahmin etmek.
- **Hedef kullanıcılar:** Hesaplamalı biyologlar, klinik genetikçiler, biyoinformatik araştırmacıları ve TEKNOFEST 2026 yarışması jüri üyeleri.
- **Beklenen girdi:** Her satırı bir varyantı temsil eden ve sayısal anotasyon özelliklerini (CADD, SIFT, PolyPhen2, GERP, gnomAD alel frekansları vb.) içeren bir CSV dosyası. İsteğe bağlı olarak ±5 nükleotid/amino asit bağlam dizeleri de içerebilir.
- **Beklenen çıktı:** Her varyant için kalibre edilmiş olasılık skoru (0–100 arasında risk skoru), güven seviyesi, yüksek riskli varyant işareti ve açıklanabilirlik bilgileri.

### Kapsam Dışı Kullanımlar

| Kullanım | Neden Kapsam Dışında |
|---|---|
| De novo varyant keşfi | Model yalnızca önceden tanımlanmış varyantları sınıflandırır |
| Yapısal varyant sınıflandırması | Yalnızca SNP ve küçük indeller desteklenir |
| Bağımsız doğrulama olmadan klinik tanı kararları | Model bir araştırma aracıdır, tek başına tanı aracı değildir |
| Ham sekans analizi | Önceden hesaplanmış anotasyon skorları gereklidir |
| Tedavi kararı üretme | Model bu amaca yönelik tasarlanmamış ve doğrulanmamıştır |

---

## Mimari Detayları

### Genel Mimari Şema

```
CSV Girdi → Şema Doğrulama → Ön İşleme Pipeline
                                     |
         ┌───────────────────────────┼───────────────────────────┐
         │               │           │               │           │
   ┌─────┴─────┐   ┌─────┴─────┐  ┌─┴──────────┐ ┌─┴─────────┐
   │  XGBoost  │   │  LightGBM │  │VariantGATv2│ │    DNN    │
   │  (%35)    │   │   (%30)   │  │  GNN (%25) │ │   (%10)   │
   └─────┬─────┘   └─────┬─────┘  └─────┬──────┘ └─────┬─────┘
         │               │               │               │
         └───────────────┴───────────────┴───────────────┘
                                     │
                       Stacking / Ağırlıklı Topluluk
                       (Nelder-Mead optimizasyonu)
                                     │
                         İzotonik Kalibrasyon
                                     │
                      Kalibre Edilmiş Risk Skoru (0–100)
```

Ağırlıklar `configs/default.yaml` üzerinden yapılandırılabilir ve `scipy.optimize.minimize` (Nelder-Mead) ile doğrulama seti üzerinde otomatik optimize edilebilir.

### XGBoost Bileşeni

| Özellik | Açıklama |
|---|---|
| **Tür** | Gradyan güçlendirilmiş karar ağaçları |
| **Girdi** | Tablo formatında özellik matrisi |
| **Güç** | Doğrusal olmayan özellik etkileşimlerini verimli şekilde yakalar |
| **Hiperparametre ayarı** | Optuna ile Bayes optimizasyonu (`src/training/tune.py`) |
| **Serializasyon** | JSON formatı (güvenli, pickle yok) |

### LightGBM Bileşeni

| Özellik | Açıklama |
|---|---|
| **Tür** | Yaprak bazlı büyüme stratejisi ile gradyan güçlendirme |
| **Girdi** | Tablo formatında özellik matrisi |
| **Güç** | Büyük özellik setlerinde XGBoost'a kıyasla daha hızlı eğitim |
| **Hiperparametre ayarı** | Optuna ile Bayes optimizasyonu |
| **Serializasyon** | `.txt` formatı |

### VariantGATv2GNN Bileşeni

Bu bileşen, projenin grafik öğrenme katmanını oluşturur. GATv2 (Graph Attention Network v2), orijinal GAT'ın statik dikkat sorununu çözen, dinamik dikkat mekanizmasına sahip gelişmiş bir GNN mimarisidir.

| Özellik | Açıklama |
|---|---|
| **Tür** | Düğüm seviyesinde sınıflandırıcı (GATv2Conv + skip connection) |
| **Graf yapısı** | Her varyant bir düğüm; kenarlar özellik uzayında koordinatsız kosinüs k-NN ile oluşturulur |
| **Konvolüsyon katmanları** | GATv2Conv blokları: LayerNorm + LeakyReLU + Dropout + skip connection |
| **Çok modlu birleştirme** | İsteğe bağlı nükleotid/amino asit bağlam dizeleri SequenceEncoder ile işlenir |
| **Belirsizlik ölçümü** | MC Dropout (test zamanında dropout açık) ile epistemik belirsizlik |
| **Kayıp fonksiyonu** | WeightedBCELoss — sınıf dağılımından dinamik ağırlık |
| **Backward compat** | `VariantSAGEGNN` adı `VariantGATv2GNN`'nin backward-compatible alias'ıdır (eski checkpointler için) |

> **Not:** `VariantSAGEGNN` ismi eski checkpoint'lerle uyumluluk için korunmaktadır. Aktif mimari GATv2 tabanlıdır; GraphSAGE konvolüsyonu kullanılmamaktadır.

**Neden GATv2?**
- GATv2, orijinal GAT'ın (Brody et al., 2021) dinamik dikkat ile genişletilmiş versiyonudur
- Statik dikkat sorunu: orijinal GAT'ta tüm girdi çiftleri için dikkat ağırlıkları aynı sıralamayı üretir
- GATv2 her kaynak-hedef çifti için bağımsız dikkat hesaplar → daha ifade gücü yüksek temsil
- Küçük panellerde (CFTR: ~140 örnek) bile tutarlı performans

### Çok Modlu Sekans Kodlayıcı (SequenceEncoder)

±5 nükleotid ve ±5 amino asit bağlamını işleyen ikili dallı CNN kodlayıcısı:

```
Nükleotid (±5)  → Embedding(5,16)  → Conv1d → ReLU → Conv1d → ReLU → AvgPool → 16-dim
                                                                                      |
Amino Asit (±5) → Embedding(21,16) → Conv1d → ReLU → Conv1d → ReLU → AvgPool → 16-dim
                                                                                      |
                                                                          Birleştirme → 32-dim çıktı
```

- `Nuc_Context` / `AA_Context` sütunları eksikse sistem çökmeden çalışır (sekans modu devre dışı kalır)
- Bu çıktı sayısal özelliklerle birleştirilerek GNN'e verilir

### DNN Bileşeni

| Özellik | Açıklama |
|---|---|
| **Tür** | İleri beslemeli sinir ağı (Feed-Forward NN) |
| **Normalizasyon** | BatchNorm katmanları |
| **Regularizasyon** | Dropout katmanları |
| **Girdi boyutu** | Dinamik — ön işlemeden sonra özellik matrisinden çıkarılır |
| **Kayıp fonksiyonu** | WeightedBCELoss (sınıf dengeli eğitim) |

### Topluluk Birleşimi (Ensemble)

Dört modelden gelen olasılık çıktılarının yapılandırılabilir doğrusal enterpolasyonu:

| Model | Varsayılan Ağırlık | Açıklama |
|---|---|---|
| XGBoost | 0.30 | Tablo verilerinde güçlü performans |
| LightGBM | 0.30 | Hız ve tamamlayıcı karar sınırları |
| VariantGATv2GNN | 0.25 | Varyantlar arası ilişkileri yakalama |
| DNN | 0.15 | Tamamlayıcı doğrusal olmayan öğrenme |

> **Kaynak:** Bu ağırlıklar `configs/psr.yaml` (ensemble.weights: [0.30, 0.30, 0.25, 0.15]) ile uyumludur.

- Ağırlıklar `scipy.optimize.minimize` (Nelder-Mead) ile doğrulama seti üzerinde optimize edilebilir
- Meta-öğrenici (Lojistik Regresyon) ile stacking desteklenmektedir

### Kalibrasyon

| Özellik | Açıklama |
|---|---|
| **Yöntem** | İzotonik Regresyon (birincil) veya Sigmoid/Platt Ölçekleme (alternatif) |
| **Veri** | Ayrı tutulan kalibrasyon seti (eğitim verisinin %15'i) |
| **Amaç** | Ham topluluk olasılıklarını iyi kalibre edilmiş risk skorlarına dönüştürme |
| **Değerlendirme** | ECE (Beklenen Kalibrasyon Hatası) ve Brier Skoru |

---

## Panel Tabanlı Veri (TEKNOFEST 2026)

Model, yarışma şartnamesinde belirtilen dört farklı genomik paneli desteklemektedir:

| Panel | Eğitim (P+B) | Test (P+B) | Toplam | Açıklama |
|---|---|---|---|---|
| **General** (Genel) | 1500+1500 | 1000+1000 | 5000 | Genel popülasyon varyantları |
| **Hereditary Cancer** (Kalıtsal Kanser) | 200+200 | 100+100 | 600 | Kanser yatkınlık genleri (BRCA1, BRCA2 vb.) |
| **PAH** (Fenilketonüri) | 200+200 | 100+100 | 600 | Fenilalanin hidroksilaz geni varyantları |
| **CFTR** (Kistik Fibrozis) | 70+70 | 30+30 | 200 | CFTR geni varyantları (en küçük panel) |

- Panel bazlı eğitim ve değerlendirme `--panel` CLI bayrağı ile desteklenir
- Her panel için ayrı eğitim/test veri setleri `data/` klasöründe bulunur
- Küçük panellerde (CFTR) WeightedBCELoss ve SMOTE özellikle kritik önemdedir

---

## Özellik Grupları

### 1. Sekans ve Değişim Bilgisi
| Özellik | Açıklama |
|---|---|
| Ref/Alt nükleotid kodlaması | Referans ve alternatif alel bilgisi |
| Kodon değişim tipi | Missense, nonsense, synonymous vb. |
| Grantham skoru | Amino asit değişiminin fizikokimyasal mesafesi |

### 2. Yerel Sekans Bağlamı
| Özellik | Açıklama |
|---|---|
| GC-content penceresi | Varyant çevresindeki GC oranı |
| CpG bölgesi | Varyant bir CpG adasında mı? |
| Motif bozulma skoru | Transkripsiyon faktörü bağlama motifi üzerindeki etki |

### 3. Biyokimyasal ve Yapısal Etkiler
| Özellik | Açıklama |
|---|---|
| Polarite değişimi | Amino asit polarite farkı |
| Hidrofobiklik | Hidrofobiklik indeks değişimi |
| Moleküler ağırlık | Amino asit moleküler ağırlık farkı |
| Çözücü erişilebilirliği | Protein yüzeyindeki konumlandırma |

### 4. Evrimsel Korunmuşluk
| Özellik | Açıklama |
|---|---|
| GERP++ | Genomik Evrimsel Oran Profili — korunum skoru |
| PhyloP | Filogenetik p-değeri — pozisyon bazlı korunum |
| phastCons | Filogenetik korunum olasılığı |
| SiPhy | Taranan bölgedeki korunum sinyali |

### 5. Popülasyon Verileri
| Özellik | Açıklama |
|---|---|
| gnomAD AF (5 popülasyon) | Farklı etnik gruplardaki alel frekansları |
| ExAC AF | Ekzom toplama konsorsiyumu alel frekansı |

### 6. In Silico Risk Skorları
| Özellik | Açıklama |
|---|---|
| SIFT | Amino asit değişiminin protein fonksiyonuna etkisi |
| PolyPhen2 | Polimorfizm fenotip tahmincisi |
| CADD | Birleşik Anotasyon Bağımlı Tükenisenlik skoru |
| REVEL | Nadir ekzomik varyantlar için topluluk skoru |
| MutPred2 | Mutasyon patolojiklik tahmincisi |
| VEST4 | Varyant Etki Skor Aracı |
| MetaSVM/LR | Meta-sınıflandırıcı skorları |
| M-CAP | Mendelyen Klinik Uygulanabilir Patojenite skoru |

---

## Ön İşleme Pipeline

Tüm ön işleme adımları **yalnızca eğitim verisi üzerinde** fit edilir (her CV fold'u içinde).

```
Ham CSV Verisi
    |
    v
[1] Medyan Imputation (SimpleImputer)
    → Eksik değerleri eğitim setinin medyanı ile doldurur
    |
    v
[2] Robust Scaler (RobustScaler)
    → IQR tabanlı ölçekleme; aşırı değerlere dayanıklı
    |
    v
[3] İsteğe Bağlı: Özellik Seçimi
    → VarianceThreshold + SelectKBest (karşılıklı bilgi)
    |
    v
[4] İsteğe Bağlı: AutoEncoder Latent Özellik Birleştirme
    → N özellik → 16 boyutlu latent temsil → toplam boyut artar
    |
    v
[5] SMOTE Over-sampling
    → Azınlık sınıfını sentetik örneklerle dengeler
    → YALNIZCA fold içinde uygulanır (sızıntı önleme)
    |
    v
[6] Kosinüs k-NN Graf Yapılandırma (VariantGATv2GNN için)
    → Her varyant bir düğüm; en yakın k komşuları kenar olarak bağlandırılır
    → Koordinat gerektirmez; yalnızca özellik vektörleri kullanılır
```

---

## Eğitim Detayları

| Ayar | Değer | Açıklama |
|---|---|---|
| Çapraz doğrulama | Stratified K-Fold (k=5 varsayılan) | Sınıf dağılımını koruyarak bölen |
| Model seçim metriği | **Macro F1** (doğruluk değil) | Dengesiz sınıflar için daha adil |
| Kalibrasyon bölmesi | Eğitim verisinin %15'i | İzotonik regresyon için ayrı tutulan set |
| Test bölmesi | Veri setinin %20'si | Son performans değerlendirmesi için |
| Rastgele tohum | 42 (tüm bileşenler) | Tekrarlanabilirlik için sabit tohum |
| Kayıp fonksiyonu | WeightedBCELoss (sınıf dengeli) | GNN + DNN için dinamik sınıf ağırlık |
| Erken durdurma | Doğrulama Macro F1 (sabır=5 epoch) | Aşırı uyumlanmayı önler |

---

## Değerlendirme Metrikleri

| Metrik | Açıklama | Yön |
|---|---|---|
| **Macro F1** | Birincil metrik; sınıf dengeli F1 skoru | Yüksek = iyi |
| **ROC-AUC** | ROC eğrisi altındaki alan | Yüksek = iyi |
| **PR-AUC** | Hassasiyet-Duyarlılık eğrisi altındaki alan | Yüksek = iyi |
| **MCC** | Matthews Korelasyon Katsayısı | Yüksek = iyi |
| **Brier Skoru** | Ortalama karesi alınmış olasılık hatası | Düşük = iyi |
| **ECE** | Beklenen Kalibrasyon Hatası | Düşük = iyi |

### Dış Doğrulama (External Validation)

```bash
python main.py --mode external_val --test_file data/test_variants.csv
```

### Adversarial Validation

```bash
python main.py --mode adversarial_val --data_file data/train_variants.csv --test_file data/test_variants.csv
# AUC ≈ 0.5 → eğitim ve test dağılımı benzer (iyi)
# AUC > 0.7 → alan kayması (domain shift) riski var
```

---

## Veri Gereksinimleri

### Girdi Sütunları

| Sütun | Tip | Zorunluluk | Açıklama |
|---|---|---|---|
| `Variant_ID` | String | Zorunlu | Benzersiz tanımlayıcı; pipeline boyunca korunur, **asla özellik olarak kullanılmaz** |
| Sayısal anotasyon özellikleri | Float/Int | Zorunlu | Fonksiyonel anotasyon skorları |
| `Label` | 0/1 | Eğitim için zorunlu, tahmin için isteğe bağlı | 0=Benign, 1=Patojenik |
| `Panel` | String | İsteğe bağlı | Panel tanımlayıcısı (General, Hereditary_Cancer, PAH, CFTR) |
| `Nuc_Context` | String | İsteğe bağlı | ±5 nükleotid bağlam dizesi |
| `AA_Context` | String | İsteğe bağlı | ±5 amino asit bağlam dizesi |

### Anonim Kolon Desteği

Şartname, kolon isimlerinin açıklanmayabileceği senaryoları kapsar. Sistem hem adlandırılmış hem de anonim özellik modunu destekler. Ayrıntılar için `data/contracts/` klasörüne bakınız.

---

## Belirsizlik Ölçümleme (Uncertainty Quantification)

| Özellik | Açıklama |
|---|---|
| **Yöntem** | MC Dropout — test zamanında dropout açık bırakarak N ileri geçiş |
| **İleri geçiş sayısı** | 30 (varsayılan) |
| **Çıktı** | Tahmin entropisi → [0, 1] arasında normalize belirsizlik |
| **Klinik kategoriler** | Yüksek Güven / Orta Güven / Düşük Güven |

---

## Açıklanabilirlik (XAI)

### 1. SHAP
- **TreeExplainer** ile XGBoost ve LightGBM için yerel ve global açıklamalar
- 6 biyolojik kategori gruplandırması ile grup SHAP

### 2. LIME
- Yerel pertürbasyon tabanlı açıklamalar
- Bireysel varyant tahminlerinin yorumu

### 3. GNN Açıklayıcı (GNNExplainer)
- Graf düğüm ve kenar önemliliği maskeleri
- Hangi komşuların tahmine katkı sağladığını gösterir

### 4. Klinik İçgörüler (Türkçe)
- SHAP değerlerinden otomatik Türkçe yorum üretimi
- Risk bölge sınıflandırması: KRİTİK / YÜKSEK / ORTA / DÜŞÜK

---

## Sınırlamalar

| Sınır | Açıklama |
|---|---|
| Sınıf dengesizliği | SMOTE + WeightedBCELoss ile ele alınır; aşırı dengesiz veri setlerinde performans düşebilir |
| Önceden hesaplanmış skorlar gerekli | Ham sekans analizi yapmaz; CADD, SIFT vb. önceden hesaplanmış olmalıdır |
| VUS desteği | VUS örnekleri mevcut eğitim setinden çıkarılmıştır; VUS tahmini için ayrı etiketli veri gerekir |
| Küçük panel performansı | CFTR gibi küçük panellerde (~140 örnek) istatistiksel güç sınırlı olabilir |
| Gerçek klinik validasyon eksik | Bağımsız klinik kohort üzerinde prospektif doğrulama yapılmamıştır |
| Tek dilli XAI | Klinik içgörüler şu an yalnızca Türkçe |

---

## Etik Değerlendirmeler

### Klinik Kullanım Uyarısı

> **Bu sistem ARASTIRMA VE YARIŞMA PROTOTİPİDİR.**
>
> - Klinik tanı koyamaz, tedavi kararı üretemez.
> - Klinik kullanıma hazır değildir.
> - Uzman değerlendirmesinin yerine geçmez.
> - Bağımsız klinik validasyon gerektirir.
> - Klinik kararın tek dayanağı olarak kullanılmamalıdır.
> - İnsan uzman denetimi zorunludur.

### Adillik ve Önyargı

- Performans, gnomAD alel frekans özelliklerinin bileşimi nedeniyle **soy grupları arasında farklılık gösterebilir**
- Eğitim verisinde temsil edilmeyen popülasyonlar için tahmin güvenilirliği düşük olabilir
- Adversarial validation ile eğitim/test dağılım uyumu düzenli olarak kontrol edilmelidir

### Şeffaflık

- SHAP, LIME ve GNNExplainer ile tahmin açıklanabilirliği sağlanmıştır
- Belirsizlik ölçümleme, modelin güven seviyesini raporlamaktadır
- Kaynak kodu MIT lisansı altında açık erişimlidir

---

## Tekrarlanabilirlik

| Bileşen | Yöntem |
|---|---|
| Rastgele tohum | `set_global_seed(42)` — tüm bileşenler (NumPy, PyTorch, Python random) |
| Veri bölmeleri | Sabit `random_state=42` ile StratifiedKFold |
| Model ağırlıkları | Deterministik başlatma (seed kontrollü) |
| Ortam | `requirements.txt` ile sabitlenmiş bağımlılık sürümleri |
| Docker | Dockerfile ile tam tekrarlanabilir ortam |

---

## Komut Satırı Kullanımı

```bash
# Tam eğitim pipeline
python main.py --mode train

# Panel bazlı eğitim
python main.py --mode train --panel cftr --data_file data/train_cftr.csv

# Çapraz doğrulama
python main.py --mode crossval --data_file data/train_variants.csv

# Tahmin (etiket gerektirmez)
python main.py --mode predict --test_file data/test_variants_blind.csv

# Değerlendirme (etiket gerektirir)
python main.py --mode eval --data_file data/test_variants.csv

# Dış doğrulama
python main.py --mode external_val --test_file data/test_variants.csv

# Adversarial validation
python main.py --mode adversarial_val --data_file data/train_variants.csv --test_file data/test_variants.csv

# Açıklanabilirlik
python main.py --mode explain --data_file data/train_variants.csv

# Streamlit arayüzü
streamlit run app.py
```

---

## İlgili Dosyalar

| Dosya | Açıklama |
|---|---|
| `src/core/gnn.py` | VariantGATv2GNN model tanımlaması (GATv2Conv tabanlı) |
| `src/core/models/gnn.py` | VariantGATv2GNN + backward-compat alias'lar |
| `src/models/ensemble.py` | Topluluk birleşimi ve ağırlık optimizasyonu |
| `src/training/trainer.py` | Ana eğitim döngüsü ve çapraz doğrulama |
| `src/inference/pipeline.py` | Uçtan uca tahmin pipeline |
| `src/api/uncertainty.py` | MC Dropout belirsizlik ölçümleme |
| `src/explainability/` | SHAP, LIME, GNN Explainer, klinik içgörüler |
| `src/calibration/calibrator.py` | İzotonik kalibrasyon modülü |
| `configs/default.yaml` | Tüm yapılandırma parametreleri |
| `data_contracts/variant_schema.py` | Pydantic v2 veri şeması |
| `data/contracts/` | JSON veri sözleşmeleri ve şemalar |

---

*Son güncelleme: Nisan 2026 — VARIANT-GNN v3.1 (TEKNOFEST 2026 PDR Sürümü)*

*Bu model kartının kısa versiyonu için kök dizindeki `MODEL_CARD.md` dosyasına bakınız.*

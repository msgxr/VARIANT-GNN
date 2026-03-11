# Model Karti — VARIANT-GNN

## Model Genel Bakis

| Alan | Deger |
|---|---|
| **Model adi** | VARIANT-GNN Hibrit Topluluk (Hybrid Ensemble) |
| **Surum** | 3.0.0 (TEKNOFEST 2026 Finale Surumu) |
| **Mimari** | XGBoost + VariantSAGEGNN (GraphSAGE) + DNN — Cok Modlu Topluluk Modeli |
| **Gorev** | Ikili siniflandirma — Genomik varyant patojenisite tahmini (Benign / Patojenik) |
| **Lisans** | MIT |
| **Calistirma ortami** | Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.5.3 |
| **Egitim suresi** | ~5-15 dakika (CPU), ~2-5 dakika (GPU) |

---

## Amac ve Kullanim Alani

### Birincil Kullanim

- **Ana amac:** Bir genomik varyantin (SNP/indel) Benign (zararsiz) veya Patojenik (hastaliga neden olan) olup olmadigini, onceden hesaplanmis fonksiyonel anotasyon skorlari kullanarak tahmin etmek.
- **Hedef kullanicilar:** Hesaplamali biyologlar, klinik genetikciler, biyoinformatik arastirmacilari ve TEKNOFEST 2026 yarismasi juri uyeleri.
- **Beklenen girdi:** Her satiri bir varyanti temsil eden ve sayisal anotasyon ozelliklerini (CADD, SIFT, PolyPhen2, GERP, gnomAD alel frekanslari vb.) iceren bir CSV dosyasi. Istege bagli olarak ±5 nukleotid/amino asit baglam dizeleri de icerebilir.
- **Beklenen cikti:** Her varyant icin kalibre edilmis olasilik skoru (0-100 araliginda risk skoru), guven seviyesi, yuksek riskli varyant isareti ve klinik karar destek bilgileri.

### Kapsam Disi Kullanimlar

| Kullanim | Neden Kapsam Disinda |
|---|---|
| De novo varyant kesfi | Model yalnizca onceden tanimlanmis varyantlari siniflandirir |
| Yapisal varyant siniflandirmasi | Yalnizca SNP ve kucuk indeller desteklenir |
| Bagimsiz dogrulama olmadan klinik tani kararlari | Model bir arastirma aracidir, tek basina tani araci degildir |
| Ham sekans analizi | Onceden hesaplanmis anotasyon skorlari gereklidir |

> **Uyari:** Bu model klinik tani amacli degildir. Tahminler, klinik veriler, aile oykusu ve uzman genetik degerlendirmesi ile birlikte yorumlanmalidir.

---

## Mimari Detaylari

### Genel Mimari Sema

```
CSV Girdi → Pydantic Dogrulama → On Isleme Pipeline
                                        |
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────┴─────┐     ┌──────┴──────┐     ┌─────┴─────┐
              │  XGBoost  │     │ VariantSAGE │     │    DNN    │
              │  (%40)    │     │   GNN (%40) │     │   (%20)   │
              └─────┬─────┘     └──────┬──────┘     └─────┬─────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                          Agirlikli Topluluk Birlesimi
                                        │
                          Izotonik Kalibrasyon
                                        │
                          Kalibre Edilmis Risk Skoru (0-100)
```

### XGBoost Bileseni

| Ozellik | Aciklama |
|---|---|
| **Tur** | Gradyan guclendirilmis karar agaclari |
| **Girdi** | Tablo formatinda ozellik matrisi (43 ozellik) |
| **Guc** | Dogrusal olmayan ozellik etkilesimlerini verimli sekilde yakalar |
| **Hiperparametre ayari** | Optuna ile Bayes optimizasyonu (`src/training/tune.py`) |
| **Serializasyon** | JSON formati (guvenli, pickle yok) |

### VariantSAGEGNN Bileseni (TEKNOFEST 2026 Birincil Model)

Bu bilesen, projenin en yenilikci parcasidir ve TEKNOFEST 2026 sartnamesine ozel olarak tasarlanmistir:

| Ozellik | Aciklama |
|---|---|
| **Tur** | Induktif, dugum seviyesinde siniflandirici (GraphSAGE) |
| **Graf yapisi** | Her varyant bir dugum; kenarlar ozellik uzayinda koordinatsiz kosinüs k-NN ile olusturulur |
| **Konvolusyon katmanlari** | 3 SAGEConv blogu: BatchNorm + skip connection + Dropout(0.3) |
| **Cok modlu birlestirme** | Istege bagli olarak nukleotid ve amino asit sekans ozelliklerini SequenceEncoder'dan birlestirme |
| **Kayip fonksiyonu** | WeightedBCELoss — sinif dagilimindan dinamik olarak agirlik hesaplar |
| **Induktivlik** | Yeni, gorulmemis varyantlar icin graf yeniden olusturmadan tahmin yapabilir |

**Neden GraphSAGE?**
- Geleneksel GNN'ler (GCN, GAT) transduktiftir — yeni dugumler icin tum grafi yeniden egitmek gerekir
- GraphSAGE induktiftir — komsuluk orneklemesi ile yeni varyantlari ogrenilmis agirliklarla siniflandirir
- Kucuk panellerde (CFTR: 70+70 ornek) bile kararlı performans gosterir

### Cok Modlu Sekans Kodlayici (SequenceEncoder)

±5 nukleotid ve ±5 amino asit baglamini isleyen ikili dallanmali CNN kodlayicisi:

```
Nukleotid Dizesi (±5)  →  Embedding(5,16) → Conv1d(3,pad=1) → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d → 16-dim
                                                                                                              |
Amino Asit Dizesi (±5) →  Embedding(21,16) → Conv1d(3,pad=1) → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d → 16-dim
                                                                                                              |
                                                                                                    Birlestirme → 32-dim cikti
```

- **Cikti:** 32 boyutlu ozellik vektoru, sayisal ozelliklerle birlestirilerek toplam girdi boyutunu arttirir
- **Biyolojik motivasyon:** Varyant cevresimdeki nukleotid/amino asit baglami, mutasyonun fonksiyonel etkisi hakkinda onemli bilgi tasir

### DNN Bileseni

| Ozellik | Aciklama |
|---|---|
| **Tur** | Ileri beslemeli sinir agi (Feed-Forward NN) |
| **Normalizasyon** | BatchNorm katmanlari |
| **Regularizasyon** | Dropout katmanlari |
| **Girdi boyutu** | Dinamik — on islemeden sonra ozellik matrisinden cikarilir |
| **Kayip fonksiyonu** | WeightedBCELoss (sinif dengeli egitim) |

### Topluluk Birlesimi (Ensemble)

Uc modelden gelen olasilik ciktilarinin yapilandırılabilir dogrusal enterpolasyonu:

| Parametre | Varsayilan Deger | Aciklama |
|---|---|---|
| XGBoost agirligi | 0.40 | Tablo verilerinde guclu performans |
| GNN agirligi | 0.40 | Varyantlar arasi iliskileri yakalama |
| DNN agirligi | 0.20 | Tamamlayici dogrusal olmayan ogrenme |

- Agirliklar `scipy.optimize.minimize` (Nelder-Mead) ile dogrulama seti uzerinde optimize edilebilir
- Optimizasyon, bireysel modellerin zayif yonlerini telafi ederek macro F1'i maksimize eder

### Kalibrasyon

| Ozellik | Aciklama |
|---|---|
| **Yontem** | Izotonik Regresyon (birincil) veya Sigmoid/Platt Olcekleme (alternatif) |
| **Veri** | Ayri tutulan kalibrasyon seti (egitim verisinin %15'i) |
| **Amac** | Ham topluluk olasilarini iyi kalibre edilmis risk skorlarina donusturme |
| **Degerlendirme** | ECE (Beklenen Kalibrasyon Hatasi) ve Brier Skoru |

---

## Panel Tabanli Veri (TEKNOFEST 2026)

Model, yarisma sartnamesinde belirtilen dort farkli genomik paneli desteklemektedir:

| Panel | Egitim (P+B) | Test (P+B) | Toplam | Aciklama |
|---|---|---|---|---|
| **General** (Genel) | 1500+1500 | 1000+1000 | 5000 | Genel populasyon varyantlari |
| **Hereditary Cancer** (Kalitsal Kanser) | 200+200 | 100+100 | 600 | Kanser yatkinlik genleri (BRCA1, BRCA2 vb.) |
| **PAH** (Fenilketonuri) | 200+200 | 100+100 | 600 | Fenilalanin hidroksilaz geni varyantlari |
| **CFTR** (Kistik Fibrozis) | 70+70 | 30+30 | 200 | CFTR geni varyantlari (en kucuk panel) |

- Panel bazli egitim ve degerlendirme `--panel` CLI bayragı ile desteklenir
- Her panel icin ayri egitim/test veri setleri `data/` klasorunde bulunur
- Kucuk panellerde (CFTR) WeightedBCELoss ve SMOTE ozellikle kritik oneme sahiptir

---

## Ozellik Gruplari (43 Sayisal Ozellik)

### 1. Sekans ve Degisim Bilgisi
| Ozellik | Aciklama |
|---|---|
| Ref/Alt nukleotid kodlamasi | Referans ve alternatif alel bilgisi |
| Kodon degisim tipi | Missense, nonsense, synonymous vb. |
| Grantham skoru | Amino asit degisiminin fizikokimyasal mesafesi |

### 2. Yerel Sekans Baglami
| Ozellik | Aciklama |
|---|---|
| GC-content penceresi | Varyant cevresindeki GC orani |
| CpG bolgesi | Varyant bir CpG adasinda mi? |
| Motif bozulma skoru | Transkripsiyon faktoru baglama motifi uzerindeki etki |

### 3. Biyokimyasal ve Yapisal Etkiler
| Ozellik | Aciklama |
|---|---|
| Polarite degisimi | Amino asit polarite farki |
| Hidrofobiklik | Hidrofobiklik indeks degisimi |
| Molekuler agirlik | Amino asit molekuler agirlik farki |
| Protein etkisi | Protein yapisina tahmini etki |
| Cozucu erisilebilirligi | Protein yuzeyindeki konumlandirma |

### 4. Evrimsel Korunmusluk
| Ozellik | Aciklama |
|---|---|
| GERP++ | Genomik Evrimsel Oran Profili — korunum skoru |
| PhyloP | Filogenetik p-degeri — pozisyon bazli korunum |
| phastCons | Filogenetik korunum olasiligi |
| SiPhy | Taranan bolgedeki korunum sinyali |

### 5. Populasyon Verileri
| Ozellik | Aciklama |
|---|---|
| gnomAD AF (5 populasyon) | Farkli etnik gruplardaki alel frekanslari |
| ExAC AF | Exon aglama konsorsiyumu alel frekansı |

> **Not:** Populasyon frekanslari, nadir varyantlarin patojenik olma olasiligiyla ters orantilidir.

### 6. In Silico Risk Skorlari
| Ozellik | Aciklama |
|---|---|
| SIFT | Amino asit degisiminin protein fonksiyonuna etkisi (dusuk = zarar) |
| PolyPhen2 | Polimorfizm fenotipi tahmincisi (yuksek = zarar) |
| CADD | Birlesik Anotasyon Bagimli Tukenisenlik skoru (yuksek = zarar) |
| REVEL | Nadir ekzomik varyantlar icin topluluk skoru |
| MutPred2 | Mutasyon patolojiklik tahmincisi |
| VEST4 | Varyant Etki Skor Araci |
| PROVEAN | Protein Varyasyon Etki Analizcisi |
| MutationTaster | Mutasyon hastalik potansiyeli tahmincisi |
| MetaSVM/LR | Meta-siniflandirici skorlari (destek vektor makinesi / lojistik regresyon) |
| M-CAP | Mendelyen Klinik Uygulanabilir Patojenisite skoru |

---

## On Isleme Pipeline

Tum on isleme adimlari **yalnizca egitim verisi uzerinde** fit edilir (her CV fold'u icinde). Bu, veri sizintisini (data leakage) onleyen altin standart yaklasimdir.

```
Ham CSV Verisi
    |
    v
[1] Medyan Imputation (SimpleImputer)
    → Eksik degerleri egitim setinin medyani ile doldurur
    |
    v
[2] Robust Scaler (RobustScaler)
    → IQR tabanli olcekleme; aşırı degerlere dayanikli
    |
    v
[3] Istege Bagli: Ozellik Secimi
    → VarianceThreshold + SelectKBest (karsilikli bilgi)
    → Dusuk varyanslı ve bilgisiz ozellikleri eler
    |
    v
[4] Istege Bagli: AutoEncoder Latent Ozellik Birlestirme
    → 43 ozellik → 16 boyutlu latent temsil → toplam 59 boyut
    → Bottleneck mimarisi ile ozellik siksitirma
    |
    v
[5] SMOTE Over-sampling
    → Azinlik sinifini sentetik orneklerle dengeler
    → YALNIZCA fold icinde uygulanir (sizinti onleme)
    |
    v
[6] Kosinüs k-NN Graf Yaplandirma (VariantSAGEGNN icin)
    → Her varyant bir dugum; en yakin k komsulari kenar olarak baglantilandi rilir
    → Koordinat gerektirmez; yalnizca ozellik vektorleri kullanilir
```

> **Kritik:** SMOTE, AutoEncoder ve ozellik secimi adimlarinin YALNIZCA egitim fold'u icinde uygulanmasi, test setinin herhangi bir sekilde egitim surecini etkilememesini garanti eder.

---

## Egitim Detaylari

| Ayar | Deger | Aciklama |
|---|---|---|
| Capraz dogrulama | Stratified K-Fold (k=5 varsayilan) | Sinif dagılımını koruyarak bolen |
| Model secim metrigi | **Macro F1** (dogruluk degil) | Dengesiz siniflar icin daha adil |
| Kalibrasyon bolmesi | Egitim verisinin %15'i | Izotonik regresyon icin ayri tutulan set |
| Test bolmesi | Veri setinin %20'si | Son performans degerlendirmesi icin |
| Rastgele tohum | 42 (tum bilesenler) | Tekrarlanabilirlik icin sabit tohum |
| Kayip fonksiyonu | WeightedBCELoss (sinif dengeli) | SAGE + DNN icin dinamik sinif agirlik |
| Erken durdurma | Dogrulama Macro F1 (sabir=5 epoch) | Asiri uyumlanamayi (overfitting) onler |
| Ogrenme orani | Yapilandirmadan (varsayilan: 1e-3) | GNN ve DNN icin |
| Epoch sayisi | Maksimum 100 (erken durdurma ile) | Pratikte ~20-40 epoch'ta durur |

### Hiperparametre Optimizasyonu

Optuna kutuphanesi ile Bayes tabanli hiper parametre arama:

| Parametre | Aralik | Olcek |
|---|---|---|
| max_depth | [3, 10] | Tamsayi |
| learning_rate | [0.01, 0.3] | Logaritmik |
| n_estimators | [100, 500] | Tamsayi |
| subsample | [0.6, 1.0] | Surekli |
| colsample_bytree | [0.6, 1.0] | Surekli |

- Varsayilan deneme sayisi: 30
- Optimizasyon metrigi: Macro F1 (capraz dogrulama ortalaması)

---

## Degerlendirme Metrikleri

Tum metrikler, kalibrasyondan sonra ayri tutulan test seti uzerinde raporlanir:

| Metrik | Aciklama | Yon |
|---|---|---|
| **Macro F1** | Birincil metrik; sinif dengeli F1 skoru | Yuksek = iyi |
| **ROC-AUC** | ROC egrisi altindaki alan; ayirt edicilik gucu | Yuksek = iyi |
| **PR-AUC** | Hassasiyet-Duyarlilik egrisi altindaki alan | Yuksek = iyi |
| **MCC** | Matthews Korelasyon Katsayisi; dengeli ikili metrik | Yuksek = iyi |
| **Brier Skoru** | Ortalama karesi alinmis olasilik hatasi | Dusuk = iyi |
| **ECE** | Beklenen Kalibrasyon Hatasi (10 esit genislikli kutu) | Dusuk = iyi |

### Dis Dogrulama (External Validation)

Dis dogrulama modu, onceden egitilmis bir modeli yeni test verileri uzerinde degerlendirir:

```bash
python main.py --mode external_val --test-data data/test_variants_blind.csv
```

Cikti olarak F1, ROC-AUC, Brier Skoru, Hassasiyet, Duyarlilik hesaplanir ve JSON raporu disa aktarilir.

### Adversarial Validation (Rakip Dogrulama)

Egitim ve test veri dagilimlarinin benzerligini olcmek icin adversarial dogrulama modulu bulunur:

```bash
# src/evaluation/adversarial_validation.py
# AUC ≈ 0.5 → egitim ve test dagilimi benzer (iyi)
# AUC > 0.7 → alan kaymasi (domain shift) riski var (kotu)
```

---

## Veri Gereksinimleri

### Girdi Sutunlari

| Sutun | Tip | Zorunluluk | Aciklama |
|---|---|---|---|
| `Variant_ID` | String | Zorunlu | Benzersiz tanimlayici; pipeline boyunca korunur, **asla ozellik olarak kullanilmaz** |
| Sayisal anotasyon ozellikleri | Float/Int | Zorunlu | 43 adet fonksiyonel anotasyon skoru |
| `Label` | 0/1 | Egitim icin zorunlu, tahmin icin istege bagli | 0=Benign, 1=Patojenik |
| `Panel` | String | Istege bagli | Panel tanimlayicisi (General, Hereditary_Cancer, PAH, CFTR) |
| `Nuc_Context` | String | Istege bagli | ±5 nukleotid baglam dizesi (ornek: "ACGTACGTAC") |
| `AA_Context` | String | Istege bagli | ±5 amino asit baglam dizesi (ornek: "MVLSPADKTN") |

### Veri Sozlesmesi

Tum girdiler islenmeden once `data_contracts/variant_schema.py` dosyasindaki Pydantic v2 semasina karsi dogrulanir:

```python
from data_contracts.variant_schema import validate_dataset

result = validate_dataset(df)
if not result.is_valid:
    raise ValueError(f"Sema dogrulama hatasi: {result.errors}")
```

---

## Belirsizlik Olcumleme (Uncertainty Quantification)

Model, MC Dropout (Monte Carlo Dropout) yontemi ile epistemik belirsizlik tahmini saglar:

| Ozellik | Aciklama |
|---|---|
| **Yontem** | Test zamaninda dropout acik birakarak N ileri gecis |
| **Ileri gecis sayisi** | 30 (varsayilan) |
| **Cikti** | Tahmini entropi → [0, 1] araliginda normalize belirsizlik |
| **Klinik kategoriler** | Yuksek Guven / Orta Guven / Dusuk Guven |

Bu sayede model, "Bu tahmin hakkinda ne kadar emin?" sorusuna yanitlayabilir — bu, klinik karar destek sistemleri icin kritik oneme sahiptir.

---

## Aciklanabilirlik (XAI — Explainable AI)

Model, dort farkli aciklanabilirlik katmani sunar:

### 1. SHAP (SHapley Additive exPlanations)
- **TreeExplainer** ile XGBoost icin yerel ve global aciklamalar
- Her ozelligin tahmini nasil etkiledigini gosteren ozet ve selale grafikleri
- Ozellik onemi siralamasi

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Yerel pertürbasyon tabanli aciklamalar
- Bireysel varyant tahminlerinin neden yapildigini aciklar
- 500 ornek varsayilan pertürbasyon sayisi

### 3. GNN Aciklayici (GNNExplainer)
- Graf dugum ve kenar onemliligi maskeleri
- Hangi komsularin tahmine katki sagladigini gosterir
- PyTorch Geometric entegrasyonu

### 4. Klinik Icgoruler (Turkce)
- SHAP degerlerinden otomatik Turkce klinik yorum uretimi
- 43 ozellik → 6 biyolojik kategori eslemesi
- Risk bolge siniflandirmasi: KRITIK / YUKSEK / ORTA / DUSUK
- Her risk bolgesi icin klinik eylem onerileri

---

## Sinirlamalar

| Sinir | Aciklama | Olasi Cozum |
|---|---|---|
| Sinif dengesizligi | SMOTE + WeightedBCELoss ile ele alinir, ancak asiri dengesiz veri setlerinde performans dusebilir | FocalLoss alternatifi mevcut (henuz varsayilan degil) |
| Onceden hesaplanmis skorlar gerekli | Ham sekans analizi yapmaz; CADD, SIFT vb. onceden hesaplanmis olmalidir | VEP veya ANNOVAR ile on isleme gerekli |
| VUS destegi | Belirsiz Onemi Bilinmeyen Varyantlar (VUS) mimari olarak desteklenir | Etiketli VUS egitim verisi gereklidir |
| Kucuk panel performansi | CFTR gibi kucuk panellerde (140 ornek) istatistiksel guc sinirli olabilir | Daha fazla veri toplama veya transfer ogrenme |
| GPU bagimliligi | GNN ve DNN bilesenleri GPU ile onemli olcude hizlanir | CPU modunda da calişir ancak yavas |
| Tek dilli XAI | Klinik icgoruler yalnizca Turkce | Dil destek modulu genişletilebilir |

---

## Etik Degerlendirmeler

### Klinik Kullanim Uyarisi

> **Bu model bir ARASTIRMA ARACIDIR ve klinik tani kararlarinin tek temeli olarak kullanilmamalidir.**

- Tahminler, klinik veriler, aile oykusu ve uzman klinik genetik degerlendirmesi ile birlikte yorumlanmalidir
- Model, bagimsiz klinik dogrulama olmadan hasta yonetim kararlarini yonlendirmek icin kullanilmamalidir

### Adillik ve Onyargi

- Performans, gnomAD alel frekans ozelliklerinin bilesimi nedeniyle **soy gruplari arasinda farklilik gosterebilir**
- Egitim verisinde temsil edilmeyen populasyonlar icin tahmin guvenirligi dusuk olabilir
- Adversarial validation ile egitim/test dagilim uyumu duzenli olarak kontrol edilmelidir

### Seffaflik

- Model karti, tum mimari ve egitim detaylarini acik olarak belgelemektedir
- SHAP, LIME ve GNNExplainer ile tahmin aciklanabilirligi saglanmistir
- Belirsizlik olcumleme, modelin guven seviyesini raporlamaktadir
- Kaynak kodu MIT lisansi altinda acik erisimlidir

---

## Tekrarlanabilirlik

| Bilesen | Yontem |
|---|---|
| Rastgele tohum | `set_global_seed(42)` — tum bilesenler (NumPy, PyTorch, Python random) |
| Veri bolmeleri | Sabit `random_state=42` ile StratifiedKFold |
| Model agirliklari | Deterministik baslatma (seed kontrollü) |
| Ortam | `requirements.txt` ile sabitlenmis bagimlilik surümleri |
| Docker | Dockerfile ile tam tekrarlanabilir ortam |

---

## Komut Satiri Kullanimi

```bash
# Tam egitim pipeline (varsayilan: General panel)
python main.py --mode train

# Panel bazli egitim
python main.py --mode train --panel cftr

# Dis dogrulama
python main.py --mode external_val --test-data data/test_variants_blind.csv

# Streamlit arayuzu
streamlit run app.py
```

---

## Ilgili Dosyalar

| Dosya | Aciklama |
|---|---|
| `src/models/gnn.py` | VariantSAGEGNN model tanimlamasi |
| `src/models/ensemble.py` | Topluluk birlesimi ve agirlik optimizasyonu |
| `src/training/trainer.py` | Ana egitim dongusu ve capraz dogrulama |
| `src/inference/pipeline.py` | Uctan uca tahmin pipeline |
| `src/inference/uncertainty.py` | MC Dropout belirsizlik olcumleme |
| `src/explainability/` | SHAP, LIME, GNN Explainer, klinik icgorüler |
| `src/calibration/calibrator.py` | Izotonik kalibrasyon modulu |
| `configs/default.yaml` | Tum yapilandirma parametreleri |
| `data_contracts/variant_schema.py` | Pydantic v2 veri semasi |

---

*Son guncelleme: Mart 2026 — VARIANT-GNN v3.0 (TEKNOFEST 2026 Finale Surumu)*

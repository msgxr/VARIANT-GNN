# VARIANT-GNN — Türkçe Proje Özeti

**TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Üniversite ve Üzeri Kategorisi**

---

## Proje Kimliği

| Alan | Değer |
|---|---|
| **Proje Adı** | VARIANT-GNN |
| **Takım** | XYRA3 — ID: #909249 |
| **Başvuru ID** | #4865399 |
| **Kategori** | TEKNOFEST 2026 Sağlıkta Yapay Zekâ — Üniversite ve Üzeri |
| **PSR Puanı** | 93.00 / 100 — GEÇİLDİ |
| **Güncel Aşama** | PDR (Proje Detay Raporu) Geliştirmesi |

---

## Projenin Tanımı

VARIANT-GNN, TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması Üniversite ve Üzeri kategorisinde tanımlanan göreve uygun olarak geliştirilmiş bir **araştırma ve yarışma prototipidir.**

**Görev:** Missense genetik varyantların Patojenik / Benign olarak sınıflandırılması.

Sistem, dört farklı makine öğrenmesi modelinin hibrit topluluk mimarisini kullanan, uçtan uca kalibre edilmiş bir tahmin pipelineı sunmaktadır.

> **Önemli Uyarı:** Bu sistem klinik tanı koymaz, tedavi kararı üretmez ve klinik kullanıma hazır değildir. Araştırma ve yarışma prototipi niteliğindedir. Uzman değerlendirmesinin yerine geçmez; bağımsız klinik validasyon gerektirir.

---

## Sistem Mimarisi

### Dört Modlu Hibrit Topluluk

```
Ham Varyant Profili (CSV)
         |
         v
   Ön İşleme Pipeline
   (Imputation → Scaler → Özellik Seçimi → AutoEncoder → SMOTE)
         |
    ┌────┴────┬──────────┬────────────┐
    │         │          │            │
 XGBoost  LightGBM  GATv2GNN      DNN
  (%35)    (%30)    (%25)         (%10)
    │         │          │            │
    └────┬────┴──────────┴────────────┘
         |
    Topluluk Birleşimi
    (Nelder-Mead ağırlık optimizasyonu)
         |
    İzotonik Kalibrasyon
         |
    Kalibre Edilmiş Risk Skoru (0–100)
    + MC Dropout Belirsizlik
    + SHAP/LIME/GNNExplainer
```

### Model Bileşenleri

| Model | Ağırlık | Teknoloji |
|---|---|---|
| **XGBoost** | %35 | Gradyan güçlendirilmiş karar ağaçları (JSON serializasyon) |
| **LightGBM** | %30 | Yaprak bazlı gradyan güçlendirme |
| **VariantGATv2GNN** | %25 | GATv2 dikkat mekanizmalı grafik sinir ağı |
| **DNN** | %10 | İleri beslemeli derin sinir ağı |

### Destekleyici Katmanlar

| Katman | Açıklama |
|---|---|
| İzotonik Kalibrasyon | Olasılıkların gerçeğe uygunluğunu artırır |
| MC Dropout Belirsizlik | 30 ileri geçişle epistemik belirsizlik ölçümü |
| SHAP Açıklanabilirlik | Özellik önem analizi (global + yerel) |
| LIME Açıklanabilirlik | Bireysel varyant kararı açıklaması |
| GNNExplainer | Grafik düğüm/kenar önem analizi |
| Panel Değerlendirme | General, Hereditary Cancer, PAH, CFTR |
| External Validation | Bağımsız test seti değerlendirme |
| Adversarial Validation | Eğitim/test dağılım uyumu kontrolü |

---

## Veri Mimarisi

### Panel Yapısı (TEKNOFEST 2026)

| Panel | Eğitim (P+B) | Test (P+B) | Toplam |
|---|---|---|---|
| Genel | 1500+1500 | 1000+1000 | 5000 |
| Herediter Kanser | 200+200 | 100+100 | 600 |
| PAH | 200+200 | 100+100 | 600 |
| CFTR | 70+70 | 30+30 | 200 |

### Veri Özellikleri

- **Özellik türü:** Önceden hesaplanmış fonksiyonel anotasyon skorları
- **Temel özellikler:** CADD, SIFT, PolyPhen2, GERP, PhyloP, gnomAD AF, REVEL, MutPred2, vb.
- **İsteğe bağlı:** ±5 nükleotid / amino asit bağlam dizeleri (`Nuc_Context`, `AA_Context`)
- **Anonim kolon desteği:** Şartname anonim özellik verirse sistem buna uyum sağlar
- **Gizlilik:** KVKK + GDPR uyumlu; ham klinik veri içermez

---

## Kurulum

```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# CPU (Geliştirme / CI)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

pip install -r requirements.txt
```

---

## Kullanım

```bash
# Eğitim (genel panel)
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
python main.py --mode adversarial_val \
  --data_file data/train_variants.csv \
  --test_file data/test_variants.csv

# Açıklanabilirlik raporu
python main.py --mode explain --data_file data/train_variants.csv

# Web arayüzü
streamlit run app.py
```

---

## Eğitim Protokolü

| Parametre | Değer |
|---|---|
| Çapraz doğrulama | Stratified K-Fold (k=5) |
| Birincil metrik | **Macro F1** |
| Kalibrasyon seti | Eğitim verisinin %15'i |
| Test seti | Veri setinin %20'si |
| Rastgele tohum | 42 (tüm bileşenler) |
| Veri sızıntısı kontrolü | SMOTE ve preprocessing yalnızca fold içinde fit edilir |

---

## Performans Değerlendirme Metrikleri

| Metrik | Açıklama |
|---|---|
| **Macro F1** | Birincil; sınıf dengeli F1 |
| **ROC-AUC** | Ayırt edicilik gücü |
| **PR-AUC** | Hassasiyet-Duyarlılık dengesi |
| **MCC** | Matthews Korelasyon Katsayısı |
| **Brier Skoru** | Olasılık kalibrasyonu |
| **ECE** | Beklenen Kalibrasyon Hatası |

---

## Testler

```bash
# Smoke testler
pytest tests/smoke/ -v

# Unit testler
pytest tests/unit/ -v

# Integration testler
pytest tests/integration/ -v
```

---

## Dizin Yapısı

```
VARIANT-GNN/
├── main.py                    # Ana CLI giriş noktası
├── app.py                     # Streamlit web arayüzü
├── MODEL_CARD.md              # Kısa model kartı
├── DATA_CARD.md               # Veri kartı
├── PROJECT_STATUS.md          # Proje olgunluk durumu
├── TECHNICAL_DEBT.md          # Teknik borç listesi
├── ROADMAP.md                 # Geliştirme yol haritası
├── configs/                   # Yapılandırma dosyaları
├── data/                      # Veri + sözleşmeler
│   ├── contracts/             # JSON veri şemaları
│   └── samples/               # Örnek/sentetik veri
├── src/                       # Kaynak kod
│   ├── core/                  # Model tanımları (GNN, DNN)
│   ├── models/                # Ensemble ve model proxy'leri
│   ├── training/              # Eğitim pipeline
│   ├── inference/             # Tahmin pipeline
│   ├── evaluation/            # Metrikler, validasyon
│   ├── explainability/        # SHAP, LIME, GNNExplainer
│   ├── calibration/           # İzotonik kalibrasyon
│   ├── features/              # Preprocessing, autoencoder
│   └── ui/                    # Streamlit bileşenleri
├── tests/                     # Test altyapısı
│   ├── smoke/                 # Import testleri
│   ├── unit/                  # Birim testler
│   └── integration/           # Entegrasyon testleri
├── docs/                      # Dokümantasyon
│   ├── MODEL_CARD.md          # Detaylı model kartı
│   ├── clinical/              # Klinik uyarı belgeleri
│   └── submission/            # TEKNOFEST teslim belgeleri
├── reports/                   # Üretilen raporlar ve grafikler
└── submission/                # Final teslim paketi
```

---

## Klinik Kullanım Uyarısı

> **Bu sistem TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması için geliştirilmiş bir araştırma ve yarışma prototipidir.**
>
> - Patojenik / Benign sınıflandırma sinyali üretir; kesin klinik tanı koymaz.
> - Uzman değerlendirmesinin yerine geçmez.
> - Bağımsız klinik validasyon gerektirir.
> - Klinik kararın tek dayanağı olarak kullanılmamalıdır.
> - İnsan uzman denetimi zorunludur.

---

## Lisans

MIT Lisansı — detaylar için `LICENSE` dosyasına bakınız.

---

## İletişim

- **Takım:** XYRA3 (#909249)
- **Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zekâ
- **GitHub:** [msgxr/VARIANT-GNN](https://github.com/msgxr/VARIANT-GNN)

# Jüri Teslim Rehberi — VARIANT-GNN

> **Şartname Referansı:** TEKNOFEST 2026 §7.2 (External Validation), §7.3
> (F1 değerlendirmesi), §7.5 (jüri kod re-run yetkisi).

Bu rehber, **TEKNOFEST 2026 jürisinin** sıfırdan başlayarak VARIANT-GNN
projesini **yerel ortamlarında çalıştırıp** beyan edilen sonuçları
doğrulamasını sağlar.

---

## 1. Sistem Gereksinimleri

| Gereksinim | Asgari | Önerilen |
|------------|--------|----------|
| Python | 3.10 | 3.11 / 3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| GPU (opsiyonel) | yok | NVIDIA CUDA 11.8 |
| OS | macOS / Linux / Windows + WSL2 | Ubuntu 22.04 |

> **Not:** Yarışma çıkarımı tamamen offline ve CPU üzerinde çalışır;
> GPU yalnızca yeniden eğitim için gereklidir.

---

## 2. Hızlı Başlangıç (5 Dakika)

```bash
# 1. Repo'yu klonla
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# 2. Sanal ortam kur (Python >=3.10,<3.13)
python3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükle (sabit versiyonlar — §7.5)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Testleri çalıştır (418 test fonksiyonu, 39 dosya; toplanan item sayısı CI junit artefaktında)
pytest tests/ -q

# 5. Eğitim (gerçek yarışma verisi paylaşıldıktan sonra)
python main.py --mode train --config configs/pdr.yaml \
               --data_file data/train_variants.csv

# 6. Jüri inference (offline, label kullanmaz)
python submission/predict.py \
    --input <jury_test.csv> \
    --model_dir models \
    --output submission/predictions.csv \
    --config configs/pdr.yaml
```

---

## 3. Inference (Tahmin Üretme)

### 3.1 Standart Path (Şartname §7.2)

> **Jüri CSV formatı — tek kaynak (canonical):** Teslim edilecek jüri CSV'sinin
> kolon seti yalnızca koddan tanımlıdır → `src/scientific/submission_validator.py`
> `JURY_COLUMNS`:
> `Variant_ID, prediction_label, pathogenic_probability, calibrated_risk, confidence_level, uncertainty_score, expert_review_flag` (7 kolon).
>
> **Resmi submission dosya formatı HENÜZ duyurulmadı (UNVERIFIED).** Bu nedenle
> GÜVENLİ varsayılan teslim biçimi `--jury_minimal` modudur: yalnız
> `Variant_ID + prediction_label` (ikili 0/1). 7-kolonlu zengin format iç
> analiz/doğrulama içindir; resmi format açıklanınca güncellenecektir.

**Güvenli teslim (önerilen — 2 kolon):**
```bash
python submission/predict.py \
    --input  <BLIND_TEST_CSV> \
    --model_dir models \
    --output submission/predictions.csv \
    --config configs/pdr.yaml \
    --jury_minimal
```

**Zengin çıktı (iç analiz — `JURY_COLUMNS`, 7 kolon):**
```bash
python submission/predict.py \
    --input  <BLIND_TEST_CSV> \
    --model_dir models \
    --output submission/predictions.csv \
    --config configs/pdr.yaml
```

**Üretilen çıktı:**
- `submission/predictions.csv` — jüri tahmin CSV'i (`--jury_minimal`: 2 kolon; aksi halde `JURY_COLUMNS` 7 kolon)
- `submission/reports/leakage_report_inference.json` — sızıntı raporu
- `submission/reports/inference_summary.json` — özet istatistikler

**Jüri CSV kolonları (`JURY_COLUMNS`, canonical — 7 kolon):**

| Kolon | Tip | Açıklama |
|-------|-----|---------|
| `Variant_ID` | string | Varyant kimliği |
| `prediction_label` | int {0,1} | İkili tahmin (1 = Pathogenic, 0 = Benign) |
| `pathogenic_probability` | float [0,1] | Ham ensemble olasılığı |
| `calibrated_risk` | float [0,100] | Isotonic kalibrasyon sonrası risk skoru |
| `confidence_level` | float [0,100] | Güven seviyesi (1 − belirsizlik) |
| `uncertainty_score` | float [0,1] | MC-Dropout std (yüksek = belirsiz) |
| `expert_review_flag` | bool | Yüksek belirsizlik → uzman gözden geçirme önerisi (araştırma amaçlı; klinik karar DEĞİL) |

> **SUPERSEDED:** Önceki 10-kolonlu çıktı şeması
> (`Panel`, `Prediction`, `Benign_Probability`, eski triyaj-bayrağı,
> `Model_Version`, `Inference_Timestamp` dahil) artık jüri teslim formatı
> DEĞİLDİR — kod ile uyumsuzdu. Triyaj bayrağı araştırma-güvenli
> `expert_review_flag` (uzman gözden geçirme önerisi) olarak yeniden
> adlandırılmıştır. Canonical jüri CSV'si yukarıdaki `JURY_COLUMNS` listesidir.
> (`runner` iç çalışma dosyası ek alanlar üretebilir; bunlar jüriye sunulan
> teslim CSV'sine yazılmaz.)

### 3.2 Anonim Kolon Senaryosu (§3.2)

Şartname §3.2 öznitelik kolon isimlerinin verilmeyebileceğini belirtir.
VARIANT-GNN bunu otomatik tespit eder:

```bash
python -c "
from src.api.pipeline import InferencePipeline
from src.inference.anonymous_inference import predict_anonymous_csv

pipe = InferencePipeline().load()
predict_anonymous_csv(
    csv_path='data/anonymous_test.csv',
    pipeline=pipe,
    output_csv='submission/predictions_anonymous.csv',
)
"
```

**Otomatik tespit kuralları:**
- `Variant_ID` → uniqueness ≥ %99
- `Panel` → değerleri {General, Hereditary_Cancer, PAH, CFTR} ⊆ ile %60+ eşleşme
- `Nuc_Context` → IUPAC alfabesi + 5–25 karakter
- `AA_Context` → amino asit alfabesi + 5–25 karakter
- Sayısal öznitelikler → distributional signature ile training-time
  şablonuna eşlenir (`feature_signatures` preprocessor'da kaydedilir).

---

## 4. Modeli Yeniden Eğitme (Reproducibility — §7.5)

```bash
# 1. Eğitim verisini yerleştir
cp <jury_train.csv> data/train_variants.csv

# 2. Eğitim çalıştır (5-fold CV + final fit + kalibrasyon)
python main.py --mode train --config configs/pdr.yaml

# 3. External validation
python main.py --mode external_val \
               --test_file data/test_variants.csv

# 4. Reproducibility manifest üret
python -c "
from src.utils.reproducibility_manifest import ManifestBuilder
from src.config import get_settings
from pathlib import Path

builder = ManifestBuilder(seed=42)
builder.with_data(train_csv='data/train_variants.csv')
builder.with_settings(get_settings())
builder.with_artifacts(Path('models'))
builder.save(Path('models/reproducibility_manifest.json'))
"
```

**Beklenen Çıktı:**
- `models/*.json` + `*.pth` + `*.pkl` — model artefaktları
- `models/reproducibility_manifest.json` — §7.5 imzalı manifest
- `reports/cv_report.json` — 5-fold CV özeti
- `reports/external_validation_report.json` — panel kırılımlı F1

---

## 5. Manifest Doğrulama

Jüri, kayıtlı manifest'in artefaktlarla uyumlu olduğunu doğrulayabilir:

```python
from pathlib import Path
from src.utils.reproducibility_manifest import verify_manifest_chain

result = verify_manifest_chain(
    manifest_path = Path("models/reproducibility_manifest.json"),
    models_dir    = Path("models"),
)
print(result)
# → {'manifest_integrity': True, 'all_match': True, ...}
```

---

## 6. Ek Modlar (Tartışma + Doğrulama)

```bash
# Çapraz panel genelleştirme matrisi (§3.2 4 panel)
python main.py --mode panel_transfer --data_file data/train_variants.csv

# Etiket kalitesi (Confident Learning)
python main.py --mode label_quality --data_file data/train_variants.csv

# Ablation analizi (model + preprocessing katkısı)
python main.py --mode ablation --data_file data/train_variants.csv

# Adversarial validation (train/test domain shift)
python main.py --mode adversarial_val \
               --data_file data/train_variants.csv \
               --test_file data/test_variants.csv

# Açıklanabilirlik (SHAP + GNNExplainer + ACMG)
python main.py --mode explain --data_file data/train_variants.csv
```

---

## 7. Streamlit Demo Arayüzü (Görsel İnceleme)

```bash
streamlit run app.py
# → http://localhost:8501
```

UI bileşenleri:
- Tek/toplu CSV varyant analizi
- Risk haritası, kalibrasyon eğrisi, model karşılaştırma grafikleri
- SHAP, LIME, GNNExplainer açıklama panelleri
- ACMG kriter raporu, araştırma amaçlı PDF analiz raporu üretimi (FPDF2 ile; klinik tanı/karar DEĞİL)

---

## 8. Sorun Giderme

| Sorun | Çözüm |
|------|-------|
| `ModuleNotFoundError: torch_geometric` | `pip install torch-geometric==2.5.3` |
| GPU bulunmadı uyarısı | Sistem otomatik CPU'ya düşer; sorun değil |
| `Required artifact missing` | Önce `python main.py --mode train` çalıştırın |
| Test düşüklük | `pytest tests/ -q -v` ile detaylı çıktı alın |
| Streamlit port çakışması | `streamlit run app.py --server.port 8502` |

---

## 9. İletişim

Bu sistemin sahibi takım:
- **Takım Adı:** XYRA3
- **Takım ID:** 909249
- **Başvuru ID:** 5200240

Yarışma organizasyonel sorular için: `iletisim@teknofest.org` (§11).

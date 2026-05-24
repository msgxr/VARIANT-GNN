# Final Reproducibility Checklist — VARIANT-GNN

**Son güncelleme:** 2026-05-19  
**Doğrulayan:** XYRA3 (#909249)  
Senaryo: Jüri, finale kalan takımların reposunu açarak modeli yeniden çalıştırmak istiyor.
Bu checklist, o senaryoda her adımın sorunsuz geçmesi için hazırlanmıştır.

---

## 1. Ortam Kurulumu

- [x] requirements.txt veya environment.yml mevcut → `requirements.txt`, `environment.yml`, `environment-gpu-cu118.yml`
- [x] Python versiyonu açıkça belirtilmiş → Python 3.12 (`pyproject.toml`, `Dockerfile`)
- [x] CUDA / CPU uyumluluğu dokümante edilmiş → `configs/default.yaml`: `device: auto`; CPU'da çalışır
- [x] Kurulum komutu README'de tek satırda verilmiş → `pip install -r requirements.txt`
- [x] Ortam kurulumu temiz sanal ortamda test edilmiş → `Dockerfile` ile izole ortam

---

## 2. Bağımlılıklar

- [x] Tüm kütüphane versiyonları sabitlenmiş → `requirements.txt` pin'li sürümler
- [x] Versiyon çakışması olmayan bağımlılık ağacı → CI pipeline doğruladı
- [x] Özel/yerel kütüphane varsa kurulum talimatı eklendi → yok (tüm PyPI)
- [x] PyTorch Geometric versiyonu belirtilmiş → `requirements.txt`: `torch-geometric==2.5.*`

---

## 3. Seed Yönetimi

- [x] Global seed (NumPy, PyTorch, Python random) sabitlenmiş → `src/utils/reproducibility.py`, `src/utils/seeds.py`
- [x] Her deney için aynı seed ile aynı sonuç alınabiliyor → seed_stability_report: F1 std=0.0013 (5 seed)
- [x] CUDA deterministik modu etkinleştirilmiş → `torch.backends.cudnn.deterministic=True` (`src/utils/reproducibility.py`)
- [x] Seed değeri README ve config dosyasında belirtilmiş → `configs/default.yaml`: `seed: 42`

**Kanıt:** `reports/seed_stability.json` — 5 seed × CV F1 = 0.8667 ± 0.0013

---

## 4. Model Dosyaları

- [x] Eğitilmiş model ağırlıkları kaydedilmiş → `models/` klasörü (gnn_model.pth, xgb_model.json, dnn_model.pth, preprocessor.pkl, calibrator.pkl)
- [x] Her panel için: tek model + `--panel` argümanı ile panel-aware inference
- [x] Model dosyaları repo içinde → `models/` klasörü (küçük boyutlu, git ile takip edilebilir)
- [x] SHA256 doğrulandı → `submission/teknofest/checksums.json` ve `artifact_manifest.json`
- [x] Model dosyası ile eğitim kodunun versiyon uyumu → `models/PROVENANCE.json`

**SHA256 Referansı:**
```
models/gnn_model.pth:    c00346d7fa879a0062b672ca53d874e1f6d946519041f93b8cf40de63567e3d7
models/xgb_model.json:   cf63b96f3fc93257e99dd543190f6372a349778eb0e77f65683a1ba2b5a28690
models/preprocessor.pkl: 333f8d69e1e9f6ac7e95383dd1881cda08a7f978710b3f5a4155518bda5afb0f
models/calibrator.pkl:   81ca0d8ee1a95994be60b217ad55ba16383e563229c01f8d6d22d535fb0c88d4
```

---

## 5. İnferans Komutu

- [x] Tahmin için tek komut yeterli → `python submission/predict.py --panel General --input test.csv`
- [x] Her panel için komut README'de örneklenmiş → `docs/submission_guide.md`
- [x] Çıktı formatı belirtilmiş → `submission/teknofest/jury_predictions.csv` formatı: `Variant_ID,Predicted_Label,Probability`
- [x] Çıktıdaki kolon isimleri dokümante edilmiş → `src/inference/prediction_schema.py`

**Örnek:**
```bash
python submission/predict.py \
    --panel General \
    --input data/synthetic/test_general.csv \
    --output predictions_general.csv
```

---

## 6. Tahmin Çıktısı

- [x] Çıktı dosyası yarışmanın istediği formatta → CSV, kolon: Variant_ID, Predicted_Label, Probability
- [x] Patojenik / Benign etiketleri doğru kodlanmış → Pathogenic=1, Benign=0 (`data/contracts/label_mapping.json`)
- [x] Olasılık skoru çıktıda mevcut → `Probability` kolonu
- [x] Test seti sırası korunuyor → inference pipeline'da shuffle yok

---

## 7. Log Dosyası

- [x] Eğitim logu mevcut → `logs/training_log.md`
- [x] 5-fold CV sonuçları kaydedilmiş → `reports/cv_report.json`
- [x] Seed stabilite raporu → `reports/seed_stability.json`
- [x] Panel bazlı metrikler → `reports/cross_panel_eval.json`
- [x] Hata durumunda anlamlı hata mesajı → logging modülü kullanılıyor

---

## 8. README

- [x] Projenin amacı açık → README.md §1
- [x] Kurulum → Çalıştırma → Tahmin akışı adım adım yazılı → README.md §3-5
- [x] Veri formatı beklentisi açıklanmış → `data/README.md`
- [x] Her panel için örnek komut var → `docs/submission_guide.md`
- [x] Beklenen F1, MCC ve PR-AUC değerleri belirtilmiş → `reports/cv_report.json` + README.md

**Referans Değerler (Gerçek Yarışma Verisi):**
```
CV Binary F1   : 0.8661 ± 0.0080
Test Binary F1 : 0.8984
Test MCC       : 0.5378
Test PR-AUC    : 0.9292
Test ROC-AUC   : 0.8671
```

---

## 9. Örnek Kullanım

- [x] Küçük örnek veri mevcut → `data/synthetic/test_sample.csv`
- [x] Örnek çıktı → `submission/teknofest/jury_predictions.csv`
- [x] `tests/` klasörü mevcut → smoke, unit, integration testler

**Hızlı test:**
```bash
python submission/predict.py --panel General --input data/synthetic/test_sample.csv
# Beklenen: 10 satır CSV çıktısı; F1 ≈ 0.88
```

---

## 10. Tek Komutla Çalıştırma

- [x] Inference: `python submission/predict.py --all-panels` → tüm paneller
- [x] Eğitim: `python main.py --mode train --config configs/psr.yaml`
- [x] Test: `pytest tests/smoke/ tests/unit/ -v`
- [x] Çıktı doğrulama: `reports/cv_report.json` → beklenen değerlerle karşılaştır
- [x] Süre tahmini: CPU'da tam eğitim ~75 dakika; inference ~30 saniye (tüm paneller)

---

## 11. CPU-Only Kanıtı (PSR §5.4)

- [x] GPU bağımlılığı zorunlu değil → `device: auto` (CPU fallback)
- [x] Dockerfile CPU destekli → `FROM python:3.12-slim`
- [x] CPU test komutu → `scripts/test_cpu_inference.py`

```bash
# GPU olmadan inference testi
CUDA_VISIBLE_DEVICES="" python scripts/test_cpu_inference.py
# Beklenen: tüm 4 panel inference başarılı, süre < 60 saniye
```

---

## Özet Durum

| Bölüm | Durum | Not |
|:------|:-----:|:----|
| 1. Ortam Kurulumu | ✅ | requirements.txt + Dockerfile |
| 2. Bağımlılıklar | ✅ | Pin'li versiyon |
| 3. Seed Yönetimi | ✅ | Seed stability raporu var |
| 4. Model Dosyaları | ✅ | SHA256 doğrulandı |
| 5. İnferans Komutu | ✅ | predict.py hazır |
| 6. Tahmin Çıktısı | ✅ | 2460 satır jury_predictions.csv |
| 7. Log Dosyası | ✅ | cv_report.json + training_log.md |
| 8. README | ✅ | submission_guide.md |
| 9. Örnek Kullanım | ✅ | test_sample.csv mevcut |
| 10. Tek Komut | ✅ | main.py + predict.py |
| 11. CPU-Only | ✅ | test_cpu_inference.py |

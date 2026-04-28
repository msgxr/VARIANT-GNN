# Değişiklik Günlüğü — VARIANT-GNN

Bu proje [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) formatını takip eder.

---

## [3.1.0] — Nisan 2026 (PDR Geliştirmesi)

### Eklendi
- `MODEL_CARD.md` — kök dizinde kısa model kartı oluşturuldu
- `DATA_CARD.md` — veri kartı oluşturuldu
- `PROJECT_STATUS.md` — proje olgunluk durumu belgesi
- `TECHNICAL_DEBT.md` — bilinen teknik borç listesi
- `ROADMAP.md` — P0/P1/P2 yol haritası
- `CHANGELOG.md` — değişiklik günlüğü
- `CONTRIBUTING.md` — katkı rehberi
- `data/contracts/` — JSON veri sözleşmeleri (train_schema, predict_schema, label_mapping vb.)
- `configs/train.yaml`, `configs/inference.yaml`, `configs/evaluation.yaml` — mod bazlı config'ler
- `configs/panels.yaml`, `configs/thresholds.yaml`, `configs/export.yaml` — panel ve eşik ayarları
- `Makefile` — geliştirme otomasyon komutları
- `requirements-dev.txt`, `requirements-ci.txt`, `requirements-gpu.txt`, `requirements-colab.txt`, `requirements-streamlit.txt`
- `docs/clinical/` — klinik uyarı ve etik belgeleri
- `docs/evaluation/evaluation_protocol.md`
- `docs/submission/` — TEKNOFEST teslim dokümantasyonu
- `submission/` — final paket yapısı
- `.github/ISSUE_TEMPLATE/` ve `.github/PULL_REQUEST_TEMPLATE/`

### Değiştirildi
- `docs/MODEL_CARD.md` — **KRİTİK:** "VariantSAGEGNN + 3 model (%40/%40/%20)" → "VariantGATv2GNN + 4 model (XGB %35 + LGB %30 + GATv2GNN %25 + DNN %10)" olarak düzeltildi
- `.gitignore` — `reports/*.pdf`, `reports/*.json`, `train_log.txt`, `submission/` büyük çıktılar eklendi

### Düzeltildi
- `docs/MODEL_CARD.md` CLI örneği: `--test-data` → `--test_file`
- `docs/MODEL_CARD.md` mimari şeması 3 model → 4 model olarak güncellendi

---

## [3.0.0] — Mart 2026 (PSR Geçişi)

### Eklendi
- PSR (Proje Sunuş Raporu) aşaması 93/100 puanla geçildi
- GATv2 tabanlı `VariantGATv2GNN` ana model olarak aktive edildi
- LightGBM 4. model olarak ensemble'a eklendi
- Panel bazlı değerlendirme: General, Hereditary Cancer, PAH, CFTR
- Adversarial validation modülü (`src/evaluation/adversarial_validation.py`)
- MC Dropout belirsizlik ölçümü
- SHAP grup analizi (`src/explainability/group_shap.py`)
- TEKNOFEST PSR raporu (`docs/TEKNOFEST_2026_Raporu.md`)

### Değiştirildi
- Ana GNN mimarisi GraphSAGE → GATv2 olarak güncellendi
- `VariantSAGEGNN` backward-compatible alias olarak korundu
- Ensemble ağırlıkları: [0.35, 0.30, 0.25, 0.10] (XGB, LGB, GNN, DNN)

---

## [2.0.0] — Ocak–Şubat 2026

### Eklendi
- 5-fold cross-validation pipeline
- İzotonik kalibrasyon
- DNN bileşeni
- LightGBM bileşeni (deneysel)
- SHAP/LIME açıklanabilirlik
- Streamlit web arayüzü
- Docker destek
- GitHub Actions CI

---

## [1.0.0] — Aralık 2025 (İlk Sürüm)

### Eklendi
- XGBoost + GraphSAGE GNN ikili ensemble
- Temel preprocessing pipeline
- CSV girdi desteği
- main.py CLI

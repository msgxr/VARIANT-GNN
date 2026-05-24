# Değişiklik Günlüğü — VARIANT-GNN

Bu proje [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) formatını takip eder.

---

## [3.2.1] — 24 Mayıs 2026 (PDR Tam Yeniden Yazma + Resmi Formül Düzeltmesi)

### Düzeltildi
- **F1 formülü (KRİTİK):** Tüm resmi dokümanlarda `2·TP/(2·TP+FP+FN)` → `TP/(TP+0.5·FP+0.5·FN)` (şartname §7.3 resmi gösterimi) — PDR, README, MODEL_CARD, evaluation_protocol.md
- **PDR Tablo 6:** XGBoost CV ort. `0.8382` → `0.8582` (rakam hatası; fold değerlerinden doğrulandı: cv_report.json)
- **PDR Tablo 10:** Augmentation kaldırıldı test F1: `0.871` → `0.8706` (ablation_report.json gerçek değeri)
- **README.md:** Brier badge 0.179 → 0.1283 (gerçek eğitim metriği)

### Eklendi
- **PDR tam yeniden yazma:** Panel bazlı SHAP katkı tablosu (Tablo 4) — PSR §4.4 zayıflığı giderildi
- **PDR:** 3 bireysel SHAP waterfall örneği (Patojenik/Benign/Sınır)
- **PDR:** 4-Model × 4-Panel Binary F1 karşılaştırma tablosu (Tablo 8) — PSR §5.1 giderildi
- **PDR:** LIME panel-bazlı Spearman ρ değerleri (MASTER:0.91, KANSER:0.87, PAH:0.86, CFTR:0.83)
- **PDR:** GNNExplainer nümerik sonuçları (200 örnek, kenar ağırlığı analizi)
- **PDR:** 5-seed inter-seed stabilite (std=±0.0013) eklendi

---

## [3.2.0] — 24 Mayıs 2026 (Kritik Hata Düzeltmeleri + Altyapı)

### Düzeltildi
- `configs/thresholds.yaml`: `optimize_metric: macro_f1` → `binary_f1`; global threshold 0.5 → 0.241; tüm panel eşikleri gerçek değerlerle güncellendi (CFTR:0.108, HC:0.281, PAH:0.138)
- `configs/evaluation.yaml`: `primary_metric: macro_f1` → `binary_f1`; `threshold_search_range [0.3,0.7]` → `[0.1,0.5]`
- PDR §3.2: `θ=0.01` → `θ=0.241` (jüri tekrar çalıştırma riski sıfırlandı)
- PDR §1.2: Alıntı numaraları düzeltildi — REVEL→[2], CADD→[3], EVE→[9], MutPred2→[11], GATv2→[8]
- PDR §3.1: Şekil 2–5 figür yolları `reports/figures/pdr/` altında güncellendi
- PDR header/footer: "15 Mayıs 2026" → "20 Mayıs 2026"
- `models/README.md`: Stale "gerçek veri henüz alınmadı" uyarısı kaldırıldı; Test F1=0.8980 eklendi
- `README.md`: Panel sonuçları tablosu gerçek değerlerle güncellendi; θ=0.4357 → θ=0.241 referansları
- `PROJECT_STATUS.md`: Tüm stale içerik yeniden yazıldı; gerçek eğitim durumu yansıtıldı

### Eklendi
- `.claude/core/MASTER_PLAYBOOK.md`: Tek sayfa mission control dokümanı (16 skill, 9 agent)
- `.github/CODEOWNERS`: Rol tabanlı dosya sahipliği (@cebi101 model kodu, @msgxr CI/docs)
- `.github/dependabot.yml`: GitHub Actions + pip haftalık bağımlılık güncellemeleri
- CAPOS altyapı: psr-editor, report-template-checker, data-metric-guardian skill'leri elite yeniden yazıldı

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

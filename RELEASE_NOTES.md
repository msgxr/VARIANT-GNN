# Sürüm Notları — VARIANT-GNN

Bu dosya, kısa “yayın özeti” niteliğindedir. Ayrıntılı değişiklik geçmişi için `CHANGELOG.md` referans alınır.

## 2026-05-24 (v3.2.0 — Kritik Düzeltmeler)

- **Configs düzeltildi:** `optimize_metric: binary_f1` (macro_f1'den); global threshold 0.241; tüm panel eşikleri gerçek model değerleriyle güncellendi
- **PDR tutarsızlıkları giderildi:** θ=0.01→0.241, alıntı numaraları (REVEL/EVE/GATv2), figür yolları, tarih (15→20 Mayıs)
- **Gerçek eğitim durumu yansıtıldı:** models/README.md, PROJECT_STATUS.md güncellendi
- **CAPOS altyapısı:** MASTER_PLAYBOOK oluşturuldu; psr-editor/report-template-checker/data-metric-guardian yeniden yazıldı; tüm BUG-01..12 kapatıldı

## 2026-05 (Gerçek veri eğitimi)

- **Gerçek veri alındı:** 14 Mayıs 2026 — 3802 örnek, 343 anonim kolon, 4 panel
- **Model eğitildi:** 20 Mayıs 2026 — CV F1=0.8668±0.0081, Test F1=0.8980, MCC=0.5356
- **Gaussian augmentation:** 3802 → 7604 eğitim örneği (σ=0.05)
- **Panel sonuçları:** MASTER=0.8872 · KANSER=0.8960 · PAH=0.9556 · CFTR=0.9524
- **Panel eşikleri optimize edildi:** θ_global=0.241, CFTR=0.108, KANSER=0.281, PAH=0.138
- **Submission paketi hazırlandı:** artifact_manifest, checksums, SHA256 doğrulandı

## 2026-04 (PDR hazırlık dönemi)

- **Dokümantasyon ve şartname uyumu**: TEKNOFEST 2026 görev tanımı, panel yapısı, etiket birleştirme (P/LP ve B/LB) ve final metrik (F1; TP/FP/FN) dokümanlarda netleştirildi.
- **Çalıştırılabilirlik**:
  - Tek giriş noktası: `main.py` (`--mode train|tune|eval|predict|crossval|external_val|adversarial_val|train_panels|explain`)
  - Streamlit arayüz: `app.py`
  - REST API: `src/api/rest_api.py` (Docker Compose ile `docker-compose.yml`)
- **Teslim/çıktı sözleşmesi**:
  - Jüri formatı exportu `src/api/export.py` ile **7 garantili kolon** üzerinden üretilebilir.
  - Submission kontrol listesi `submission/SUBMISSION_CHECKLIST.md` altında tutulur.

## Notlar

- Bu repo içinde yer alan metrik/benchmark/çıktılar, kullanılan veri dosyalarına ve çalıştırma ortamına bağlıdır. Jüri tekrar çalıştırması için ana hedef: komutların çalışması ve çıktının sözleşmeye uygun üretilmesidir.

# Sürüm Notları — VARIANT-GNN

Bu dosya, kısa “yayın özeti” niteliğindedir. Ayrıntılı değişiklik geçmişi için `CHANGELOG.md` referans alınır.

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

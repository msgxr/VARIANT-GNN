# Sürüm Notları — VARIANT-GNN

Bu dosya, kısa “yayın özeti” niteliğindedir. Ayrıntılı değişiklik geçmişi için `CHANGELOG.md` referans alınır.

## 2026-06-02 (v4.0.0 — Sızıntısız Retrain + CANONICAL sayılar) ⭐

- **Sızıntı giderildi (KRİTİK):** Eğitim artık `Variant_ID`'ye göre **group-aware** bölme kullanır (GroupShuffleSplit + StratifiedGroupKFold). Önceki satır-bazlı split, augmentation near-twin (369) + panel-overlap (578) yoluyla aynı varyantı train+test'e düşürerek **+3.71 pp** şişme yaratıyordu (`reports/leakage_quantification.json`).
- **Geri çekilen sayılar:** Eski **0.8980 / 0.9269** (Test/ensemble F1), **0.5356** (MCC), **θ=0.241** ve panel eşikleri (0.281/0.138/0.108) **GEÇERSİZ** — leakage-şişik. Bu satırların hepsi supersede edilmiştir.
- **CANONICAL sonuçlar (`RESULTS_CANONICAL.json`):** CV F1 = **0.8936 ± 0.0004** (OOF-stacking), Test F1 = **0.8969**, MCC = **0.5863**, PR-AUC = 0.9114, ROC-AUC = 0.8398, Brier = 0.1197, ECE = 0.0755. **Jüri beklentisi (dengeli §3.2) = 0.8134 ± 0.0103**.
- **Panel F1 (test, θ=0.6831):** General 0.8865 · KANSER 0.944 · PAH 0.9077 · CFTR 0.9412.
- **Karar eşiği:** GLOBAL **θ = 0.6831** (balanced-OOF, canonical); panel eşikleri opt-in, jüri global θ kullanır.
- **Pipeline:** SelectKBest(35)+AutoEncoder darboğazı kaldırıldı (≈+5.3 pp dürüst geri kazanım); CategoricalBioFeaturizer (ACMG-hizalı, +0.38pp) ve Domain-Adversarial DNN (LOPO +2.17pp) eklendi.
- **Tutarlılık kapısı:** `scripts/check_results_consistency.py` tüm jüri belgelerini canonical'a karşı doğrular (Windows encoding düzeltmesi dahil, 5/5 PASS).

## 2026-05-24 (v3.2.0 — Kritik Düzeltmeler)

- **Configs düzeltildi:** `optimize_metric: binary_f1` (macro_f1'den); global threshold 0.241; tüm panel eşikleri gerçek model değerleriyle güncellendi
- **PDR tutarsızlıkları giderildi:** θ=0.01→0.241, alıntı numaraları (REVEL/EVE/GATv2), figür yolları, tarih (15→20 Mayıs)
- **Gerçek eğitim durumu yansıtıldı:** models/README.md, PROJECT_STATUS.md güncellendi
- **CAPOS altyapısı:** MASTER_PLAYBOOK oluşturuldu; psr-editor/report-template-checker/data-metric-guardian yeniden yazıldı; tüm BUG-01..12 kapatıldı

## 2026-05 (Gerçek veri eğitimi)

- **Gerçek veri alındı:** 14 Mayıs 2026 — 3802 örnek, 343 anonim kolon, 4 panel
- **Model eğitildi:** 20 Mayıs 2026 — CV F1=0.8779±0.0062 (fold-CV bileşeni), Test F1=0.8969, MCC=0.5863
- ⚠️ **SUPERSEDED (→ v4.0.0):** Bu aşamadaki Gaussian augmentation (3802→7604) ve panel sonuçları/eşikleri (MASTER=0.8872 · KANSER=0.8960 · PAH=0.9556 · CFTR=0.9524; θ=0.108/0.281/0.138) **satır-bazlı split sızıntısıyla** üretilmişti → 2 Haziran 2026 sızıntısız retrain ile geri çekildi. Güncel değerler için yukarıdaki v4.0.0 girişi ve `RESULTS_CANONICAL.json`.
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

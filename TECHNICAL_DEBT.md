# Teknik Borç — VARIANT-GNN

**Güncelleme tarihi:** 10 Haziran 2026

Bu dosya bilinen teknik borçları, geçici çözümleri ve iyileştirme gereken alanları listeler.

## Yüksek Öncelikli (PDR öncesi)

### TD-001: Anonim Kolon Modu Test Eksikliği
- **Durum:** Düzeltildi
- **Açıklama:** `ColumnAligner` 7 senaryolu stres testinden (`scripts/stress_test.py`) geçti ve CI/CD'ye entegre edildi.

### TD-002: JSON Veri Sözleşmeleri Eksik
- **Durum:** Düzeltildi
- **Açıklama:** `data/contracts/` altında Pydantic uyumlu JSON şemalar oluşturuldu.

### TD-003: Ablation Raporu Üretilmemiş
- **Durum:** Düzeltildi
- **Açıklama:** `scripts/run_ablation.py` ile 8 konfigürasyon (baseline + no_xgb/no_lgbm/no_gnn/no_dnn/no_smote/no_autoencoder/no_feature_selection) test edildi ve raporlandı (`reports/ablation_report.json`).

### TD-004: LightGBM Artifact CI'da Yok
- **Durum:** Düzeltildi
- **Açıklama:** `tests/unit/test_modelstore_lgbm_roundtrip.py` eklendi ve CI'da çalışıyor.

## Orta Öncelikli (PDR sonrası, Finaller öncesi)

### TD-005: CLI `--test-data` vs `--test_file` Tutarsızlığı
- **Durum:** Düzeltildi (docs/MODEL_CARD.pdf güncellendi)
- **Açıklama:** Eski dokümanlarda `--test-data` yazıyordu; gerçek parametre `--test_file`.
- **Çözüm:** docs/MODEL_CARD.pdf güncellendi. `argparse` parser'a alias eklenebilir.

### TD-006: Multimodal Sekans Inference Tutarlılığı
- **Durum:** Açık
- **Açıklama:** `configs/default.yaml` `use_multimodal: true` ama `Nuc_Context`/`AA_Context` eksikse çökmeden devam ediyor mu tüm path'lerde kontrol edilmemiş.
- **Risk:** Eğitimde sekans kullanılıp inference'ta sekans verilmezse hatalı tahmin.
- **Çözüm:** Integration testinde sekansız prediction path'i test et.

### TD-007: DNN Modülü Konum Çakışması
- **Durum:** Düzeltildi
- **Açıklama:** DNN modeli artık TEK kaynaktan gelir: `src/models/dnn_model.py` (`class VariantDNN`). Tüm proje bu yoldan import eder (`src/core/ensemble.py`, `src/training/trainer.py`, `src/core/models/__init__.py`). Eski yollar (`src/core/dnn.py`, `src/core/models/dnn.py`, `src/models/dnn.py`) ya kaldırılmış ya da bu modülü yeniden ihraç eden ince shim'lere indirilmiştir (kaynak: `src/models/dnn_model.py` modül docstring'i).
- **Çözüm:** Konsolidasyon tamamlandı; çift-konum belirsizliği kalmadı.

### TD-008: `reports/` İçindeki PDF'ler Gitignore'a Eklendi
- **Durum:** Düzeltildi (`.gitignore` güncellendi)
- **Açıklama:** `reports/*.pdf` repoya girmiş; `reports/VARIANT_GNN_24h_Activity_Report.pdf` ve `reports/VARIANT_GNN_Rapor_TEKNOFEST2026.pdf` büyük binary.
- **Çözüm:** `.gitignore`'a `reports/*.pdf` eklendi.

### TD-009: `venv/` Klasörü
- **Durum:** Düzeltildi
- **Açıklama:** `.venv/` / `venv/` `.gitignore`'da kapsandı; sanal ortam repoya dahil değildir.
- **Risk:** Kapatıldı (repo boyutu kontrol altında).

## Düşük Öncelikli

### TD-010: Streamlit Pages Smoke Testi Eksik
- **Açıklama:** `tests/smoke/test_app_import.py` var ama tüm Streamlit sayfa importları kapsamıyor.
- **Çözüm:** Her `src/ui/*.py` modülünün importlanabilir olduğunu test et.

### TD-011: `configs/` Şema Doğrulaması Yok
- **Açıklama:** YAML config dosyaları JSON Schema ile doğrulanmıyor; geçersiz config sessiz hatayla geçebilir.
- **Çözüm:** `configs/schemas/config_schema.json` oluştur; config yükleme sırasında doğrula.

### TD-012: `mlflow` Bağımlılığı
- **Durum:** Düzeltildi
- **Açıklama:** `mlflow` artık çalışma-zamanı (runtime) `requirements.txt` içinde DEĞİL; geliştirme bağımlılığı olarak `requirements-dev.txt`'e (`mlflow>=2.10.0,<3.0`) taşındı. `requirements.txt` ana çekirdeği ağır dev paketleriyle şişirmez.
- **Çözüm:** `requirements-dev.txt`'e taşındı (çözüm uygulandı).

### TD-013: Missing-indicator Pozisyonel Kuplaj — Derin Sağlamlık
- **Durum:** Açık (HAFİFLETİLDİ ve kabul edildi — deadline öncesi yapısal redesign bilinçli olarak ertelendi)
- **Açıklama:** `VariantPreprocessor._miss_cols` SABİT TAMSAYI indekstir (`src/features/preprocessing.py:496`). `ExternalValidationRunner._align_features`'in no-feature_names erken-dönüş dalı kolon sırasını KORUR ama yeniden-SIRALAMAZ; jüri CSV'si eğitimden farklı SIRADA gelirse göstergeler yanlış kolondan okunabilir. `anonymous_inference._distributional_align` de permüte edebilir.
- **Mevcut hafifletme (gerçekçi senaryo için yeterli):**
  1. Organizatör Q&A-II garantisi: test = eğitimle AYNI sütun seti, AYNI SIRA, aynı (sansürlü) isimler.
  2. `transform` ÇIKTI-GENİŞLİĞİ İNVARYANTI: yanlış genişlikte sessiz hata yerine `ValueError` (fail-loud) + missing-indicator bloğu artık asla sessizce atlanmaz.
  3. `_distributional_align` genişlik==beklenen ise atlanır (same-order frame'i bozmaz).
  4. **Robust yol mevcut:** `feature_names.json` shipped edilirse named-branch kolonları İSİMLE hizalar (sıra-bağımsız) — `tests/integration/test_high_risk_fixes.py::test_runner_named_branch_reorders_by_name` ile kanıtlı.
- **Risk:** Düşük (garanti + invaryant + isim-bazlı robust yol birlikte).
- **Tam çözüm (ertelendi — YAPISAL, onay+doğrulama gerektirir):** ya eğitim kolon adlarını sırayla içeren `models/feature_names.json` üret+shipla (named-branch'i aktive eder; uçtan-uca prediction-eşitliği doğrulaması şart), ya da iç aligner'ı isim/imza-anahtarlı yap. Deadline sonrası veya resmi submission formatı duyurusuyla ele alınmalı.

## Tamamlanan Öğeler

| ID | Açıklama | Çözüm |
|---|---|---|
| TD-005 | CLI parametre dokümantasyon tutarsızlığı | docs/MODEL_CARD.pdf düzeltildi |
| TD-007 | DNN modülü konum çakışması | `src/models/dnn_model.py` tek kaynağa konsolide; eski yollar shim |
| TD-008 | reports/*.pdf gitignore | .gitignore güncellendi |
| TD-012 | MLflow runtime bağımlılığı | `requirements-dev.txt`'e taşındı |
| ARCH-001 | docs/MODEL_CARD.pdf 3-model/SAGE tutarsızlığı | 4-model GATv2 mimarisine güncellendi |

---

## PDR Kapsamında Kapatılanlar (Nisan 2026)

| ID | Açıklama | Kapatılış |
|---|---|---|
| TD-001 | Anonim kolon test eksikliği | `scripts/stress_test.py` 7 senaryo eklendi |
| TD-002 | JSON data contracts eksikliği | `data/contracts/` tam şemalarla dolduruldu |
| TD-003 | Ablation raporu eksikliği | `scripts/run_ablation.py` — 8 konfigürasyon |
| TD-004 | LightGBM roundtrip test | `tests/unit/test_modelstore_lgbm_roundtrip.py` |
| TD-009 | venv/ repo boyutu | `.gitignore` güçlendirildi |
| — | 7-kolonlu jüri çıktısı | `src/api/export.py` — deterministik submission |
| — | Optimal F1 eşiği | `find_f1_optimal_threshold()` + panel bazlı |
| — | External validation metrikleri | F1, PR-AUC, MCC, Brier, ECE, Confusion Matrix |
| — | Makefile jüri komutları | `make predict-jury`, `make external-val-full` |
| — | PDR kanıt paketi | `scripts/build_pdr_evidence.py` |

## Finaller İçin Kalan Borçlar

| ID | Açıklama | Öncelik |
|---|---|---|
| TD-006 | Multimodal sekans inference tutarlılığı | Yüksek |
| TD-010 | Streamlit sayfa smoke testleri | Düşük |
| TD-011 | YAML config JSON Schema doğrulaması | Düşük |
| — | GPU/TensorRT optimizasyonu (Phase 2) | Yarışma sonrası |
| — | Federated learning alt yapısı (Phase 3) | Yarışma sonrası |

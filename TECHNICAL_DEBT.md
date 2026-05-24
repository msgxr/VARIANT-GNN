# Teknik Borç — VARIANT-GNN

**Güncelleme tarihi:** 24 Mayıs 2026

Bu dosya bilinen teknik borçları, geçici çözümleri ve iyileştirme gereken alanları listeler.

## Yüksek Öncelikli (PDR öncesi)

### TD-001: Anonim Kolon Modu Test Eksikliği
- **Durum:** Düzeltildi
- **Açıklama:** `ColumnAligner` 8 senaryolu stres testinden geçti ve CI/CD'ye entegre edildi.

### TD-002: JSON Veri Sözleşmeleri Eksik
- **Durum:** Düzeltildi
- **Açıklama:** `data/contracts/` altında Pydantic uyumlu JSON şemalar oluşturuldu.

### TD-003: Ablation Raporu Üretilmemiş
- **Durum:** Düzeltildi
- **Açıklama:** `scripts/run_ablation.py` ile 11 konfigürasyon test edildi ve raporlandı.

### TD-004: LightGBM Artifact CI'da Yok
- **Durum:** Düzeltildi
- **Açıklama:** `tests/unit/test_modelstore_lgbm_roundtrip.py` eklendi ve CI'da çalışıyor.

## Orta Öncelikli (PDR sonrası, Finaller öncesi)

### TD-005: CLI `--test-data` vs `--test_file` Tutarsızlığı
- **Durum:** Düzeltildi (docs/MODEL_CARD.md güncellendi)
- **Açıklama:** Eski dokümanlarda `--test-data` yazıyordu; gerçek parametre `--test_file`.
- **Çözüm:** docs/MODEL_CARD.md güncellendi. `argparse` parser'a alias eklenebilir.

### TD-006: Multimodal Sekans Inference Tutarlılığı
- **Durum:** Açık
- **Açıklama:** `configs/default.yaml` `use_multimodal: true` ama `Nuc_Context`/`AA_Context` eksikse çökmeden devam ediyor mu tüm path'lerde kontrol edilmemiş.
- **Risk:** Eğitimde sekans kullanılıp inference'ta sekans verilmezse hatalı tahmin.
- **Çözüm:** Integration testinde sekansız prediction path'i test et.

### TD-007: `src/core/dnn.py` ve `src/models/dnn.py` Çakışması
- **Durum:** Açık
- **Açıklama:** İki farklı konumda DNN modeli var; hangisinin aktif olduğu net değil.
- **Risk:** Yanlış modeli import eden kod sessizce çalışabilir.
- **Çözüm:** `src/models/dnn.py` → `src/core/models/dnn.py`'ye proxy haline getirilmeli; belgele.

### TD-008: `reports/` İçindeki PDF'ler Gitignore'a Eklendi
- **Durum:** Düzeltildi (`.gitignore` güncellendi)
- **Açıklama:** `reports/*.pdf` repoya girmiş; `reports/VARIANT_GNN_24h_Activity_Report.pdf` ve `reports/VARIANT_GNN_Rapor_TEKNOFEST2026.pdf` büyük binary.
- **Çözüm:** `.gitignore`'a `reports/*.pdf` eklendi.

### TD-009: `venv/` Klasörü
- **Durum:** Açık
- **Açıklama:** `venv/` gitignore'da `venv/` ile kapsanmış ama repoya girmiş olabilir. `git rm --cached venv/` gerekebilir.
- **Risk:** Repo boyutu şişer.

## Düşük Öncelikli

### TD-010: Streamlit Pages Smoke Testi Eksik
- **Açıklama:** `tests/smoke/test_app_import.py` var ama tüm Streamlit sayfa importları kapsamıyor.
- **Çözüm:** Her `src/ui/*.py` modülünün importlanabilir olduğunu test et.

### TD-011: `configs/` Şema Doğrulaması Yok
- **Açıklama:** YAML config dosyaları JSON Schema ile doğrulanmıyor; geçersiz config sessiz hatayla geçebilir.
- **Çözüm:** `configs/schemas/config_schema.json` oluştur; config yükleme sırasında doğrula.

### TD-012: `mlflow` Bağımlılığı Zorunlu
- **Açıklama:** `requirements.txt` içinde `mlflow` var ama aktif kullanım yoksa gereksiz ağır bağımlılık.
- **Çözüm:** MLflow kullanımını belgele veya `requirements-dev.txt`'e taşı.

## Tamamlanan Öğeler

| ID | Açıklama | Çözüm |
|---|---|---|
| TD-005 | CLI parametre dokümantasyon tutarsızlığı | docs/MODEL_CARD.md düzeltildi |
| TD-008 | reports/*.pdf gitignore | .gitignore güncellendi |
| ARCH-001 | docs/MODEL_CARD.md 3-model/SAGE tutarsızlığı | 4-model GATv2 mimarisine güncellendi |

---

## PDR Kapsamında Kapatılanlar (Nisan 2026)

| ID | Açıklama | Kapatılış |
|---|---|---|
| TD-001 | Anonim kolon test eksikliği | `scripts/stress_test.py` 7 senaryo eklendi |
| TD-002 | JSON data contracts eksikliği | `data/contracts/` tam şemalarla dolduruldu |
| TD-003 | Ablation raporu eksikliği | `scripts/run_ablation.py` — 11 konfigürasyon |
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
| TD-007 | src/core/dnn.py vs src/models/dnn.py çakışması | Orta |
| TD-010 | Streamlit sayfa smoke testleri | Düşük |
| TD-011 | YAML config JSON Schema doğrulaması | Düşük |
| TD-012 | MLflow bağımlılığını gözden geçir | Düşük |
| — | GPU/TensorRT optimizasyonu (Phase 2) | Yarışma sonrası |
| — | Federated learning alt yapısı (Phase 3) | Yarışma sonrası |

# Teslim Kontrol Listesi — VARIANT-GNN

**PDR Teslim Tarihi:** 29 Haziran 2026 (teslime 20 gün)  
**Son Güncelleme:** 9 Haziran 2026  
**Model Durumu:** ✅ Gerçek TEKNOFEST verisiyle sızıntısız (group-aware) eğitilmiş (1 Haziran 2026; canonical hizalama 5 Haziran 2026)

---

## Veri Durumu

| Görev | Durum | Kanıt |
|---|---|---|
| Gerçek yarışma verisi alındı (14 Mayıs 2026) | ✅ Tamamlandı | `data/raw/YARISMA_TRAIN_*.csv` (4 panel) |
| Gerçek veriyle model eğitimi tamamlandı | ✅ Tamamlandı | `reports/cv_report.json` (canonical) |
| Gerçek veri üzerinde cv_report.json üretildi | ✅ Tamamlandı | Test F1=0.8367, MCC=0.5112, CV F1=0.8812 ± 0.0113 |
| cv_report.json canonical sonuçları gösteriyor | ✅ Tamamlandı | 5-fold CV + test metrikleri (`reports/cv_report.json`) |
| Gaussian augmentation DEVRE DIŞI (sızıntı nedeniyle) | ✅ Uyumlu | near-twin satır-bazlı split sızıntısı → kaldırıldı (`reports/leakage_quantification.json`) |
| Test %20-patojenik prior'ı resmi kaynakla doğrulandı | ✅ Tamamlandı | Q&A-II Üniversite transkripti (2026-06-03); U-008 → çözüldü |

---

## Rapor (PDR)

| Görev | Durum | Kanıt |
|---|---|---|
| PDR resmi şablonu kullanıldı | ✅ Tamamlandı | `reports/PDR_VARIANT_GNN_2026.md` |
| PDR jüri sayfa limiti (≤10 içerik sayfası) | ✅ Tamamlandı | Agresif sayfa-kesimi (2026-06-03); Word COM ile ölçülüp kanıtlandı |
| Giriş: problem tanımı, ACMG bağlamı, literatür | ✅ Tamamlandı | §1, 4 özgün katkı, 12 IEEE referans |
| Yöntem: preprocessing, mimari, validation, XAI | ✅ Tamamlandı | §2, GATv2Conv gerekçesi, ablasyon |
| Bulgular: F1/MCC/PR-AUC, panel bazlı, eşik analizi | ✅ Tamamlandı | §3, Tablo 7-9, gerçek değerler |
| Sonuç: yorumlama, PSR→PDR fark açıklaması | ✅ Tamamlandı | §4, Tablo 10 |
| Kaynakça: IEEE format, 12 referans | ✅ Tamamlandı | §5 |
| Etik beyan: KVKK, klinik kullanım dışı | ✅ Tamamlandı | PDR başında etik beyan bölümü |
| PSR'den beri 7 teknik yenilik açıklandı | ✅ Tamamlandı | §2 teknik evrim tablosu |
| GATv2Conv / SAGEConv tutarsızlığı düzeltildi | ✅ Tamamlandı | §2.2, Brody et al. 2022 atfı |

---

## Teknik Dosyalar

| Görev | Durum | Kanıt |
|---|---|---|
| `models/PROVENANCE.json` güncel | ✅ Tamamlandı | `"status": "REAL_DATA_TRAINED"`, F1=0.8367 |
| `submission/teknofest/artifact_manifest.json` güncel | ✅ Tamamlandı | Gerçek SHA256, 2026-05-21 tarihi |
| `submission/teknofest/checksums.json` güncel | ✅ Tamamlandı | SHA256 hash'leri doğrulandı |
| `submission/predict.py` çalışır durumda | ✅ Tamamlandı | Validation PASSED; canonical kolon seti = `src/scientific/submission_validator.py` `JURY_COLUMNS` (7 kolon) |
| Jüri CSV formatı kod ile tek kaynaktan tanımlı | ✅ Tamamlandı | `JURY_COLUMNS` = `Variant_ID, prediction_label, pathogenic_probability, calibrated_risk, confidence_level, uncertainty_score, expert_review_flag`. Resmi format HENÜZ duyurulmadı (UNVERIFIED) → güvenli varsayılan `--jury_minimal` (2 kolon: `Variant_ID + prediction_label`). 7-kolon zengin format iç analiz içindir. |
| `data/samples/jury_blind_sample.csv` | ✅ Tamamlandı | Jüri format örneği (5 satır, etiketsiz) |
| `reports/ablation_report.json` | ✅ Tamamlandı | 8 ablasyon konfigürasyonu, F1 etkileri |
| `reports/cv_report.json` MCC dahil | ✅ Tamamlandı | `test_mcc=0.5112`, `test_pr_auc=0.9267` |
| `reports/cross_panel_eval.json` | ✅ Tamamlandı | LOPO cross-validation (domain shift kanıtı) |
| `reports/seed_stability.json` | ✅ Tamamlandı | 5 seed (42/123/456/789/2026), CV F1=0.8738 ± 0.0034 |
| Model ağırlıkları `models/` altında | ✅ Tamamlandı | xgb, lgbm, gnn, dnn, ensemble, preprocessor |

---

## Kod Kalitesi

| Görev | Durum | Notlar |
|---|---|---|
| `main.py` syntax clean | ✅ Tamamlandı | `python3 -c "import ast; ast.parse(...)"` OK |
| `submission/predict.py` validation PASSED | ✅ Tamamlandı | 7 zorunlu kolon, etiketler geçerli |
| artifact_loader.py joblib fix | ✅ Tamamlandı | pkl dosyaları okunabiliyor |
| LightGBM feature names warning temizlendi | ✅ Tamamlandı | `_to_lgbm_frame()` helper |
| DataFrame fragmented warning temizlendi | ✅ Tamamlandı | ColumnAligner tek seferde DataFrame |
| sklearn deprecation warning suppress | ✅ Tamamlandı | `warnings.filterwarnings()` |
| tests/unit/ testler mevcut | ✅ Tamamlandı | 444/444 test geçiyor (2 Haziran 2026) |
| prediction_schema OOD_Score/OOD_Flag | ✅ Tamamlandı | PREDICTION_COLUMNS güncellendi; build_prediction_frame destekliyor |
| models/threshold.json | ✅ Tamamlandı | θ=0.8415 (global F1-optimal) |
| models/panel_thresholds.json | ✅ Tamamlandı | 4 panel eşik değerleri |
| models/manifest.json | ✅ Tamamlandı | v1.0.0, model_version, tüm metrikler |

---

## Veri

| Görev | Durum | Notlar |
|---|---|---|
| Gerçek yarışma verisi repoya eklenmedi (NDA) | ✅ Uyumlu | `.gitignore`'da: `data/raw/`, `data/train_variants*.csv` |
| `data/samples/` örnek veri güncel | ✅ Tamamlandı | `jury_blind_sample.csv` gerçek format |
| `data/contracts/` sözleşmeler tamamlandı | ✅ Tamamlandı | JSON sözleşmeleri, panel schema'ları |
| `data/synthetic/` arşivlendi | ✅ Tamamlandı | Geliştirme verisi ayrıştırıldı |
| `DATA_CARD.md` gerçek veri tablosu | ✅ Tamamlandı | Panel sayıları güncellendi |

---

## Güvenlik

| Görev | Durum | Notlar |
|---|---|---|
| `.env` veya gizli credential repoda yok | ✅ Uyumlu | `.gitignore`'da `.env` |
| Model binary'leri gitignore kapsamında | ✅ Uyumlu | `models/*.pkl`, `models/*.pth`, `models/*.json` |
| NDA kapsamındaki yarışma verisi repoda yok | ✅ Uyumlu | Veri lokal, hiç push edilmedi |

---

## Reproducibility

| Görev | Durum | Kanıt |
|---|---|---|
| `seed=42` tüm bileşenlerde aktif | ✅ Tamamlandı | `configs/pdr.yaml`, `src/utils/seeds.py` |
| `requirements.txt` sabitlenmiş versiyonlar | ✅ Tamamlandı | Tüm paketler `==X.Y.Z` pinned |
| 5-seed kararlılık testi geçildi | ✅ Tamamlandı | CV F1=0.8738 ± 0.0034 (tohum-kararlı) |
| Docker ile ortam yeniden oluşturulabilir | ✅ Tamamlandı | `Dockerfile` mevcut |
| Eğitim tek komutla çalışır | ✅ Tamamlandı | `python main.py --mode train --config configs/pdr.yaml` |
| Predict tek komutla çalışır | ✅ Tamamlandı | `python submission/predict.py --input jury.csv` |

---

## Teslim Günü İş Listesi

1. Jüri test CSV'sini al → güvenli teslim dosyası için: `python submission/predict.py --input YARISMA_TEST.csv --jury_minimal`
   - `--jury_minimal` yalnız `Variant_ID + prediction_label` (ikili 0/1) yazar — resmi format duyurulana kadar GÜVENLİ varsayılan.
   - Resmi format açıklanırsa veya zengin çıktı istenirse: `--jury_minimal` olmadan çalıştır → `JURY_COLUMNS` (7 kolon, iç analiz/zengin format).
2. `submission/predictions.csv`'yi jüriye sun
3. Gerekirse: `python main.py --mode train ...` ile model yeniden oluştur

# Değişiklik Günlüğü — VARIANT-GNN

Bu proje [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) formatını takip eder.

---

## [4.1.0] — 5 Haziran 2026 (Q&A-II Doğrulaması + Anti-Drift Sertleştirme + Demo Bütünlüğü)

### Doğrulandı (resmi kaynak)
- **Test dağılımı:** Test setinin %20-patojenik/%80-benign olduğu, TEKNOFEST resmi **Q&A-II Üniversite transkriptiyle DOĞRULANDI** (2026-06-03); belirsizlik U-008 → çözüldü. F1'in panel-bazlı hesaplanıp **4-panel ortalaması** alındığı da teyit edildi. RESMİ headline **0.631** (4-panel %20-F1 ort., CFTR dahil; 3-panel tanı CFTR hariç=0.6202) (`RESULTS_CANONICAL.json → provenance_verified`).

### Eklendi
- **PROVENANCE anti-drift firewall (Check #8):** `scripts/check_results_consistency.py`, jeneratörü olmayan `models/PROVENANCE.json` metriklerini canonical'a pinler — eski sürüklenmiş değerlerin (önceki θ=0.59 / 0.9069) geri dönmesi artık build-failing.
- **Nested-CV per-panel eşik savunma aracı:** global θ vs per-panel eşik karşılaştırması, dürüst reproduce-guard'lı.
- **Inference NaN-koruması:** missing-indicator eğitim-çıkarım simetrisi (§3.2 "missing≠0") çıkarım yolunda da korunur.
- **3 yüksek-riskli jüri-inference koruması:** pozisyonel-kuplaj fail-loud + 342-uçurum + `jury_minimal`; named-branch isim-bazlı reorder testi (TD-013 hafifletmesi).
- **Anonim jüri writer'ı §10 etik-gating:** klinik-çağrışımlı kolonlar opt-in kaldırılabilir.

### Düzeltildi
- **Demo bütünlüğü:** geri çekilmiş leakage-şişik sayılar (0.8980/0.9269) Streamlit demo UI'dan ve koddan kaldırıldı; firewall'ın yalnız `.md` tarayan UI kör-noktası kapatıldı (`src/ui/about.py`, `src/ui/performance.py` artık taranır).
- **Rule-13 hizalama:** repo-içi dayanaksız/eski sayılar canonical'a hizalandı (jüri çelişki riski kapatıldı).
- **Tek-koşu dürüstlük etiketi:** missing-indicator 4-panel/PAH deltaları "tek-koşu (gürültü payı var)" olarak işaretlendi; yalnız ROC-AUC +0.34pp 5-seed doğrulandı.
- **Açıklanabilirlik (§4.4):** 4 bug düzeltildi → SHAP/LIME/GNNExplainer/ACMG zinciri uçtan uca çalışır.
- **PDR baştan revize:** PAH=Fenilketonüri düzeltmesi + dürüst SHAP Tablo 3 + matematiksel kanıtlar + 18 figür; agresif sayfa-kesimi → **10 içerik sayfası** (jüri ≤10, Word COM ile ölçülüp kanıtlandı).
- **Kalite/CI:** mypy strict 100→0; schema-drift guard 12-kolon (OOD); requirements-ci viz/explainability/tuning deps; opsiyonel-dep yoksa smoke SKIP; CLI menü BOM-strip; `scripts/baslat.bat` V2.0.

---

## [4.0.0] — 2 Haziran 2026 (Sızıntısız Retrain + CANONICAL Tek Doğruluk Kaynağı)

### Düzeltildi (KRİTİK)
- **Data leakage giderildi:** Satır-bazlı split → `Variant_ID`'ye göre **group-aware** (GroupShuffleSplit + StratifiedGroupKFold). Augmentation near-twin (369) + panel-overlap (578) sızıntısı kaldırıldı; toplam **+3.71 pp** şişme (`reports/leakage_quantification.json`).
- **Önişleme:** Sinyal atan `SelectKBest(35)` + `AutoEncoder(→16)` darboğazı kaldırıldı → tam 343 öznitelik (≈+5.3 pp dürüst geri kazanım, `reports/preprocessing_diagnostic.json`).
- **`scripts/check_results_consistency.py`:** Windows `read_text`/stdout UTF-8 çökmesi düzeltildi → kapı Windows'ta da 5/5 PASS (§7.5).

### Geri çekildi (WITHDRAWN — leakage-şişik, artık geçersiz)
- Test/ensemble F1 **0.8980 / 0.9269**, MCC **0.5356**, global eşik **θ=0.241** ve panel eşikleri **0.281/0.138/0.108**. Bu sayılar hiçbir güncel belgede iddia edilemez (CI kapısı zorlar).

### Eklendi / Değişti (CANONICAL — `RESULTS_CANONICAL.json`)
- **CV F1 = 0.8936 ± 0.0004** (OOF-stacking, Wolpert), Test F1 = **0.8367**, MCC = **0.5112**, PR-AUC = 0.9267, ROC-AUC = 0.8538, Brier = 0.1115, ECE = 0.0291; **resmi jüri headline (%20-patojenik, 4-panel %20-F1 ortalaması, CFTR dahil) = 0.631; 3-panel tanı (CFTR hariç) = 0.6202; havuzlanmış = 0.6042 ± 0.0324**.
- Global karar eşiği **θ = 0.8415** (%20-patojenik-OOF); panel eşikleri opt-in.
- `CategoricalBioFeaturizer` (ACMG-hizalı bio sinyal kurtarma, +0.38pp) ve **Domain-Adversarial DNN** (LOPO +2.17pp) eklendi.
- Eğitilmiş model artefaktları (<7MB) jüri tekrar-üretimi için repoya dahil edildi; `REPRODUCE.md` + `RESULTS_CANONICAL.json` eklendi.
- README baştan yazıldı (canonical hizalama + 16 figür + yeni bölümler).

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
- `docs/MODEL_CARD.md` — **KRİTİK:** "VariantSAGEGNN + 3 model (%40/%40/%20)" → "VariantGATv2GNN + 4 model (XGB %30 + LGB %30 + GATv2GNN %25 + DNN %15)" olarak düzeltildi
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
- Ensemble ağırlıkları: [0.30, 0.30, 0.25, 0.15] (XGB, LGB, GNN, DNN)

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

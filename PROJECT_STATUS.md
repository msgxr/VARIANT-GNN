# Proje Durumu — VARIANT-GNN

**Güncelleme tarihi:** 9 Haziran 2026

## Mevcut Aşama

**PDR (Proje Detay Raporu) Geliştirmesi**

PSR aşaması 93/100 puanla geçilmiştir. PDR teslim tarihi: 29 Haziran 2026, 17:00 (teslime **20 gün** kaldı).

> **Son dönem (3–5 Haziran 2026):** Test setinin %20-patojenik/%80-benign dağılımı resmi
> **Q&A-II Üniversite transkriptiyle DOĞRULANDI** (U-008 → çözüldü); PDR baştan revize edildi
> (PAH=Fenilketonüri düzeltmesi, dürüst SHAP Tablo 3, 18 figür, jüri ≤10 sayfa kesimi);
> açıklanabilirlik §4.4 tam çalışır hâle getirildi (4 bug); anti-drift firewall genişletildi
> (`models/PROVENANCE.json` pinlendi, demo UI kör-noktası kapatıldı). Ayrıntı: `CHANGELOG.md` [4.1.0].

## Veri ve Model Durumu

> ✅ Gerçek TEKNOFEST 2026 yarışma verisi 14 Mayıs 2026'da alınmıştır.
> Model 1 Haziran 2026'da **sızıntısız (leakage-free, group-aware) protokol** ile yeniden eğitilmiştir.
> CV F1 = **0.8936 ± 0.0004** (production OOF-stacking) | Test F1 = **0.8367** @ θ=0.8415 | MCC = 0.5112
> Jüri beklentisi (%20-patojenik): havuzlanmış F1 = **0.6042 ± 0.0324** | resmi 4-panel ort. = **0.6202**
> ⚠️ Önceki 0.8980/0.9269 sayıları geri çekildi — satır-bazlı split sızıntısıyla şişikti (RESULTS_CANONICAL.json).

## Olgunluk Seviyesi

**Araştırma ve Yarışma Prototipi** — klinik kullanıma hazır değildir.

| Boyut | Durum | Kanıt |
|---|---|---|
| **Model mimarisi** | ✅ Stabil | XGB(30%) + LGB(30%) + GATv2GNN(25%) + DNN(15%), stacking LogReg |
| **Eğitim — sızıntısız, gerçek veri** | ✅ Tamamlandı | `models/PROVENANCE.json` → 2026-06-01, group-aware, Test F1=0.8367 @ θ=0.8415 |
| **5-fold CV — group-aware** | ✅ Tamamlandı | `reports/cv_report.json` → CV F1=0.8936±0.0004 (StratifiedGroupKFold) |
| **Leakage guard** | ✅ Geçti | 0 Variant_ID train/test'i çaprazlamıyor (`src/cli/modes/train.py`) |
| **Panel bazlı metrikler (test @ θ=0.8415)** | ✅ Tamamlandı | General=0.8185 · KANSER=0.906 · PAH=0.912 · CFTR=0.7143 |
| **Eğitim pipeline (kod)** | ✅ Çalışıyor | `python main.py --mode train --config configs/pdr.yaml` |
| **Inference pipeline** | ✅ Çalışıyor | Batch + tekli tahmin, belirsizlik desteği |
| **Açıklanabilirlik** | ✅ Çalışıyor | SHAP, LIME, GNNExplainer, Türkçe rapor |
| **Panel değerlendirme** | ✅ Çalışıyor | General, Hereditary Cancer, PAH, CFTR |
| **External validation** | ✅ Çalışıyor | `--mode external_val` |
| **Adversarial validation** | ✅ Çalışıyor | `--mode adversarial_val`, AUC≈0.50 (tüm paneller) |
| **Kalibrasyon** | ✅ Çalışıyor | İsotonik Regresyon, Brier=0.1115, ECE=0.0291 |
| **MC Dropout belirsizlik** | ✅ Çalışıyor | 10 forward pass |
| **Ablation analizi** | ✅ Tamamlandı | `reports/ablation_report.json`, 8 konfigürasyon |
| **Submission artifact'ları** | ✅ Tamamlandı | `submission/teknofest/`: manifest, checksums, predict.py |
| **Streamlit UI** | ✅ Çalışıyor | `streamlit run app.py` |
| **CI pipeline** | ✅ Çalışıyor | GitHub Actions: lint, typecheck, test, security |
| **Docker** | ✅ Mevcut | CPU ve GPU destekli |
| **Test altyapısı** | ✅ 416 test fonksiyonu (statik `def test_` sayımı, 39 dosya); parametrize sonrası toplanan item sayısı CI junit artefaktıyla doğrulanır | Smoke, unit, integration testler (2 Haziran 2026) |
| **configs/ — binary F1** | ✅ Düzeltildi | `optimize_metric: binary_f1`, GLOBAL cal-türevli eşik θ=0.8415 (canonical; panel-spesifik test'te daha kötü, opt-in) |
| **PDR belge hataları** | ✅ Tümü kapatıldı | BUG-01..12 CLOSED (24 Mayıs 2026) |
| **Bağımsız klinik validasyon** | ❌ Kapsam dışı | Araştırma prototipi; kasıtlı kapsam dışı |
| **VUS sınıflandırma** | ❌ Yok | Etiketli VUS verisi gerektirir |
| **Deployment (üretim)** | ❌ Yok | Araştırma prototipi; üretim dağıtımı planlanmıyor |

## PDR için Kalan Görevler

1. **[Gerekli]** PDR'yi resmi DOCX şablonuna aktar ve teslim et (29 Haziran 2026, 17:00)
2. **[Önerilen]** LIME-SHAP örtüşme görselini PDR §2.4'e ekle (ρ=0.89 belgelendi; görsel final)

> ✅ **Tamamlanan PSR→PDR güçlendirmeleri (06-03 revizyonu):** SHAP waterfall (§4.4),
> deney günlüğü + öğrenme eğrisi + ablasyon tablosu (§4.5), 4-model × 4-panel Binary F1
> karşılaştırma tablosu (§5.1). Detay: README §25, `CHANGELOG.md` [3.2.1]/[4.1.0].

## Bilinen Sınırlamalar

- MASTER (General) paneli MCC=0.4951: sınıf dengesizliği (2.75:1), %20-patojenik jüri seti etkisi; PDR §4.2'de açıklandı
- Klinik validasyon kapsamı dışındadır; bu sistem klinik tanı amacıyla kullanılamaz
- VUS (Önemi Belirsiz Varyant) desteği bulunmamaktadır
- Sentetik `jury_predictions.csv` placeholder'ı **silindi** (2026-06-02); gerçek tahminler jüri kör test setini sağladığında `submission/predict.py` ile üretilir

## PSR → PDR Puanı Hedef

| Kriter | PSR | PDR Hedef |
|---|---|---|
| §4.4 Açıklanabilirlik | 3.33/5 | 5/5 (SHAP waterfall + GNNExplainer) |
| §4.5 Teknik Evrim | 3.33/5 | 5/5 (deney günlüğü + ablasyon) |
| §5.1 Mimari Gerekçe | 4.00/5 | 5/5 (5×4 ablasyon tablosu) |
| **TOPLAM** | **93/100** | **≥97/100** |

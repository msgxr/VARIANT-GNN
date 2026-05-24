# Proje Durumu — VARIANT-GNN

**Güncelleme tarihi:** 24 Mayıs 2026

## Mevcut Aşama

**PDR (Proje Detay Raporu) Geliştirmesi**

PSR aşaması 93/100 puanla geçilmiştir. PDR teslim tarihi: 29 Haziran 2026, 17:00.

## Veri ve Model Durumu

> ✅ Gerçek TEKNOFEST 2026 yarışma verisi 14 Mayıs 2026'da alınmıştır.
> Model 20 Mayıs 2026'da gerçek veriyle yeniden eğitilmiştir.
> Test F1 = **0.8980** | CV F1 = 0.8668 ± 0.0081 | MCC = 0.5356

## Olgunluk Seviyesi

**Araştırma ve Yarışma Prototipi** — klinik kullanıma hazır değildir.

| Boyut | Durum | Kanıt |
|---|---|---|
| **Model mimarisi** | ✅ Stabil | XGB(30%) + LGB(30%) + GATv2GNN(25%) + DNN(15%), stacking LogReg |
| **Eğitim — gerçek yarışma verisi** | ✅ Tamamlandı | `models/PROVENANCE.json` → 2026-05-20, Test F1=0.8980 |
| **5-fold CV — gerçek veri** | ✅ Tamamlandı | `reports/cv_report.json` → CV F1=0.8668±0.0081 |
| **Panel bazlı metrikler** | ✅ Tamamlandı | MASTER=0.8872 · KANSER=0.8960 · PAH=0.9556 · CFTR=0.9524 |
| **Eğitim pipeline (kod)** | ✅ Çalışıyor | `python main.py --mode train --config configs/pdr.yaml` |
| **Inference pipeline** | ✅ Çalışıyor | Batch + tekli tahmin, belirsizlik desteği |
| **Açıklanabilirlik** | ✅ Çalışıyor | SHAP, LIME, GNNExplainer, Türkçe rapor |
| **Panel değerlendirme** | ✅ Çalışıyor | General, Hereditary Cancer, PAH, CFTR |
| **External validation** | ✅ Çalışıyor | `--mode external_val` |
| **Adversarial validation** | ✅ Çalışıyor | `--mode adversarial_val`, AUC≈0.50 (tüm paneller) |
| **Kalibrasyon** | ✅ Çalışıyor | İsotonik Regresyon, Brier=0.1283, ECE=0.0788 |
| **MC Dropout belirsizlik** | ✅ Çalışıyor | 10 forward pass |
| **Ablation analizi** | ✅ Tamamlandı | `reports/ablation_report.json`, 8 konfigürasyon |
| **Submission artifact'ları** | ✅ Tamamlandı | `submission/teknofest/`: manifest, checksums, predict.py |
| **Streamlit UI** | ✅ Çalışıyor | `streamlit run app.py` |
| **CI pipeline** | ✅ Çalışıyor | GitHub Actions: lint, typecheck, test, security |
| **Docker** | ✅ Mevcut | CPU ve GPU destekli |
| **Test altyapısı** | ✅ 278/278 test | Smoke, unit, integration testler (22 Mayıs 2026) |
| **configs/ — binary F1** | ✅ Düzeltildi | `optimize_metric: binary_f1`, θ=0.241 (24 Mayıs 2026) |
| **PDR belge hataları** | ✅ Tümü kapatıldı | BUG-01..12 CLOSED (24 Mayıs 2026) |
| **Bağımsız klinik validasyon** | ❌ Kapsam dışı | Araştırma prototipi; kasıtlı kapsam dışı |
| **VUS sınıflandırma** | ❌ Yok | Etiketli VUS verisi gerektirir |
| **Deployment (üretim)** | ❌ Yok | Araştırma prototipi; üretim dağıtımı planlanmıyor |

## PDR için Kalan Görevler

1. **[Gerekli]** PDR'yi resmi DOCX şablonuna aktar (teslim formatı)
2. **[Önerilen]** SHAP waterfall görseli PDR §2.4'e ekle (§4.4 puanı için)
3. **[Önerilen]** Deney günlüğü tablosu PDR §4.5'e ekle (§4.5 puanı için)
4. **[Önerilen]** 5×4 model-panel ablasyon tablosu (§5.1 puanı için)

## Bilinen Sınırlamalar

- MASTER paneli MCC=0.507: sınıf dengesizliği (2.75:1), normal ve PDR §4.2'de açıklandı
- Klinik validasyon kapsamı dışındadır; bu sistem klinik tanı amacıyla kullanılamaz
- VUS (Önemi Belirsiz Varyant) desteği bulunmamaktadır
- `jury_predictions.csv` sentetik placeholder — gerçek jüri verisiyle `predict.py` yeniden çalıştırılmalı

## PSR → PDR Puanı Hedef

| Kriter | PSR | PDR Hedef |
|---|---|---|
| §4.4 Açıklanabilirlik | 3.33/5 | 5/5 (SHAP waterfall + GNNExplainer) |
| §4.5 Teknik Evrim | 3.33/5 | 5/5 (deney günlüğü + ablasyon) |
| §5.1 Mimari Gerekçe | 4.00/5 | 5/5 (5×4 ablasyon tablosu) |
| **TOPLAM** | **93/100** | **≥97/100** |

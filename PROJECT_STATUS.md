# Proje Durumu — VARIANT-GNN

**Güncelleme tarihi:** Mayıs 2026

## Mevcut Aşama

**PDR (Proje Detay Raporu) Geliştirmesi**

PSR aşaması 93/100 puanla geçilmiştir. PDR teslim tarihi: 29 Haziran 2026.

## Veri Durumu

> ⚠️ Mevcut `data/train_variants.csv` ve `data/test_variants.csv` dosyaları, gerçek yarışma verisi
> alınmadan önce geliştirme amacıyla kullanılan **gerçekçi sentetik pilot veri**dir
> (`data/README.md` satır 96–98). Gerçek TEKNOFEST 2026 yarışma verisiyle eğitim ve
> değerlendirme henüz tamamlanmamıştır. Gerçek veri üzerinde elde edilmiş performans
> kanıtı bulunamamıştır.

## Olgunluk Seviyesi

**Araştırma ve Yarışma Prototipi** — klinik kullanıma hazır değildir.

| Boyut | Durum | Notlar |
|---|---|---|
| **Model mimarisi** | ✅ Stabil | XGB + LGB + GATv2GNN + DNN, 4 model ensemble |
| **Eğitim pipeline (kod)** | ✅ Çalışıyor | `python main.py --mode train --config configs/psr.yaml` |
| **5-fold CV (kod)** | ✅ Konfigüre | `configs/psr.yaml: cv_folds=5`; son train_log.txt'te 3-fold çalışma var (eski run) |
| **Eğitim — gerçek yarışma verisi** | ❌ Kanıt yok | Gerçek veri üzerinde eğitim tamamlanmamış |
| **CV sonuçları — gerçek yarışma verisi** | ❌ Kanıt yok | `cv_report.json` sentetik pilot veri sonucu |
| **Inference pipeline** | ✅ Çalışıyor | Batch + tekli tahmin, belirsizlik desteği |
| **Açıklanabilirlik** | ✅ Çalışıyor | SHAP, LIME, GNNExplainer, Türkçe rapor |
| **Panel değerlendirme** | ✅ Çalışıyor | General, Hereditary Cancer, PAH, CFTR |
| **External validation** | ✅ Çalışıyor | `--mode external_val` |
| **Adversarial validation** | ✅ Çalışıyor | `--mode adversarial_val` |
| **Kalibrasyon** | ✅ Çalışıyor | İzotonik Regresyon |
| **MC Dropout belirsizlik** | ✅ Çalışıyor | 30 ileri geçiş |
| **Streamlit UI** | ✅ Çalışıyor | `streamlit run app.py` |
| **CI pipeline** | ✅ Çalışıyor | GitHub Actions: lint, typecheck, test, security |
| **Docker** | ✅ Mevcut | CPU ve GPU destekli |
| **Test altyapısı** | ✅ Temel seviye | Smoke, unit, integration testler |
| **Veri sözleşmeleri** | ✅ Kısmi | Pydantic şema + JSON contract'lar (`data/contracts/`) |
| **Ablation analizi** | ❌ Kanıt yok | Kod altyapısı mevcut; `reports/ablation_report.json` üretilmemiş |
| **Submission artifact'ları** | ❌ Kanıt yok | `submission/teknofest/` içinde jury_predictions.csv, manifest, checksums yok |
| **Model kartı PDF** | ❌ Kanıt yok | PDF üretilmemiş |
| **Bağımsız klinik validasyon** | ❌ Kapsam dışı | Araştırma prototipi; kasıtlı kapsam dışı |
| **VUS sınıflandırma** | ❌ Yok | Etiketli VUS verisi gerektirir |
| **Deployment (üretim)** | ❌ Yok | Araştırma prototipi; üretim dağıtımı planlanmıyor |

## PDR için Kalan Ana Görevler

1. **[Kritik]** Gerçek yarışma verisiyle `python main.py --mode train --config configs/psr.yaml` çalıştır
2. **[Kritik]** `reports/cv_report.json`'u gerçek veri sonuçlarıyla güncelle; README ve MODEL_CARD'a aktar
3. **[Kritik]** `submission/teknofest/` içinde jury_predictions.csv, artifact_manifest.json, checksums.json üret
4. Ablation analizi raporunu üret (`reports/ablation_report.json`)
5. GNNExplainer görselleştirmesini arayüze entegre et
6. LIME analizi SHAP ile karşılaştırmalı çalıştır
7. CFTR stabilizasyon sürecini karşılaştırmalı belgele
8. docs/ alt yapısını tamamla (clinical/, evaluation/, submission/)

## Bilinen Sınırlamalar

- Mevcut metrikler sentetik pilot veri üzerinde hesaplanmıştır; gerçek yarışma performansını yansıtmaz
- Klinik validasyon kapsamı dışındadır; bu sistem klinik tanı amacıyla kullanılamaz
- Anonim kolon/alias hizalaması gerçek yarışma formatında yeniden doğrulanmalıdır (`src/data/column_aligner.py`, `data/contracts/`)
- VUS (Önemi Belirsiz Varyant) desteği bulunmamaktadır
- CI bazı ortamlarda PyTorch Geometric kurulum sorunları yaşanabilir
- `data/pretrain_100k.csv.dvc`: Büyük bir ön-eğitim veri seti DVC ile takip ediliyor; şartname kapsamında kullanımı doğrulanmalıdır

# Değerlendirme Protokolü — VARIANT-GNN

> **Şartname Referansı:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması
> §7.3 (Üniversite ve Üzeri Final Değerlendirmesi).

---

## 1. Birincil Yarışma Metriği (§7.3)

> "Yarışma sıralamasını belirleyecek temel metrik, TP (Doğru Pozitif),
> FP (Yanlış Pozitif) ve FN (Yanlış Negatif) değerleri üzerinden
> hesaplanan **F1 Skoru** olacaktır."

```
F1 = TP / (TP + 0.5·FP + 0.5·FN)
```

Bu, Patojenik (positive=1) sınıfı için **Binary F1** skorudur. Sistemde
`src/scientific/metrics/metrics.py:evaluate()` tarafından `binary_f1` alanı olarak
hesaplanır ve raporlanır (canlı CLI bu kopyayı import eder: `src/cli/modes/train.py`, `evaluate.py`).

## 2. Tamamlayıcı Metrikler

| Metrik | Amaç | Modül |
|--------|------|-------|
| **Binary F1** | §7.3 birincil sıralama | `evaluate.binary_f1` |
| **Macro F1** | Sınıflar arası dengeli görünüm | `evaluate.macro_f1` |
| **Precision** | Patojenik tahmin doğruluğu | `evaluate.precision` |
| **Recall** | Patojenik yakalama oranı | `evaluate.recall` |
| **MCC** | Dengesiz veri için robust korelasyon | `evaluate.mcc` |
| **ROC-AUC** | Eşikten bağımsız sıralama gücü | `evaluate.roc_auc` |
| **PR-AUC** | Pozitif sınıf-merkezli sıralama | `evaluate.pr_auc` |
| **Brier Score** | Olasılık kalibrasyonu (alt = iyi) | `evaluate.brier_score` |
| **ECE** | Beklenen Kalibrasyon Hatası | `evaluate.ece` |

---

## 3. Değerlendirme Katmanları

### 3.1 Cross-Validation (5-fold Stratified)

```bash
python main.py --mode crossval --data_file data/train_variants.csv
```

- **Yöntem:** **StratifiedGroupKFold** (`k=5`, sabit seed, `Variant_ID`'ye göre **group-aware** — aynı varyant fold'lar arasında sızmaz)
- **Birincil metrik:** Binary F1 (Pathogenic, §7.3)
- **Tamamlayıcı:** Macro F1, ROC-AUC, MCC, Brier, ECE
- **Çıktı:** `reports/cv_report.json`

### 3.2 Hold-Out Test Set Değerlendirme

```bash
python main.py --mode train --data_file data/train_variants.csv
# → GROUP-AWARE GroupShuffleSplit (test_size=0.20, Variant_ID) + tüm metrikler
```

- **Çıktı:** `reports/cv_report.json` `test_metrics` alanı
- **Optimal threshold:** F1-optimal threshold otomatik kaydedilir
  (`models/threshold.json`).

### 3.3 Panel Bazlı Değerlendirme (§3.2 dört panel)

Şartname §3.2 dört bağımsız test seti tanımlar:

| Panel | Test Patojenik | Test Benign | Toplam |
|-------|----------------|-------------|--------|
| General | 1000 | 1000 | 2000 |
| Hereditary_Cancer | 100 | 100 | 200 |
| PAH | 100 | 100 | 200 |
| CFTR | 30 | 30 | 60 |

`evaluate_per_panel()` her panel için bağımsız Binary F1 + Macro F1
hesaplar. Panel-bazlı eşik optimizasyonu da otomatik yapılır
(`reports/panel_thresholds.json`).

### 3.4 External Validation (§7.2)

> "TEKNOFEST 2026 festival alanında yapılacak olan yarışma sürecinde,
> model performansının klinik gerçeklik içinde sınanabilmesi amacıyla
> external validasyon yapılacaktır."

```bash
python main.py --mode external_val --test_file <jury_test_csv>
```

- **Eşik:** Eğitim sırasında kaydedilen F1-optimal eşik kullanılır
  (test verisi üzerinde re-tune **YASAKTIR** — aksi halde §7.3 metriği şişer).
- **Panel kırılımı:** `panel_metrics` alanında her panelin Binary F1'i
  raporlanır.
- **Çıktı:** `reports/external_validation_report.json`.

### 3.5 Cross-Panel Generalization (§3.2)

```bash
python main.py --mode panel_transfer --data_file data/train_variants.csv
```

- **Çıktı:** 4×4 F1 matrisi (`reports/panel_transfer_matrix.json`).
- Diyagonal = in-distribution, off-diagonal = transfer kabiliyeti.
- Transfer gap < 0.10 idealdir.

### 3.6 Adversarial Validation

```bash
python main.py --mode adversarial_val \
    --data_file data/train_variants.csv \
    --test_file data/test_variants.csv
```

- **Yöntem:** Train + test birleştirilir, kaynak kümeyi tahmin eden ikinci
  model eğitilir.
- **AUC ≈ 0.5:** dağılımlar benzer (iyi).
- **AUC > 0.7:** ciddi dağılım kayması (kötü, model dikkatli kullanılmalı).
- **Çıktı:** `reports/adversarial_validation_report.json`.

### 3.7 Ablation (Komponent Katkısı)

```bash
python main.py --mode ablation --data_file data/train_variants.csv
```

| Ablation | Beklenti |
|----------|----------|
| baseline | Tam ensemble Binary F1 |
| no_xgb | Genelde -0.5 ile -2 puan |
| no_lgbm | Genelde -0.5 ile -1.5 puan |
| no_gnn | Genelde -1 ile -3 puan (small panellerde daha çok) |
| no_dnn | Genelde -0.3 ile -1 puan |
| no_smote | Dengeli veride etki yok |
| no_autoencoder | -0.5 ile -1.5 puan |
| no_feature_selection | Genelde küçük etki |

- **Çıktı:** `reports/ablation_report.json`.

### 3.8 Conformal Prediction Coverage

```python
from src.scientific.conformal_prediction import ConformalPredictor

cp = ConformalPredictor(alpha=0.10, method="APS")
cp.fit(cal_proba, cal_labels)
sets, _ = cp.predict_sets(test_proba)
report = cp.evaluate_coverage(sets, test_labels)
print(report.summary())
```

**Hedef:** Empirical coverage ≥ 0.88 (target=0.90, tolerance=0.02).

### 3.9 Label Quality (Confident Learning)

```bash
python main.py --mode label_quality --data_file data/train_variants.csv
```

- **Çıktı:** `reports/label_quality_report.json` — şüpheli etiketli
  varyantları işaretler. ClinVar 3-4 yıldızlı etiketlerde bile küçük
  oranda gürültü olabilir; bu modül onları yakalar.

---

## 4. Threshold Optimizasyonu

```bash
# Validation set üzerinde F1-optimal eşik bul ve kaydet
# (otomatik olarak --mode train sırasında yapılır)
```

- **Aralık:** [0.01, 0.99] / 100 adım
- **Optimizasyon metriği:** Binary F1 (Pathogenic)
- **Kayıt:** `models/threshold.json` (key: `classification_threshold`)

External validation'da bu eşik **olduğu gibi** kullanılır; test verisinde
yeniden hesaplama §7.3'ü ihlal eder.

---

## 5. Reproducibility (§7.5)

> "Yarışma jürisi, finale kalan takımların kodlarını tekrar
> çalıştırmasını ve beyan ettikleri sonuçları bulmalarını isteme
> yetkisine sahiptir."

Tüm değerlendirme deneyleri:
1. `seed=42` ile çalışır (`src/utils/reproducibility.py`)
2. Sabit paket versiyonları (`requirements.txt`)
3. Deterministik CUDA algoritmaları (cudnn.deterministic=True)
4. Model + config + veri SHA256 ile imzalı manifest
   (`models/reproducibility_manifest.json`)

`verify_manifest_chain()` jüri tarafından artefakt bütünlüğünü doğrulamak
için kullanılabilir.

---

## 6. Metrik Hesaplama Kanonik Yolu

```python
from src.evaluation.metrics import evaluate, find_best_threshold

# y_prob: (N, 2) array — [P(Benign), P(Pathogenic)]
# y_true: (N,)  binary labels

# Optimal eşik bul (training-set sırasında)
best_thr, best_f1 = find_best_threshold(y_true, y_prob[:, 1], metric="f1")

# Tam rapor üret
report = evaluate(y_true, y_prob, threshold=best_thr)
print(f"§7.3 Binary F1: {report.binary_f1:.4f}")
print(f"Macro F1:        {report.macro_f1:.4f}")
print(f"Precision:       {report.precision:.4f}")
print(f"Recall:          {report.recall:.4f}")
```

---

## 7. Görselleştirme

`src/scientific/metrics/plots.py` `save_all_plots()` fonksiyonu, bir
EvaluationReport için aşağıdaki figürleri otomatik üretir:

- ROC eğrisi
- PR eğrisi
- Confusion matrix (sayılar + yüzdeler)
- Calibration plot (perfect line ile karşılaştırma)
- F1 vs threshold eğrisi
- Panel-başına F1 bar grafiği

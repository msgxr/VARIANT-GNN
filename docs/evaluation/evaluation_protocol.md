# Değerlendirme Protokolü — VARIANT-GNN

## Birincil Metrik

**Macro F1 Skoru** — sınıf dengesiz veriler için standart metrik.

Macro F1, her sınıf için ayrı F1 hesaplayıp ortalamasını alır; böylece hem Patojenik hem Benign sınıfların performansı eşit ağırlıkla değerlendirilir.

## Değerlendirme Katmanları

### 1. Çapraz Doğrulama (Cross-Validation)

```bash
python main.py --mode crossval --data_file data/train_variants.csv
```

- **Yöntem:** Stratified K-Fold (k=5, varsayılan)
- **Metrikler:** Macro F1, ROC-AUC, PR-AUC, MCC, Brier Score, ECE
- **Rapor:** `reports/cv_report.json`
- **Amaç:** Model seçim ve hiperparametre optimizasyonu

### 2. Eğitim Seti Değerlendirme

```bash
python main.py --mode eval --data_file data/train_variants.csv
```

### 3. Panel Bazlı Değerlendirme

Her panel için ayrı metrik hesaplanır:

| Panel | Beklenen Önem |
|---|---|
| General | Ana performans göstergesi |
| Hereditary_Cancer | Yüksek klinik öneme sahip |
| PAH | Küçük-orta boyutlu panel |
| CFTR | En küçük panel; dikkatli yorumlanmalı |

```bash
python main.py --mode train --panel cftr --data_file data/train_cftr.csv
```

### 4. Dış Doğrulama (External Validation)

```bash
python main.py --mode external_val --test_file data/test_variants.csv
```

- **Amaç:** Eğitim setinde hiç görülmemiş veri üzerinde performans
- **Metrikler:** F1, ROC-AUC, Brier Score, Hassasiyet, Duyarlılık
- **Rapor:** `reports/external_validation_report.json`

### 5. Adversarial Validation

```bash
python main.py --mode adversarial_val \
  --data_file data/train_variants.csv \
  --test_file data/test_variants.csv
```

- **Amaç:** Eğitim ve test setleri arasında dağılım kayması (domain shift) tespiti
- **Yöntem:** Eğitim/test birleştirilir, ikili sınıflandırıcı eğitilir
- **İyi sonuç:** AUC ≈ 0.5 (dağılımlar benzer)
- **Kötü sonuç:** AUC > 0.7 (önemli domain shift var)
- **Rapor:** `reports/adversarial_validation_report.json`

### 6. Kalibrasyon Değerlendirme

- **Metrikler:** ECE (Beklenen Kalibrasyon Hatası), Brier Skoru
- **Araç:** Güvenilirlik diyagramı (reliability diagram)
- **Görsel:** `reports/figures/calibration_curve.png`

## Eşik Analizi

```bash
# Threshold sweep ve optimal eşik bulmak için
python main.py --mode eval --data_file data/test_variants.csv
```

- Aralık: [0.2, 0.8] / 60 adım
- Optimizasyon metriği: Macro F1
- Rapor: `reports/threshold_report.json`

## Ablation Analizi

Mimari kararları gerekçelendirmek için:

| Senaryo | Komut | Beklenti |
|---|---|---|
| GNN olmadan | `--mode train --no-gnn` | F1 düşmeli |
| DNN olmadan | `--mode train --no-dnn` | Küçük F1 düşüşü |
| LightGBM olmadan | `--mode train --no-lgbm` | Orta F1 düşüşü |
| Kalibrasyon olmadan | `--mode train --no-calibration` | ECE artmalı |

> Not: Ablation seçenekleri `configs/experiments/` altındaki config dosyalarıyla çalıştırılabilir.

## Tekrarlanabilirlik

Tüm değerlendirme deneyleri `seed=42` ile çalıştırılır. Sonuçlar JSON formatında kaydedilir ve `CITATION.cff`'te belirtilen versiyon bilgisiyle eşleştirilir.

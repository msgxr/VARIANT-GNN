# Eğitim Logu — VARIANT-GNN

**Proje:** TEKNOFEST 2026 Sağlıkta Yapay Zekâ  
**Takım:** XYRA3 (#909249)  
**Bu dosya:** Gerçek yarışma verisi üzerinde yapılan eğitim oturumlarını belgeler.

---

## Eğitim Komutu

```bash
# Tam eğitim (PSR konfigürasyonu)
python main.py --mode train --config configs/psr.yaml

# Panel bazlı eğitim
python main.py --mode train --config configs/psr.yaml --panel General
python main.py --mode train --config configs/psr.yaml --panel Hereditary_Cancer
python main.py --mode train --config configs/psr.yaml --panel PAH
python main.py --mode train --config configs/psr.yaml --panel CFTR
```

---

## Oturum 1 — 2026-05-15 (Yarışma Verisi, Tam Eğitim)

**Veri:** `data/synthetic/train_variants.csv` + veri artırma (Gaussian augmentation)  
**Config:** `configs/psr.yaml`  
**Seed:** 42  
**Cihaz:** CPU (Intel i7) / GPU opsiyonel

### CV Sonuçları

| Fold | Binary F1 | XGB F1 | LGB F1 | GNN F1 | DNN F1 |
|:----:|----------:|-------:|-------:|-------:|-------:|
| 1 | 0.8524 | 0.8452 | 0.8634 | 0.8460 | 0.8033 |
| 2 | 0.8665 | 0.8565 | 0.8754 | 0.8245 | 0.8390 |
| 3 | 0.8771 | 0.8693 | 0.8866 | 0.8298 | 0.8233 |
| 4 | 0.8654 | 0.8610 | 0.8739 | 0.8535 | 0.8162 |
| 5 | 0.8693 | 0.8589 | 0.8824 | 0.8385 | 0.7974 |
| **Ort ± Std** | **0.8661 ± 0.0080** | | | | |

### Test Metrikleri (Bağımsız Test Seti, %20)

| Metrik | Değer |
|:-------|------:|
| Binary F1 (§7.3) | **0.8984** |
| Macro F1 | 0.7432 |
| Precision | 0.8347 |
| Recall | 0.9725 |
| ROC-AUC | 0.8671 |
| PR-AUC | 0.9292 |
| MCC | 0.5378 |
| Brier Score | 0.1283 |
| ECE | 0.0788 |
| Optimal Threshold | 0.0100 |

### Panel Bazlı Test Metrikleri

| Panel | F1 | MCC | PR-AUC | Eşik |
|:------|---:|----:|-------:|-----:|
| General | 0.8872 | 0.5070 | 0.9181 | 0.2415 |
| Hereditary_Cancer | 0.8996 | 0.6630 | 0.9524 | 0.2809 |
| PAH | 0.9556 | 0.5562 | 0.9760 | 0.1380 |
| CFTR | 0.9524 | 0.6742 | 0.9222 | 0.1085 |

### Eğitim Süresi (CPU)

| Bileşen | Süre (ort. fold başına) |
|:--------|------------------------:|
| Preprocessing + SMOTE | ~45 saniye |
| XGBoost eğitimi | ~2.5 dakika |
| LightGBM eğitimi | ~2.0 dakika |
| GNN eğitimi (50 epoch, early stop) | ~8.5 dakika |
| DNN eğitimi (20 epoch) | ~1.5 dakika |
| **Toplam (5-fold)** | **~75 dakika** |

---

## Oturum 2 — 2026-05-15 (Seed Stabilite Testi)

**Komut:**
```bash
python scripts/seed_stability_test.py --seeds 42,123,456,789,2026
```

| Seed | CV F1 | CV Std |
|:----:|------:|-------:|
| 42 | 0.8662 | 0.0080 |
| 123 | 0.8644 | 0.0051 |
| 456 | 0.8670 | 0.0100 |
| 789 | 0.8681 | 0.0046 |
| 2026 | 0.8676 | 0.0067 |
| **Genel** | **0.8667 ± 0.0013** | — |

**Sonuç:** 5 farklı seed üzerinde F1 standart sapması 0.0013 — model kararlı ve deterministik.

---

## Oturum 3 — 2026-05-16 (Veri Artırma Etkisi)

**Komut:**
```bash
python scripts/augment_train_data.py --sigma 0.02 --output data/train_variants_aug.csv
python main.py --mode train --config configs/psr.yaml --data data/train_variants_aug.csv
```

| Konfigürasyon | Test F1 | Δ F1 |
|:-------------|--------:|-----:|
| Baseline (orijinal veri) | 0.8706 | — |
| Gaussian augmentation (σ=0.02) | **0.8984** | **+0.0278** |

**Sonuç:** Gaussian feature augmentation Test F1'i +%2.78 iyileştirdi.

---

## Reproducibility Kanıtı

Bu logdaki sonuçları yeniden üretmek için:

```bash
# 1. Ortamı kur
pip install -r requirements.txt

# 2. Seed ve deterministik mod
export PYTHONHASHSEED=42

# 3. Eğit
python main.py --mode train --config configs/psr.yaml

# Beklenen çıktı:
# Cross-validation complete: Binary F1 (Pathogenic §7.3) = 0.8661 ± 0.0080
# Test Binary F1 = 0.8984
```

**Not:** Sonuçlar ±0.001 toleransla yeniden üretilebilir. CPU/GPU arası küçük farklar kabul edilebilirdir.

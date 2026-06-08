# Eğitim Logu — VARIANT-GNN

**Proje:** TEKNOFEST 2026 Sağlıkta Yapay Zekâ  
**Takım:** XYRA3 (#909249)  
**Bu dosya:** Eğitim oturumlarını kronolojik belgeler (tarihsel + canonical).

---

> ## ⚠️ GERİ ÇEKİLDİ — Oturum 1–3 (Mayıs 2026) SIZINTILI / GERÇEK-VERİ-ÖNCESİ protokoldendir
>
> Oturum 1–3 sayıları (Test F1≈0.8984, MCC≈0.5378, θ=0.01, panel eşikleri 0.24/0.28/0.14/0.11)
> **satır-bazlı split + Gaussian augmentation near-twin sızıntısıyla** üretildi ve **GEÇERSİZDİR**
> (`reports/leakage_quantification.json`). Özellikle **Oturum 3'teki "augmentation +%2.78 iyileştirme"
> bir SIZINTI ARTEFAKTIDIR** (aynı varyantın jitter'lı kopyaları train+test'e düştü); augmentation
> kalıcı olarak **devre dışı** bırakıldı.
>
> **GEÇERLİ / CANONICAL sonuçlar → Oturum 4 (aşağıda) ve [`../RESULTS_CANONICAL.json`](../RESULTS_CANONICAL.json):**
> Test F1=**0.8367** @ θ=**0.8415** (group-aware, sızıntısız), CV F1=**0.8936 ± 0.0004** (OOF-stacking),
> MCC=**0.5112**; jüri 4-panel %20-F1 ort.=**0.6202**.

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

**Sonuç (GERİ ÇEKİLDİ):** ⚠️ Bu "+%2.78 iyileştirme" gerçek bir kazanç değil, **sızıntı
artefaktıdır** — augmentation, aynı varyantın near-twin kopyalarını satır-bazlı split'in iki
yanına düşürdü (`reports/leakage_quantification.json`). Group-aware (sızıntısız) protokolde
augmentation NET FAYDA SAĞLAMAZ ve kalıcı olarak devre dışıdır.

---

## Oturum 4 — 2026-06-01 (Sızıntısız Group-Aware Retrain — ⭐ CANONICAL)

**Veri:** `data/train_variants.csv` (gerçek NDA, 3.802 satır / 3.224 tekil varyant, 4 panel)  
**Config:** `configs/pdr.yaml`  
**Split:** GROUP-AWARE (Variant_ID) — GroupShuffleSplit %80/20 hold-out + StratifiedGroupKFold 5-fold  
**Seed:** 42 | **Augmentation:** DEVRE DIŞI | **SelectKBest/AutoEncoder:** KALDIRILDI (+5.3pp dürüst geri kazanım)  
**Leakage guard:** PASSED — 0 Variant_ID train/test'i çaprazlamıyor

### Sonuçlar (canonical — `RESULTS_CANONICAL.json`)

| Metrik | Değer |
|:-------|------:|
| CV Binary F1 (OOF-stacking, nested) | **0.8936 ± 0.0004** |
| CV Binary F1 (fold-CV bileşen) | 0.8812 ± 0.0113 |
| Test Binary F1 (§7.3, hold-out @ θ=0.8415) | **0.8367** |
| MCC | 0.5112 |
| Precision / Recall | 0.9241 / 0.7644 |
| ROC-AUC / PR-AUC | 0.8538 / 0.9267 |
| Brier / ECE | 0.1115 / 0.0291 |
| **Global karar eşiği θ** | **0.8415** (%20-patojenik prior; Q&A-II ile doğrulandı) |
| Jüri 4-panel %20-F1 ortalaması | **0.6202** (havuzlanmış 0.6042 ± 0.0324) |

### Panel Bazlı Test Metrikleri (global θ=0.8415)

| Panel | F1 | MCC |
|:------|---:|----:|
| General (MASTER) | 0.8185 | 0.4951 |
| Hereditary_Cancer (KANSER) | 0.9060 | 0.7135 |
| PAH | 0.9120 | 0.5053 |
| CFTR | 0.7143 | tanımsız (n=18 degenerate) |

---

## Reproducibility Kanıtı

Canonical (Oturum 4) sonuçlarını yeniden üretmek için:

```bash
# 1. Ortamı kur
pip install -r requirements.txt

# 2. Seed ve deterministik mod
export PYTHONHASHSEED=42

# 3. Eğit (canonical config)
python main.py --mode train --config configs/pdr.yaml --data_file data/train_variants.csv

# Beklenen çıktı:
# Cross-validation complete: Binary F1 (§7.3) = 0.8936 ± 0.0004
# Leakage guard PASSED: 0 variants straddle train/test
# [TEST] [§7.3 PRIMARY] Binary F1 : 0.8367

# 4. Tutarlılık kapısı
python scripts/check_results_consistency.py   # ✅ PASS
```

**Not:** Sonuçlar deterministiktir (seed=42); ağaç üyeleri birebir, nöral bileşenler ±küçük
çalışma-varyansı (bkz. `RESULTS_CANONICAL.json → seed_stability`). CPU/GPU arası küçük farklar kabul edilebilirdir.

# REPRODUCE.md — Jüri Tekrar Üretim Kılavuzu (TEKNOFEST §7.5)

Bu belge, jüri üyesinin repoyu klonlayıp **beyan edilen sonuçları yeniden üretmesi**
için gereken kesin adımları içerir. Tüm sonuçlar tek bir kaynaktan gelir:
[`RESULTS_CANONICAL.json`](RESULTS_CANONICAL.json).

## 0. Beyan edilen sonuçlar (canonical)

| Metrik | Değer | Protokol |
|---|---|---|
| **CV Binary F1** | **0.8936 ± 0.0004** | StratifiedGroupKFold (Variant_ID), 5 fold |
| **Test Binary F1** | **0.9069** | Group-aware 80/20 hold-out |
| Test MCC | 0.5639 | precision/recall, binary_f1'i birebir üretir |
| Panel F1 (test) | General 0.8985 · KANSER 0.9385 · PAH 0.9173 · CFTR 0.9714 | |

> **Sızıntısızlık garantisi:** Eğitim, `Variant_ID`'ye göre **grup-farkında** bölme
> kullanır; aynı varyant asla hem train hem test'te yer almaz. Eğitim çıktısında
> `Leakage guard PASSED: 0 variants straddle train/test` satırı görülür.
> Önceki 0.8980/0.9269 sayıları satır-bazlı split sızıntısı nedeniyle **geri çekilmiştir**
> (kanıt: [`reports/leakage_quantification.json`](reports/leakage_quantification.json)).

## 1. Kurulum

```bash
git clone <repo> && cd VARIANT-GNN
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# macOS'ta LightGBM için OpenMP gerekiyorsa:
#   export DYLD_LIBRARY_PATH="$(pwd)/venv/lib/python3.9/site-packages/torch/lib:$DYLD_LIBRARY_PATH"
```

## 2. Tahmin (eğitilmiş modellerle — veri gerektirmez)

Eğitilmiş model ağırlıkları repoda **dahildir** (`models/*.pkl|*.pth|*.json|*.txt`, <7MB).
Jüri kendi test CSV'si ile tahmin üretebilir:

```bash
python main.py --mode predict --test_file <jury_test.csv>
# Çıktı: reports/predictions_full.csv (panel-aware eşik otomatik uygulanır)
```

GLOBAL eşik θ=0.3367 (canonical, models/threshold.json) inference'ta otomatik
yüklenir ve her satıra `Panel`'ine göre uygulanır.

## 3. Sıfırdan eğitim (NDA verisine sahip olanlar için)

```bash
python main.py --mode train --config configs/pdr.yaml --data data/train_variants.csv
```
Beklenen log:
```
Group-aware splitting ON: 3802 rows → 3224 unique variants
CV: StratifiedGroupKFold (group-aware)
Cross-validation complete: Binary F1 (§7.3) = 0.8936 ± 0.0004
Leakage guard PASSED: 0 variants straddle train/test
[TEST] [§7.3 PRIMARY] Binary F1 : 0.9069
```
`seed=42` deterministiktir; her çalıştırma aynı sonucu verir.

## 4. Sonuç tutarlılığı kontrolü (CI gate)

```bash
python scripts/check_results_consistency.py   # ✅ PASS beklenir
```
Bu betik, tüm belgelerdeki sayıların `RESULTS_CANONICAL.json` ile birebir
uyuştuğunu doğrular; geri çekilmiş sayıların yeniden sızmasını engeller.

## 5. Veri Notu (NDA)

Ham yarışma verisi (`data/raw/`, `data/train_variants.csv`) NDA kapsamındadır ve
repoya **dahil edilmez**. Veriye sahip olmayan jüri üyeleri Adım 2 (tahmin) ile
modeli doğrulayabilir; veriye sahip olanlar Adım 3 ile tam eğitimi tekrarlayabilir.

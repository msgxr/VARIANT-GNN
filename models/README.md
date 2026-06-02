# Model Artefaktları

Bu dizindeki **eğitilmiş model ağırlıkları (toplam <7 MB) jüri tekrar-üretimi için repoya dahil
edilmiştir** — bkz. [`../REPRODUCE.md`](../REPRODUCE.md) Adım 2. Jüri, veriye sahip olmadan da
bu artefaktlarla tahmin üretebilir. Yalnızca ham NDA verisi (`data/raw/`) gitignore kapsamındadır.

## Mevcut Durum — Gerçek Veri ile Sızıntısız Eğitim Tamamlandı

`PROVENANCE.json` dosyası, mevcut model ağırlıklarının hangi veriyle eğitildiğini belgeler.

> **Durum (2026-06-02, canonical):** Gerçek TEKNOFEST yarışma verisi (3.802 örnek, 3.224 tekil
> varyant, 343 anonim kolon) kullanılarak **sızıntısız (group-aware, Variant_ID)** protokolle
> eğitim tamamlanmıştır. **CV Binary F1 = 0.8936 ± 0.0004** (OOF-stacking nested-CV; bileşen
> fold-CV = 0.8779 ± 0.0062), **Test F1 = 0.833**, MCC = 0.5863. Tüm sayılar
> [`../RESULTS_CANONICAL.json`](../RESULTS_CANONICAL.json) ile tutarlıdır.
> Model artefaktları Şeyma'nın Mac'inde üretilmiş ve bu dizine taşınmıştır.
> Tahminler yalnızca yarışma/araştırma amaçlıdır — klinik tanı için kullanılamaz.

## Jüri Tekrar Çalıştırma (Tahmin)

```bash
# Tahmin üret (jüri için) — sadece test dosyası gerekir
python main.py --mode predict --test_file data/<test_blind.csv> --output submission/predictions.csv
```

## Yeniden Eğitim Gerekirse

```bash
# Sıfırdan eğit (tüm modeller ve preprocessor otomatik güncellenir)
python main.py --mode train --data_file data/train_variants.csv

# Çapraz doğrulama
python main.py --mode crossval --data_file data/train_variants.csv
```

## Artefakt Açıklamaları

| Dosya | Açıklama |
|---|---|
| `preprocessor.pkl` | VariantPreprocessor (imputer + RobustScaler + CategoricalBioFeaturizer; SMOTE sadece train fold). AutoEncoder/SelectKBest KAPALI |
| `xgb_model.json` | XGBoost sınıflandırıcı (%30) |
| `lgbm_model.txt` | LightGBM sınıflandırıcı (%30) |
| `gnn_model.pth` | VariantGATv2GNN ağırlıkları (%25) |
| `dnn_model.pth` | VariantDNN — Domain-Adversarial (%15) |
| `meta_learner.pkl` | OOF-stacking LogisticRegression meta-öğrenici (Wolpert) |
| `ensemble.pkl` / `ensemble_config.json` | Ensemble nesnesi / optimize ağırlıklar |
| `calibrator.pkl` | EnsembleCalibrator (isotonik olasılık kalibrasyonu) |
| `ood_detector.pkl` | OOD dedektörü (train-fit; inference'ta sadece `detect()`) |
| `threshold.json` | Global F1-optimal sınıflandırma eşiği θ=0.8514 (canonical) |
| `panel_thresholds.json` | 4 panel × opt-in eşik (varsayılan KAPALI; jüri global θ kullanır) |
| `metadata.json` / `manifest.json` | SHA256 + versiyon + artifact manifesti |
| `PROVENANCE.json` | Eğitim verisi kaynağı ve durum belgesi |

## Not

Hafif eğitilmiş artefaktlar (toplam <7 MB) §7.5 jüri tekrar-üretimi için repoya **dahildir**
(bkz. [`../REPRODUCE.md`](../REPRODUCE.md)). Yalnızca ham NDA verisi (`data/raw/`,
`data/train_variants*.csv`) ve büyük/geçici çıktılar `.gitignore` kapsamındadır.

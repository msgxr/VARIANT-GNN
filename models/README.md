# Model Artefaktları

Bu dizindeki dosyalar **git tarafından takip edilmez** (`.gitignore` ile hariç tutulmuştur).
Model dosyaları eğitim sonrasında otomatik olarak bu dizine kaydedilir.

## Mevcut Durum — Gerçek Veri ile Eğitim Tamamlandı

`PROVENANCE.json` dosyası, mevcut model ağırlıklarının hangi veriyle eğitildiğini belgeler.

> **Durum (2026-05-20):** Gerçek TEKNOFEST yarışma verisi (3.802 örnek, 343 anonim kolon)
> kullanılarak eğitim tamamlanmıştır. Test F1=0.8969, CV F1=0.8779±0.0062.
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
| `preprocessor.pkl` | VariantPreprocessor (imputer, scaler, SMOTE, feature selection) |
| `xgb_model.json` | XGBoost sınıflandırıcı |
| `gnn_model.pth` | VariantGATv2GNN ağırlıkları |
| `dnn_model.pth` | VariantDNN ağırlıkları |
| `ensemble_config.json` | Ensemble ağırlıkları |
| `calibrator.pkl` | EnsembleCalibrator (olasılık kalibrasyonu) |
| `threshold.json` | F1-optimal sınıflandırma eşiği |
| `PROVENANCE.json` | Eğitim verisi kaynağı ve durum belgesi |

## Not

Bu dizin `.gitignore` tarafından korunmaktadır. `git add -f` ile model dosyalarını
asla git'e ekleme. Model dosyaları büyüktür ve versiyon kontrolüne ait değildir.

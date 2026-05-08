# Model Artefaktları

Bu dizindeki dosyalar **git tarafından takip edilmez** (`.gitignore` ile hariç tutulmuştur).
Model dosyaları eğitim sonrasında otomatik olarak bu dizine kaydedilir.

## Mevcut Durum

`PROVENANCE.json` dosyası, mevcut model ağırlıklarının hangi veriyle eğitildiğini belgeler.

> **UYARI:** Gerçek TEKNOFEST verisi henüz alınmamıştır. Mevcut ağırlıklar
> geliştirme/test amaçlı sentetik veriyle üretilmiştir. Tahminler klinik
> olarak geçersizdir.

## Gerçek Veri Geldiğinde

```bash
# 1. Eğitim verisini data/ klasörüne yerleştir
cp <teknofest_train.csv> data/train_variants.csv

# 2. Sıfırdan eğit (tüm modeller ve preprocessor otomatik güncellenir)
python main.py --mode train --data_file data/train_variants.csv

# 3. Çapraz doğrulama (isteğe bağlı)
python main.py --mode crossval --data_file data/train_variants.csv

# 4. Tahmin üret (jüri için)
python main.py --mode predict --test_file data/<test_blind.csv> --output submission/predictions.csv
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

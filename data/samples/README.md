# data/samples/

Bu klasör jüri için **format örneği** içerir.
Gerçek yarışma verisi değildir.

## Dosyalar

- `jury_blind_sample.csv` — 5 satır, etiketsiz örnek CSV (gerçek test formatı)

## Kullanım

Jüri kendi test CSV'sini bu formatta sağlayacak.
Prediction komutu:

```bash
python submission/predict.py --input data/jury_test.csv
```

Çıktı: `submission/predictions.csv`

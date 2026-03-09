# Veri Seti Klasörü

Bu klasör, VARIANT-GNN projesinin gerçek yarışma veri setleri için ayrılmıştır.

## Beklenen Dosya Yapısı

```
data/
├── train.csv          ← Etiketli eğitim verisi (Label sütunu içermeli)
├── test.csv           ← Etiketsiz test verisi (tahmin için)
└── README.md          ← Bu dosya
```

## Veri Formatı

### Eğitim CSV'si (train.csv)
- Özellik sütunları (sayısal, biyoinformatik varyant özellikleri)
- `Label` sütunu: `Pathogenic`, `Likely Pathogenic`, `Benign`, `Likely Benign` (veya 0/1)

Örnek:
```
Feature_1,Feature_2,...,Feature_N,Label
0.85,1.22,...,0.31,Pathogenic
0.10,0.55,...,0.80,Benign
```

### Test CSV'si (test.csv)
- Yalnızca özellik sütunları (Label sütunu olmayabilir)
- Tahmin çıktısı `reports/submission.csv` olarak kaydedilir

## Kullanım

```bash
# Gerçek veri ile eğitim:
python main.py --mode train --data_file data/train.csv

# Gerçek test verisi ile tahmin:
python main.py --mode predict --test_file data/test.csv

# Cross-validation:
python main.py --mode crossval --data_file data/train.csv
```

## Not

Veri seti yüklenmezse `main.py` otomatik olarak **sentetik veri** üretir (geliştirme/test amaçlı).

# Veri Seti Durumu

## Mevcut Dosyalar: Sentetik Geliştirme Verisi

Bu dizindeki CSV dosyaları (`train_*.csv`, `test_*.csv`) **gerçek TEKNOFEST verisi değildir**.
TEKNOFEST 2026 şartnamesine uygun yapıda üretilmiş sentetik geliştirme verileridir.

### Amaç

- Eğitim pipeline'ının şartnameye uygunluğunu doğrulamak
- Kolon şeması ve özellik gruplarının doğruluğunu test etmek
- CI/CD testlerinin çalışmasını sağlamak

### Şartname Uyumlu Yapı

| Dosya | Satır | Dağılım |
|---|---:|---|
| `train_general.csv` | 3 000 | 1 500 Patojenik + 1 500 Benign |
| `train_hereditary_cancer.csv` | 400 | 200 + 200 |
| `train_pah.csv` | 400 | 200 + 200 |
| `train_cftr.csv` | 140 | 70 + 70 |
| `test_general.csv` | 2 000 | 1 000 + 1 000 |
| `test_hereditary_cancer.csv` | 200 | 100 + 100 |
| `test_pah.csv` | 200 | 100 + 100 |
| `test_cftr.csv` | 60 | 30 + 30 |

> **Not:** Gerçek TEKNOFEST test verisi etiketsiz (blind) verilecektir.
> Mevcut test dosyalarındaki `Label` kolonu sentetik olup gerçek değerlendirme
> için kullanılamaz.

## Gerçek Veri Geldiğinde

```bash
# Resmi TEKNOFEST verisini bu dizine yerleştir:
# data/train_variants.csv   ← eğitim seti (tüm paneller birleşik)
# data/test_blind.csv        ← jüri test seti (etiketsiz)

# Sentetik dosyaları sil veya yeniden adlandır:
mv data/train_variants.csv data/train_variants_SYNTHETIC.csv

# Yeni veriyle yeniden eğit:
python main.py --mode train --data_file data/train_variants.csv
```

## Önemli

- Sentetik veri ID formatı: `VAR_XXXXXX` — gerçek TEKNOFEST verisi farklı ID formatında olabilir
- Genomik koordinatlar (chr, pos) şartname gereği gizlenmiştir — sentetik veride de yoktur
- Model ağırlıkları bu sentetik veriyle eğitilmiştir ve klinik olarak geçersizdir

# TEKNOFEST 2026 Teslim Rehberi — VARIANT-GNN

## Yarışma Takvimi

| Kilometre Taşı | Tarih |
|---|---|
| PSR Teslim | 25 Mart 2026 (17:00) |
| PSR Sonuçları | 22 Nisan 2026 |
| Veri Dağıtımı | 5 Mayıs 2026 |
| PDR Teslim | **29 Haziran 2026 (17:00)** |
| Finalistler | 20 Temmuz 2026 |
| Finaller | Ağustos–Eylül 2026 |
| TEKNOFEST Şanlıurfa | 30 Eylül – 4 Ekim 2026 |

## PSR Durumu

PSR aşaması **93.00 / 100** puanla geçilmiştir.
Takım: XYRA3 (#909249) | Başvuru ID: #4865399

## PDR Teslim Checklist

### Zorunlu İçerik

- [ ] PDR raporu (resmi şablon ile hazırlanmış PDF)
- [ ] Takım üye bilgileri güncellenmiş
- [ ] Veri metodolojisi açıklanmış (Bölüm 3)
- [ ] Deney tasarımı ve sonuçlar (Bölüm 4)
- [ ] Yaklaşım gerekçesi (Bölüm 5)
- [ ] Referanslar (en az 5 uluslararası makale)

### Teknik Döküman

- [ ] Model kartı güncel (`docs/MODEL_CARD.md`)
- [ ] Veri kartı güncel (`DATA_CARD.md`)
- [ ] Reproducibility checklist tamamlanmış
- [ ] Panel bazlı metrik raporu üretilmiş

### Kod Kalitesi

- [ ] CI tüm testler geçiyor
- [ ] Gerçek yarışma verisiyle tam eğitim pipeline çalıştırıldı
- [ ] Artifact manifest ve checksum üretildi
- [ ] Jüri CSV export formatı doğrulandı

## Jüri Export Formatı

Tahmin çıktısı aşağıdaki formatı izlemelidir:

```csv
Variant_ID,Predicted_Label,Predicted_Class,Pathogenic_Probability,Risk_Score
VAR_001,1,Pathogenic,0.8731,87.31
VAR_002,0,Benign,0.1245,12.45
```

Üretmek için:
```bash
python submission/predict.py --input <jury_test.csv> --output submission/predictions.csv
# Çıktı: reports/predictions.csv
```

## Teslim Paketi İçeriği

```
submission/teknofest/
├── jury_predictions.csv       # Jüri formatında tahminler
├── technical_report.pdf       # PDR raporu
├── model_card.pdf             # Model kartı (PDF)
├── artifact_manifest.json     # Artifact listesi ve hash'leri
├── checksums.json             # Dosya bütünlük doğrulaması
└── submission_bundle.zip      # Tüm paket (sıkıştırılmış)
```

Paketi oluşturmak için:
```bash
# Manuel
cp reports/predictions.csv submission/teknofest/jury_predictions.csv

# Artifact manifest üretimi
python scripts/artifacts/create_manifest.py
```

## Reproducibility

Sonuçları yeniden üretmek için:
```bash
# 1. Ortamı kur
make install

# 2. Gerçek yarışma verisiyle eğit
python main.py --mode train --data_file data/train_variants.csv

# 3. Tahmin üret
python submission/predict.py --input <jury_test.csv> --output submission/predictions.csv

# 4. Değerlendirme raporunu oluştur
python main.py --mode eval --data_file data/test_variants.csv
```

Tüm adımlar sabit tohum (42) ile deterministik sonuç üretir.

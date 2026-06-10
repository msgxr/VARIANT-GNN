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

> **Tek kaynak (canonical):** Jüri CSV kolon seti yalnızca koddan tanımlanır —
> `src/scientific/submission_validator.py` → `JURY_COLUMNS`. Aşağıdaki sıra
> birebir o listedir:
>
> `Variant_ID, prediction_label, pathogenic_probability, calibrated_risk, confidence_level, uncertainty_score, expert_review_flag`
>
> **Resmi submission dosya formatı HENÜZ duyurulmadı (UNVERIFIED).** Bu yüzden
> GÜVENLİ varsayılan `--jury_minimal` modudur: yalnız `Variant_ID + prediction_label`
> (ikili 0/1). 7-kolonlu zengin çıktı iç analiz/doğrulama içindir; resmi format
> açıklanınca kolon seti güncellenecektir.

**Güvenli teslim (2 kolon — varsayılan öneri):**
```bash
python submission/predict.py --input <jury_test.csv> --output submission/predictions.csv --jury_minimal
```

```csv
Variant_ID,prediction_label
VAR_001,1
VAR_002,0
```

**Zengin çıktı (7 kolon — iç analiz; `JURY_COLUMNS`):**
```bash
python submission/predict.py --input <jury_test.csv> --output submission/predictions.csv
```

```csv
Variant_ID,prediction_label,pathogenic_probability,calibrated_risk,confidence_level,uncertainty_score,expert_review_flag
VAR_001,1,0.8731,87.31,92.40,0.0760,False
VAR_002,0,0.1245,12.45,95.10,0.0490,False
```

> **SUPERSEDED:** Eski 5-kolonlu örnek
> (`Variant_ID,Predicted_Label,Predicted_Class,Pathogenic_Probability,Risk_Score`)
> artık geçersizdir — kod ile uyumsuzdu. Yukarıdaki `JURY_COLUMNS` canonical'dır.

## Teslim Paketi İçeriği

**Şu an repoda mevcut** (`submission/teknofest/`):
```
submission/teknofest/
├── artifact_manifest.json     # Artifact listesi ve hash'leri
└── checksums.json             # Dosya bütünlük doğrulaması
```

**Teslim günü üretilir** (jüri kör test seti sağlandığında — şu an repoda yok):
```
jury_predictions.csv                  # python submission/predict.py ile üretilir
VARIANT_GNN_jury_package_<tarih>.zip  # kod + model ağırlıkları + requirements (paket ZIP)
technical_report.pdf                  # PDR PDF (resmi şablondan dışa aktarılır)
model_card.pdf                        # docs/MODEL_CARD.md → PDF
```

Jüri teslim paketini (kod + model ağırlıkları) oluşturmak için:
```bash
# Tam jüri paketi ZIP'i (submission/teknofest/VARIANT_GNN_jury_package_<tarih>.zip)
bash scripts/create_jury_package.sh

# Jüri tahmin CSV'i (jüri kör test seti sağlandıktan sonra)
python submission/predict.py --input <jury_test.csv> --output submission/predictions.csv --jury_minimal
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

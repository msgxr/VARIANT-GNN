# Veri Kartı — VARIANT-GNN

> **Ayrıntılı veri kartı:** [`data/README.md`](data/README.md) ve `data/contracts/` (şema sözleşmeleri)

## Özet

| Alan | Değer |
|---|---|
| **Veri türü** | Missense genetik varyant anotasyon profilleri |
| **Etiket** | Patojenik (1) / Benign (0) |
| **Kaynak** | TEKNOFEST 2026 yarışma verisi (NDA kapsamında) |
| **Gizlilik** | KVKK/GDPR ve TEKNOFEST NDA bağlamında; genomik adres bilgileri gizlenmiş, ham hasta verisi içermez |
| **Format** | CSV (nümerik özellikler + isteğe bağlı sekans bağlamı) |

## Panel Yapısı

| Panel | Eğitim Seti | Test Seti | Toplam |
|---|---|---|---|
| General (Genel) | 3000 (1500P + 1500B) | 2000 (1000P + 1000B) | 5000 |
| Hereditary Cancer | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| PAH | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| CFTR | 140 (70P + 70B) | 60 (30P + 30B) | 200 |

> **Şartname uyumu:** Pathogenic + Likely Pathogenic tek sınıfta; Benign + Likely Benign tek sınıfta birleştirilir. Ground truth etiketleri ACMG-referenced etiketlerden türetilmiştir.

## Veri Politikası

- Yarışma verisi TEKNOFEST NDA kapsamında olduğundan, bu repo içeriğinde **ham yarışma verisi paylaşımı** yapılmamalıdır.
- `data/samples/` dizini yalnızca örnek/sentetik veri içerecek şekilde tasarlanmıştır.
- NDA/gizlilik kısıtları varsa bu açıkça belirtilmiştir.
- `data/contracts/` dizinindeki JSON sözleşmeleri veri şemasını tanımlar.

## KLİNİK VERİ UYARISI

> Bu projede kullanılan veri; klinik hasta verisi değildir. Yarışma kapsamında sağlanan anonimleştirilmiş, in-silico anotasyon profilleri içermektedir. Genomik adres/kromozom/pozisyon bilgileri gizlenmiştir. Bu nedenle özellik isimleri ve bazı alanlar anonimleştirilmiş olabilir.

### Anonim Kolonlar ve Column Alignment

- **Neden gerekli?** Şartname gereği genomik adres bilgileri ve bazı kolon isimleri gizlenebilir.
- **Repo karşılığı**: `src/data/column_aligner.py` ve `data/contracts/column_aliases.json` ile, beklenen şema/kolon eşleme mantığı yönetilir.
- **Önerilen yaklaşım**: Jüri koşumunda gelen dosyada kolon isimleri farklı/anonim ise, sözleşme dosyalarındaki alias/grup tanımları ile hizalama yapılır; `Variant_ID` gibi kimlik alanları **özellik olarak kullanılmaz**.

### Veri Sızıntısı (Leakage) Riskleri

- **Leakage kaynağı**: Test verisinin ön işleme adımlarına (imputer/scaler/feature selection/SMOTE/autoencoder/graf kurma) yanlışlıkla dahil edilmesi.
- **Repo karşılığı**: eğitim tarafında tüm fit işlemleri fold içinde yapılacak şekilde tasarlanmıştır (`main.py`, `src/training/*`, `src/features/preprocessing.py`).

## İlgili Dosyalar

- [`data/contracts/train_schema.json`](data/contracts/train_schema.json) — Eğitim şeması
- [`data/contracts/predict_schema.json`](data/contracts/predict_schema.json) — Tahmin şeması
- [`data/contracts/label_mapping.json`](data/contracts/label_mapping.json) — Etiket eşlemesi
- [`data_contracts/variant_schema.py`](data_contracts/variant_schema.py) — Pydantic v2 şeması
- [`data/README.md`](data/README.md) — Veri dosyaları ve örnek kullanım

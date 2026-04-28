# Veri Kartı — VARIANT-GNN

> **Ayrıntılı veri kartı:** [`docs/data/data_card.md`](docs/data/data_card.md)

## Özet

| Alan | Değer |
|---|---|
| **Veri türü** | Missense genetik varyant anotasyon profilleri |
| **Etiket** | Patojenik (1) / Benign (0) |
| **Kaynak** | TEKNOFEST 2026 yarışma verisi (NDA kapsamında) |
| **Gizlilik** | KVKK + GDPR uyumlu; ham sekans verisi içermez |
| **Format** | CSV (nümerik özellikler + isteğe bağlı sekans bağlamı) |

## Panel Yapısı

| Panel | Eğitim Seti | Test Seti | Toplam |
|---|---|---|---|
| General (Genel) | 3000 (1500P + 1500B) | 2000 (1000P + 1000B) | 5000 |
| Hereditary Cancer | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| PAH | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| CFTR | 140 (70P + 70B) | 60 (30P + 30B) | 200 |

## Veri Politikası

- Gerçek yarışma verisi bu repoya eklenmemektedir.
- `data/samples/` dizini yalnızca örnek/sentetik veri içerir.
- NDA/gizlilik kısıtları varsa bu açıkça belirtilmiştir.
- `data/contracts/` dizinindeki JSON sözleşmeleri veri şemasını tanımlar.

## KLİNİK VERİ UYARISI

> Bu projede kullanılan veri; klinik hasta verisi değildir. Yarışma kapsamında sağlanan anonimleştirilmiş, in-silico anotasyon profilleri içermektedir. Veri gizliliği ve güvenliği için `docs/data/data_card.md` ve `data/DATA_PRIVACY.md` dosyalarına bakınız.

## İlgili Dosyalar

- [`data/contracts/train_schema.json`](data/contracts/train_schema.json) — Eğitim şeması
- [`data/contracts/predict_schema.json`](data/contracts/predict_schema.json) — Tahmin şeması
- [`data/contracts/label_mapping.json`](data/contracts/label_mapping.json) — Etiket eşlemesi
- [`data_contracts/variant_schema.py`](data_contracts/variant_schema.py) — Pydantic v2 şeması
- [`docs/data/data_card.md`](docs/data/data_card.md) — Ayrıntılı veri kartı

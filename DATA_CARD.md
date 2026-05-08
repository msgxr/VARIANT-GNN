# Veri Kartı — VARIANT-GNN

> **Ayrıntılı veri kartı:** [`data/README.md`](data/README.md) ve `data/contracts/` (şema sözleşmeleri)

## Özet

| Alan | Değer |
|---|---|
| **Veri türü** | Missense genetik varyant anotasyon profilleri |
| **Etiket** | Patojenik (1) / Benign (0) |
| **Mevcut veri durumu** | Gerçekçi sentetik pilot veri (bkz. `data/README.md`) |
| **Yarışma verisi durumu** | Gerçek TEKNOFEST 2026 yarışma verisiyle değiştirme bekleniyor |
| **Gizlilik** | KVKK/GDPR ve TEKNOFEST NDA bağlamında; genomik adres bilgileri gizlenmiş, ham hasta verisi içermez |
| **Format** | CSV (nümerik özellikler + isteğe bağlı sekans bağlamı) |

## Panel Yapısı

| Panel | Eğitim Seti | Test Seti | Toplam |
|---|---|---|---|
| General (Genel) | 3000 (1500P + 1500B) | 2000 (1000P + 1000B) | 5000 |
| Hereditary Cancer | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| PAH | 400 (200P + 200B) | 200 (100P + 100B) | 600 |
| CFTR | 140 (70P + 70B) | 60 (30P + 30B) | 200 |

> **Şartname uyumu:** Pathogenic + Likely Pathogenic tek sınıfta (Etiket: 1); Benign + Likely Benign tek sınıfta (Etiket: 0) birleştirilir. Ground truth etiketleri ACMG-referenced etiketlerden türetilmiştir.

## Mevcut Veri Durumu

> ⚠️ Mevcut `data/train_variants.csv` ve `data/test_variants.csv` dosyaları, gerçek yarışma verisi
> alınmadan önce geliştirme amacıyla üretilmiş **gerçekçi sentetik pilot veri**dir.
> Dosya yapısı, kolon şeması ve panel boyutları şartnameyle uyumludur; ancak içeriğin
> gerçek ClinVar/gnomAD kayıtlarından türetilmediği unutulmamalıdır.
> Gerçek yarışma verisi alındığında CSV dosyaları değiştirilerek pipeline yeniden çalıştırılacaktır.

## Etiket Kaynakları

- **Patojenik sınıf (1):** ClinVar + ClinGen "Expert Panel" veya "Practice Guideline" değerlendirmeli kayıtlar. Güvenilirlik: 3–4 yıldız. Kapsam: Pathogenic + Likely Pathogenic.
- **Benign sınıf (0):** ClinVar (Benign + Likely Benign) + gnomAD sağlıklı popülasyon varyantları.
- **Dışlanan:** VUS (Variant of Uncertain Significance) — etiket güvenilirliği yetersiz olduğundan çıkarıldı.

## Genomik Adres Gizleme Kuralı

- Şartname gereği: `chromosome`, `position`, `chr`, `rsid`, `hgvs_genomic` gibi tüm koordinat bilgileri veri setinde yer almaz.
- `src/data/leakage_firewall.py` — COORDINATE_COLUMNS blocklist ile bu sütunların pipeline'a girmesi engellenir.
- `src/data/competition_sanitizer.py` — Hem eğitim hem inference modunda otomatik temizleme yapar.
- CI `schema-drift` job'u bu blocklist'lerin bütünlüğünü her push'ta doğrular.

## Veri Politikası

- Gerçek yarışma verisi TEKNOFEST NDA kapsamında olduğundan bu repo içeriğinde paylaşılamaz.
- `data/samples/` dizini yalnızca örnek/sentetik veri içerecek şekilde tasarlanmıştır.
- `data/contracts/` dizinindeki JSON sözleşmeleri veri şemasını tanımlar.

## KLİNİK VERİ UYARISI

> Bu projede kullanılan veri; klinik hasta verisi değildir. Yarışma kapsamında sağlanan
> anonimleştirilmiş, in-silico anotasyon profilleri içermektedir. Genomik adres/kromozom/pozisyon
> bilgileri gizlenmiştir. Bu sistem klinik tanı, tedavi veya bağımsız tıbbi karar destek
> amacıyla kullanılamaz. Model çıktıları yalnızca araştırma, eğitim ve yarışma değerlendirmesi
> kapsamında yorumlanmalıdır.

## Anonim Kolonlar ve Column Alignment

- **Neden gerekli?** Şartname gereği genomik adres bilgileri ve bazı kolon isimleri gizlenebilir.
- **Repo karşılığı:** `src/data/column_aligner.py` ve `data/contracts/column_aliases.json` ile beklenen şema/kolon eşleme mantığı yönetilir.
- **Önerilen yaklaşım:** Jüri koşumunda gelen dosyada kolon isimleri farklı/anonim ise, sözleşme dosyalarındaki alias/grup tanımları ile hizalama yapılır; `Variant_ID` gibi kimlik alanları **özellik olarak kullanılmaz**.

## Veri Sızıntısı (Leakage) Riskleri

- **Leakage kaynağı:** Test verisinin ön işleme adımlarına (imputer/scaler/feature selection/SMOTE/autoencoder/graf kurma) yanlışlıkla dahil edilmesi.
- **Repo karşılığı:** Tüm fit işlemleri fold içinde yapılacak şekilde tasarlanmıştır (`src/training/trainer.py`, `src/features/preprocessing.py`).
- **CI doğrulaması:** `leakage-audit` job'u sentetik kirli veri üzerinde otomatik test çalıştırır.

## İlgili Dosyalar

- [`data/contracts/train_schema.json`](data/contracts/train_schema.json) — Eğitim şeması
- [`data/contracts/predict_schema.json`](data/contracts/predict_schema.json) — Tahmin şeması
- [`data/contracts/label_mapping.json`](data/contracts/label_mapping.json) — Etiket eşlemesi
- [`data_contracts/variant_schema.py`](data_contracts/variant_schema.py) — Pydantic v2 şeması
- [`data/README.md`](data/README.md) — Veri dosyaları ve özellik listesi

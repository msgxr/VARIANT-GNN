# Veri Kartı — VARIANT-GNN

> **Ayrıntılı veri kartı:** [`data/README.md`](data/README.md) ve `data/contracts/` (şema sözleşmeleri)

## Özet

| Alan | Değer |
|---|---|
| **Veri türü** | Missense genetik varyant anotasyon profilleri |
| **Etiket** | Patojenik (1) / Benign (0) |
| **Mevcut veri durumu** | ✅ Gerçek TEKNOFEST 2026 yarışma verisi (14 Mayıs 2026 alındı) |
| **Yarışma verisi durumu** | ✅ Model 20 Mayıs 2026'da gerçek veriyle yeniden eğitildi — Test F1 = 0.8969 |
| **Gizlilik** | KVKK/GDPR ve TEKNOFEST NDA bağlamında; genomik adres bilgileri gizlenmiş, ham hasta verisi içermez |
| **Format** | CSV (nümerik özellikler + isteğe bağlı sekans bağlamı) |

## Panel Yapısı

### Şartname beklentisi (§3.2)

| Panel | Eğitim Seti (şartname) | Test Seti (jüri günü gelecek) |
|---|---|---|
| General (Genel) | 3000 (1500P + 1500B) | 2000 (1000P + 1000B) |
| Hereditary Cancer | 400 (200P + 200B) | 200 (100P + 100B) |
| PAH | 400 (200P + 200B) | 200 (100P + 100B) |
| CFTR | 140 (70P + 70B) | 60 (30P + 30B) |

### Gerçek TEKNOFEST verisi (14 Mayıs 2026'da alındı)

| Panel | Patojenik | Benign | Toplam | Oran |
|---|---|---|---|---|
| General | 2149 | 782 | 2931 | 2.75:1 |
| Hereditary Cancer | 268 | 120 | 388 | 2.23:1 |
| PAH | 310 | 62 | 372 | 5.00:1 |
| CFTR | 90 | 21 | 111 | 4.29:1 |
| **Toplam** | **2817** | **985** | **3802** | **2.86:1** |

> **Not:** Gerçek veri şartnamenin vaat ettiği 1:1 dengesinden farklı geldi. Bunu karşılamak için eğitim pipeline'ında SMOTE (sadece training fold içinde) ve sınıf-ağırlıklı kayıp fonksiyonu kullanıldı.
>
> **Augmentation (KAPALI):** Önceden materyalize edilmiş Gaussian jitter'lı `train_variants_aug.csv`
> (3802→7604), near-twin kopyalarla satır-bazlı split'te **train/test sızıntısı** yaratıyordu →
> devre dışı bırakıldı. Eğitim 3802 orijinal örnek + `Variant_ID` **group-aware** split ile yapılır
> (`reports/leakage_quantification.json`).

> **Şartname uyumu:** Pathogenic + Likely Pathogenic tek sınıfta (Etiket: 1); Benign + Likely Benign tek sınıfta (Etiket: 0) birleştirildi. Ground truth etiketleri ACMG-referenced etiketlerden türetilmiştir.

## Veri Durumu

> ✅ **Gerçek TEKNOFEST 2026 yarışma verisi 14 Mayıs 2026'da alındı.** Model 20 Mayıs 2026'da bu veriyle yeniden eğitildi.
>
> Eski sentetik geliştirme verisi `data/synthetic/` altında arşivlenmiştir.
> Gerçek veri `data/raw/` klasöründe TEKNOFEST NDA kapsamında lokal olarak tutulmaktadır (GitHub'a yüklenmez).
>
> **Kolon yapısı:** 354 kolon — Variant_ID, Panel, AL_1..AL_351 (anonim sayısal özellikler, §3.2), CAT_1..6, EK_1..9, AA_1..2, Label.
> Genomik adres (chr/pos) mevcut değil, şartname gereği gizlenmiş.

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

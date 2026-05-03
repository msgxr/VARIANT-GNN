# Model Kartı — VARIANT-GNN

> **Ayrıntılı model kartı:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)

## Özet

| Alan | Değer |
|---|---|
| **Proje** | VARIANT-GNN |
| **Mimari** | XGBoost + LightGBM + VariantGATv2GNN + DNN — Dört Modlu Hibrit Topluluk |
| **Görev** | Missense genetik varyantların Patojenik / Benign sınıflandırması |
| **Yarışma** | TEKNOFEST 2026 Sağlıkta Yapay Zekâ — Üniversite ve Üzeri |
| **Durum** | Araştırma ve yarışma prototipi |

## Model Bileşenleri ve Ağırlıklar

| Model | Ağırlık | Açıklama |
|---|---|---|
| XGBoost | %30 | Gradyan güçlendirilmiş karar ağaçları |
| LightGBM | %30 | Yaprak bazlı gradyan güçlendirme |
| VariantGATv2GNN | %25 | GATv2 dikkat mekanizmalı grafik sinir ağı |
| DNN | %15 | İleri beslemeli sinir ağı |

Ağırlıklar `configs/default.yaml` üzerinden yapılandırılabilir ve otomatik optimize edilebilir.

## Destekleyici Katmanlar

- **Kalibrasyon:** İzotonik Regresyon
- **Belirsizlik:** MC Dropout (30 ileri geçiş)
- **Açıklanabilirlik:** SHAP, LIME, GNNExplainer, Türkçe klinik rapor
- **Değerlendirme:** Panel bazlı metrikler, external validation, adversarial validation

## KLİNİK KULLANIM UYARISI

> Bu sistem TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması için geliştirilmiş bir **araştırma ve yarışma prototipidir.**
>
> - Klinik tanı koyamaz.
> - Tedavi kararı üretemez.
> - Klinik kullanıma hazır değildir.
> - Uzman değerlendirmesinin yerine geçmez.
> - Bağımsız klinik validasyon gerektirir.
> - Klinik kararın tek dayanağı olarak kullanılmamalıdır.
> - İnsan uzman denetimi zorunludur.

## Hızlı Başlangıç

```bash
# Eğitim
python main.py --mode train

# Tahmin
python main.py --mode predict --test_file data/test_variants_blind.csv

# Açıklanabilirlik
python main.py --mode explain --data_file data/train_variants.csv
```

Ayrıntılı kullanım için [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) ve [`README.md`](README.md) dosyalarına bakınız.

## Notlar (TEKNOFEST 2026 Şartname Uyumlu)

- **Final değerlendirme metriği**: F1 skoru (TP/FP/FN üzerinden). Repo içinde `main.py` eğitim çıktısı olarak `reports/cv_report.json` üretir ve metrik alanlarını kaydeder.
- **Etiket birleştirme**: Pathogenic + Likely Pathogenic → 1; Benign + Likely Benign → 0 (konfig: `configs/psr.yaml` → `schema.label_mapping`).
- **Klinik kullanım uyarısı**: Çıktılar araştırma/yarışma bağlamındadır; klinik kararların tek dayanağı olamaz. Human-in-the-loop yaklaşımı belirsizlik (MC Dropout) üzerinden işaretleme yapar.

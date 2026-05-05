# Teslim Kontrol Listesi — VARIANT-GNN

**PDR Teslim Tarihi:** 29 Haziran 2026

---

## Veri Durumu

| Görev | Durum |
|---|---|
| Gerçek yarışma verisi alındı (5 Mayıs 2026) | ⏳ Bekliyor |
| Gerçek veriyle model eğitimi tamamlandı | ❌ Kanıt bulunamadı |
| Gerçek veri üzerinde cv_report.json üretildi | ❌ Kanıt bulunamadı |
| train_log.txt gerçek veri eğitimini gösteriyor | ❌ Kanıt bulunamadı |

> Mevcut `cv_report.json` ve `train_log.txt` sentetik pilot veri üzerindeki çalışmaları yansıtmaktadır.

---

## Rapor (PDR)

| Görev | Durum |
|---|---|
| PDR resmi şablonu kullanıldı | ⏳ Bekliyor |
| Takım şeması dolduruldu | ⏳ Bekliyor |
| 10 uluslararası makale özeti yazıldı | ⏳ Bekliyor |
| Veri seti ve etiketler açıklandı (Bölüm 3.1) | ⏳ Bekliyor |
| Veri ön işleme detaylandırıldı (Bölüm 3.3) | ⏳ Bekliyor |
| Sınıf dengesi stratejisi açıklandı (Bölüm 3.5) | ⏳ Bekliyor |
| Deney protokolü ve bölme stratejisi açıklandı (Bölüm 4.1) | ⏳ Bekliyor |
| Metrikler ve panel bazlı sonuçlar raporlandı (Bölüm 4.2) | ❌ Gerçek veri sonucu yok |
| Hata analizi yapıldı (Bölüm 4.3) | ❌ Gerçek veri sonucu yok |
| Açıklanabilirlik yaklaşımı anlatıldı (Bölüm 4.4) | ⏳ Bekliyor |
| Mimari seçim gerekçesi yazıldı (Bölüm 5.1) | ⏳ Bekliyor |
| Ablation sonuçları eklendi (Bölüm 5.2–5.3) | ❌ `reports/ablation_report.json` yok |
| Hesaplama kaynakları belirtildi (Bölüm 5.4) | ⏳ Bekliyor |

---

## Teknik Dosyalar

| Görev | Durum | Kanıt |
|---|---|---|
| `submission/teknofest/jury_predictions.csv` üretildi | ❌ Yok | Dosya mevcut değil |
| `submission/teknofest/artifact_manifest.json` üretildi | ⚠️ Placeholder | Tüm SHA256 ve tarih alanları PLACEHOLDER |
| `submission/teknofest/checksums.json` üretildi | ❌ Yok | Dosya mevcut değil |
| Model kartı PDF olarak hazırlandı | ⚠️ Kısmi | `reports/VARIANT_GNN_Rapor_TEKNOFEST2026.pdf` var; içerik doğrulanmadı |
| Reproducibility checklist tamamlandı | ❌ Tamamlanmadı | |

---

## Kod Kalitesi

| Görev | Durum | Notlar |
|---|---|---|
| `pytest tests/smoke/` | ⏳ Çalıştırılmalı | CI'de tanımlı |
| `pytest tests/unit/` | ⏳ Çalıştırılmalı | CI'de tanımlı |
| `pytest tests/integration/` | ⏳ Çalıştırılmalı | CI'de tanımlı |
| `ruff check src/` | ⏳ Çalıştırılmalı | CI'de tanımlı |
| `mypy src/` | ⏳ Çalıştırılmalı | CI'de tanımlı |
| CI pipeline yeşil | ⏳ Doğrulanmalı | GitHub Actions |

---

## Veri

| Görev | Durum |
|---|---|
| Gerçek yarışma verisi repoya eklenmedi (NDA) | ✅ Uyumlu |
| `data/samples/` örnek veri güncel | ⏳ Doğrulanmalı |
| `data/contracts/` sözleşmeler tamamlandı | ✅ JSON sözleşmeleri mevcut |
| Panel bazlı dosyalar doğru yerleştirildi | ✅ `train_*.csv` / `test_*.csv` mevcut |
| `data/pretrain_100k.csv.dvc` kaynağı belgelendi | ❌ Belirsiz |

---

## Güvenlik

| Görev | Durum |
|---|---|
| `.env` veya gizli credential repoda yok | ✅ `.env.example` var, `.env` yok |
| Model binary'leri gitignore kapsamında | ⏳ Doğrulanmalı |
| NDA kapsamındaki yarışma verisi repoda yok | ✅ Uyumlu |

---

## Reproducibility Checklist

| Görev | Durum |
|---|---|
| `seed=42` tüm bileşenlerde aktif | ✅ `configs/psr.yaml`, `src/utils/reproducibility.py` |
| `requirements.txt` sabitlenmiş versiyonlar içeriyor | ⏳ Doğrulanmalı |
| Docker ile ortam yeniden oluşturulabilir | ✅ `Dockerfile` mevcut |
| README kurulum adımları test edildi | ⏳ Doğrulanmalı |
| Gerçek veriyle eğitim tekrarlanabilirliği kanıtlandı | ❌ Gerçek veri eğitimi tamamlanmadı |

---

## Teslim Öncesi Kesin Yapılması Gereken Son 5 İş

1. **Gerçek yarışma verisiyle** `python main.py --mode train --config configs/psr.yaml` çalıştır ve `cv_report.json`'u güncelle.
2. `python submission/predict.py --input <gerçek_blind_test.csv> --model_dir models/final --output submission/teknofest/jury_predictions.csv --config configs/pdr.yaml` çalıştır.
3. `submission/teknofest/artifact_manifest.json` içindeki tüm PLACEHOLDER alanlarını gerçek SHA256 ve tarih değerleriyle doldur; `checksums.json` üret.
4. Ablation analizini çalıştır ve `reports/ablation_report.json` üret.
5. CI pipeline'ını son kez çalıştır; tüm job'ların yeşil olduğunu doğrula.

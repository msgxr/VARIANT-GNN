# Güvenlik Politikası — VARIANT-GNN

Bu belge, `VARIANT-GNN` deposu için güvenlik bildirim sürecini ve temel güvenlik ilkelerini tanımlar. Proje, **TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması** kapsamında **NDA/gizlilik** yükümlülükleri bulunan veri akışlarıyla çalışabilecek şekilde tasarlanmıştır.

## Kapsam

- **Kod güvenliği**: `src/`, `main.py`, `app.py`, `src/api/rest_api.py`
- **Artefakt güvenliği**: model dosyaları (`models/`), çıktı dosyaları (`reports/`, `submission/`)
- **Veri gizliliği**: `data/` altındaki dosyalar ve sözleşmeler (`data/contracts/`)

## Güvenlik Açığı Bildirimi

- **Lütfen güvenlik açıklarını herkese açık GitHub Issue olarak açmayın.**
- Bildirim için tercih edilen yöntem: depo sahibi/maintainer ile **özel iletişim** (GitHub profilindeki iletişim kanalları).
- Bildirim içeriği:
  - Etkilenen dosya/yol (örn. `src/utils/serialization.py`)
  - Minimum yeniden üretim adımları
  - Etki analizi (C/I/A)
  - Varsa PoC (zararsız, veri sızdırmayan)

## Veri Gizliliği ve NDA (TEKNOFEST)

- TEKNOFEST yarışma verisi NDA kapsamında olabilir. Bu nedenle:
  - Ham yarışma verisi repoya **eklenmemelidir**.
  - Issue/PR içeriğinde **ham veri satırı**, genomik adres, kişisel veri veya NDA kapsamına girecek içerik paylaşılmamalıdır.
- Repo içinde şema sözleşmeleri `data/contracts/` altında tutulur; anonim kolon/alias eşleme bu sözleşmelere dayanır.

## Model Dosyaları ve Güvenli Yükleme

- PyTorch ağırlıkları güvenli yükleme ile okunur (`src/utils/serialization.py` içinde `torch.load(..., weights_only=True)` kullanımı mevcuttur; sürüm uyumsuzsa geriye dönük fallback uygulanır).
- XGBoost modeli **JSON** formatında saklanır (`models/xgb_model.json`), pickle tabanlı model yükleme önerilmez.
- `preprocessor.pkl` ve `calibrator.pkl` gibi dosyalar **yalnızca güvenilir kaynaklardan** alınmalı ve yüklenmelidir (pickle tabanlı format riski).

## CI Güvenlik Kontrolleri

- GitHub Actions iş akışı: `.github/workflows/ci.yml`
- Bandit taraması CI içinde çalışır (repo yolları: `src/`, `data_contracts/`, `main.py`, `app.py`).

## Desteklenen Sürümler

Bu depo için “desteklenen sürüm” kavramı, yarışma takvimi gereği **PDR/Final** aşamalarında çalıştırılabilirlik ve tekrar üretilebilirlik hedefleriyle sınırlıdır. Sürüm geçmişi için `CHANGELOG.md` ve `RELEASE_NOTES.md` dosyalarına bakınız.

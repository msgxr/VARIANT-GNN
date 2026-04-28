# Katkı Rehberi — VARIANT-GNN

VARIANT-GNN'e katkıda bulunmak istiyorsanız bu rehberi okuyunuz.

## Geliştirme Ortamı Kurulumu

```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# CPU (geliştirme için)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Testleri Çalıştırma

```bash
# Smoke testler (hızlı, import kontrolü)
pytest tests/smoke/ -v

# Unit testler
pytest tests/unit/ -v

# Integration testler
pytest tests/integration/ -v

# Tüm testler
pytest tests/ -v
```

## Kod Kalitesi

```bash
# Lint
ruff check src/ tests/ main.py app.py

# Otomatik düzeltme
ruff check --fix src/ tests/ main.py app.py

# Type check
mypy src/ --ignore-missing-imports

# Güvenlik taraması
bandit -r src/ -ll
```

Veya Makefile ile:

```bash
make lint
make typecheck
make test
make security
```

## Geliştirme Kuralları

### Kod Stili
- Ruff ile uyumlu Python kodu
- Tip anotasyonları (`from __future__ import annotations`)
- Sınıf/fonksiyon docstring'leri yalnızca WHY açıklamaları için
- WHAT açıklamaları iyi isimlendirilmiş değişkenlerle

### Test Yazma
- Her yeni özellik için unit test
- Gerçek yarışma verisi test fixture olarak kullanılmamalı
- `data/samples/` altındaki sentetik veri kullanılmalı
- Smoke testleri import seviyesinde tutulmalı

### Commit Mesajları
```
<tip>: <kısa açıklama>

[opsiyonel gövde]
```
Tipler: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

### Branch Stratejisi
- `main` — stabil, yarışma sürümü
- `develop` — aktif geliştirme
- `feature/xyz` — yeni özellik branch'leri
- `fix/xyz` — hata düzeltme branch'leri

## Kritik Kurallar

1. **Gerçek yarışma verisi** repoya eklenmez.
2. **Model binary dosyaları** (.pth, .pkl) gitignore kapsamında tutulur.
3. **Klinik iddialar** — "tanı koyar", "tedavi önerir" gibi ifadeler kullanılmaz.
4. **Veri sızıntısı** — preprocessing tüm fold içinde fit edilir.
5. **Backward compat** — `VariantSAGEGNN` gibi alias'lar belgelenerek korunur.

## Pull Request Süreci

1. Branch oluştur: `git checkout -b feature/yeni-ozellik`
2. Değişiklikleri yap ve test et
3. PR oluştur: `main` branch'e karşı
4. CI'ın geçmesini bekle (lint + typecheck + test + security)
5. İnceleme ve merge

## Güvenlik Bildirimi

Güvenlik açığı tespit ederseniz `docs/SECURITY.md` dosyasındaki prosedürü izleyiniz.

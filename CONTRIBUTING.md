# Katkı Rehberi — VARIANT-GNN

**Proje:** VARIANT-GNN — Missense Varyant Patojenisite Tahmini  
**Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**Resmi Kaynak:** https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/  
**Şartname:** 2026 Sağlıkta Yapay Zeka Türkçe Şartname v4

> Bu belgede yer alan tüm kısıtlamalar, yukarıdaki resmi TEKNOFEST şartnamesinden
> doğrudan alınmıştır. Çelişki durumunda şartname geçerlidir.

---

## 1. Kimler Katkı Yapabilir?

Bu proje TEKNOFEST 2026 yarışma süreci kapsamında geliştirilmektedir.

**XYRA3 Takım Üyeleri:**  
Şartname §5 gereği, yarışma sürecinde üretilen tüm kod ve modeller takım
üyelerinin emeğinin ürünüdür. Katkı sağlayanlar bu kurala uymakla yükümlüdür.

**Harici Katkılar:**  
Hata bildirimi, dokümantasyon iyileştirmesi ve teknik öneri kabul edilir.
Ancak şartname yükümlülükleri ve NDA kapsamı dahilinde değerlendirme yapılır.

---

## 2. Önce Bunları Oku — Şartname Yükümlülükleri

### 2.1. Yarışma Verisi Kısıtı — Şartname §4

> *"Yarışmacılar, yarışmada paydaşlar tarafından sağlanacak verilere
> ancak 'Gizlilik Sözleşmesini' imzalı olarak sunmaları halinde erişim
> sağlayabilecek ve yarışmaya katılabileceklerdir."*

```
✗  Ham yarışma eğitim/test verisi REPOYA EKLENEMEz
✗  Sınıf etiketleri veya ground truth dosyaları paylaşılamaz
✗  Issue / PR / commit mesajında ham veri satırı yazılamaz
✗  Genomik adres (Chr/Pos) içeren hiçbir çıktı paylaşılamaz
```

### 2.2. Genomik Adres Yasağı — Şartname §3.2

> *"Yarışma veri setinde varyantların genomik adres (kromozom ve pozisyon)
> bilgileri... tamamen gizlenmiştir. Yarışmacıların patojenite tahminlerini
> harici veri kaynaklarına başvurmaksızın... yapmaları sağlanmaktadır."*

```
✗  ClinVar / gnomAD API ile etiket araması yapılamaz
✗  Genomik adres üzerinden tersine mühendislik yapılamaz
✗  Dış veri kaynağıyla etiket sızıntısı (leakage) oluşturulamaz
```

### 2.3. Klinik Kullanım Yasağı — Şartname §10

> *"Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar,
> herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla
> kullanılamaz. Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."*

Kod, yorum veya dokümantasyonda şu ifadeler **kullanılamaz**:

```
✗  "tanı koyar / koyabilir"
✗  "tedavi önerir"
✗  "%100 doğru"
✗  "klinik olarak kanıtlanmıştır"
✗  "doktor yerine geçer"
```

### 2.4. Tekrarlanabilirlik Zorunluluğu — Şartname §7.5

> *"Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını
> ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."*

Her katkı bu garantiyi korumalıdır:

```
✅  random_state=42 — tüm stokastik işlemlerde sabit seed
✅  python main.py --mode train — tek komutla çalıştırılabilir
✅  requirements.txt — sabit versiyonlarla kilitli
✅  CV F1 = 0.8936 ± 0.0004 (OOF-stacking)  |  Test F1 = 0.8367 @ θ=0.8415
```

---

## 3. Geliştirme Ortamı Kurulumu

```bash
# 1. Klonla
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# 2. Sanal Ortam
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Bağımlılıklar
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Doğrulama
python -c "from src.core.gnn import VariantGATv2GNN; print('GNN OK')"
python -c "from src.core.ensemble import HybridEnsemble; print('Ensemble OK')"
pytest tests/smoke/ -q
```

### PyTorch Versiyonu

Bu proje `torch==2.2.1` ve `torch-geometric==2.5.3` kullanır.
`requirements.txt` sabit versiyonlar içerir — değiştirme.

---

## 4. Testleri Çalıştırma

```bash
# Smoke testler — hızlı import ve yapı kontrolü
pytest tests/smoke/ -v

# Unit testler
pytest tests/unit/ -v

# Integration testler
pytest tests/integration/ -v

# Tüm testler
pytest tests/ -v --tb=short

# Belirli modül
pytest tests/unit/test_preprocessing.py -v
```

**Test kuralları:**

```
✅  Testlerde sentetik veri kullan (data/samples/)
✅  Her yeni özellik için unit test ekle
✗   Gerçek yarışma verisi test fixture olarak kullanılamaz (NDA §4)
✗   Test çıktısında genomik adres veya etiket verisi bulunmamalı
```

---

## 5. Kod Kalitesi

```bash
# Lint
ruff check src/ tests/ main.py app.py

# Otomatik düzeltme
ruff check --fix src/ tests/ main.py app.py

# Type check
mypy src/ --ignore-missing-imports

# Güvenlik taraması (Bandit)
bandit -r src/ main.py app.py -ll

# Bağımlılık güvenlik taraması
pip-audit
```

Makefile kısayolları:

```bash
make lint       # ruff check
make typecheck  # mypy
make test       # pytest
make security   # bandit + pip-audit
make all        # hepsi birden
```

---

## 6. Geliştirme Kuralları

### 6.1. Kod Stili

```python
# ✅ Doğru
from __future__ import annotations
from typing import Optional

def train_fold(X: np.ndarray, y: np.ndarray, fold: int) -> FoldResult:
    """GATv2 ve ensemble'ı tek fold'da eğitir."""
    ...

# ✗ Yanlış — gereksiz yorum (WHAT anlatıyor, WHY değil)
# X array'ini y etiketleriyle fold numarasında eğit
def train_fold(X, y, fold):
    ...
```

- Ruff ile uyumlu Python
- `from __future__ import annotations` her dosyada
- Docstring yalnızca WHY (neden) için — WHAT iyi isimler anlatır
- Sihirli sabit yok: `threshold = 0.8415` değil, `cfg.thresholds.decision_threshold`

### 6.2. Veri Sızıntısı Önleme (Kritik)

Şartname §3.2 ve PSR §3.2 uyumu:

```python
# ✅ DOĞRU — preprocessing SADECE eğitim fold'unda fit edilir
preprocessor = VariantPreprocessor()
X_train_proc, y_res = preprocessor.fit_resample_train(X_train, y_train)
X_val_proc          = preprocessor.transform(X_val)   # fit YOK

# ✗ YANLIŞ — tüm veriyi fit eder → veri sızıntısı
preprocessor.fit(X_all)   # KESİNLİKLE YAPMA
```

### 6.3. Commit Mesajı Formatı

```
<tip>: <kısa açıklama (50 karakter max)>

[opsiyonel gövde — neden bu değişiklik?]

Co-Authored-By: İsim <email>
```

**Tipler:**

| Tip | Ne Zaman |
|:---|:---|
| `feat` | Yeni özellik |
| `fix` | Hata düzeltme |
| `docs` | Sadece dokümantasyon |
| `test` | Test ekleme/güncelleme |
| `refactor` | Davranış değişikliği olmadan yapı değişikliği |
| `ci` | CI/CD değişiklikleri |
| `chore` | Bağımlılık güncelleme, temizlik |
| `security` | Güvenlik düzeltmesi |
| `legal` | LICENSE/SECURITY/CONTRIBUTING |

### 6.4. Branch Stratejisi

```
main          ← stabil, yarışma sürümü (doğrudan push yapılmaz)
develop       ← aktif geliştirme
feature/xyz   ← yeni özellik
fix/xyz       ← hata düzeltme
docs/xyz      ← dokümantasyon
```

---

## 7. Pull Request Süreci

```bash
# 1. Branch oluştur
git checkout -b feature/panel-threshold-optimization

# 2. Değişiklik yap, test et
pytest tests/ -q
make lint
make security

# 3. Commit
git add src/evaluation/metrics.py
git commit -m "feat: panel bazlı F1-optimal threshold optimizasyonu"

# 4. Push
git push origin feature/panel-threshold-optimization

# 5. PR aç — main'e karşı
```

**PR Kontrol Listesi:**

```
[ ] CI geçiyor (lint + typecheck + test + security)
[ ] Gerçek yarışma verisi eklenmedi
[ ] Klinik iddia içermiyor (§10)
[ ] Genomik adres veya etiket verisi yok (§3.2)
[ ] Veri sızıntısı riski kontrol edildi
[ ] random_state=42 korunuyor (§7.5 tekrarlanabilirlik)
[ ] Test eklenmiş (yeni özellikler için)
[ ] Commit mesajı formatına uygun
```

---

## 8. Kritik Kurallar Özeti

| Kural | Kaynak | Sonuç |
|:---|:---:|:---|
| Gerçek yarışma verisi repoya eklenmez | §4 NDA | Diskalifiye riski |
| Klinik iddia kullanılamaz | §10 | Şartname ihlali |
| Genomik adres araması yapılamaz | §3.2 | Şartname ihlali |
| random_state=42 korunur | §7.5 | Jüri tekrarı garantisi |
| Preprocessing fold içinde fit | §3.2 PSR | Sonuç geçersizleşir |
| Model binary'leri gitignore'da | Güvenlik | SHA256 doğrulaması |
| Danışman eser sahibi olamaz | §5 | Başvuru geçersizleşir |

---

## 9. Sık Yapılan Hatalar

```bash
# ✗ YANLIŞ — veri tüm sette ön işleniyor
scaler.fit(X_all)

# ✗ YANLIŞ — SMOTE bölmeden önce uygulanıyor
X_res, y_res = SMOTE().fit_resample(X_all, y_all)
X_train, X_test = train_test_split(X_res, y_res)

# ✗ YANLIŞ — test seti threshold ayarlamada kullanılıyor
best_thr = optimize_threshold(y_test, preds_test)

# ✗ YANLIŞ — klinik iddia
"Bu model %87 doğrulukla tanı koyabilmektedir."
```

---

## 10. Güvenlik Bildirimi

Güvenlik açığı tespit ederseniz `SECURITY.md` dosyasındaki prosedürü izleyin.  
**Güvenlik açıklarını herkese açık Issue olarak açmayın.**

Doğrudan güvenlik iletişimi: **SECURITY.md → GitHub Security Advisory**

---

*Resmi kaynak: https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/*

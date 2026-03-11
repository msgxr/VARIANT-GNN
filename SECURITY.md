# Guvenlik Politikasi

VARIANT-GNN projesi, genomik varyant siniflandirma alaninda hassas saglik verileriyle calistigi icin guvenlik en yuksek oncelikli konulardan biridir. Bu belge, projede uygulanan guvenlik onlemlerini, bilinen duzeltmeleri, tehdit modelini ve guvenlik acigi bildirme surecini detayli olarak aciklamaktadir.

---

## Desteklenen Surumler

| Surum | Destek Durumu | Aciklama |
|---|---|---|
| 3.x (guncel) | Evet — aktif destek | TEKNOFEST 2026 finale uyumlu, tum guvenlik yamalari uygulanmis |
| 2.x | Kismi — yalnizca kritik yamalar | Moduler mimari mevcut, ancak yeni ozellikler eklenmeyecek |
| 1.x (orijinal) | Hayir — bilinen guvenlik aciklari var | Guvenli olmayan pickle serializasyon, veri sizintisi sorunlari iceriyor |

> **Onemli:** v1.x kullanan kurulumlar derhal v3.x surumune gecmelidir. v1.x surumunde uzaktan kod calistirma (RCE) riski bulunmaktadir.

---

## v2.0 ve v3.0 ile Gelen Guvenlik Duzeltmeleri

### CVE Iliskili Kritik Duzeltmeler

| Sorun Tipi | v1 (Savunmasiz) | v2/v3 (Duzeltilmis) | Risk Seviyesi |
|---|---|---|---|
| Guvenli olmayan deserialization | `torch.load(path)` — pickle uzerinden keyfi kod calistirma | `torch.load(path, weights_only=True)` (`src/utils/serialization.py`) | **Kritik** |
| Guvenli olmayan deserialization | XGBoost modeli `.pkl` olarak kaydediliyor | XGBoost `.json` formatinda (pickle yok) | **Kritik** |
| Veri sizintisi (butunluk) | Preprocessor/SMOTE tum veri seti uzerinde fit ediliyor | Yalnizca CV fold icinde egitim verisi uzerinde fit ediliyor | **Yuksek** |
| Giris dogrulama eksikligi | CSV dosyalari dogrulanmadan isleniyor | Pydantic v2 sema dogrulamasi zorunlu | **Yuksek** |
| Bagimlilık guvenligi | Otomatik tarama yok | Bandit + pip-audit CI pipeline icerisinde | **Orta** |

### Bagimlilik Guvenlik Taramasi

Projede kullanilan ucuncu parti kutuphanelerin guvenlik aciklari duzenlı olarak taranmaktadir:

```bash
# Bagimlilik guvenlik taramasi (duzenli olarak calistirilmalidir)
pip-audit -r requirements.txt

# Alternatif: safety araci ile kontrol
safety check --file requirements.txt

# Statik kod analizi (CI pipeline icinde otomatik calisir)
bandit -r src/ -ll
```

- **Bandit** statik guvenlik taramasi CI pipeline icinde otomatik olarak calisir (`.github/workflows/ci.yml`)
- **pip-audit** ile bagimlilik zafiyetleri kontrol edilir
- Her PR oncesinde guvenlik taramasi zorunludur

---

## Model Serializasyon Guvenligi

Makine ogrenmesi modellerinin kaydedilmesi ve yuklenmesi sirasinda **pickle tabanli saldirilar** en buyuk tehditlerden biridir. Kotu niyetli bir `.pth` veya `.pkl` dosyasi, yuklenme sirasinda keyfi Python kodu calistirabilir. Bu riski onlemek icin tum model artefaktlari guvenli serializasyon yontemleri kullanmaktadir:

### PyTorch Modelleri (GNN ve DNN)

```python
# src/utils/serialization.py

# GUVENLI: weights_only=True parametresi keyfi kod calistirmayi onler
# Yalnizca tensor agirliklari yuklenir, pickle nesneleri reddedilir
torch.load(path, map_location=device, weights_only=True)

# Model kaydetme — yalnizca state_dict kaydedilir (tam model nesnesi degil)
torch.save(model.state_dict(), path)
```

**Neden `weights_only=True` kritik?**
- Standart `torch.load()` Python pickle protokolunu kullanir
- Pickle dosyalari icerisine keyfi Python kodu gomulabilir
- `weights_only=True` yalnizca tensor verilerini yukler, kod calistirmaz
- PyTorch 2.6+ surumlerinde varsayilan davranis bu yondedir

### XGBoost Modeli

```python
# GUVENLI: JSON formati — pickle yok, kod calistirma riski yok
model.save_model("models/xgb_model.json")   # kaydetme
model.load_model("models/xgb_model.json")   # yukleme
```

**Neden JSON?**
- JSON salt metin formatindadir ve kod icermez
- Insanlar tarafindan okunabilir ve dogrulanabilir
- Pickle formatinin aksine keyfi nesne olusturmaz

### On Islemci ve Kalibrator

```python
# Dahili kullanim icin kabul edilebilir — joblib serializasyonu
import joblib
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.load("models/preprocessor.pkl")
```

> **Uyari:** Joblib dosyalari pickle tabanlidir. Bu dosyalar yalnizca guvenilir kaynaklardan yuklenmelidir. Dis kaynaklardan alinan `.pkl` dosyalari **asla** dogrudan yuklenenmemelidir.

### Checksum Dogrulama

Modeller ag uzerinden aliniyorsa, SHA-256 checksum ile dogrulanmalidir:

```bash
# Model dosyasinin checksum'ini olusturma
sha256sum models/gnn_model.pth

# Dogrulama (beklenen hash ile karsilastirma)
echo "beklenen_hash  models/gnn_model.pth" | sha256sum --check
```

---

## Giris Dogrulama (Input Validation)

Sisteme giren tum veriler islenmeden once Pydantic v2 semasina karsi dogrulanir. Bu, bozuk, eksik veya kotu niyetli verilerin pipeline icine girmesini onler.

### Dogrulama Sureci

```python
from data_contracts.variant_schema import validate_dataset

# Veri setini dogrula — sema ihlallerinde ValueError firlatir
result = validate_dataset(df)

if not result.is_valid:
    for error in result.errors:
        print(f"Hata: {error}")
```

### Uygulanan Dogrulama Kontrolleri

| Kontrol | Aciklama | Basarisizlik Durumu |
|---|---|---|
| Sutun tipi kontrolu | Tum ozellik sutunlari sayisal (float/int) olmalidir | `ValueError` firlatilir |
| Etiket degerleri | Etiketler `{0, 1}` kumesi icinde olmalidir (Benign/Pathogenic) | `ValueError` firlatilir |
| Eksik deger kontrolu | %100 eksik degere sahip sutunlar kabul edilmez | `ValueError` firlatilir |
| Sutun sayisi uyumu | Beklenen ozellik sayisi ile girdi eslesmelidir (43 ozellik) | Uyari veya hata |
| Sayisal aralik | RobustScaler ile asiri degerler (outlier) normalize edilir | Otomatik duzeltme |
| NaN islem stratejisi | Eksik degerler medyan imputation ile doldurulur | Otomatik duzeltme |

### Ek Guvenlik Katmanlari

- **RobustScaler**: Numerik overflow'a neden olabilecek asiri degerleri yonetir (IQR tabanli olcekleme)
- **NaN Imputation**: Eksik degerler medyan ile doldurulur, boylelikle NaN propagasyonu onlenir
- **SMOTE Sinif Kontrolu**: Azinlik sinifinda yeterli ornek (>=6) yoksa SMOTE uygulanmaz

---

## Veri Sizintisi Onleme (Data Leakage Prevention)

Veri sizintisi, makine ogrenmesi modellerinin gercek performansindan daha iyi gorunmesine neden olan kritik bir metodolojik hatdir. VARIANT-GNN'de bu sorun sistematik olarak ele alinmistir:

### Uygulanan Onlemler

| Bilesen | Sizinti Riski | Cozum |
|---|---|---|
| Preprocessor (Imputer + Scaler) | Test verisi dagiliminin egitime sizmasi | Yalnizca egitim fold'unda `fit`, test fold'unda `transform` |
| SMOTE (Over-sampling) | Sentetik orneklerin test setine karismasi | SMOTE yalnizca egitim fold'u icinde uygulanir |
| AutoEncoder | Test verisinin latent uzay ogrenimine katilmasi | AutoEncoder yalnizca egitim verisinde egitilir |
| Ozellik Secimi | Test verisinin ozellik onem sirasini etkilemesi | SelectKBest yalnizca egitim verisinde fit edilir |
| Graf Yapilandirma | Test dugumlerinin egitim graf yapisini degistirmesi | k-NN grafi yalnizca egitim verisinden olusturulur |
| Kalibrasyon | Kalibrasyon setinin egitim/test ile cakismasi | Ayri %15 held-out kalibrasyon seti kullanilir |
| Hiperparametre Ayari | Parametre seciminin test performansina gore yapilmasi | Optuna yalnizca CV fold'lari icerisinde calisir |

---

## Guvenlik Acigi Bildirimi

Lutten guvenlik aciklarini **herkese acik GitHub issue'larinda bildirmeyin**. Aciklar kotu niyetli kisiler tarafindan istismar edilebilir.

### Bildirme Sureci

1. **Dogrudan iletisim**: Repository sahibine e-posta gonderin (profil iletisim bilgilerine bakiniz)
2. **Detayli aciklama**: Guvenlik aciginin tanimi, etkilenen bilesenler ve yeniden uretim adimlari
3. **Kanitlar**: Mumkunse PoC (Proof of Concept) kodu veya ekran goruntuleri ekleyin
4. **Etki analizi**: Acigin potansiyel etkisini ve kimleri etkileyebilecegini belirtin

### Yanit Sureci

| Asama | Sure | Eylem |
|---|---|---|
| Ilk yanit | En fazla 14 gun | Bildirim alindi onayi ve ilk degerlendirme |
| Onceliklendirme | 7 gun icinde | Kritik/Yuksek/Orta/Dusuk seviye belirleme |
| Yama gelistirme | Kritik: 7 gun, Yuksek: 14 gun, Orta: 30 gun | Guvenlik yamasinin hazirlanmasi |
| Yayinlama | Yama sonrasi 48 saat | Yeni surum yayinlanmasi ve duyuru |

> Tum gecerli guvenlik bildirimleri ciddiye alinacak ve en kisa surede yamalanacaktir.

---

## Tehdit Modeli

Asagidaki tablo, projede tanimlanan varlik-tehdit-onlem uctusunu detayli olarak gostermektedir:

| Varlik | Tehdit | Saldiri Senaryosu | Onlem | Risk Seviyesi |
|---|---|---|---|---|
| Model dosyalari (`.pth`) | Kotu niyetli pickle kodu gomma | Saldirgan, icine keyfi Python kodu gomulmus bir `.pth` dosyasi paylasiyor | `weights_only=True` ile yukleme; SHA-256 checksum dogrulama | **Kritik** |
| XGBoost modeli | Pickle RCE (uzaktan kod calistirma) | `.pkl` formatindaki model dosyasi uzerinden sistem ele gecirme | JSON formati kullanilir (pickle yok) | **Kritik** |
| Giris CSV dosyasi | Numerik overflow / NaN enjeksiyonu | Asiri buyuk float degerleri veya ozel karakterler | RobustScaler + NaN imputation + Pydantic dogrulama | **Yuksek** |
| Giris CSV dosyasi | Sutun enjeksiyonu | Beklenen format disinda sutunlar ekleme | Sema dogrulamasi ile yalnizca bilinen sutunlar kabul edilir | **Yuksek** |
| Yapilandirma YAML | Path traversal saldirisi | `../../etc/passwd` gibi yol girdileri | Yollar proje kok dizinine gore cozumlenir; kullanici tarafindan saglanan yollar uretimde kullanilmaz | **Orta** |
| Bagimlilik zinciri | Savunmasiz ucuncu parti paket | Bilinen CVE'ye sahip bir kutuphanenin kullanilmasi | Bandit statik analiz + pip-audit duzenli tarama + CI pipeline | **Orta** |
| Streamlit arayuzu | Buyuk dosya yuklemesi ile bellek tasirmasi (DoS) | Cok buyuk CSV dosyasi yukleyerek sunucuyu cokertme | Dosya boyutu sinirlamasi onerisi (yapilandirmada ayarlanabilir) | **Orta** |
| Egitim verileri | Veri zehirleme (data poisoning) | Kasitli olarak yanlis etiketlenmis varyantlarin veri setine eklenmesi | Veri sema dogrulama + istatistiksel anomali kontrolu | **Dusuk** |

---

## Guvenli Gelistirme Pratikleri

### Kod Guvenligi

- **Statik analiz**: Ruff + Bandit ile her commit'te otomatik kod taramasi
- **Tip guvenligi**: mypy ile statik tip kontrolu, `from __future__ import annotations` kullanımı
- **Bagimlilık yonetimi**: Surumler `pyproject.toml` icinde sabitlenmis (pinned versions)
- **Gizli bilgi yonetimi**: Kodda hardcoded sifre veya API anahtari bulunmaz; yapilandirma YAML uzerinden yapilir

### CI/CD Guvenlik Pipeline

```
Her PR / Push icin:
  1. ruff lint        → Kod kalitesi ve guvenlik kontrolleri
  2. bandit -r src/   → Statik guvenlik taramasi (B310, B614 vb.)
  3. pytest           → Birim ve entegrasyon testleri
  4. pip-audit        → Bagimlilik zafiyet taramasi (onerilen)
```

### Docker Guvenligi

```dockerfile
# Onerilen guvenlik iyilestirmeleri:
# 1. Non-root kullanici ile calistirma
RUN useradd -m appuser
USER appuser

# 2. Minimal base image kullanma
FROM python:3.10-slim

# 3. Gereksiz paketlerin kaldirilmasi
RUN apt-get purge -y --auto-remove && rm -rf /var/lib/apt/lists/*
```

---

## OWASP Top 10 Uyumlulugu

| OWASP Kategorisi | Proje Durumu | Detay |
|---|---|---|
| A01: Bozuk Erisim Kontrolu | Uygulanabilir degil | Tek kullanicili bilimsel uygulama; cok kullanicili erisim yok |
| A02: Kriptografik Hatalar | Uyumlu | Model dosyalari icin SHA-256 checksum onerisi |
| A03: Enjeksiyon | Uyumlu | Pydantic giris dogrulamasi; SQL kullanilmiyor |
| A04: Guvenli Olmayan Tasarim | Uyumlu | Veri sizintisi onleme, guvenli serializasyon |
| A05: Guvenlik Yapilandirma Hatasi | Kismi | Docker non-root user onerilen ama zorunlu degil |
| A06: Savunmasiz Bilesenler | Uyumlu | pip-audit + Bandit ile duzenli tarama |
| A07: Kimlik Dogrulama Hatalari | Uygulanabilir degil | Kimlik dogrulama sistemi yok |
| A08: Veri Butunlugu Hatalari | Uyumlu | `weights_only=True`, JSON serializasyon |
| A09: Guvenlik Kaydi Eksikligi | Kismi | Kapsamli logging mevcut; yapılandırılmış log formatı oneriliyor |
| A10: SSRF | Uygulanabilir degil | Dis URL cagrisi yalnizca ClinVar API (guvenilir kaynak) |

---

*Son guncelleme: Mart 2026 — VARIANT-GNN v3.0 (TEKNOFEST 2026 Finale Surumu)*

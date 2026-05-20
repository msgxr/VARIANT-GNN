# Güvenlik Politikası — VARIANT-GNN

**Proje:** VARIANT-GNN — Missense Varyant Patojenisite Tahmini  
**Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**Resmi Kaynak:** https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/  
**Şartname:** 2026 Sağlıkta Yapay Zeka Türkçe Şartname v4

> Bu belgede yer alan tüm kısıtlamalar ve yükümlülükler, yukarıdaki resmi
> TEKNOFEST şartnamesinden doğrudan alınmıştır. Çelişki durumunda şartname
> geçerlidir.

---

## 1. Güvenlik Açığı Bildirimi

**Güvenlik açıklarını herkese açık GitHub Issue olarak açmayın.**

| Kanal | Adres |
|:---|:---|
| E-posta (tercih edilen) | sinagun93@gmail.com |
| GitHub özel mesaj | @msgxr |

Bildiriminizde şunlar bulunmalıdır:

```
1. Etkilenen dosya ve satır numarası
   Örnek: src/utils/serialization.py:124

2. Açığın türü
   Örnek: pickle deserialization / path traversal / veri sızıntısı riski

3. Minimum yeniden üretim adımları

4. Etki analizi
   - Gizlilik (C): Yarışma verisi veya kişisel veri ifşası riski var mı?
   - Bütünlük (I): Model ağırlıkları veya sonuçlar değiştirilebilir mi?
   - Erişilebilirlik (A): Pipeline çökme veya servis reddi riski var mı?

5. PoC (opsiyonel — zararsız, veri sızdırmayan)
```

Bildirimlerinize **72 saat** içinde yanıt verilecektir.

---

## 2. Şartname Kaynaklı Veri Güvenliği Yükümlülükleri

### 2.1. Gizlilik Sözleşmesi (NDA) — Şartname §1 ve §4

Şartname §1 tanımlar bölümü:

> *"Gizlilik Sözleşmesi: Sağlıkta Yapay Zekâ Yarışması kapsamında
> T.C. Sağlık Bakanlığı/TÜSEB tarafından yarışmacıların modellerini
> eğitmek ve/veya test etmek amacıyla paylaşılan anonimleştirilmiş
> bilgi/belge/veriyi kullanabilmeleri için yarışmacıların imzaladıkları
> 'Kurumsal Gizlilik Taahhütnamesi'ni"*

Şartname §4:

> *"Yarışmacılar, yarışmada paydaşlar tarafından sağlanacak verilere
> ancak 'Gizlilik Sözleşmesini' imzalı olarak sunmaları halinde erişim
> sağlayabilecek ve yarışmaya katılabileceklerdir."*

**Bu yükümlülükler gereği repoya KESİNLİKLE eklenmemesi gerekenler:**

```
✗  Ham yarışma veri setleri (eğitim veya test)
✗  Sınıf etiketleri veya ground truth dosyaları
✗  Hasta verisi veya genetik kişisel veri
✗  Genomik adres içeren herhangi bir çıktı
✗  NDA kapsamında edinilen belge veya tablo
✗  Issue / PR / commit mesajında ham veri satırı
```

### 2.2. Genomik Adres ve Anonimleştirme — Şartname §3.2

Şartname §3.2:

> *"Yarışma veri setinde varyantların genomik adres (kromozom ve pozisyon)
> bilgileri, katılımcıların dış veri kaynaklarına başvurarak etiketi
> doğrudan bulmalarını engellemek amacıyla tamamen gizlenmiştir."*

> *"Bu kısıtlamanın amacı; yarışmacıların patojenite tahminlerini harici
> veri kaynaklarına başvurmaksızın, yalnızca yarışma komitesi tarafından
> sağlanan varyant profilleri üzerinden yapmalarını sağlamak ve kamuya
> açık veri tabanlarından elde edilebilecek hazır etiket bilgisinin
> kullanımını engellemektir."*

**Yasak işlemler:**

```
✗  Gizlenmiş Chr/Pos bilgisini tersine mühendislikle çıkarmak
✗  ClinVar/gnomAD API'si ile genomik adres üzerinden etiket aramak
✗  Test seti etiketlerini dış kaynaklardan dolaylı yoldan elde etmek
✗  Yarışma verisini harici veri tabanlarıyla birleştirerek
   etiket sızıntısı (leakage) oluşturmak
```

**Kodda uygulanan koruma** (`src/explainability/clinvar_api.py`):

```python
# ClinVar API eğitim ve tahmin sırasında kilitlenir
set_inference_mode(True)  # main.py başlangıcında çağrılır
```

### 2.3. Veri Sorumlusu ve KVKK — Şartname §10 ve §12.1

Şartname §10:

> *"Yarışma kapsamında sağlanan veri setlerine ilişkin veri sorumlusu
> TEKNOFEST organizasyonudur."*

> *"Yarışmacılar, ilgili verileri yalnızca organizasyon tarafından
> belirlenen kapsamda ve veri işleyen sıfatıyla kullanmakla yükümlüdür."*

> *"Yarışmacılara sunulan veri setleri, Kişisel Verilerin Korunması Kanunu
> (KVKK) ve uluslararası GDPR standartlarına uygun olarak, katılımcıların
> kimliğini ifşa edebilecek hiçbir Kişisel Tanımlayıcı Bilgi (PII)
> içermemektedir."*

Şartname §12.1:

> *"Yarışmaya başvuran kişilerin kişisel verileri (TCKN, e-posta, telefon
> no, IBAN, Nüfus Kayıt Örneği), ödül ödemesi ve tanıtım faaliyetleri
> kapsamında TÜSEB tarafından işlenebilir ve aktarılabilir."*

---

## 3. Klinik Kullanım Güvenliği — Şartname §10

Şartname §10 (tam alıntı):

> *"Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar,
> herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla
> kullanılamaz. Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."*

Bu model:

```
✗  Klinik tanı koymaz
✗  Tedavi önermez
✗  Hasta yönetimi için kullanılamaz
✗  Klinik karar destek sistemi değildir
✗  Tıbbi cihaz veya regülasyon onaylı ürün değildir
```

---

## 4. Kod ve Artefakt Güvenliği

### 4.1. Model Yükleme Güvenliği

**PyTorch modelleri** (`src/utils/serialization.py`):

```python
# CVE güvenli yükleme — weights_only=True arbitrary code execution'ı engeller
torch.load(path, map_location=device, weights_only=True)
# PyTorch < 2.0 fallback: sadece güvenilir kaynaklardan yüklendiğinde
```

**XGBoost** — JSON formatında saklanır, pickle kullanılmaz:

```
models/xgb_model.json   ← güvenli (JSON, keyfi kod yürütme riski yok)
models/lgbm_model.txt   ← güvenli (LightGBM text format)
```

**Pickle formatı riski** (joblib):

```
models/preprocessor.pkl  ← SADECE güvenilir kaynaktan yükleyin
models/calibrator.pkl    ← SADECE güvenilir kaynaktan yükleyin
models/ood_detector.pkl  ← SADECE güvenilir kaynaktan yükleyin
```

> ⚠️ Pickle/joblib dosyaları arbitrary code execution içerebilir.
> Yalnızca bu repo'nun ürettiği, SHA256 sağlaması doğrulanmış
> dosyaları yükleyin. Sağlamalar `models/metadata.json` içindedir.

### 4.2. Güvenlik Tarama Araçları

```bash
# Static güvenlik analizi (Bandit)
bandit -r src/ main.py app.py -ll

# Bağımlılık güvenlik taraması
pip-audit

# Gizli bilgi taraması (commit öncesi)
git secrets --scan
```

CI/CD: `.github/workflows/ci.yml` — Bandit her PR'da otomatik çalışır.

### 4.3. Repoya Eklenmemesi Gerekenler

```
# .gitignore kontrolü — bunların repoda OLMAMASI gerekir:
data/train_variants.csv        ← yarışma eğitim verisi (NDA)
data/test_variants*.csv        ← yarışma test verisi (NDA)
.env                           ← API anahtarları
*.key / *.pem                  ← sertifikalar
*_secret* / *_password*        ← kimlik bilgileri
logs/*.log                     ← etiket sızıntısı içerebilir
```

---

## 5. Jüri Tekrar Çalıştırma Güvenliği — Şartname §7.5

Şartname §7.5:

> *"Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını
> ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."*

Bu gereksinim için güvenlik garantileri:

| Gereksinim | Uygulama |
|:---|:---|
| Deterministik çıktı | `random_state=42` tüm RNG kaynaklarında |
| Tekrarlanabilir ortam | `requirements.txt` sabit versiyonlarla kilitli |
| Tek komut çalıştırma | `python main.py --mode train` |
| Model sağlama | SHA256 → `models/metadata.json` |
| Artifact versiyonlama | `models/manifest.json` |

---

## 6. Güvenli Geliştirme Kılavuzu

### Commit Öncesi Kontrol Listesi

```bash
# 1. Gizli bilgi taraması
grep -r "password\|secret\|api_key\|token\|IBAN\|TCKN" src/ --include="*.py"

# 2. Yarışma verisi kontrolü
ls data/*.csv 2>/dev/null && echo "UYARI: data/ içinde CSV var!"

# 3. Bandit taraması
bandit -r src/ -ll -q

# 4. Büyük dosya kontrolü (model ağırlıkları kazara ekleniyor olabilir)
git diff --cached --name-only | xargs -I{} du -h {} | sort -h
```

### Hassas Dosyalar İçin Pre-commit Hook

```bash
# .git/hooks/pre-commit içine ekleyin
#!/bin/bash
if git diff --cached --name-only | grep -E "^data/.*\.csv$|\.env$|\.pem$"; then
  echo "HATA: Hassas dosya commit'e eklenmeye çalışılıyor!"
  exit 1
fi
```

---

## 7. Desteklenen Sürümler

Bu proje TEKNOFEST 2026 yarışma takvimi kapsamında geliştirilmektedir:

| Aşama | Tarih | Güvenlik Garantisi |
|:---|:---:|:---|
| PSR | 25.03.2026 | ✅ Tamamlandı |
| Veri Paylaşımı | 05.05.2026 | ✅ NDA imzalandı |
| **PDR** | **29.06.2026** | 🔄 Aktif geliştirme |
| Final | Ağu–Eyl 2026 | — |

Sürüm geçmişi: `CHANGELOG.md`

---

## 8. Sorumluluk Reddi — Şartname §12

Şartname §12:

> *"TEKNOFEST ve Paydaş Kurumlar, yarışmacıların teslim etmiş olduğu
> herhangi bir üründen veya yarışmacıdan kaynaklanan herhangi bir
> yaralanma veya hasardan hiçbir şekilde sorumlu değildir."*

Bu projenin model çıktıları tıbbi karar için kullanılamaz (§10).
Klinik yanlış yorumlama, veri ihlali veya üçüncü taraf zararından
XYRA3 Takımı ve katkı sağlayıcıları sorumlu tutulamaz.

---

*Resmi kaynak: https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/*  
*İletişim: sinagun93@gmail.com*

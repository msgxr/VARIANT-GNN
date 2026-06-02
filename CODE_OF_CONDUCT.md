# Davranış Kuralları — VARIANT-GNN

**Proje:** VARIANT-GNN — Missense Varyant Patojenisite Tahmini  
**Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**Resmi Kaynak:** https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/  
**Şartname:** 2026 Sağlıkta Yapay Zeka Türkçe Şartname v4

> Bu belgede yer alan tüm yükümlülükler, yukarıdaki resmi TEKNOFEST şartnamesinden
> doğrudan alınmıştır. Çelişki durumunda şartname geçerlidir.

---

## 1. Şartname Kaynaklı Zorunlu Kurallar

Aşağıdaki kurallar TEKNOFEST şartnamesinden doğrudan türetilmiş olup
ihmal edilemez, tartışılamaz ve istisnasız uygulanır.

### 1.1. Bilimsel Dürüstlük — Şartname §7.3 ve §7.5

Şartname §7.3:

> *"Yarışma sıralamasını belirleyecek temel metrik, TP (Doğru Pozitif),
> FP (Yanlış Pozitif) ve FN (Yanlış Negatif) değerleri üzerinden
> hesaplanan F1 Skoru olacaktır."*

Şartname §7.5:

> *"Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını
> ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."*

Bu yükümlülükler gereği:

```
✗  İç validasyon skoru final yarışma skoru gibi sunulamaz
✗  Pilot veya sentetik veri sonuçları gerçek yarışma başarısı gibi gösterilemez
✗  Tekrar üretilemeyen metrik veya sonuç beyan edilemez
✗  "Kesin başarı", "%100 doğru", "mükemmel sonuç" gibi ifadeler kullanılamaz
✅  Beyan edilen sonuçlar: CV F1=0.8779±0.0062 | Test F1=0.833 | θ=0.8514 (global)
```

### 1.2. Klinik İddia Yasağı — Şartname §10

Şartname §10 (tam alıntı):

> *"Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar,
> herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla
> kullanılamaz. Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."*

Issue, PR, yorum veya dokümantasyonda kesinlikle kullanılamaz:

```
✗  "tanı koyar / koyabilir"
✗  "tedavi önerir"
✗  "doktor yerine geçer"
✗  "klinik olarak kanıtlanmıştır"
✗  "hastanede kullanılabilir"
```

### 1.3. Yarışma Verisi ve Gizlilik — Şartname §1 ve §4

Şartname §1 — Gizlilik Sözleşmesi tanımı:

> *"T.C. Sağlık Bakanlığı/TÜSEB tarafından yarışmacıların modellerini
> eğitmek ve/veya test etmek amacıyla paylaşılan anonimleştirilmiş
> bilgi/belge/veriyi kullanabilmeleri için yarışmacıların imzaladıkları
> 'Kurumsal Gizlilik Taahhütnamesi'ni"*

Şartname §4:

> *"Yarışmacılar, yarışmada paydaşlar tarafından sağlanacak verilere
> ancak 'Gizlilik Sözleşmesini' imzalı olarak sunmaları halinde erişim
> sağlayabilecek ve yarışmaya katılabileceklerdir."*

```
✗  Ham yarışma verisi issue/PR/yorum içinde paylaşılamaz
✗  Sınıf etiketleri veya test seti içeriği ifşa edilemez
✗  Genomik adres (Chr/Pos) içeren herhangi bir içerik paylaşılamaz
✗  NDA kapsamındaki belge veya tablo kamuya açılamaz
```

### 1.4. Genomik Adres Kısıtı — Şartname §3.2

Şartname §3.2:

> *"Bu kısıtlamanın amacı; yarışmacıların patojenite tahminlerini
> harici veri kaynaklarına başvurmaksızın, yalnızca yarışma komitesi
> tarafından sağlanan varyant profilleri üzerinden yapmalarını sağlamak
> ve kamuya açık veri tabanlarından elde edilebilecek hazır etiket
> bilgisinin kullanımını engellemektir."*

```
✗  ClinVar/gnomAD API'si ile etiket araması yapılamaz
✗  Genomik adres üzerinden tersine mühendislik önerilemez
✗  Dış veri kaynağıyla leakage oluşturmaya yönelik yöntem tartışılamaz
```

### 1.5. Veri Kullanım Kapsamı — Şartname §10

Şartname §10:

> *"Yarışmacılar, ilgili verileri yalnızca organizasyon tarafından
> belirlenen kapsamda ve veri işleyen sıfatıyla kullanmakla yükümlüdür.
> Verilerin kullanımı ticari bir amaç gütmemekte, tamamen bilimsel
> algoritma geliştirme amacı taşımaktadır."*

```
✗  Veri ticari amaçla kullanılamaz
✗  Organizasyonun belirlediği kapsam dışında kullanılamaz
```

---

## 2. Genel Davranış Kuralları

### 2.1. Beklenen Davranış

- **Saygılı ve profesyonel iletişim** — teknik tartışmalar kişi değil
  içerik üzerinden yürütülür
- **Bilimsel dürüstlük** — doğrulanmamış metrik veya sonuç kesin başarı
  gibi sunulmaz (§7.3/§7.5)
- **Teknik açıklık** — tekrar üretim için gerekli komut ve dosyalar
  gerçek ve çalışır olmalı; olmayan yol veya komut uydurulmamalı
- **Gizlilik duyarlılığı** — NDA, KVKK, GDPR kapsamında gizli olabilecek
  içerik paylaşılmaz (§1, §4)
- **Yapıcı eleştiri** — kod inceleme yorumları teknik kanıta dayanır

### 2.2. Kabul Edilemez Davranış

- Hakaret, ayrımcılık, tehdit veya taciz
- Kasıtlı yanlış yönlendirme (sahte sonuç, sahte benchmark)
- Yarışma verisi veya NDA kapsamlı içeriğin issue/PR/yorumda paylaşılması
- Klinik kullanım iddiası içeren ifadeler (§10)
- Güvenlik açıklarını herkese açık kanalda ifşa etmek (→ `SECURITY.md`)
- Genomik adres veya etiket içeren veri sızıntısı oluşturmak (§3.2)

---

## 3. Kapsam

Bu kurallar şu kanalların tamamında geçerlidir:

```
• GitHub Issues
• GitHub Pull Requests
• Kod inceleme (code review) yorumları
• Dokümantasyon katkıları
• Commit mesajları
• Wiki ve tartışma sayfaları
• Proje ile ilgili her türlü yazılı iletişim
```

---

## 4. Uygulama

İhlal durumlarında depo yöneticileri aşağıdaki adımları uygulayabilir:

| Adım | Durum |
|:---|:---|
| Uyarı | İlk ihlal, kasıtsız |
| İçerik kaldırma | Gizli veri veya kural ihlali içeren yorum/PR |
| Erişim kısıtlama | Tekrarlayan ihlal |
| Kalıcı engelleme | Ağır ihlal (NDA ihlali, güvenlik açığı ifşası) |

Şartname §12 uyarınca TEKNOFEST/Paydaş Kurumlar da yarışmacı davranışını
değerlendirme yetkisine sahiptir.

---

## 5. İletişim

| Konu | Kanal |
|:---|:---|
| Güvenlik açığı | `SECURITY.md` → sinagun93@gmail.com |
| Katkı süreci | `CONTRIBUTING.md` |
| Davranış ihlali | sinagun93@gmail.com |
| Yarışma soruları | TEKNOFEST resmi mail grubu (Şartname §11) |

---

*Resmi kaynak: https://teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/*

# PDR Şablon Özeti — Doğrulanmış Gereksinimler

Kaynak: 2026_PDR_Şablon_Universite_TR — resmî şablon dokümanı

## Görev Başlığı (şablonda belirtilen)

"Klinik Genomik Verilerde Yapay Zeka Tabanlı Varyant Patojenite Tahmini"

## Bölüm Yapısı ve Puan Dağılımı

| Bölüm | Puan |
|---|---|
| 1. Giriş | 10 |
| 2. Yöntem | 25 |
| 3. Bulgular | **30** (en büyük bölüm) |
| 4. Sonuç | 25 |
| 5. Kaynakça ve Rapor Düzeni | 10 |
| **Toplam** | **100** |

---

## 1. Giriş (10 puan) — Beklentiler

- Missense varyant patojenite tahmini problemi açık ve anlaşılır tanımlanmalı
- Klinik ve genomik bağlamdaki önemi vurgulanmalı
- Sınıf dengesizliği problemi ve model başarımı üzerindeki etkileri açıklanmalı
- Konuya ilişkin **5–10 güncel uluslararası çalışma** incelenmeli
  - Kullanılan yöntemler
  - Veri kaynakları (ClinVar, gnomAD)
  - Raporlanan başarım ölçütleri

---

## 2. Yöntem (25 puan) — Beklentiler

**Veri mühendisliği:**
- Asimetrik ve şifreli genomik veri kümesinin yapısı
- Eksik değerlerin tamamlanma yöntemi
- Aykırı değerlerin yönetimi
- Dış kaynaklardan veri eklendi mi?
- Yeni özellik üretimi (varsa)

**Model geliştirme:**
- Denenen algoritmalar ve seçilme gerekçeleri
- Hiperparametre belirleme yöntemi ve arama uzayı
- Genelleme gücüne etkisi
- Çapraz doğrulama yaklaşımları
- Overfitting önleme önlemleri
- **Açıklanabilirlik yöntemleri**
- **Karar eşiği belirleme süreci**

---

## 3. Bulgular (30 puan) — ZORUNLU METRİKLER

### Resmî şablondan birebir alıntı:
> "Başarım ölçütü olarak en azından **F1 skoru, Matthews korelasyon katsayısı ve kesinlik-duyarlılık eğrisi altında kalan alan** ölçütleri raporlanmalıdır."

**PDR'de zorunlu metrikler (her panel için ayrı ayrı):**
- F1 Skoru
- Matthews Korelasyon Katsayısı (MCC)
- Kesinlik-Duyarlılık Eğrisi Altında Kalan Alan (PR-AUC)

**Ek zorunlu unsurlar:**
- Karmaşıklık matrisi (Confusion Matrix) ve karşılaştırma grafikleri
- Farklı karar eşiklerinin test sonuçları + en doğru eşik değeri

### Panel Adları (PDR şablonunda belirtilen):

| PDR Şablon Adı | Karşılık |
|---|---|
| **MASTER** | Genel Veri Seti |
| **KANSER** | Herediter Kanser Paneli |
| **CFTR** | Kistik Fibrozis Paneli |
| **PAH** | Fenilketonüri Paneli |

Panel bazlı raporlama zorunlu: "Model başarımları yalnızca genel veri kümesi üzerinde ve ayrıca farklı alt gruplar (MASTER, KANSER, CFTR ve PAH) bazında ayrı ayrı raporlanmalıdır."

---

## 4. Sonuç (25 puan) — Beklentiler

- Bulgular yorumlanmalı, çalışmanın genel katkısı ortaya konulmalı
- Modelin başarılı olduğu ve yetersiz kaldığı durumlar analiz edilmeli
- **Yanlış pozitif ve yanlış negatif sonuçlar ayrıntılı incelenmeli**
- Modelin **hangi özellik gruplarında zorlandığı** belirlenmeli
- Hataların **klinik veya biyolojik anlamı** yorumlanmalı
- Çalışmanın literatürdeki yeri
- "Yarışmanın son basamağında karşılaşılabilecek zorluklar tartışılmalı"

---

## 5. Kaynakça ve Rapor Düzeni (10 puan)

- Referanslar: **IEEE formatı**, doğru ve eksiksiz
- Görsel iletişim: grafik ve tablolar yeterli ve anlaşılır
- Akademik yazım ve Türkçe dil kuralları

---

## Biçim Kuralları (KESİN — Aşılırsa değerlendirmeye alınmaz)

| Parametre | Değer |
|---|---|
| Yazı tipi | **Aptos** |
| Punto | 12 |
| Başlık yazı tipi | Aptos |
| Başlık punto | 14 |
| Satır aralığı | 1.15 |
| Hizalama | İki tarafa yaslı |
| Kenar boşluğu üst | 2.8 cm |
| Kenar boşluğu alt-sağ-sol | 2.5 cm |
| **Sayfa sınırı** | **10 sayfa** (kapak + içindekiler hariç) |

**10 sayfayı geçen raporlar değerlendirmeye alınmayacaktır.**

---

## Raporun Zorunlu Bölümleri

- Kapak sayfası
- İçindekiler sayfası
- Sayfa numaralandırması (birbirini takip edecek şekilde)
- Giriş → Yöntem → Bulgular → Sonuç → Kaynakça

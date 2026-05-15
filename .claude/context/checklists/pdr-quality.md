# PDR Odaklı Kalite Kuralları

Son güncelleme: 2026-05-15 (PDR şablon okundu — metrikler ve panel adları kesinleşti)

## 1. Şablona Bağlılık

PDR metni yalnızca resmî "Proje Detay Raporu Üniversite ve Üzeri Seviyesi Şablonu"na göre yapılandırılır. Şablonda olmayan bölümler eklenmez. Şablondaki zorunlu bölümler atlanamaz.

## 2. Bölüm Dengesi ve Puan Ağırlıkları

| Bölüm | Puan | Öncelik |
|---|---|---|
| Giriş | 10 | Orta |
| Yöntem | 25 | Yüksek |
| **Bulgular** | **30** | **En Yüksek** |
| Sonuç | 25 | Yüksek |
| Kaynakça ve Düzen | 10 | Orta |

## 3. Yöntem Bölümü Zorunlu Unsurları

- [ ] Asimetrik ve şifreli genomik veri kümesinin yapısı
- [ ] Eksik değer tamamlama (hangi split'te, hangi yöntemle)
- [ ] Aykırı değer yönetimi
- [ ] Dış kaynak veri eklendi mi?
- [ ] Yeni özellik üretimi (varsa)
- [ ] Algoritmalar ve seçilme gerekçeleri (deneysel kanıtla)
- [ ] Hiperparametre optimizasyonu (yöntem, arama uzayı, genelleme etkisi)
- [ ] Çapraz doğrulama yaklaşımları
- [ ] Overfitting önleme önlemleri
- [ ] **Açıklanabilirlik yöntemleri** (PSR'de 3.33/5 — PDR'de güçlendirilmeli)
- [ ] **Karar eşiği belirleme süreci**

## 4. Bulgular Bölümü — PDR ŞABLONUNDAN ZORUNLU METRİKLER

### Resmî şablon metni (birebir):
> "Başarım ölçütü olarak en azından **F1 skoru, Matthews korelasyon katsayısı ve kesinlik-duyarlılık eğrisi altında kalan alan** ölçütleri raporlanmalıdır."

**Zorunlu metrikler (her panel için ayrı):**
- [ ] **F1 Skoru** — zorunlu (şartname final metriği)
- [ ] **Matthews Korelasyon Katsayısı (MCC)** — zorunlu (PDR şablonu)
- [ ] **Kesinlik-Duyarlılık Eğrisi Altında Kalan Alan (PR-AUC)** — zorunlu (PDR şablonu)
- [ ] **Confusion Matrix** ve karşılaştırma grafikleri — zorunlu (PDR şablonu)
- [ ] Farklı karar eşiklerinin test sonuçları + en doğru eşik — zorunlu (PDR şablonu)

**PSR'de kullanılan — PDR'de de yer almalı:**
- [ ] ROC-AUC (PSR Tablo 3'te birincil metrik)
- [ ] Brier Score (kalibrasyon kalitesi)

### Panel Adları — PDR Şablonunda Belirtilen:

| PDR Adı | Açıklama |
|---|---|
| **MASTER** | Genel Veri Seti |
| **KANSER** | Herediter Kanser Paneli |
| **CFTR** | Kistik Fibrozis Paneli |
| **PAH** | Fenilketonüri Paneli |

### Görseller:
- [ ] Kesinlik-Duyarlılık (PR) Eğrisi — zorunlu (PR-AUC için)
- [ ] Confusion Matrix görseli (her panel)
- [ ] Karşılaştırma grafikleri
- [ ] Baseline karşılaştırma tablosu
- [ ] GNN vs. diğer yöntem karşılaştırması

**PR-AUC PSR'de hesaplanmamıştı. PDR için ayrıca üretilmesi gerekiyor.**

## 5. Sonuç Bölümü Zorunlu Unsurları

- [ ] Bulgular yorumu ve çalışmanın katkısı
- [ ] **Yanlış pozitif ve yanlış negatif ayrıntılı inceleme**
- [ ] Modelin hangi özellik gruplarında zorlandığı
- [ ] Hataların klinik veya biyolojik anlamı
- [ ] Çalışmanın literatürdeki yeri
- [ ] Yarışmanın son basamağında karşılaşılabilecek zorluklar

## 6. Biçim Kuralları (KESİN — Aşılırsa değerlendirmeye alınmaz)

- Yazı tipi: **Aptos** (Arial veya Times değil!)
- Punto: 12 (başlıklar: 14)
- Satır aralığı: 1.15
- Hizalama: İki tarafa yaslı
- Kenar boşluğu: Üst 2.8 cm, alt-sağ-sol 2.5 cm
- **Sayfa sınırı: 10 sayfa** (kapak + içindekiler hariç)
- **10 sayfayı geçen raporlar değerlendirmeye alınmaz!**
- IEEE format referanslar

## 7. Zorunlu Rapor Bölümleri

- Kapak sayfası
- İçindekiler
- Sayfa numaralandırması
- Giriş → Yöntem → Bulgular → Sonuç → Kaynakça

## 8. Etik Sınır Cümlesi (şartnameden alınan resmî metin)

"Bu çalışma TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında gerçekleştirilmiş olup geliştirilen model ve çıktılar yalnızca araştırma ve eğitim amaçlıdır; klinik tanı veya tıbbi karar desteği amacıyla kullanılamaz."

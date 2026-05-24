# Kesin Yasaklar ve Karar Verme Prensipleri — VARIANT-GNN

---

## BÖLÜM 1: KESIN YASAKLAR

### Yarışma Şartnamesi Açısından

- Resmî dokümanda yer almayan yarışma şartı üretme.
- Değerlendirme metriği olarak F1 dışında başka bir metriğin final için belirleyici olduğunu varsayma.
- PDR değerlendirme kriterlerini resmî şablon dışından tahmin etme.
- Kolon anlamlarını kesin biliyormuş gibi davranma (kolonlar anonimdir).
- Test verisi etiketli varmış gibi yorum yapma.

### Etik Açıdan

- Modelin klinik tanı, tıbbi karar destek veya hastaya yönelik uygulama için uygun olduğunu yazan veya ima eden cümle kurma.
- Sağlık beyanı niteliği taşıyan herhangi bir ifade kullanma.
- Yarışma kapsamında geliştirilen modelin gerçek klinik durumu güvenilir biçimde tahmin edebildiğini iddia etme.

### Teknik Açıdan

- Veri sızıntısı riskini ihmal etme (normalizasyon, imputation, encoding hangi split'te yapıldı?).
- Anonim kolon analizini "kesin bilgi" olarak sunma.
- Kod çalıştırmadan kesin metrik sonucu yazma.
- Tek bir metrikle (sadece F1) sonucu yeterli gösterme; destekleyici metrikler olmadan PDR'yi tamamlama.
- Overfitting bulgusunu görmezden gelme.

### Rapor Yazımı Açısından

- Şablonun sayfa ve biçim sınırlarını aşma.
- "Benzersiz", "devrimsel", "en iyi", "mükemmel" gibi savunulamaz süperlatifleri yazma.
- Referanssız, doğrulanamaz akademik iddia kurma.
- Basın bülteni dili veya pazarlama metni kullanma.
- Bulgular bölümünde görsel veya tablo olmadan salt metin bırakma.

### Claude Sistemi Açısından

- Resmî kaynaktan doğrulanmamış yarışma bilgisini kesinmiş gibi verme.
- "Varsayım" etiketlemeden spekülasyon yapma.
- Projenin teknik kararlarını sorgulamadan onaylama.
- Jüri sorularına "kolay" muamelesi yapma.

---

## BÖLÜM 2: KARAR VERME PRENSİPLERİ

### 1. Resmî Doküman Önceliği

Her teknik veya stratejik karar, önce resmî TEKNOFEST şartnamesiyle uyumlu mu diye kontrol edilir. Çelişki varsa şartname kazanır.

### 2. Kanıt Temelli Teknik Karar

"GNN daha iyidir" → Kanıt gerekir: baseline karşılaştırma tablosu, CV metrikleri.
"Ensemble kullanmak mantıklı" → Gerekçe gerekir: ablation study veya literatür referansı.
Spekülasyon ile deneysel bulgu arasındaki fark her zaman açık tutulur.

### 3. F1 Hizalı Ama PDR Eksiksiz

Final metriği F1'dir. Ancak PDR'de yalnızca F1 yeterli değildir. Precision, recall, confusion matrix, ROC/PR eğrisi de yer almalıdır.

### 4. Panel Bazlı Düşünme

Hiçbir karar dört paneli tek blok olarak ele alamaz. CFTR az veri riski, Herediter Kanser spesifik örüntü, Genel Panel çeşitlilik — ayrı teknik değerlendirme gerektirir.

### 5. Yeniden Üretilebilirlik Kırmızı Çizgi

Her metodolojik karar "bu adım yeniden üretilebilir mi?" sorusuyla test edilir. Sabit seed, kaydedilmiş model dosyaları, tek komutla inferans.

### 6. Risk Önceliklendirme

1. Finale geçişi engelleyen kritik riskler (PDR format, etik ihlal)
2. Final F1'i düşüren teknik riskler (veri sızıntısı, overfitting)
3. Savunmayı zorlaştıran anlatı riskleri (gerekçesiz mimari seçimi)

### 7. Jüri Denetimiyle Uyum

Her teknik karar için: "Jüri bunu sorsa nasıl cevaplanır?" Cevap verilemeyen kararlar ya revize edilir ya da PDR'de sınırlılık olarak belirtilir.

### 8. Takım İçi Karar Akışı

| Karar Türü | Sorumlu |
|---|---|
| Resmî yarışma ve iletişim kararları | Şeyma Nur Çebi (Kaptan) |
| Model geliştirme ve deneysel kararlar | Şeyma Nur Çebi koordinasyonunda |
| Biyolojik / veri kalitesi kararları | Şahin Kara |
| Yazılım mimarisi ve kod kararları | Burak Küçükcengiz |
| Sistem kurgusu, PDR uyumu, jüri anlatısı | Muhammed Sina Gün |
| Akademik rehberlik | Pınar Karadayı Ataş |

Claude, analiz zemini hazırlar ve denetler. Nihai kararlar takıma aittir.

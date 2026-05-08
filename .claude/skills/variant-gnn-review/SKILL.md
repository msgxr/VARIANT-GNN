---
name: teknofest-2026-health-ai-compliance
description: Use when reviewing, improving, documenting, testing, refactoring, or preparing the VARIANT-GNN repository for full compliance with the TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması şartnamesi, especially the Üniversite ve Üzeri genetic variant pathogenicity prediction task.
---

# TEKNOFEST 2026 Sağlıkta Yapay Zeka Şartname Uyum Skill'i

Bu skill aktif olduğunda VARIANT-GNN reposunu, TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması şartnamesine göre katı, teknik, akademik ve dosya temelli şekilde denetle.

Amaç, projenin şartnameye mümkün olan en yüksek düzeyde uyumlu hale getirilmesidir. Genel tavsiye verme. Önce ilgili dosyaları oku, sonra kanıta dayalı değerlendirme yap. Görmediğin şeyi varmış gibi kabul etme. Eksik bilgi varsa açıkça “kanıt bulunamadı” de.

## 1. Resmi Yarışma Bağlamı

Projeyi aşağıdaki resmi bağlama göre değerlendir:

- Yarışma: TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması
- Kategori: Üniversite ve Üzeri Seviyesi
- Alan: Genetik
- Görev: Klinik durumu bilinmeyen varyantların “Patojenik” veya “Benign” olarak sınıflandırılması
- Varyant tipi: Missense varyantlar
- Referans sınıflandırma: ACMG rehberleri ve kriterleri
- Ground truth: Kaynak veri tabanlarındaki ACMG uyumlu mevcut etiketler
- Temel başarı metriği: F1 skoru
- Final puan etkisi: Fiziki final/görev performansı %90, final sunumu %10
- Kod beklentisi: Çalışabilir, yeniden üretilebilir ve açık şekilde dokümante edilmiş olmalıdır
- Jüri yetkisi: Finale kalan takımlardan kodlarını tekrar çalıştırmaları ve beyan edilen sonuçları üretmeleri istenebilir

Bu bağlam dışına çıkma. Proje, doğrudan tıbbi tanı aracı gibi sunulmamalıdır.

## 2. Şartnameye Göre Veri Seti Gereksinimleri

Aşağıdaki veri seti yapısına göre projeyi denetle:

### Eğitim Setleri

1. Genel Veri Seti:
   - 1500 Patojenik varyant
   - 1500 Benign varyant

2. Kalıtsal / Herediter Kanser Paneli:
   - 200 Patojenik varyant
   - 200 Benign varyant

3. Fenilketonüri / PAH Gen Paneli:
   - 200 Patojenik varyant
   - 200 Benign varyant

4. Kistik Fibrozis / CFTR Gen Paneli:
   - 70 Patojenik varyant
   - 70 Benign varyant

### Test Setleri

1. Genel Veri Seti:
   - 1000 Patojenik varyant
   - 1000 Benign varyant

2. Kalıtsal / Herediter Kanser Paneli:
   - 100 Patojenik varyant
   - 100 Benign varyant

3. Fenilketonüri / PAH Gen Paneli:
   - 100 Patojenik varyant
   - 100 Benign varyant

4. Kistik Fibrozis / CFTR Gen Paneli:
   - 30 Patojenik varyant
   - 30 Benign varyant

Test setinin yarışma sırasında etiketsiz verileceğini dikkate al. Reponun test etiketlerine erişiyormuş gibi davranan, etiketi tahmin yerine dolaylı bulan veya dış kaynaklardan doğrudan etiket çıkaran her yaklaşımını kritik uyumsuzluk olarak işaretle.

## 3. Şartnameye Göre Sınıf Tanımları

Şu sınıf mantığını kontrol et:

### Patojenik Sınıf

- ClinVar ve ClinGen kaynaklı olmalıdır.
- Expert Panel ve Practice Guideline inceleme statüsüne sahip güvenilir varyantlar dikkate alınmalıdır.
- 3 ve 4 yıldız güvenilirlik seviyesi bağlamı açıklanmalıdır.
- “Pathogenic” ve “Likely Pathogenic” tek Patojenik sınıf altında birleştirilmelidir.

### Benign Sınıf

- ClinVar verilerine ek olarak gnomAD sağlıklı popülasyon varyantları bağlamı açıklanmalıdır.
- “Benign” ve “Likely Benign” tek Benign sınıf altında birleştirilmelidir.
- Sınıf dengesizliğini azaltma amacı belirtilmelidir.

Eğer README, DATA_CARD, MODEL_CARD veya raporlarda bu sınıf tanımları eksikse kritik dokümantasyon eksiği olarak yaz.

## 4. Genomik Adres ve Veri Sızıntısı Kontrolü

Şartnameye göre genomik adres, kromozom ve pozisyon bilgileri; yarışmacıların dış veri kaynaklarından etiketi doğrudan bulmasını engellemek için gizlenmiştir.

Bu nedenle şu kontrolleri yap:

- Kod genomik adres, kromozom veya pozisyon bilgisini kullanıyor mu?
- Harici veri tabanlarından etiket çekiliyor mu?
- ClinVar/ClinGen/gnomAD üzerinden doğrudan sınıf etiketi bulunabilecek bir pipeline var mı?
- Veri zenginleştirme işlemi etiketi sızdırıyor olabilir mi?
- Train/test ayrımı yapılmadan preprocessing, scaler, imputer, feature selection veya SMOTE uygulanıyor mu?
- SelectKBest, scaler, imputer, AutoEncoder veya graph construction tüm veri üzerinde fit ediliyor mu?
- Panel bilgisi model için dolaylı etiket sinyali oluşturuyor mu?
- Aynı veya çok benzer varyantlar train ve validation/test bölümlerine düşüyor mu?

Veri sızıntısı şüphesi varsa bunu “Kritik Sorun” olarak değerlendir.

## 5. Şartnameye Göre Özellik Grupları

Repo, veri açıklaması ve model mimarisi aşağıdaki özellik gruplarını açıklamalıdır:

1. Sekans ve değişim bilgisi
   - Referans nükleotid
   - Alternatif nükleotid
   - Kodon değişimi
   - Amino asit dönüşümü

2. Yerel sekans ve çevresel bağlam
   - Varyant öncesi ve sonrası 5 nükleotid
   - İlgili amino asit öncesi ve sonrası 5 amino asit

3. Biyokimyasal ve yapısal etkiler
   - Hidrofobiklik
   - Polarite
   - Moleküler ağırlık değişimi
   - Proteinin 3B yapısına olası etki

4. Evrimsel korunmuşluk
   - Filogenetik çeşitlilik
   - İnsan popülasyonları arası korunmuşluk
   - Korunmuşluk skorları

5. Popülasyon verileri
   - Minör allel frekansı
   - Popülasyon görülme sıklıkları

6. In silico risk skorları
   - Farklı algoritmalar tarafından hesaplanmış risk skorları

Her özellik grubu için şunları denetle:

- Veri kartında açıklanmış mı?
- Model kartında kullanım amacı belirtilmiş mi?
- README’de anlaşılır anlatılmış mı?
- Kodda preprocessing karşılığı var mı?
- Eksik değer stratejisi var mı?
- Kategorik/sayısal dönüşüm net mi?
- Yarışma test senaryosunda üretilebilir mi?
- Etiket sızıntısı riski taşıyor mu?

## 6. İncelenecek Dosya ve Klasörler

Önce şu dosya ve klasörleri incele:

- README.md
- DATA_CARD.md
- MODEL_CARD.md
- PROJECT_STATUS.md
- CHANGELOG.md
- CITATION.cff
- LICENSE
- SECURITY.md
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- pyproject.toml
- requirements.txt
- requirements-dev.txt
- requirements-ci.txt
- requirements-gpu.txt
- environment.yml
- environment-ci.yml
- environment-gpu-cu118.yml
- environment-gpu-cu121.yml
- Dockerfile
- Dockerfile.api
- docker-compose.yml
- Makefile
- app.py
- main.py
- trainer.py
- pipeline.py
- configs/
- src/
- tests/
- docs/
- reports/
- notebooks/
- models/
- data/
- data_contracts/
- .github/
- ci_pipeline_new.yml

Dosya yoksa “eksik” de. Dosya varsa içeriğine göre değerlendir. Sadece dosya adı var diye yeterli kabul etme.

## 7. README Uyum Kontrolü

README şu bölümleri açık ve profesyonel şekilde içermelidir:

1. Proje başlığı
2. TEKNOFEST 2026 yarışma bağlamı
3. Üniversite ve üzeri genetik varyant görevi
4. Patojenik / Benign sınıflandırma tanımı
5. Missense varyant odağı
6. Veri seti yapısı
7. Eğitim ve test seti ayrımı
8. Genomik adres gizleme kuralı
9. Kullanılan özellik grupları
10. Model mimarisi
11. Kurulum
12. Eğitim komutu
13. Validasyon komutu
14. Inference/tahmin komutu
15. Docker ile çalıştırma
16. Testleri çalıştırma
17. F1 skoru hesaplama
18. Panel bazlı değerlendirme
19. Reproducibility açıklaması
20. Sınırlılıklar
21. Klinik kullanım uyarısı
22. Etik ve gizlilik beyanı
23. Lisans
24. Atıf bilgisi
25. Takım/proje bilgisi

Eksik bölümleri tek tek yaz ve her biri için doğrudan eklenebilir düzeltme önerisi ver.

## 8. MODEL_CARD Uyum Kontrolü

MODEL_CARD aşağıdaki bilgileri içermelidir:

- Modelin amacı
- Kullanım kapsamı
- Kullanılmaması gereken durumlar
- Girdi formatı
- Çıktı formatı
- Patojenik / Benign sınıf yorumu
- Model mimarisi
- Eğitim verisi özeti
- Validasyon yöntemi
- F1, precision, recall, confusion matrix
- Panel bazlı performans
- Kalibrasyon açıklaması
- Belirsizlik tahmini açıklaması
- Açıklanabilirlik yöntemi
- Veri sızıntısı önlemleri
- Klinik sınırlılıklar
- İnsan uzman denetimi gerekliliği
- Etik ve gizlilik notu
- Bilinen hatalar
- Sürüm bilgisi

Eksikse puan kır.

## 9. DATA_CARD Uyum Kontrolü

DATA_CARD aşağıdaki bilgileri içermelidir:

- Veri kaynakları
- ClinVar açıklaması
- ClinGen açıklaması
- gnomAD açıklaması
- ACMG referans bilgisi
- Patojenik sınıf tanımı
- Benign sınıf tanımı
- Likely sınıfların nasıl birleştirildiği
- Eğitim seti sayıları
- Test seti sayıları
- Panel dağılımları
- Genomik adreslerin neden gizlendiği
- Özellik kolonlarının açıklaması
- Kolon isimlerinin yarışmada verilmeyeceği bilgisi
- Eksik değerler
- Sınıf dengesi
- Bias riskleri
- Veri sızıntısı riskleri
- KVKK/GDPR uyumu
- İkincil veri kullanımı
- Araştırma ve eğitim amacı
- Klinik kullanım dışı sınır

Eksikse açıkça belirt.

## 10. Model ve Deney Tasarımı Kontrolü

Aşağıdaki bileşenleri dosya temelli incele:

- GNN
- GATv2
- XGBoost
- LightGBM
- DNN
- Stacking ensemble
- Logistic regression meta learner
- AutoEncoder
- SelectKBest
- RobustScaler
- SMOTE
- k-NN graph construction
- Isotonic calibration
- MC Dropout
- SHAP
- GNNExplainer

Her bileşen için şunu yaz:

- Kodda var mı?
- README’de anlatılmış mı?
- MODEL_CARD’da açıklanmış mı?
- Deneysel sonucu var mı?
- Şartname görevine gerçek katkısı var mı?
- Gereksiz karmaşıklık oluşturuyor mu?
- Veri sızıntısı riski var mı?

## 11. F1 Skoru ve Değerlendirme Kontrolü

Şartnameye göre finalde temel metrik F1 skorudur.

Kontrol et:

- F1 doğru hesaplanıyor mu?
- Binary classification için TP, FP, FN mantığı doğru mu?
- Accuracy gereğinden fazla öne çıkarılıyor mu?
- Precision ve recall ayrı veriliyor mu?
- Confusion matrix var mı?
- Genel veri seti ve üç panel ayrı değerlendiriliyor mu?
- Validation sonucu final başarısı gibi sunuluyor mu?
- Test etiketi bilinmeyen final senaryosu dikkate alınmış mı?
- Threshold seçimi açıklanmış mı?
- Calibration sonrası F1 değişimi gösterilmiş mi?

F1 odaklı olmayan değerlendirme yapısını uyumsuzluk olarak işaretle.

## 12. Tekrarlanabilirlik Kontrolü

Bir jüri üyesi projeyi sıfırdan çalıştırmak istediğinde şu adımlar net olmalıdır:

1. Repository clone
2. Ortam kurulumu
3. Veri yerleşimi
4. Config seçimi
5. Eğitim
6. Validasyon
7. Tahmin üretimi
8. F1 hesaplama
9. Test dosyası oluşturma
10. Docker ile çalıştırma
11. API/arayüz başlatma
12. Log ve çıktıların doğrulanması

Şunları kontrol et:

- Python sürümü açık mı?
- Bağımlılıklar çakışıyor mu?
- Conda ve pip yolu net mi?
- GPU/CPU ayrımı açık mı?
- Seed sabit mi?
- Config merkezi mi?
- Veri yolları hard-coded mı?
- Windows/Linux uyumu var mı?
- Örnek veri veya synthetic demo var mı?
- Model ağırlıkları açıklanmış mı?
- Kod tek komutla smoke test verebiliyor mu?

Eksikse teslim riski olarak yaz.

## 13. Test ve CI Kontrolü

Şu testlerin varlığını kontrol et:

- Unit test
- Preprocessing test
- Data schema test
- Leakage prevention test
- Train/validation split test
- F1 metric test
- Inference contract test
- Model smoke test
- API test
- Docker build test
- Config loading test
- Panel bazlı değerlendirme testi
- Random seed determinism test

Eksik testler için önerilen dosya adlarını yaz:

- tests/test_data_schema.py
- tests/test_preprocessing_no_leakage.py
- tests/test_metrics_f1.py
- tests/test_inference_contract.py
- tests/test_panel_evaluation.py
- tests/test_reproducibility_seed.py
- tests/test_docker_smoke.py
- tests/test_config_loading.py

## 14. Klinik Güvenlik, Etik ve KVKK/GDPR Kontrolü

Şartnameye göre yarışmada kullanılan veriler kamuya açık, anonimleştirilmiş ve ikincil veri kullanımı kapsamındadır. Yarışma çıktıları klinik tanı, tedavi veya tıbbi karar destek amacıyla kullanılamaz.

Bu nedenle kontrol et:

- Repo açıkça “tanı aracı değildir” diyor mu?
- “Yalnızca araştırma ve eğitim amaçlıdır” ifadesi var mı?
- İnsan uzman denetimi zorunluluğu belirtilmiş mi?
- KVKK/GDPR uyumu kanıtlı ve abartısız mı?
- PII içermediği doğru bağlamda açıklanmış mı?
- Genomik adreslerin kaldırılması re-identification riskini azaltma amacıyla açıklanmış mı?
- Klinik kullanım için regülasyon ve harici validasyon gerektiği belirtilmiş mi?
- Hasta veya hekime doğrudan karar öneren riskli ifadeler var mı?

Riskli ifadeleri güvenli ifadelerle değiştir.

Örnek güvenli ifade:

“Bu sistem klinik tanı, tedavi veya bağımsız tıbbi karar verme amacıyla kullanılamaz. Model çıktıları yalnızca araştırma, eğitim ve yarışma değerlendirmesi kapsamında yorumlanmalıdır. Klinik kullanım için bağımsız validasyon, regülasyon uygunluğu ve uzman hekim değerlendirmesi gereklidir.”

## 15. Rapor Uyum Kontrolü

PSR ve PDR için şu şartname beklentilerini kontrol et:

### Proje Sunuş Raporu

- Genel problem tanımı
- Literatür taraması
- Önerilen çözüm yöntemi
- Veri yaklaşımı
- Ön model/plan
- Şartname görevine açık bağlantı

### Proje Detay Raporu

- Geliştirilen model mimarisi
- Eğitim süreçleri
- İç test / validasyon sonuçları
- Değerlendirme metodolojisi
- Kodun çalıştırılabilirliği
- Kullanılan veri setleri
- Sonuç dosyaları
- F1 odaklı başarı analizi

Eksikse hangi rapor bölümünün nasıl yazılması gerektiğini öner.

## 16. Puanlama Rubriği

Projeyi 100 üzerinden değerlendir:

| Kategori | Maksimum Puan |
|---|---:|
| Şartnameye Uygunluk | 20 |
| Bilimsel Geçerlilik | 20 |
| F1 ve Değerlendirme Metodolojisi | 15 |
| Veri Sızıntısı Önlemleri | 15 |
| Tekrarlanabilirlik | 10 |
| Kod Kalitesi | 10 |
| Dokümantasyon | 5 |
| Klinik Güvenlik ve Etik | 5 |

Puanı şişirme. Kanıt yoksa puan kır. “Dosyada kanıt bulunamadı” ifadesini kullan.

## 17. Nihai Çıktı Formatı

Cevabı mutlaka şu formatta ver:

## Yönetici Özeti

5-8 cümleyle projenin şartnameye uyum durumunu açıkla.

## Şartname Uyum Matrisi

| Şartname Gereksinimi | Repo Durumu | Kanıt Dosyası | Risk | Düzeltme |
|---|---|---|---|---|

## Kritik Uyumsuzluklar

Her madde için:

- Sorun
- Etkilenen dosya
- Şartname bağlantısı
- Risk seviyesi
- Net düzeltme

## Orta Seviye Uyumsuzluklar

Her madde için aynı formatı kullan.

## Güçlü Yönler

Sadece dosyada kanıtı olan güçlü yönleri yaz.

## Veri Seti ve Etiket Uyumu

Eğitim/test setleri, sınıf tanımları, panel yapısı ve ground truth mantığını değerlendir.

## Veri Sızıntısı Analizi

Genomik adres, dış veri kaynağı, preprocessing, feature selection, SMOTE, graph construction ve validation risklerini incele.

## F1 Skoru ve Final Değerlendirmesi Uyumu

F1 hesaplama, TP/FP/FN mantığı, panel bazlı sonuçlar ve final senaryosu uyumunu değerlendir.

## README Düzeltme Planı

Eksik README bölümlerini yaz ve doğrudan eklenebilir metin öner.

## MODEL_CARD Düzeltme Planı

Eksikleri ve düzeltmeleri yaz.

## DATA_CARD Düzeltme Planı

Eksikleri ve düzeltmeleri yaz.

## Kod ve Pipeline İncelemesi

Dosya bazlı teknik öneriler ver.

## Test ve CI İncelemesi

Eksik testleri ve önerilen test dosyalarını yaz.

## Klinik Güvenlik ve Etik İnceleme

Riskli ifadeleri ve güvenli alternatiflerini yaz.

## Tekrarlanabilirlik İncelemesi

Jürinin projeyi sıfırdan çalıştırırken nerede takılacağını açıkla.

## Teslim Hazırlığı

Projeyi şu kategorilerden biriyle sınıflandır:

- Hazır
- Kısmen hazır
- Riskli
- Hazır değil

Gerekçesini yaz.

## Önceliklendirilmiş Aksiyon Planı

### İlk 24 Saatte Düzeltilmeli

### Teslim Öncesi Kesin Düzeltilmeli

### Kalite Artırıcı İyileştirmeler

## Nihai Puan

| Kategori | Puan | Maksimum | Gerekçe |
|---|---:|---:|---|

Toplam puanı ver.

## Son Karar

Şu soruları net cevapla:

1. Repo mevcut haliyle teslim edilmeli mi?
2. Şartnameye göre en büyük 5 eksik nedir?
3. En büyük bilimsel risk nedir?
4. En büyük mühendislik riski nedir?
5. En büyük dokümantasyon riski nedir?
6. 100 puana yaklaşmak için ilk yapılacak 10 iş nedir?

## 18. Yazım Kuralları

- Türkçe yaz.
- Resmi ve akademik dil kullan.
- Mühendislik disipliniyle değerlendir.
- Genel tavsiye verme.
- Dosya temelli konuş.
- Kanıt yoksa “kanıt bulunamadı” de.
- Şartnameye açık bağlantı kur.
- Abartılı övgü kullanma.
- Belirsiz ifadeler kullanma.
- Kritik hataları yumuşatma.
- Her soruna uygulanabilir çözüm yaz.

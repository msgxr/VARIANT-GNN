# Veri Panel Haritası — VARIANT-GNN

## Panel Genel Tablosu

| Panel | Eğitim Pat. | Eğitim Ben. | Test Pat. | Test Ben. | Toplam Eğitim | Toplam Test |
|---|---|---|---|---|---|---|
| Genel | 1500 | 1500 | 1000 | 1000 | 3000 | 2000 |
| Herediter Kanser | 200 | 200 | 100 | 100 | 400 | 200 |
| PAH | 200 | 200 | 100 | 100 | 400 | 200 |
| CFTR | 70 | 70 | 30 | 30 | 140 | 60 |

## Panel Bazlı Teknik Analiz

### Panel 1: Genel Veri Seti
- **Veri hacmi:** En büyük panel (3000 eğitim). Genel genelleme kapasitesini ölçer.
- **Sınıf dengesi:** Mükemmel denge (1500/1500).
- **Teknik zorluk:** Çeşitli gen bağlamları bir arada olduğundan model, genel patojenite sinyallerini öğrenmek zorundadır.
- **Dikkat:** Bu panelde yüksek F1 elde etmek diğer panellerde düşük F1 riskini gizleyebilir; panel bazlı raporlama zorunlu.

### Panel 2: Herediter Kanser Paneli
- **Veri hacmi:** Orta (400 eğitim). Hastalık-spesifik örüntüler beklenir.
- **Teknik zorluk:** Az veriyle gen-spesifik sinyal öğrenmek. Overfitting riski Genel Panel'e göre daha yüksek.
- **Dikkat:** Bu panel için ayrı CV stratejisi ve hiperparametre ayarı değerlendirilmeli.

### Panel 3: PAH (Fenilketonüri — PAH Geni)
- **Veri hacmi:** Orta (400 eğitim). Tek gen hastalığı bağlamı.
- **Teknik zorluk:** Tek gen üzerindeki varyantlar benzer özellik uzayında kümelenmiş olabilir (varsayım — EDA gerekli).
- **Dikkat:** Test setinde beklenmedik özellik dağılımı varsa (train-test distribution shift) F1 düşebilir.

### Panel 4: CFTR (Kistik Fibrozis — CFTR Geni)
- **Veri hacmi:** En küçük panel (140 eğitim, 60 test). En yüksek risk.
- **Teknik zorluk:** 70 örnekle model eğitimi ciddi overfitting riski taşır.
- **Dikkat:** Bu panelde 1 hatalı test tahmini yaklaşık %3.3 F1 değişimine yol açar. Karar eşiği hassasiyeti kritik.
- **PDR'de:** CFTR panelinin küçük hacmi ve ona özel risk yönetimi açıkça tartışılmalı.

## Resmî Öznitelik Kategorileri (Anonim Kolon Çerçevesi)

Kolonlar anonim olsa da resmî şartname aşağıdaki kategorileri tanımlamaktadır:

1. Sekans ve değişim bilgisi
2. Yerel sekans ve çevresel bağlam bilgisi
3. Biyokimyasal ve yapısal etkiler
4. Evrimsel korunmuşluk
5. Popülasyon verileri
6. In silico risk skorları

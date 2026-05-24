# Resmî Yarışma Bağlamı ve Şartname Özeti

Kaynak: 2026 Sağlıkta Yapay Zeka Yarışması Şartnamesi V1.4 (14.04.2026) + TEKNOFEST resmî sayfası

---

## Yarışmanın Amacı (§2)

Sağlık alanında yapay zeka uygulamalarını geliştirmek, bu alanda nitelikli insan kaynağı oluşturmak ve sağlık sorunlarına yönelik çözümler üretmek. Genetik ve Kardiyoloji alanlarında yapay zekâ çözümleri geliştirmek. Üniversite seviyesinde: missense varyantların Patojenik / Benign sınıflandırması.

## Üniversite ve Üzeri Seviyesi Görevi (§3.2)

**Görev:** Bilinen az sayıdaki varyantın biyolojik ve hesaplamalı özelliklerini kullanarak, klinik durumu bilinmeyen varyantların Patojenik veya Benign olma durumuna yönelik tahmin modelleri geliştirmek.

**Etiket kaynakları:**
- Patojenik: ClinVar + ClinGen "Expert Panel"/"Practice Guideline" 3-4 yıldız, "Pathogenic"+"Likely Pathogenic" (2909 varyant)
- Benign: ClinVar (1381) + gnomAD (~1500) sağlıklı popülasyon varyantları

**Veri sayıları (şartnamede kesin):**

| Panel | Eğitim Pat. | Eğitim Ben. | Test Pat. | Test Ben. |
|---|---|---|---|---|
| Genel | 1500 | 1500 | 1000 | 1000 |
| Herediter Kanser | 200 | 200 | 100 | 100 |
| PAH | 200 | 200 | 100 | 100 |
| CFTR | 70 | 70 | 30 | 30 |

**Veri kısıtları:**
- Genomik adres (kromozom, pozisyon) tamamen gizlenmiş
- Öznitelik kolon isimleri verilmeyecek
- Model yalnızca yarışma komitesi tarafından sağlanan varyant profilleri üzerinden çalışmalı

**Öznitelik kategorileri (şartnamede belirtilen 6 kategori):**
1. Sekans ve Değişim Bilgisi
2. Yerel Sekans ve Çevresel Bağlam Bilgisi (±5 nükleotid, ±5 amino asit)
3. Biyokimyasal ve Yapısal Etkiler
4. Evrimsel Korunmuşluk
5. Popülasyon Verileri (MAF vb.)
6. In Silico Risk Skorları

## Final Değerlendirme Metriği (§7.3) — KESİN

**"Yarışma sıralamasını belirleyecek temel metrik, TP (Doğru Pozitif), FP (Yanlış Pozitif) ve FN (Yanlış Negatif) değerleri üzerinden hesaplanan F1 Skoru olacaktır."**

Şartnamede yalnızca F1 belirtilmiştir. MCC, PR-AUC, Brier Score şartname final metriği değildir. Bunlar PDR şablonunda istenebilir.

## PDR İçeriği (§4 — Katılım Koşulları)

"Proje Detay Raporunda ise, geliştirilen modelin **mimarisi, eğitim süreçleri ve iç test (validasyon) sonuçları** detaylı olarak sunulacaktır."

## Puanlama (§7.5) — KESİN

| Aşama | Ağırlık |
|---|---|
| PSR | %0 (yalnızca eleme) |
| PDR | %0 (yalnızca eleme) |
| Final Fiziki Yarışma | %90 |
| Final Sunum | %10 |

## Jüri Kodu Yeniden Çalıştırma (§7.5) — RESMİ DOĞRULAMA

**"Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."**

Reproducibility şartı şartnamede resmî olarak var. Bu varsayım değil, kesin kuraldır.

## Metrik Değişiklik Hakkı (§7.5) — ÖNEMLİ UYARI

**"TÜSEB, yarışma değerlendirme aşamasında kullanılan değerlendirme metrikleri ve değerlendirme oranlarında değişiklik yapma hakkını saklı tutar."**

F1 metriği değişebilir. Bu riski takip et.

## Yarışma Takvimi (§6) — KESİN

| Tarih | Açıklama |
|---|---|
| 28.02.2026 | Son başvuru |
| 23.03.2026 | Soru-Cevap I |
| 25.03.2026, 17:00 | PSR teslim |
| 22.04.2026 | PSR sonuçları |
| 05.05.2026 | Veri paylaşımı |
| **18.05.2026** | **Soru-Cevap II** |
| **29.06.2026, 17:00** | **PDR teslim** |
| 20.07.2026 | PDR sonuçları ve finalistler |
| Ağustos–Eylül 2026 | Finallar |
| 30 Eylül–4 Ekim 2026 | TEKNOFEST Şanlıurfa |

## Etik Sınır (§10) — RESMİ METİN

"Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar, herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla kullanılamaz. Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."

## Ödüller (§8)

| Derece | Üniversite | Danışman |
|---|---|---|
| 1. | 180.000 ₺ | 15.000 ₺ |
| 2. | 150.000 ₺ | 12.000 ₺ |
| 3. | 130.000 ₺ | 10.000 ₺ |

## Gizlilik Sözleşmesi (§4)

Yarışmacılar verilere ancak "Kurumsal Gizlilik Taahhütnamesi" imzalayarak erişebilir. Bu belge PDR tesliminden önce teslim edilmiş olmalı.

## Finale Ulaşım Desteği (§5)

Takım başı 3 kişi (danışman dahil). Komite değişiklik hakkını saklı tutar.

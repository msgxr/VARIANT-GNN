# OFFICIAL_REFERENCES.md — VARIANT-GNN

**Son Kontrol:** 2026-05-24  
**Beyan:** Bu dosyada listelenen kaynaklar dışında üçüncü taraf kaynak yarışma kuralları için kullanılmamıştır.

---

## 1. Ana Resmi Sayfa

**URL:** https://www.teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/  
**Kullanım:** Yarışma genel bilgisi, takvim, ödüller, bağlantılar  
**Kontrol Tarihi:** 2026-05-24

---

## 2. Şartname (Birincil Kaynak — TIER 1)

**Türkçe Şartname v4:**  
https://cdn.teknofest.org/media/upload/userFormUpload/2026-_Sağlıkta_Yapay_Zeka_Türkçe_Şartname_v4_6k439.pdf  
**Durum:** BİRİNCİL KAYNAK — tüm yarışma kural kararları bu belgeden alınır. Çelişki durumunda kazanır.

**İngilizce Şartname v3:**  
https://cdn.teknofest.org/media/upload/userFormUpload/2026-_Sağlıkta_Yapay_Zeka_İngilizce_Şartname-_V3_Bwi8D.pdf  
**Durum:** Yardımcı — Türkçe ile çelişirse Türkçe esas alınır.

---

## 3. Rapor Şablonları (İkincil Kaynak — TIER 2)

**PSR Şablonu — Üniversite ve Üzeri:**  
https://cdn.teknofest.org/media/upload/userFormUpload/2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx  
**Kullanım:** PSR bölüm yapısı, zorunlu başlıklar, format

**PDR Şablonu — Üniversite ve Üzeri:**  
https://cdn.teknofest.org/media/upload/userFormUpload/2026_PDR_Şablon_Universite_TR_bCw49.docx  
**Kullanım:** PDR bölüm yapısı, zorunlu başlıklar, sayfa limiti, format gereksinimleri

> Lise / EKG / Kardiyoloji şablonları bu proje için geçersizdir.

---

## 4. Doğrulanmış Bilgiler (2026 Resmi Kaynaklardan)

| Bilgi | Değer | Kaynak | Tier |
|---|---|---|---|
| Yarışma adı | TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması | Ana sayfa | 3 |
| Kategori | Üniversite ve Üzeri | Şartname | 1 |
| Görev | Patojenik/Benign binary sınıflandırma | Şartname §3.2 | 1 |
| Birincil metrik | Binary F1 Score (pos_label=1=Patojenik) | Şartname §7.3 | 1 |
| F1 formülü | TP / (TP + 0.5×FP + 0.5×FN) | Şartname §7.3 | 1 |
| PSR ağırlığı | **%0** (yalnızca eleme) | Şartname §7.5 | 1 |
| PDR ağırlığı | **%0** (yalnızca eleme) | Şartname §7.5 | 1 |
| Final yarışma ağırlığı | **%90** | Şartname §7.5 | 1 |
| Final sunum ağırlığı | **%10** | Şartname §7.5 | 1 |
| TÜSEB metrik değişiklik hakkı | Saklıdır (§7.5) | Şartname §7.5 | 1 |
| Jüri kodu tekrar çalıştırma | Yetkisi vardır (§7.5) | Şartname §7.5 | 1 |
| PDR teslim tarihi | 29 Haziran 2026, saat 17:00 | Takvim §6 | 1 |
| PSR teslim tarihi | 25 Mart 2026, saat 17:00 | Takvim §6 | 1 |
| Veri dağıtım tarihi | 5 Mayıs 2026 | Takvim §6 | 1 |
| Final lokasyonu | Şanlıurfa | Takvim §6 | 1 |
| Final tarihi | Ağustos–Eylül 2026 (30 Eyl–4 Eki TEKNOFEST) | Takvim §6 | 1 |
| Ödül 1. | ₺180.000 (+ danışman ₺15.000) | Ana sayfa §8 | 3 |
| Ödül 2. | ₺150.000 (+ danışman ₺12.000) | Ana sayfa §8 | 3 |
| Ödül 3. | ₺130.000 (+ danışman ₺10.000) | Ana sayfa §8 | 3 |
| Takım büyüklüğü | 2–5 kişi (danışman hariç) | Şartname | 1 |
| Gizlilik taahhütü | Kurumsal Gizlilik Taahhütü zorunlu | Şartname §4 | 1 |
| KVKK uyumu | Zorunlu | Şartname | 1 |
| **PDR sayfa limiti** | **≤10 sayfa** (kapak + içindekiler hariç) | PDR Şablonu | 2 |
| PDR yazı tipi | Aptos, 12pt (başlık 14pt) | PDR Şablonu | 2 |
| PDR satır aralığı | 1.15 | PDR Şablonu | 2 |
| PDR marjin | Üst 2.8cm, diğerleri 2.5cm | PDR Şablonu | 2 |
| PDR referans formatı | IEEE | PDR Şablonu | 2 |
| PDR zorunlu metrikler | F1 + MCC + PR-AUC (her panel ayrı) | PDR Şablonu §3 | 2 |
| Etik sınır (resmi metin) | "Klinik tanı, tedavi veya tıbbi karar desteği için kullanılamaz" | Şartname §10 | 1 |
| Şeyma GitHub | cebi101 / seymanurcebi6@gmail.com | Takım doğrulaması | Verified |

---

## 5. UNVERIFIED — Doğrulanamamış Bilgiler

Kesin hüküm gibi yazılmaz. Gerekirse resmi kaynaktan kontrol edilmeli.

| Bilgi | Durum | Notlar |
|---|---|---|
| Veri seti kolon sayısı | UNVERIFIED | Veri dağıtıldı, resmi şartnamede sayı yok |
| Gerçek Patojenik/Benign oranı | UNVERIFIED | Eğitim setine göre değişebilir |
| Yayın ve tez kullanım hakları | UNVERIFIED | Şartname §10 kontrol edilmeli |
| Final sözlü sunum içeriği | UNVERIFIED | %10 ağırlık doğrulandı, format UNVERIFIED |
| PSR sayfa limiti (exact) | UNVERIFIED | PDR'de 10 sayfa doğrulandı, PSR şablondan kontrol gerekiyor |

---

## 6. Üçüncü Taraf Kaynak Yasağı

Bu projede yarışma kurallarına ilişkin hiçbir karar şu kaynaklardan alınmamıştır:  
- 2024 veya önceki yıl şartnameleri
- Blog, forum, sosyal medya, kişisel yorum
- Gayriresmi özet, üçüncü taraf doküman
- "Geçen yıl böyleydi" hafızası

Tüm yarışma kural bilgisi yalnızca §2 ve §3'te listelenen resmi belgelerden alınmıştır.

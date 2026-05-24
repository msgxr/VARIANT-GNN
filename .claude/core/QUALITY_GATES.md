# QUALITY_GATES.md — CAPOS v2.0
# Görev Sınıflandırma ve Kalite Kapıları — VARIANT-GNN

**Versiyon:** 2026-05-24

---

## Görev Sınıflandırması

### ROUTINE
**Örnekler:** Yazım hatası, yorum güncellemesi, log mesajı, minor config değeri  
**Risk:** Düşük — yarışma sonuçlarını etkilemez  
**Kapılar:** Hızlı kontrol → uygula → spot-check

### IMPORTANT
**Örnekler:** Yeni özellik, dokümantasyon bölümü, metrik ekleme, config değişikliği  
**Risk:** Orta — kapsamı veya netliği etkileyebilir  
**Kapılar:** İnspeksiyon → uygula → doğrula → raporla

### CRITICAL
**Örnekler:** Pipeline mantık değişikliği, metrik hesaplama, veri bölme mantığı, eşik değişikliği  
**Risk:** Yüksek — gerçek sonuçları değiştirebilir veya iddiaları geçersiz kılabilir  
**Kapılar:** Scientist incelemesi → uygula → tam doğrulama → çapraz dosya etkisi → raporla

### HIGH-RISK
**Örnekler:** GNN mask mantığı, SMOTE sıralaması, feature selector fitting, label encoding, ensemble ağırlıkları  
**Risk:** Kritik — fark edilmeden data leakage veya yanlış sonuç üretebilir  
**Kapılar:** Scientist + Debugger çift kontrol → uygula → verifier → smoke test → raporla

### DELIVERY-BLOCKING
**Örnekler:** PDR finalizasyonu, teslim paketi, jüri sunum materyali  
**Risk:** Misyon kritik — başarısızlık yarışma sonucunu doğrudan etkiler  
**Kapılar:** TÜM ajanlar → pre-submission-gate → mission-readiness → jury-adversary → GO/NO-GO

---

## Kapı Tanımları

### G1: Kanıt Kontrolü
**Gereksinim:** Her iddia dosya/sonuç kanıtına dayanmalı  
**Kim:** Hepsi — her zaman aktif  
**Başarısızlık:** Kanıtsız iddia → varsayım olarak işaretle, gerçek olarak sunma

### G2: Çapraz Dosya Etkisi
**Gereksinim:** Değişiklik öncesi hedef kodu kullanan diğer modüller incelenmeli  
**Kim:** architect + debugger  
**Başarısızlık:** Değişiklik downstream modülü bozar → regresyon

### G3: Bilimsel Geçerlilik
**Gereksinim:** Metrik iddiaları, pipeline kararları ve deneysel sonuçlar savunulabilir olmalı  
**Kim:** scientist  
**Başarısızlık:** Doğrulanamayan iddia → kaldır veya nitelendir

### G4: Tekrar Üretilebilirlik
**Gereksinim:** Pipeline değişikliğinden sonra seed=42 hâlâ tutarlı çıktı üretmeli  
**Kim:** verifier  
**Başarısızlık:** Non-deterministik sonuçlar → jüri için F1=0.8980 yeniden üretilemiyor

### G5: Yarışma Uyumu
**Gereksinim:** Değişiklik PDR skoru, jüri izlenimi veya şartname uyumunu olumsuz etkilememeli  
**Kim:** jury-adversary  
**Başarısızlık:** İyileştirme gibi görünen değişiklik yarışma duruşunu kötüleştiriyor

### G6: Güvenlik
**Gereksinim:** Sır commit edilmedi, yarışma verisi açığa çıkmadı, injection riski yok  
**Kim:** sentinel  
**Başarısızlık:** Veri ihlali riski veya veri kötüye kullanımı nedeniyle diskwalifikasyon

### G7: Smoke Test
**Gereksinim:** Pipeline baştan sona çalışıyor ve geçerli çıktı üretiyor  
**Kim:** verifier  
**Başarısızlık:** Kod çalışmıyor → jüri yeniden üretemez → §7.5 ihlali

### G8: Dokümantasyon Tutarlılığı
**Gereksinim:** Uygulama değişiklikleri docs/PDR'ye, tersi de koda yansımalı  
**Kim:** documentalist  
**Başarısızlık:** README/PDR bir şey söylüyor, kod başka bir şey yapıyor → güvenilirlik zararı

---

## Kapı Uygulama Matrisi

| Görev Sınıfı | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
|---|---|---|---|---|---|---|---|---|
| Routine | ✓ | — | — | — | — | — | — | — |
| Important | ✓ | ✓ | — | — | — | — | — | ✓ |
| Critical | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| High-Risk | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Delivery-Blocking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Üretici-Gözlemci Ayrımı

Critical ve üstü görevlerde bir değişikliğin üreticisi tek gözlemci olamaz.

| Görev Türü | Üretici | Gözlemci |
|---|---|---|
| Kod değişikliği | Claude (doğrudan) | debugger + verifier |
| Rapor/belge | Claude (doğrudan) | documentalist + jury-adversary |
| Deneysel iddia | scientist | scientist + jury-adversary (çelişkili) |
| Mimari karar | architect | scientist + verifier |
| Final teslim | Tüm ajanlar | pre-submission-gate (bağımsız kontrol) |

---

## Başarısızlık Yönetimi

### Kapı Başarısızlığı = Görev Tamamlanmadı
Görev yalnızca tüm gerekli kapılar geçildiğinde tamamdır.

### Başarısızlık Yanıt Protokolü
1. **Tanımla:** Hangi kapı başarısız oldu? Kanıt nedir?
2. **Değerlendir:** Engelleyici mi? Risk bayrağıyla ilerlenebilir mi?
3. **Düzelt veya İşaretle:** Başarısızlığı düzelt veya kabul edilen riski açıkça belgele
4. **Yeniden Test:** Düzeltmeden sonra başarısız kapıyı yeniden çalıştır
5. **Raporla:** Başarısızlığı ve çözümü görev raporuna kaydet

### Kabul Edilen Risk Protokolü
Bir kapı başarısızlığı deadline öncesi düzeltilemiyorsa:
1. Başarısızlığı açıkça belgele
2. Yarışma riskini değerlendir (DÜŞÜK/ORTA/YÜKSEK/KRİTİK)
3. Bu zayıflık için jüri savunması hazırla
4. Uygunsa PDR sınırlılıklar bölümüne ekle
5. Bilinen zayıflıkları asla gizleme — jüri zaten bulacaktır

---

*CAPOS Quality Gates v2.0 — 2026-05-24*

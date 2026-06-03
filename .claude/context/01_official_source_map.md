# 01_official_source_map.md — Resmi Kaynak Haritası

**Versiyon:** 2026-05-24  
**İlke:** Bu dosyadaki her bilgi yalnızca aşağıdaki resmi kaynaklardan alınmıştır.

---

## Birincil Kaynak

**TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması Şartnamesi — Türkçe v4**  
URL: https://cdn.teknofest.org/media/upload/userFormUpload/2026-_Sağlıkta_Yapay_Zeka_Türkçe_Şartname_v4_6k439.pdf  
Durum: Birincil — tüm kural kararları bu belgeden alınır.

---

## İkincil Kaynaklar

| Belge | URL | Kullanım |
|---|---|---|
| Ana yarışma sayfası | teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/ | Genel bilgi, takvim |
| PDR Şablonu (Üniversite) | cdn.teknofest.org/.../2026_PDR_Şablon_Universite_TR_bCw49.docx | PDR format |
| PSR Şablonu (Üniversite) | cdn.teknofest.org/.../2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx | PSR format |
| İngilizce Şartname v3 | cdn.teknofest.org/.../2026-_İngilizce_Şartname-_V3_Bwi8D.pdf | Yardımcı |

---

## Tier-4 — Resmi Duyuru / Q&A (şartnameyi NOKTASAL olarak günceller)

| Belge | Kaynak | Kullanım | Statü |
|---|---|---|---|
| **Q&A-II Üniversite transkripti** | Online Soru-Cevap Toplantısı (kayıt `ÜNİVERSİTE VE ÜZERİ.mp4`); takvim 18.05.2026; ekibe 2026-06-02 ulaştı. Yerel artefakt: `docs/qa/2026_QA_Universite_transkript.md` (gitignore — organizatör çekincesi) | Test dağılımı, metrik, çıktı formatı, model sayısı, öznitelik kararları | ✅ **DOĞRULANDI 2026-06-03** |

> ⚠️ ÇELİŞKİ NOTU: Şartname §3.2 test setini 1000/1000 (50/50) yazar; Q&A-II'de organizatör bunun **eski/hatalı** olduğunu, güncel kararın **test ~%20 patojenik / %80 benign** olduğunu açıkça söyledi ve şartnameyi yeniden göndereceğini belirtti (transkript 07:13, 23:37, 26:23). Çelişki Öncelik Sırası gereği **Q&A-II (Tier-4 güncel duyuru) bu noktada şartnameyi geçersiz kılar.** Şartname §7.1/§7.5 komiteye metrik/oran değiştirme hakkını zaten tanır.

---

## Kullanılmayan / Reddedilen Kaynaklar

- 2024 veya önceki yıl şartnameleri → Geçersiz (2026 kuralları farklı olabilir)
- Üçüncü taraf blog veya forum → Kabul edilmez
- Sosyal medya paylaşımları → Kabul edilmez
- Gayriresmi özet dokümanlar → Kabul edilmez

---

## Çelişki Öncelik Sırası

1. 2026 Türkçe Şartname v4 (en güncel)
2. Resmi rapor şablonları
3. Resmi yarışma web sayfası
4. KYS / resmi duyuru / yarışma grubu bilgilendirmesi

---

## ✅ ÇÖZÜLDÜ (2026-06-03) — Q&A artefaktı edinildi

**TEKNOFEST resmi Q&A-II (Üniversite) — "test seti ~%20-patojenik / %80-benign; 50/50 ESKİ/geçersiz".**

- **Durum: DOĞRULANDI.** Toplantı transkripti edinildi ve okundu (yerel artefakt: `docs/qa/2026_QA_Universite_transkript.md`). Hakem ifadesi NET ve TEKRARLI (07:13, 07:48, 08:54, 09:26, 23:37). Şartmanenin 1000/1000'i organizatör tarafından **hata** olarak kabul edildi (26:23).
- **Sonuç:** `RESULTS_CANONICAL.json` içindeki **θ=0.8415**, jüri-F1 **0.6042** ve **resmi 4-panel skor 0.6202** artık resmi dayanağa sahiptir. U-008 → Çözülmüş. `provenance_unverified` → güncellendi.
- **Aynı transkriptle ayrıca doğrulandı:** F1 patojenik-odaklı + per-panel + ortalama (34:10); ikili 0/1 çıktı (08:12); 4 ayrı model (31:30); komşu sekans/AA paylaşılmıyor (21:47); missing≠0 (17:25).

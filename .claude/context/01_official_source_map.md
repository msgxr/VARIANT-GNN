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

## ⚠️ PENDING — EYLEM GEREKLİ (eklenecek resmi kaynak)

**TEKNOFEST resmi Q&A / duyuru (iddia: 2026-06-02 — "gizli test seti %20-patojenik / %80-benign; 50/50 ESKİ/geçersiz").**

- **Durum: UNVERIFIED — artefakt henüz eklenmedi.** Repoda yalnızca bu iddiaya *atıf* var, kaynağın kendisi (ekran görüntüsü / arşiv URL / KYS mesajı) yok.
- **Bağımlılık:** Bu iddia, `RESULTS_CANONICAL.json` içindeki karar eşiği **θ=0.8415**, havuzlanmış jüri-F1 **0.6042** ve **resmi 4-panel skor 0.6202**'nin tek dayanağıdır. Bkz. `07_uncertainty_log.md` U-008 ve `RESULTS_CANONICAL.provenance_unverified`.
- **Eylem:** Q&A artefaktını edinip yukarıdaki "İkincil Kaynaklar" tablosuna Tier-4 (KYS/resmi duyuru) olarak ekleyin ve U-008'i "Çözülmüş"e taşıyın. **Edinilemezse:** %20-prior'ı resmi kural değil *modelleme varsayımı* olarak çerçeveleyin (official-source-guardian, CLAUDE.md §III.1).

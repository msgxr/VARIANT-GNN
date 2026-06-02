# 04_report_requirements.md — Rapor Gereksinimleri

**Kaynak:** TEKNOFEST 2026 PDR Şablonu (Üniversite) + PSR Şablonu (Üniversite)  
**Versiyon:** 2026-05-24

---

## PSR — Proje Sunuş Raporu

**Teslim:** 25 Mart 2026, 17:00 (TAMAMLANDI — 93/100 alındı)  
**Şablon:** 2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx

### PSR Sonuçları ve Zayıf Noktalar

| Bölüm | Puan | Not |
|---|---|---|
| §4.4 Explainability | 3.33/5 | Güçlendirilmeli |
| §4.5 Technical Evolution | 3.33/5 | GATv2Conv gerekçesi yetersizdi |
| §5.1 Algorithm Justification | 4/5 | Kısmen zayıf |
| Genel PSR | 93/100 | Ön eleme geçildi |

---

## PDR — Proje Detay Raporu

**Teslim:** 29 Haziran 2026, 17:00 (36 gün kaldı — 2026-05-24 itibarıyla)  
**Şablon:** 2026_PDR_Şablon_Universite_TR_bCw49.docx

### Zorunlu Bölümler (Şablondan)

| # | Bölüm | Puan Ağırlığı |
|---|---|---|
| 1 | Giriş | 10 puan |
| 2 | Yöntem | 25 puan |
| 3 | Bulgular | 30 puan |
| 4 | Sonuç | 25 puan |
| 5 | Kaynakça | 10 puan |

**Not:** Sayfa limiti ve ağırlıklar şablondan doğrulanmalı — UNVERIFIED (tam değer şablonda)

### PDR Kalite Gereksinimleri

- Her performans iddiasının kanıtı olmalı (JSON, log, tablo)
- Panel bazlı sonuçlar ayrı raporlanmalı (MASTER / KANSER / PAH / CFTR)
- PSR → PDR teknik evrim açıklanmalı (7 yenilik tablosu mevcut)
- GATv2Conv / SAGEConv tutarsızlığı düzeltildi (PDR §2.2'de açıklandı)
- PSR pilot sonuçları (MCC=0.892) ile gerçek veri sonuçları (MCC=0.5863, canonical) farkı §4.2'de açıklandı
- Etik beyan mevcut

### Mevcut PDR Durumu (2026-06-02 — canonical'a hizalandı)

| Alan | Durum |
|---|---|
| Dosya | reports/PDR_VARIANT_GNN_2026.md |
| Sayılar | ✅ RESULTS_CANONICAL.json ile tutarlı (check_results_consistency.py 5/5 PASS) |
| Kaynakça numaraları | ✅ Düzeltildi (REVEL[2], EVE[9], GATv2[8]) |
| §3.2 eşik değeri | ✅ GLOBAL θ=0.8514 (canonical; 0.241 supersede) |
| Figür yolları | ✅ reports/figures/pdr/* |
| Rapor tarihi | ✅ 2 Haziran 2026 (sızıntısız retrain) |

---

## Format Gereksinimleri (UNVERIFIED — şablondan kontrol)

- Yazı tipi: Aptos 12pt (mevcut PDR'de belirtilmiş)
- Başlıklar: 14pt
- Satır aralığı: 1.15
- Teslim formatı: UNVERIFIED (PDF mi, DOCX mu?)

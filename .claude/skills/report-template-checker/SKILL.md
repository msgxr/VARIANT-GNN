---
name: report-template-checker
description: Use when verifying that PSR or PDR reports match the official 2026 TEKNOFEST templates exactly. Checks section order, required headings, page limits, font/format requirements, mandatory fields, and submission format. Activate before any report submission or when "şablona uygun mu?" is asked.
---

# Skill: report-template-checker

## Purpose

PSR ve PDR raporlarının resmi TEKNOFEST 2026 şablonlarıyla birebir uyumunu denetlemek.

## Official Source Boundary

- PSR Şablonu: cdn.teknofest.org/.../2026_PSR-Üniversite_ve_Üzeri_lt7Hv.docx
- PDR Şablonu: cdn.teknofest.org/.../2026_PDR_Şablon_Universite_TR_bCw49.docx
- Lise şablonları bu proje için geçersizdir.

## Inputs

- PDR veya PSR içerik dosyası
- Resmi şablon (DOCX)

## Outputs

- Başlık uyum tablosu (mevcut vs. şablon)
- Eksik / sıra dışı bölümler
- Format hataları (yazı tipi, sayfa limiti vb.)
- GO / NO-GO kararı

## Hard Rules

1. Lise şablonu asla kullanılmaz.
2. Zorunlu bölüm eksikse rapor teslime hazır değildir.
3. Sayfa limiti aşılırsa rapor reddedilebilir.
4. Şablon dışı başlık eklemek jüri izlenimine zarar verebilir.

## Bilinen PDR Sorunları (2026-05-24)

| Sorun | Konum | Düzeltme |
|---|---|---|
| Kaynakça numaraları yanlış | §1.2 | REVEL[3]→[2], EVE[5]→[9], GATv2[7]→[8] |
| Eşik değeri hatalı | §3.2 | θ=0.01 → θ=0.241 |
| Figür yolları geçersiz | §3.1 Şekil 2-5 | reports/figures/pdr/ altındaki doğru dosyalara güncelle |
| Rapor tarihi tutarsız | Başlık | 15 Mayıs → 20 Mayıs veya sonrası |

## Step-by-Step Procedure

1. Rapor dosyasını aç.
2. Şablon bölümlerini sırayla kontrol et.
3. Her bölüm için: mevcut mu, doğru başlıkta mı, içerik yeterli mi?
4. Format kontrolü: yazı tipi, satır aralığı, sayfa sayısı.
5. Figür ve tablo referanslarını doğrula.
6. Kaynakça uyumunu kontrol et (numara-içerik eşleşmesi).
7. GO / NO-GO karar ver.

## Validation Checklist

- [ ] Üniversite PDR şablonu kullanıldı (lise değil)?
- [ ] Tüm 5 bölüm ve alt bölümler mevcut?
- [ ] Etik beyan mevcut?
- [ ] Her Şekil ve Tablo doğru yola atıfta bulunuyor?
- [ ] Kaynakça in-text numaraları ile uyuşuyor?
- [ ] Sayfa limiti aşılmadı? (UNVERIFIED — şablondan kontrol)

## Failure Conditions

- Zorunlu bölüm eksik
- Kaynakça numarası uyuşmazlığı
- Geçersiz figür yolu
- Sayfa limiti aşımı

## Escalation Rule

Kritik hata bulunursa → pre-submission-gate'e eskalasyon, teslim engellenir.

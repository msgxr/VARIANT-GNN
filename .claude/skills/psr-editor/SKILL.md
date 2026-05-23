---
name: psr-editor
description: Use when reviewing, editing, or preparing the VARIANT-GNN Project Presentation Report (PSR — Proje Sunuş Raporu). Checks compliance with the official 2026 PSR template (Üniversite ve Üzeri), section structure, technical accuracy, and measurable claims. Note: PSR was submitted 25 March 2026 (score 93/100) — this skill is used for PSR retrospective analysis and weak-point strengthening for jury defense.
---

# Skill: psr-editor

## Purpose

PSR içeriğinin resmi TEKNOFEST 2026 şablonuna uyumunu kontrol etmek ve PSR zayıf noktalarını PDR'de güçlendirmek.

## Official Source Boundary

Yalnızca 2026 PSR Şablonu (Üniversite ve Üzeri) esas alınır.  
Şablon URL: cdn.teknofest.org/.../2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx

## Inputs

- PSR içeriği veya taslağı
- Jüri geri bildirimi (varsa)
- Puan dağılımı

## Outputs

- Şablon uyum raporu
- Zayıf bölüm tespiti ve güçlendirme önerileri
- PDR'de ele alınması gereken PSR açıkları

## Hard Rules

1. Şablon başlıkları bozulmaz.
2. Ölçülemeyen ifade yazılmaz ("çok iyi", "mükemmel" gibi).
3. Her performans iddiasının kanıtı olmalı.
4. Pilot ClinVar sonuçları gerçek yarışma verisi sonucu gibi sunulamaz.

## PSR Zayıf Noktaları (Mevcut Proje)

| Bölüm | PSR Puanı | Sorun |
|---|---|---|
| §4.4 Explainability | 3.33/5 | Panel bazlı SHAP breakdown eksik |
| §4.5 Technical Evolution | 3.33/5 | GATv2Conv seçim gerekçesi nicel olarak zayıf |
| §5.1 Algorithm Justification | 4/5 | Mimari tercih gerekçesi yetersiz |

## Step-by-Step Procedure

1. PSR şablon bölümlerini listele.
2. Her bölümde eksik veya zayıf içeriği tespit et.
3. Zayıf bölümleri PSR puanlarıyla eşleştir.
4. PDR'de güçlendirilmesi gereken noktaları belirle.
5. Somut iyileştirme metni öner (ölçülebilir, kanıtlı).

## Validation Checklist

- [ ] Tüm zorunlu PSR bölümleri mevcut mu?
- [ ] Üniversite şablonu kullanıldı (lise değil)?
- [ ] Her performans iddiasının kanıtı var mı?
- [ ] Pilot sonuçlar gerçek veri sonucu gibi sunulmadı mı?
- [ ] §4.4, §4.5, §5.1 zayıklıları PDR'de ele alındı mı?

## Failure Conditions

- Şablon başlıklarının değiştirilmesi
- Kanıtsız performans iddiası
- Lise şablonunun kullanılması

## Escalation Rule

PSR'deki zayıf noktalar PDR'ye taşındıysa → pdr-editor skill'ini tetikle.

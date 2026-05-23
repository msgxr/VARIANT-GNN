---
name: competition-compliance-auditor
description: Use when auditing the full project for TEKNOFEST 2026 Sağlıkta Yapay Zeka competition compliance. Checks category correctness (Üniversite ve Üzeri), metric correctness (Binary F1), data rules, ethics, reproducibility, report structure, and submission package. Activate when user asks "şartnameye uygun mu?", "eksiğimiz var mı?", "teslime hazır mıyız?" or before any major submission.
---

# Skill: competition-compliance-auditor

## Purpose

Projenin TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri şartnamesiyle tam uyumunu denetlemek.

## Official Source Boundary

Yalnızca TEKNOFEST 2026 Türkçe Şartname v4 ve resmi şablonlar esas alınır.  
UNVERIFIED bilgiler açıkça işaretlenir.

## Inputs

- Proje repo durumu
- Mevcut PDR/PSR içeriği
- Kod ve model durumu
- Deney sonuçları

## Outputs

- Bölüm bazlı uyum raporu (GEÇTİ / RİSK / BAŞARISIZ)
- Kritik eksikler listesi
- PDR teslim öncesi eylem listesi

## Hard Rules

1. Lise kategorisi kuralları (EKG, kardiyoloji) uygulanmaz.
2. Binary F1 birincil metrik — diğerleri destekleyici.
3. Test seti etiketleri eğitimde kullanılamaz.
4. Klinik tanı/tedavi iddiası yasaktır.
5. Kanıtsız performans iddiası raporlanamaz.

## Step-by-Step Procedure

1. ERRORCHECKLIST.md'yi A'dan J'ye uygula.
2. PDR bölümlerini şablonla karşılaştır.
3. Metrik hesaplamalarını doğrula (F1 pos_label=1).
4. Veri sızıntısı kontrolü (scaler, SMOTE, encoder fit yeri).
5. Tekrar üretilebilirlik: seed, requirements, tek komut.
6. Git kimlik protokolü kontrolü.
7. Bulgular tablosunu üret.

## Validation Checklist

- [ ] Üniversite kategorisi mi (lise değil)?
- [ ] Birincil metrik Binary F1 mi?
- [ ] Dört panel ayrı raporlandı mı?
- [ ] Data leakage yok mu?
- [ ] PDR resmi şablona uygun mu?
- [ ] Etik beyan mevcut mu?
- [ ] Tekrar üretilebilirlik testi geçildi mi?
- [ ] Commit kimliği doğru mu?

## Failure Conditions

- Lise içeriği (EKG) tespit edilmesi
- F1 yerine accuracy ana metrik olarak sunulması
- Data leakage bulunması
- Test etiketlerinin eğitimde kullanılması
- Klinik tanı iddiası

## Escalation Rule

BAŞARISIZ sonuç → pre-submission-gate skill'ini tetikle, teslim bloklansın.

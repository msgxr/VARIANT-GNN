---
name: official-source-guardian
description: Use when any competition rule, deadline, metric, or requirement needs verification. Enforces the TEKNOFEST 2026 official source policy — rejects third-party sources, marks unverifiable claims as UNVERIFIED, and ensures all competition decisions trace to the official specification. Activate when a rule is cited, when a source is unclear, or when a decision depends on competition requirements.
---

# Skill: official-source-guardian

## Purpose

Tüm yarışma kuralı kararlarının yalnızca resmi TEKNOFEST 2026 kaynaklarına dayandığını denetlemek.

## Official Source Boundary

**Kabul edilen kaynaklar (öncelik sırasıyla):**
1. TEKNOFEST 2026 Türkçe Şartname v4
2. TEKNOFEST 2026 PDR Şablonu (Üniversite ve Üzeri)
3. TEKNOFEST 2026 PSR Şablonu (Üniversite ve Üzeri)
4. Resmi TEKNOFEST ana yarışma sayfası
5. Resmi KYS / duyuru / yarışma grubu bilgilendirmesi

**Kesinlikle reddedilen kaynaklar:**
- 2024 veya önceki yıl şartnameleri (kural değişmiş olabilir)
- Blog, forum, sosyal medya, kişisel yorum
- Gayriresmi özet, üçüncü taraf doküman

## Inputs

- Yarışma kuralına ilişkin herhangi bir soru, iddia veya karar noktası

## Outputs

- Her bilgi için: kaynak + bölüm numarası + doğrulanmış/UNVERIFIED etiketi
- Reddedilen kaynakların listesi
- "OFFICIAL SOURCE REQUIRED" durumunda işin durdurulması ve not düşülmesi

## Hard Rules

1. Kaynak resmi değilse bilgi kullanılmaz.
2. Doğrulanamayan bilgi UNVERIFIED olarak işaretlenir — kesin kural gibi sunulmaz.
3. Şartname, şablon, web sayfası çelişirse: Şartname > Şablon > Web sayfası.
4. Kural uydurulmaz. Emin olunmayan her şey durdurulur.

## Step-by-Step Procedure

1. İddia edilen kuralı belirle.
2. Resmi kaynaklar listesini kontrol et (OFFICIAL_REFERENCES.md).
3. İlgili şartname bölümünü bul.
4. Bilgiyi doğrula veya UNVERIFIED olarak işaretle.
5. Kaynak + bölüm numarasını belirterek yanıt ver.
6. Üçüncü taraf kaynak tespitinde "OFFICIAL SOURCE REQUIRED" yaz ve dur.

## Validation Checklist

- [ ] Bilginin kaynağı resmi TEKNOFEST belgesi mi?
- [ ] Kaynak 2026 yılına mı ait?
- [ ] Üçüncü taraf kaynak kullanılmadı mı?
- [ ] Doğrulanamayan bilgi UNVERIFIED mi?
- [ ] Şartname ile şablon çelişmiyorsa öncelik doğru mu?

## Failure Conditions

- Üçüncü taraf kaynak kullanılması
- UNVERIFIED bilginin doğrulanmış gibi sunulması
- 2024 şartname maddelerinin 2026 kuralı gibi yazılması

## Escalation Rule

Resmi kaynakta açıkça yazılmayan bir kural kesin hüküm gerektiriyorsa → işi durdur, "OFFICIAL SOURCE REQUIRED" yaz, kullanıcıya sor.

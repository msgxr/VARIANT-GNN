---
name: official-source-guardian
description: Use when any competition rule, deadline, metric, or requirement needs verification. Enforces the TEKNOFEST 2026 official source policy — rejects third-party sources, marks unverifiable claims as UNVERIFIED, and ensures all competition decisions trace to the official specification. Activate when a rule is cited, when a source is unclear, or when a decision depends on competition requirements.
---

# Official Source Guardian — VARIANT-GNN

When this skill is active, every competition-related claim must be traceable to an official TEKNOFEST 2026 document. No exceptions. No approximations. No memory of previous years.

## Source Priority Hierarchy

```
TIER 1 (Binding)   : TEKNOFEST 2026 Türkçe Şartname v4
TIER 2 (Binding)   : TEKNOFEST 2026 PDR Şablonu — Üniversite ve Üzeri
TIER 2 (Binding)   : TEKNOFEST 2026 PSR Şablonu — Üniversite ve Üzeri
TIER 3 (Reference) : teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/
TIER 4 (Reference) : KYS / resmi TEKNOFEST duyurusu / yarışma grubu

REJECTED (Zero Trust):
  - 2024 veya önceki yıl şartnameleri
  - Blog, forum, sosyal medya, kişisel yorum
  - Gayriresmi özet, üçüncü taraf doküman
  - "Geçen yıl böyleydi" hafızası
```

## Verified Facts (2026 — Confirmed from Official Source)

| Fact | Value | Source |
|---|---|---|
| Yarışma | TEKNOFEST 2026 Sağlıkta Yapay Zeka | Ana sayfa |
| Kategori | Üniversite ve Üzeri | Şartname |
| Görev | Patojenik/Benign binary sınıflandırma | Şartname |
| Birincil metrik | Binary F1 Score, pos_label=1 | Şartname §7.3 |
| PDR teslim | 29 Haziran 2026, 17:00 | Takvim |
| Veri dağıtımı | 5 Mayıs 2026 | Takvim |
| Final | Ağustos–Eylül 2026, Şanlıurfa | Takvim |
| Ödül 1. | ₺180.000 | Ana sayfa |
| Takım büyüklüğü | 2–5 kişi (danışman hariç) | Şartname |
| Gizlilik taahhütü | Kurumsal Gizlilik Taahhütü zorunlu | Şartname |
| KVKK | Zorunlu uyum | Şartname |
| PSR teslim | 25 Mart 2026, 17:00 (TAMAMLANDI) | Takvim |

## UNVERIFIED Items (Do Not Present as Fact)

| Item | Why Unverified |
|---|---|
| PDR sayfa limiti (exact) | Şablon DOCX içinde — doğrulanmadı |
| Puanlama ağırlıkları (% breakdown) | Şartname §7 tam okuma gerekiyor |
| Teslim formatı (PDF/DOCX) | Sistem duyurusundan gelecek |
| Final sözlü sunum ağırlığı | Şartname §7 |
| Mezuniyet tarih kısıtı | Şartname §3 |

## Decision Protocol

```
Step 1: Identify the claim or rule being asserted.
Step 2: Map to Source Hierarchy above.
Step 3a: Found in Tier 1–2 → state fact with section reference.
Step 3b: Found in Tier 3–4 only → state as "reference only, not binding."
Step 3c: Not found anywhere → mark UNVERIFIED, do not assert.
Step 3d: Third-party source detected → reject, write "OFFICIAL SOURCE REQUIRED."
Step 4: If rule determination is required and source is insufficient → STOP, ask user.
```

## Response Format

When verifying a claim:
```
CLAIM: [what was asserted]
SOURCE: [TEKNOFEST 2026 Şartname §X.X / UNVERIFIED]
STATUS: [VERIFIED / UNVERIFIED / REJECTED — third party]
CONFIDENCE: [HIGH / MEDIUM / LOW]
NOTE: [if any caveat]
```

## Hard Rules

1. "2024'te böyleydi" → automatic rejection, year mismatch
2. Doğrulanamayan bilgi → UNVERIFIED, not assumed true
3. Tier 1 vs Tier 3 conflict → Tier 1 wins, always
4. User asks for rule not in any source → "OFFICIAL SOURCE REQUIRED — check şartname §[relevant section]"
5. Never invent a rule, never round-trip from memory alone

## Cross-References

- Verified facts → `.claude/OFFICIAL_REFERENCES.md`
- Uncertainty log → `.claude/context/07_uncertainty_log.md`
- Application to audit → `competition-compliance-auditor`

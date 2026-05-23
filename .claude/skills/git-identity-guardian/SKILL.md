---
name: git-identity-guardian
description: Use before every git push. Ensures commits are attributed to the correct team member — never to Claude, Bot, AI, or any automation name. Because two team members (msgxr and Şeyma) push from different machines, always asks who is pushing before configuring git identity. Activate automatically before any git push, or when user says "commit at", "push yap", "gönder".
---

# Skill: git-identity-guardian

## Purpose

Her git push öncesinde commit kimliğinin doğru takım üyesine ait olduğunu garanti etmek.

## Official Source Boundary

Bu skill yarışma şartnamesiyle değil, proje iç kimlik protokolüyle çalışır.  
Referans: .claude/PROJECT_RULES.md §7 ve .claude/context/06_team_and_process_rules.md

## Inputs

- Git push isteği (herhangi bir branch, herhangi bir commit)

## Outputs

- Doğrulanmış git log çıktısı (Author ve Committer)
- Push izni (ONAYLANDI / ENGELLENDI)

## Hard Rules

1. Push öncesi **her zaman** kimin push atacağı sorulur.
2. Author veya Committer satırında şunlar **kesinlikle görünemez:**
   - Claude / claude
   - Bot / bot
   - AI / ai / artificial intelligence
   - Automation / automation
   - Assistant
   - Herhangi bir otomasyon / araç adı
3. Author yanlışsa push yapılmaz.
4. Düzeltme yapılmadan push yapılmaz.

## Takım Kimlik Tablosu

| Kişi | git user.name | git user.email |
|---|---|---|
| Kullanıcı | `msgxr` | `mgun345@icloud.com` |
| Şeyma | `cebi101` | `seymanurcebi6@gmail.com` |

## Step-by-Step Procedure

**Adım 1 — Push Sahibini Sor**
```
"Bu commit hangi takım üyesi adına atılacak?
(1) msgxr — mgun345@icloud.com
(2) cebi101 (Şeyma) — seymanurcebi6@gmail.com"
```

**Adım 2 — Git Config Ayarla**
```bash
git config user.name "<seçilen isim>"
git config user.email "<seçilen email>"
```

**Adım 3 — Kontrol Et**
```bash
git config user.name
git config user.email
```

**Adım 4 — Commit Yap**
```bash
git commit -m "<mesaj>"
```

**Adım 5 — Author Doğrula**
```bash
git log -1 --pretty=fuller
```

Beklenen çıktı:
```
Author:     msgxr <mgun345@icloud.com>
AuthorDate: ...
Commit:     msgxr <mgun345@icloud.com>
CommitDate: ...
```

**Adım 6 — Author Yanlışsa Düzelt**
```bash
git commit --amend --reset-author --no-edit
git log -1 --pretty=fuller  # tekrar kontrol
```

**Adım 7 — Her Şey Doğruysa Push**
```bash
git push
```

## Commit Mesajı Standardı

**İyi örnekler:**
- `feat: PDR §2.2 GATv2Conv justification strengthened`
- `fix: PDR kaynakça numaraları düzeltildi`
- `add: TEKNOFEST 2026 compliance skills`
- `refactor: threshold analysis figure paths updated`

**Yasak ifadeler:**
- `update`, `fix`, `changes`, `bot update`, `claude changes`, `ai commit`

## Validation Checklist

- [ ] Push sahibi soruldu mu?
- [ ] git config user.name doğru kişi mi?
- [ ] git config user.email doğru kişi mi?
- [ ] git log -1 --pretty=fuller çalıştırıldı mı?
- [ ] Author satırı doğru mu?
- [ ] Committer satırı doğru mu?
- [ ] Claude / Bot / AI adı görünmüyor mu?

## Failure Conditions

- Author veya Committer yanlış → push kesinlikle yapılmaz
- Push sahibi sorulmadan devam edilmesi → bu skill prosedürü ihlal edilmiş

## Escalation Rule

Author doğrulanamıyorsa veya kimlik belirsizse → push durdurulur, kullanıcıya bilgi verilir.

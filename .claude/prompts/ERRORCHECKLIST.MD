# /ERRORCHECKLIST — VARIANT-GNN Hata Denetimi

Bu slash command `error-checker` skill'ini tetikler ve `.claude/core/ERRORCHECKLIST.md` referansıyla Domain A–J hata taraması yapar.

**Tam kontrol listesi için bkz.:** `.claude/core/ERRORCHECKLIST.md`  
**Skill:** `.claude/skills/error-checker/SKILL.md`

---

Kullanım: `/ERRORCHECKLIST` yazınca Claude şunları yapar:
1. `error-checker` skill'ini yükler
2. Domain A–J sırasıyla tarar (Kategori → Metrik → Leakage → Gizlilik → Kaynak → Rapor → Etik → Reproducibility → Kod → Git)
3. Her domain için PASS / RISK / FAIL verir
4. Açık bulguları öncelik sırasıyla listeler (P0 → P1 → P2)

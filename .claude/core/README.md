# .claude — VARIANT-GNN Claude Agent Operating System (CAPOS v2.0)

**Proje:** VARIANT-GNN — Missense Varyant Patojenite Tahmini  
**Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**Takım:** XYRA3 | Takım ID: 909249 | Başvuru ID: 4865399  
**Git Kimliği (Bu PC):** msgxr <mgun345@icloud.com>

---

## Dizin Yapısı

```
.claude/
├── core/                        ← Proje kuralları ve referans merkezi
│   ├── README.md                ← Bu dosya — ana dizin rehberi
│   ├── PROJECT_RULES.md         ← Değiştirilemez kurallar (§1-8)
│   ├── OFFICIAL_REFERENCES.md   ← Doğrulanmış TEKNOFEST 2026 kaynakları
│   ├── ERRORCHECKLIST.md        ← Risk ve hata kontrol listesi (Domain A-J)
│   ├── QUALITY_GATES.md         ← Görev sınıflandırma ve kalite kapıları
│   └── MASTER_PLAYBOOK.md       ← Tek sayfa misyon kontrol merkezi ★
├── settings.local.json          ← Claude Code sistem dosyası
├── agents/                      ← 9 uzman ajan tanımı
│   ├── orchestrator/AGENT.md
│   ├── scientist/AGENT.md
│   ├── debugger/AGENT.md
│   ├── architect/AGENT.md
│   ├── documentalist/AGENT.md
│   ├── verifier/AGENT.md
│   ├── jury-adversary/AGENT.md
│   ├── sentinel/AGENT.md
│   └── meta-governor/AGENT.md
├── context/                     ← Yarışma ve proje bağlamı
│   ├── 01_official_source_map.md
│   ├── 02_competition_summary.md
│   ├── 03_university_category_rules.md
│   ├── 04_report_requirements.md
│   ├── 05_data_and_metric_rules.md
│   ├── 06_team_and_process_rules.md
│   ├── 07_uncertainty_log.md
│   └── archive/                 ← Eski bağlam dosyaları (referans)
├── prompts/                     ← Yeniden kullanılabilir prompt şablonları
└── skills/                      ← 16 mission skill tanımı
    ├── official-source-guardian/
    ├── competition-compliance-auditor/
    ├── psr-editor/
    ├── pdr-editor/
    ├── report-template-checker/
    ├── experiment-review/
    ├── data-metric-guardian/
    ├── reproducibility/
    ├── jury-sim/
    ├── git-identity-guardian/
    ├── error-checker/
    ├── mission-readiness/
    ├── pre-submission-gate/
    ├── variant-gnn-review/
    ├── code-change-verifier/
    └── meta-audit/
```

---

## Tek Yetkili Kaynak Hiyerarşisi

```
TIER 1 (Binding)   : TEKNOFEST 2026 Türkçe Şartname v4
TIER 2 (Binding)   : PDR Şablonu — Üniversite ve Üzeri
TIER 2 (Binding)   : PSR Şablonu — Üniversite ve Üzeri
TIER 3 (Reference) : teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/
TIER 4 (Reference) : KYS / resmi TEKNOFEST duyurusu

REJECTED           : 2024 şartname, blog, forum, sosyal medya, gayriresmi özet
```

Doğrulanamayan yarışma bilgisi → **UNVERIFIED** işareti. Hiçbir zaman kesin kural gibi sunulmaz.

---

## Skills — Hızlı Referans

| Skill | Tetikleyici |
|---|---|
| `official-source-guardian` | Yarışma kuralı sorgulandığında |
| `competition-compliance-auditor` | "Şartnameye uygun mu?", "Eksikler neler?" |
| `psr-editor` | PSR analizi, hakem skoru, PSR↔PDR farkı |
| `pdr-editor` | PDR içerik ve düzenleme |
| `report-template-checker` | Şablon uyumu, sayfa limiti, format |
| `experiment-review` | Deney sonuçları analizi |
| `data-metric-guardian` | Metrik doğruluğu, data leakage |
| `reproducibility` | Jüri tekrar çalıştırma testi |
| `jury-sim` | Jüri soru simülasyonu |
| `git-identity-guardian` | Her git push öncesi |
| `error-checker` | Hata tespiti |
| `mission-readiness` | Teslim hazırlık değerlendirmesi |
| `pre-submission-gate` | GO/NO-GO final kontrol |
| `variant-gnn-review` | Genel proje denetimi |
| `code-change-verifier` | Kod değişikliği cross-impact |
| `meta-audit` | CAPOS altyapı denetimi |

---

## Hızlı Durum (2026-05-24)

| Görev | Durum |
|---|---|
| PSR Teslimi | ✅ 25 Mart 2026 — 93/100 |
| Model Eğitimi | ✅ Test F1=0.833, CV F1=0.8936±0.0004 (sızıntısız; jüri dengeli F1=0.6063) |
| PDR Hazırlığı | ✅ Tüm BUG'lar kapalı; sayılar RESULTS_CANONICAL.json ile tutarlı |
| PDR Deadline | ⏰ 29 Haziran 2026, 17:00 |

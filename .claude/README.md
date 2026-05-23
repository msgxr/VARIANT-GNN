# .claude — VARIANT-GNN Claude Agent Operating System

**Proje:** VARIANT-GNN — Missense Varyant Patojenite Tahmini  
**Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**Takım:** XYRA3 | Takım ID: 909249 | Başvuru ID: 4865399  
**Git Kimliği:** msgxr <mgun345@icloud.com>

---

## Dizin Yapısı

```
.claude/
├── README.md                    ← Bu dosya
├── PROJECT_RULES.md             ← Projenin değişmez kuralları
├── OFFICIAL_REFERENCES.md       ← Resmi TEKNOFEST 2026 kaynakları
├── ERRORCHECKLIST.md            ← Risk ve hata kontrol listesi
├── QUALITY_GATES.md             ← Görev sınıflandırma ve kalite geçitleri
├── settings.local.json          ← Claude Code ayarları
├── agents/                      ← 9 uzman ajan tanımı
├── context/                     ← Yarışma bağlamı ve referans dosyaları
├── prompts/                     ← Yeniden kullanılabilir prompt şablonları
└── skills/                      ← 16 mission skill tanımı
```

## Tek Yetkili Kaynak

**Tüm yarışma kararları yalnızca şu resmi kaynaklara dayanır:**

1. TEKNOFEST 2026 Şartnamesi (Türkçe v4): cdn.teknofest.org/...
2. PDR Şablonu (Üniversite ve Üzeri): cdn.teknofest.org/...
3. PSR Şablonu (Üniversite ve Üzeri): cdn.teknofest.org/...
4. Ana sayfa: teknofest.org/tr/yarismalar/saglikta-yapay-zeka-yarismasi/

Üçüncü taraf kaynak kullanılmaz. Doğrulanamayan bilgi UNVERIFIED işaretlenir.

## Skills — Hızlı Referans

| Skill | Ne Zaman |
|---|---|
| `official-source-guardian` | Yarışma kuralı doğrulaması gerektiğinde |
| `competition-compliance-auditor` | Genel şartname uyum denetimi |
| `psr-editor` | PSR içerik ve format kontrolü |
| `pdr-editor` | PDR içerik ve format kontrolü |
| `report-template-checker` | Şablon uyum kontrolü |
| `experiment-review` | Deney sonuçları analizi |
| `data-metric-guardian` | Veri/metrik/etiket doğrulaması |
| `reproducibility` | Jüri tekrar çalıştırma testi |
| `jury-sim` | Jüri soru simülasyonu |
| `git-identity-guardian` | Commit kimlik doğrulaması |
| `error-checker` | Hata tespiti ve düzeltme |
| `mission-readiness` | Teslim hazırlık değerlendirmesi |
| `pre-submission-gate` | GO/NO-GO final kontrol |
| `variant-gnn-review` | Genel proje denetimi |
| `code-change-verifier` | Kod değişikliği etkisi |
| `meta-audit` | Claude altyapı denetimi |

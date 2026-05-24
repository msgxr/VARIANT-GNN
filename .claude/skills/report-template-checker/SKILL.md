---
name: report-template-checker
description: Use when verifying that the PDR matches the official 2026 TEKNOFEST template exactly. Checks section structure, scoring weights, format rules (Aptos font, 10-page hard limit), mandatory metrics per panel, all known PDR bugs (BUG-01 through BUG-09), figure paths, citation format, and ethics declaration. Activate before any PDR edit session, before submission, or when "şablona uygun mu?" is asked. Contains the definitive PDR format specification verified from the official template DOCX.
---

# Skill: report-template-checker — PDR Template Compliance

## Official Source Boundary

Primary: TEKNOFEST 2026 PDR Şablonu (Üniversite ve Üzeri)  
PDR URL: cdn.teknofest.org/.../2026_PDR_Şablon_Universite_TR_bCw49.docx  
PSR URL: cdn.teknofest.org/.../2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx  
Rejection: Lise/EKG templates, 2024 templates, third-party PDR guides, blog summaries

---

## PDR Format Specification (VERIFIED from Official Template DOCX)

| Parameter | Required Value |
|---|---|
| Body font | **Aptos** |
| Body size | **12 pt** |
| Heading font | **Aptos** |
| Heading size | **14 pt** |
| Line spacing | **1.15** |
| Alignment | **İki tarafa yaslı (justified)** |
| Top margin | **2.8 cm** |
| Other margins (left/right/bottom) | **2.5 cm** |
| **Page limit** | **≤10 pages** (kapak + içindekiler EXCLUDED from count) |
| Page numbering | Sequential, starting after cover/TOC |
| Required pages | Cover page + Table of contents (both mandatory, not in 10-page count) |

**⚠️ HARD LIMIT: Reports exceeding 10 content pages will NOT be evaluated by jury.**

---

## PDR Section Structure & Scoring (VERIFIED from Official Template)

| # | Section | Points | Content Priority |
|---|---|---|---|
| 1 | Giriş (Introduction) | 10 | Literature review, problem definition |
| 2 | Yöntem (Methodology) | 25 | Pipeline, explainability, threshold process |
| 3 | Bulgular (Results) | **30** | F1 + MCC + PR-AUC per panel — largest section |
| 4 | Sonuç (Conclusion) | 25 | FP/FN analysis, limitations, jury challenges |
| 5 | Kaynakça ve Rapor Düzeni | 10 | IEEE format, visual quality |
| | **TOTAL** | **100** | |

---

## Section Content Requirements (from Official Template)

### §1 Giriş (10 pts) — Checklist
- [ ] Missense variant pathogenicity problem clearly defined
- [ ] Clinical and genomic significance of the problem stated
- [ ] Class imbalance problem explained with its effect on model performance
- [ ] **5–10 recent international papers reviewed**, each with: methods used, data sources (ClinVar, gnomAD), reported metrics

### §2 Yöntem (25 pts) — Checklist
- [ ] Asymmetric and encoded genomic data structure described
- [ ] Missing value imputation method stated (median imputation)
- [ ] Outlier management stated (RobustScaler rationale)
- [ ] External data: disclosed (none used beyond competition data)
- [ ] Feature engineering: stated (SelectKBest k=35, AutoEncoder 43→16)
- [ ] Algorithms tried and selection rationale (XGB/LGBM/GATv2GNN/DNN)
- [ ] Hyperparameter search method and search space defined
- [ ] Cross-validation approach: Stratified 5-Fold, random_state=42
- [ ] Overfitting prevention: Dropout, BatchNorm, early stopping
- [ ] **Explainability methods**: SHAP waterfall + GNNExplainer + panel feature table
- [ ] **Decision threshold determination**: calibration set process, panel-specific values

### §3 Bulgular (30 pts) — MANDATORY METRICS (from Template)

> Official template quote: *"Başarım ölçütü olarak en azından F1 skoru, Matthews korelasyon katsayısı ve kesinlik-duyarlılık eğrisi altında kalan alan ölçütleri raporlanmalıdır."*

- [ ] **F1 Score** — all 4 panels separately (MANDATORY)
- [ ] **MCC (Matthews Correlation Coefficient)** — all 4 panels separately (MANDATORY)
- [ ] **PR-AUC (Precision-Recall AUC)** — all 4 panels separately (MANDATORY)
- [ ] Confusion matrix — per panel
- [ ] Comparison charts / visualizations
- [ ] **Threshold analysis**: multiple threshold test results + optimal threshold per panel
- [ ] All panels labeled with official names: **MASTER / KANSER / PAH / CFTR**

### §4 Sonuç (25 pts) — Checklist
- [ ] Results interpreted + overall contribution stated
- [ ] Successful cases AND failure cases analyzed
- [ ] **False positives and false negatives analyzed in detail** (template requires)
- [ ] Feature groups where model struggles identified
- [ ] Error clinical/biological meaning interpreted (with research disclaimer)
- [ ] Work positioned in literature
- [ ] "Challenges that may be encountered in the final stage" discussed

### §5 Kaynakça (10 pts) — Checklist
- [ ] **IEEE format** for all references
- [ ] All in-text citations [N] have matching reference entry
- [ ] All reference entries cited in text
- [ ] Graphs and tables sufficient, clear, labeled
- [ ] Academic Turkish writing maintained

---

## Known PDR Bugs (Must Fix Before 29.06.2026 Submission)

| ID | Location | Bug Found | Correct Value | Status |
|---|---|---|---|---|
| BUG-01 | §1.2 Kaynakça | REVEL citation is [3] | Should be **[2]** | ❌ Open |
| BUG-02 | §1.2 Kaynakça | EVE citation is [5] | Should be **[9]** | ❌ Open |
| BUG-03 | §1.2 Kaynakça | GATv2 citation is [7] | Should be **[8]** | ❌ Open |
| BUG-04 | §3.2 Threshold | θ = 0.01 written in text | θ = **0.241** correct | ❌ Open |
| BUG-05 | §3.1 Figure paths | `reports/roc_curves.png` | `reports/figures/pdr/05_roc_curves.png` | ❌ Open |
| BUG-06 | §3.1 Şekil 2-5 | All figure paths dead | Verify each path under `reports/figures/pdr/` | ❌ Open |
| BUG-07 | Header/Cover | Training date: 15 Mayıs | **20 Mayıs 2026** | ❌ Open |
| BUG-08 | §4.2 equivalent | MCC discrepancy PSR vs PDR | Explicit comparison table + explanation required | ❌ Open |
| BUG-09 | Yöntem §2 | PSR says SAGEConv | PDR must say **GATv2Conv / VariantGATv2GNN** | ❌ Open |

**All 9 bugs must be CLOSED before PDR submission. Any open bug = NO-GO.**

---

## Compliance Audit Procedure (Step-by-Step)

Activate when: user asks "şablona uygun mu?", "PDR'yi kontrol et", or before submission.

### Step 1 — Format Verification
```
□ Font: Aptos? (not Times New Roman, Calibri, Arial)
□ Body: 12pt?  Headings: 14pt?
□ Line spacing: 1.15?
□ Alignment: justified (both sides)?
□ Top margin: 2.8cm?  Other margins: 2.5cm?
□ Page count: content pages ≤10? (excluding cover + TOC)
□ Cover page present?
□ Table of contents present?
□ Sequential page numbers?
```

### Step 2 — Section Structure Verification
```
□ §1 Giriş — present? 5–10 papers cited?
□ §2 Yöntem — all subsections present? Threshold process documented?
□ §3 Bulgular — F1 + MCC + PR-AUC for ALL 4 panels?
□ §4 Sonuç — FP/FN analysis present?
□ §5 Kaynakça — IEEE format?
□ Cover page — title, team, date correct?
□ Ethics declaration — present?
```

### Step 3 — Bug Verification (Run BUG-01 through BUG-09)
```
For each bug:
  Read the relevant section of PDR document
  □ Bug still present? → FAIL — must fix
  □ Bug resolved? → PASS — mark closed
```

### Step 4 — Figure & Reference Integrity
```
□ Every figure path in document → verify file exists on disk
  Command: ls reports/figures/pdr/ (should match all paths referenced)
□ Every [N] citation → has matching entry in Kaynakça
□ Every Kaynakça entry → cited at least once in text
□ IEEE format: Author(s), "Title," Journal/Conf, vol., no., pp., year. DOI.
```

### Step 5 — Panel Completeness Check
```
□ MASTER: F1=0.8872, MCC=0.507, PR-AUC=? — all reported?
□ KANSER: F1=0.8960, MCC=0.649, PR-AUC=? — all reported?
□ PAH: F1=0.9556, MCC=0.556, PR-AUC=? — all reported?
□ CFTR: F1=0.9524, MCC=0.674, PR-AUC=? — all reported?
□ Panel-specific thresholds: General=0.241, KANSER=0.281, PAH=0.138, CFTR=0.108
```

### Step 6 — PSR Bridge Check
```
□ GATv2Conv name: correct in PDR throughout? (no SAGEConv anywhere)
□ MCC discrepancy (pilot 0.892 vs real 0.536): explained with comparison table?
□ §4.4 Explainability gap: SHAP waterfall + GNNExplainer present?
□ §4.5 Technical Evolution gap: experiment log + ablation present?
□ §5.1 Architecture: 5×4 model comparison table present?
□ §5.4 Reproducibility: concrete command sequence shown?
```

### Step 7 — Final GO/NO-GO Decision

**Automatic NO-GO conditions (any single item → block submission):**
```
- Any BUG-01 through BUG-09 still open
- Missing mandatory section (§1 through §5)
- Page count > 10 content pages
- Missing metric (F1/MCC/PR-AUC) for any panel
- Font ≠ Aptos
- Ethics declaration absent
- Lise template used
```

**RISK conditions (flag for review, not automatic NO-GO):**
```
- Figure path uncertain (check disk)
- Citation format partially non-IEEE
- PSR bridge incomplete (requires psr-editor intervention)
```

**GO:** All NO-GO conditions clear + all RISK items documented and accepted.

---

## Output Format

```
## PDR TEMPLATE COMPLIANCE AUDIT
Date: [YYYY-MM-DD]
Skill: report-template-checker

### Format Check: [PASS/FAIL]
[Findings — list any deviation from Aptos/12pt/1.15/10-page spec]

### Section Structure: [PASS/RISK/FAIL]
[Per-section status with specific gaps]

### Known Bugs: [X/9 resolved]
BUG-01: [OPEN/CLOSED]
BUG-02: [OPEN/CLOSED]
... (all 9)

### Figure Paths: [PASS/FAIL]
[List any dead paths]

### Panel Completeness: [PASS/FAIL]
[MASTER/KANSER/PAH/CFTR — each metric status]

### PSR Bridge: [PASS/RISK/FAIL]
[Discrepancy status + gap coverage]

### VERDICT: [GO / RISK-GO / NO-GO]
Priority actions (ordered by blocking severity):
1. [Most critical]
2. ...
```

---

## Hard Rules

1. Lise/EKG template → instant NO-GO (wrong category, cannot be submitted)
2. Page count > 10 → instant NO-GO (report will not be evaluated by jury)
3. Font ≠ Aptos → must fix before submission
4. Any BUG-01 through BUG-09 open → NO-GO until resolved
5. Missing metric for any panel → FAIL (§3 template requires all 4 panels)
6. Never approve a NO-GO report for submission — escalate to `pre-submission-gate`
7. Never assert a figure path is valid without checking the file exists on disk

---

## Escalation Map

| Trigger | Escalate To |
|---|---|
| BUG found, PDR text needs edit | `pdr-editor` |
| PSR discrepancy needs explanation draft | `psr-editor` |
| Metric value inconsistent | `data-metric-guardian` |
| Full pre-submission gate | `competition-compliance-auditor` → `pre-submission-gate` |
| Figure file missing on disk | `code-change-verifier` (find correct path) |

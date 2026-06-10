---
name: pdr-editor
description: Use when drafting, reviewing, or editing the VARIANT-GNN Project Detail Report (PDR). Checks compliance with the official 2026 PDR template: required metrics (F1+MCC+PR-AUC), panel names (MASTER/KANSER/CFTR/PAH), 10-page limit, Aptos font, and PSR weak points that must be strengthened.
---

# PDR Editor — VARIANT-GNN Project Detail Report Compliance

## Official Source Boundary

Yalnızca TEKNOFEST 2026 PDR Şablonu (Üniversite ve Üzeri) esas alınır.  
URL: cdn.teknofest.org/.../2026_PDR_Şablon_Universite_TR_bCw49.docx  
Üçüncü taraf kaynak, 2024 şablonu ve lise şablonu kabul edilmez.  
Doğrulanamayan şablon gereksinimleri UNVERIFIED olarak işaretlenir.

When this skill is active, review or produce PDR content strictly against the official 2026 PDR template for the Üniversite ve Üzeri Seviyesi.

## Official PDR Structure (100 points total)

| Section | Points | Priority |
|---|---|---|
| 1. Introduction | 10 | Medium |
| 2. Method | 25 | High |
| 3. Results | 30 | **Highest** |
| 4. Conclusion | 25 | High |
| 5. References & Layout | 10 | Medium |

## Mandatory Format Rules (violations = disqualification)

- Font: **Aptos** (not Arial, not Times New Roman)
- Body: 12pt | Headings: 14pt
- Line spacing: 1.15
- Alignment: Justified (two-sided)
- Margins: Top 2.8 cm | Bottom/Left/Right: 2.5 cm
- **Page limit: 10 pages** (cover + table of contents excluded)
- **Reports exceeding 10 pages are not evaluated**
- References: IEEE format
- Required: Cover page, Table of contents, page numbers

## Section 1 — Introduction (10 pts): Checklist

- [ ] Missense variant pathogenicity prediction problem defined clearly
- [ ] Clinical and genomic importance emphasized
- [ ] Class imbalance and its effect on model performance addressed
- [ ] 5–10 recent international studies reviewed with: methods, data sources (ClinVar, gnomAD), reported metrics

## Section 2 — Method (25 pts): Checklist

**Data engineering:**
- [ ] Asymmetric and anonymized genomic dataset structure described
- [ ] Missing value imputation method and split timing stated
- [ ] Outlier handling described
- [ ] External data sources documented
- [ ] Feature engineering steps (if any)

**Model development:**
- [ ] Algorithms tried + selection justification (with experimental evidence)
- [ ] Hyperparameter optimization method, search space, effect on generalization
- [ ] Cross-validation strategy
- [ ] Overfitting prevention measures
- [ ] **Explainability methods** (PSR §4.4 scored 3.33/5 — must be strengthened here)
  - Individual-level SHAP examples (waterfall plot for ≥1 pathogenic + ≥1 benign)
  - GNNExplainer subgraph visualization
  - Panel-based feature group contribution table (not just one chart)
  - LIME/SHAP agreement, quantified (canonical: Spearman ρ=0.89 over 150 samples)
- [ ] **Decision threshold determination** (PSR §4.4 weak — justify global θ=0.8415 threshold)

## Section 3 — Results (30 pts): Checklist

### Required Metrics — per PDR template (verbatim):
> "At minimum, F1 score, Matthews correlation coefficient, and area under the precision-recall curve must be reported."

- [ ] **F1 Score** — mandatory ✓
- [ ] **Matthews Correlation Coefficient (MCC)** — mandatory ✓
- [ ] **Area Under Precision-Recall Curve (PR-AUC)** — mandatory ✓ (not computed in PSR, must be added)
- [ ] **Confusion Matrix** + comparison charts — mandatory ✓
- [ ] Different threshold results + optimal threshold value — mandatory ✓

### Panel-Based Reporting — PDR template panel names:
| PDR Name | Meaning |
|---|---|
| **MASTER** | General Dataset |
| **KANSER** | Hereditary Cancer Panel |
| **CFTR** | Cystic Fibrosis Panel |
| **PAH** | Phenylketonuria Panel |

All four panels must be reported **separately**.

### Additional metrics (from PSR, include in PDR):
- [ ] ROC-AUC (PSR Table 3 primary metric)
- [ ] Brier Score (calibration quality)

### Visual requirements:
- [ ] Precision-Recall curve (panel-based)
- [ ] Confusion matrix visualization (each panel)
- [ ] Comparison chart (ensemble vs. baseline models)

## Section 4 — Conclusion (25 pts): Checklist

- [ ] Results interpreted, study's contribution stated
- [ ] When model succeeds and where it fails analyzed
- [ ] **False positive and false negative analysis in detail**
- [ ] Which feature groups the model struggles with identified
- [ ] **Clinical or biological meaning of errors** (within ethical boundary)
- [ ] Study's position in the literature
- [ ] Challenges expected in the final stage of the competition

## PSR Weak Points → PDR Strengthening Plan

| PSR Section | PSR Score | PDR Action |
|---|---|---|
| §4.4 Explainability | 3.33/5 🔴 | Individual SHAP + GNNExplainer + panel-wise table |
| §4.5 Technical Evolution | 3.33/5 🔴 | Experiment log table + ablation study (Method §2) |
| §5.1 Algorithm Justification | 4/5 🟠 | 5-model × 4-panel comparison table (Results §3) |
| §5.4 Compute Resources | 4.33/5 🟠 | Concrete commands + CPU-only test proof (Method §2) |

## Ethical Boundary (mandatory sentence)

"Bu çalışma TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında gerçekleştirilmiş olup geliştirilen model ve çıktılar yalnızca araştırma ve eğitim amaçlıdır; klinik tanı veya tıbbi karar desteği amacıyla kullanılamaz."

## Output Format

For each PDR section reviewed:
1. **COMPLIANCE** — [Full / Partial / Missing]
2. **MISSING ITEMS** — specific list
3. **PSR DELTA** — what was weak in PSR and how PDR addresses it
4. **PAGE BUDGET** — estimated page count for this section
5. **CORRECTIVE TEXT** — directly insertable draft text

## Prohibitions

- Do not suggest page count exceeding 10 total
- Do not approve "ROC-AUC only" results (MCC and PR-AUC also mandatory)
- Do not accept ERRORCHECKLIST.MD-style general advice — be PDR-template specific
- Do not write clinical diagnosis claims
- Do not use panel names other than MASTER/KANSER/CFTR/PAH in the PDR context

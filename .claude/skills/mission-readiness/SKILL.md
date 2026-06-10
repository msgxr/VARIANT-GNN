---
name: mission-readiness
description: Use when assessing VARIANT-GNN competition delivery readiness — "are we ready to submit/present?" Produces a traffic-light dashboard of: what's ready, what's at risk, what's blocking. Covers PDR deadline readiness, jury reproducibility, final presentation readiness. Activate when user asks "teslime hazır mıyız?", "kritik eksikler neler?", "ne kaldı?", or any delivery timeline question.
---

# Mission Readiness Assessment — VARIANT-GNN

When this skill is active, assess the competition delivery readiness of VARIANT-GNN as of today's date. Produce a structured, honest, traffic-light dashboard. Do not soften findings.

## Readiness Assessment Framework

### Phase Determination
First determine current competition phase:
- **PDR Prep Phase:** Before 29.06.2026 — PDR is the primary deliverable
- **Finals Prep Phase:** After PDR results (20.07.2026) — jury defense is primary
- **Competition Phase:** August–September 2026 — live performance is primary

### Readiness Dimensions

#### Dimension 1: Technical Readiness
```
[ ] Pipeline runs end-to-end without errors
[ ] Training produces F1 ≈ 0.8367 with seed=42
[ ] Inference produces submission.csv in correct format
[ ] CPU-only execution works (< threshold time)
[ ] All 4 panels produce results
[ ] Model checkpoints saved and loadable
[ ] Config files complete (primary: configs/pdr.yaml)
[ ] Requirements.txt accurate and pinned (torch==2.2.1, torch-geometric==2.5.3, streamlit==1.35.0)
```

#### Dimension 2: PDR Readiness (if PDR phase)
```
[ ] Format: Aptos 12pt, 1.15 spacing, 10-page limit
[ ] §1 Introduction: Complete with competition context
[ ] §2 Method: Architecture, pipeline, training process
[ ] §2 PSR §4.4 fix: Individual SHAP + GNNExplainer example
[ ] §2 PSR §4.5 fix: Experiment evolution table
[ ] §2 PSR §5.1 fix: GATv2 vs SAGEConv justification
[ ] §3 Results: F1 + MCC + PR-AUC + Confusion Matrix for ALL 4 panels
[ ] §3 Baseline comparison table
[ ] §3 PSR pilot vs actual results comparison + explanation
[ ] §4 Conclusion: Limitations, future work, ethics
[ ] §5 References: IEEE format, minimum 8 citations
[ ] Cover page, TOC, page numbers
```

#### Dimension 3: Scientific Defensibility
```
[ ] F1=0.8367 can be verified in code output (reports/cv_report.json → RESULTS_CANONICAL.json)
[ ] MCC gap (PSR 0.892 → actual 0.5112) has documented explanation (PDR §4.2)
[ ] MASTER (General) MCC=0.4951 explained as class imbalance 2.75:1 (PDR §4.2)
[ ] GNN architecture name inconsistency resolved in PDR (VariantGATv2GNN confirmed)
[ ] All metrics computed correctly (binary F1, GLOBAL threshold θ=0.8415 canonical; panel opt-in)
[ ] No data leakage in preprocessing pipeline
[ ] SMOTE applied only to train split
[ ] Ablation study showing GNN contribution
```

#### Dimension 4: Reproducibility (§7.5)
```
[ ] README has clear step-by-step jury re-run instructions
[ ] git clone → env setup → data placement → train → predict → F1 works
[ ] Single command produces predictions (e.g., make predict)
[ ] Python version clearly stated (3.10)
[ ] torch-geometric installation documented
[ ] Model checkpoint available for inference-only run
[ ] Expected output files listed with expected values
[ ] Estimated run time documented
```

#### Dimension 5: Documentation Completeness
```
[ ] README: All 25 required sections
[ ] MODEL_CARD: VariantGATv2GNN (not VariantSAGEGNN)
[ ] DATA_CARD: All panel counts, source descriptions, ethics
[ ] Ethics statement: "not for clinical use" in 3 locations
[ ] KVKK/GDPR compliance statement
```

#### Dimension 6: Jury Defense Readiness (if Finals phase)
```
[ ] PSR MCC discrepancy script prepared
[ ] Concrete SHAP example (pathogenic + benign)
[ ] Concrete GNNExplainer subgraph example
[ ] GATv2 vs SAGEConv justification memorized
[ ] Ensemble weight justification ready
[ ] Experiment evolution narrative prepared
[ ] CFTR small-sample limitation answer ready
[ ] Anonymous column mapping methodology explained
[ ] Live demo runs in < 5 minutes
```

## Traffic Light System

### GREEN — Ready
Checklist item verified, evidence found in files, no risk.

### YELLOW — At Risk
Checklist item partially addressed, or evidence weak, or recently changed.

### RED — Blocking
Checklist item missing, evidence not found, or active risk to score/qualification.

## Readiness Dashboard Format
```
## Mission Readiness Dashboard
Date: [today]
Competition Phase: [PDR Prep / Finals Prep / Competition]
Days to Next Deadline: [N]

### Overall Status: [GREEN/YELLOW/RED]

### Dimension Summary
| Dimension | Status | Blocking Issues |
|---|---|---|
| Technical | 🟢/🟡/🔴 | [list] |
| PDR | 🟢/🟡/🔴 | [list] |
| Scientific | 🟢/🟡/🔴 | [list] |
| Reproducibility | 🟢/🟡/🔴 | [list] |
| Documentation | 🟢/🟡/🔴 | [list] |
| Jury Defense | 🟢/🟡/🔴 | [list] |

### RED Items — Must Fix Before Deadline
1. [Item] — [Impact] — [Estimated effort]
2. ...

### YELLOW Items — Should Fix If Time Allows
1. [Item] — [Risk if not fixed]
2. ...

### GREEN Items — Confirmed Ready
[List]

### Priority Action Plan
If N days remain:
- Day 1: [Most critical actions]
- Day 2: [Second priority]
- Day N-1: [Final checks]
- Day N: [Submission day protocol]

### Competition Readiness Verdict
[GO / NO-GO / CONDITIONAL GO with conditions]
```

## Invocation Examples
This skill activates for:
- "Teslime hazır mıyız?"
- "Ne kaldı?"
- "PDR'yi göndersek olur mu?"
- "Kritik eksikler neler?"
- "1 haftam var, ne yapayım?"
- "Final öncesi ne yapmalıyız?"

---
name: experiment-review
description: Use when analyzing VARIANT-GNN experimental results: panel-based F1/MCC/PR-AUC/Brier, threshold analysis, overfitting indicators, CFTR small-data risks, and ablation studies. Produces PDR-ready result tables.
---

# Experiment Review — VARIANT-GNN Result Analysis

When this skill is active, analyze experimental results with the rigor of a senior ML reviewer. Every conclusion must be data-driven. No vague interpretations.

## Competition Context

- Final metric: **F1 Score** only (Şartname §7.3)
- PDR mandatory metrics: F1 + MCC + PR-AUC + Confusion Matrix (PDR template)
- Supporting (recommended): ROC-AUC, Brier Score
- Four panels evaluated separately: MASTER, KANSER, CFTR, PAH

## PSR Reference Results (Pilot Data — Not Competition Data)

| Panel | Macro F1 | ROC-AUC | MCC | Brier |
|---|---|---|---|---|
| MASTER (General) | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 |
| KANSER | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 |

PR-AUC: **Not computed in PSR** — must be computed for competition data and added to PDR.

## Analysis Framework

When results are shared, analyze in this order:

### Step 1: Metric Completeness Check
- [ ] F1 present for all 4 panels?
- [ ] MCC present? (PDR mandatory)
- [ ] PR-AUC present? (PDR mandatory — must be computed)
- [ ] Confusion Matrix available?
- [ ] Threshold: 0.40 confirmed? Optimal verification done?

### Step 2: Panel-by-Panel Assessment

For each panel, report:
- F1 vs. PSR pilot reference (better / worse / within variance?)
- MCC interpretation (>0.8 = excellent, 0.6–0.8 = good, <0.6 = concern)
- PR-AUC interpretation (higher = better at separating, especially with imbalance)
- Brier Score (< 0.05 = well-calibrated, > 0.10 = calibration problem)

### Step 3: CFTR Small-Data Risk
CFTR: 70+70 training, 30+30 test.
- F1 std dev: ±0.012 (PSR). Acceptable (< ±0.02).
- Each test error = ~3.3% F1 change.
- Flag if: std dev > ±0.02 OR F1 < 0.90 OR MCC < 0.80

### Step 4: Overfitting Check
Compare training F1 vs. validation F1.
- Gap > 0.05: moderate overfitting risk — report
- Gap > 0.10: serious overfitting — flag as HIGH
- PSR evidence: eğitim F1≈0.98 → doğrulama F1≈0.78 (before Dropout correction)

### Step 5: Ablation Analysis
If ablation data available:
| Model | MASTER F1 | KANSER F1 | PAH F1 | CFTR F1 |
|---|---|---|---|---|
| XGBoost only | ? | ? | ? | 0.84 (PSR) |
| LightGBM only | ? | ? | ? | ? |
| VariantGATv2GNN only | ? | ? | ? | ? |
| DNN only | ? | ? | ? | ? |
| Full Ensemble | 0.945 | 0.938 | 0.941 | 0.925 (PSR) |

Missing ablation data = flag as jury risk (§5.1 was already 4/5 in PSR).

### Step 6: Threshold Analysis
For threshold 0.40:
- PR curve must show this is near the F1-optimal point
- Compare: F1@0.40 vs F1@0.50 — is 0.40 actually better?
- CFTR-specific threshold: same or different from global?

### Step 7: PDR-Ready Output
Produce a table directly usable in PDR §3 (Results):

```
Table X: Panel-Based Performance (Competition Data, 5-Fold CV)

Panel    | F1 ± std   | MCC   | PR-AUC | ROC-AUC | Brier | Threshold
---------|------------|-------|--------|---------|-------|----------
MASTER   |            |       |        |         |       | 0.40
KANSER   |            |       |        |         |       | 0.40
PAH      |            |       |        |         |       | 0.40
CFTR     |            |       |        |         |       | 0.40
```

## Interpretation Rules

Do NOT say "the model performs well." Say:
- "CFTR F1=0.925±0.012 indicates acceptable performance given 70 training samples, but the variance implies instability that must be monitored."
- "MCC=0.852 for CFTR suggests balanced performance across both classes, better than F1 alone for small imbalanced sets."

## Output Format

```
## Experiment Review Report

### Metric Completeness: [Complete / Missing items]

### Panel Analysis
[Panel] F1: X | MCC: Y | PR-AUC: Z | Assessment: [OK/Risk/Critical]

### Key Findings
1. ...
2. ...

### Risks
- [Panel] [Risk level]: [Description]

### PDR-Ready Table
[Table]

### Missing Computations
- PR-AUC: not computed — add sklearn.metrics.average_precision_score
- [Other]

### Jury Risks
- [What a jury member would challenge based on these results]
```

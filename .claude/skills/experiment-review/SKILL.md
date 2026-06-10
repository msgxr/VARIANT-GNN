---
name: experiment-review
description: Use when analyzing VARIANT-GNN experimental results: panel-based F1/MCC/PR-AUC/Brier, threshold analysis, overfitting indicators, CFTR small-data risks, and ablation studies. Produces PDR-ready result tables.
---

# Experiment Review — VARIANT-GNN Result Analysis

## Official Source Boundary

Metrik değerlendirme standardı: TEKNOFEST 2026 Şartname §7.3 (Binary F1, pos_label=1).  
Deney sonuçları yalnızca gerçek yarışma verisiyle (2026-05-20 eğitim) raporlanır.  
PSR pilot sonuçları gerçek veri sonucu gibi sunulamaz.  
Doğrulanamayan metrik ağırlıkları UNVERIFIED olarak işaretlenir.

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
- [ ] Threshold: global θ=0.8415 confirmed (canonical)? Optimal verification done? (panel opt-in thresholds: General 0.3990 | Hereditary_Cancer 0.4532 | PAH 0.4434 | CFTR 0.1922)

### Step 2: Panel-by-Panel Assessment

For each panel, report:
- F1 vs. PSR pilot reference (better / worse / within variance?)
- MCC interpretation (>0.8 = excellent, 0.6–0.8 = good, <0.6 = concern)
- PR-AUC interpretation (higher = better at separating, especially with imbalance)
- Brier Score (< 0.05 = well-calibrated, > 0.10 = calibration problem)

### Step 3: CFTR Small-Data Risk
CFTR (actual competition data): 111 total (90 Pat / 21 Ben); test hold-out n=18.
- Test n=18 → MCC/ROC-AUC undefined (degenerate); F1/precision/recall meaningful only.
- Canonical CFTR test F1=0.7143 @ θ=0.8415. One misclassification = large F1 swing.
- (PSR pilot reference was nominal 70+70 train / 30+30 test, F1 std ±0.012 — NOT competition data.)
- Flag if: a single-error swing inverts the panel verdict, or MCC is reported despite n=18 degeneracy.

### Step 4: Overfitting Check
Compare training F1 vs. validation F1.
- Gap > 0.05: moderate overfitting risk — report
- Gap > 0.10: serious overfitting — flag as HIGH
- PSR evidence: eğitim F1≈0.98 → doğrulama F1≈0.78 (before Dropout correction)

### Step 5: Ablation Analysis
Canonical per-model fold-CV Binary F1 (overall, competition data):
| Model | CV F1 ± std |
|---|---|
| XGBoost | 0.8876 ± 0.0047 |
| LightGBM | 0.8828 ± 0.0082 |
| VariantGATv2GNN | 0.8114 ± 0.0228 |
| VariantDNN | 0.7596 ± 0.0441 |
| Full Ensemble (OOF-stacking) | 0.8936 ± 0.0004 |

> The 0.945/0.938/0.941/0.925 panel figures are PSR PILOT values (NOT competition data) — do not present as current ensemble F1. Per-panel competition F1 @ θ=0.8415: MASTER 0.8185 | KANSER 0.9060 | PAH 0.9120 | CFTR 0.7143.

Missing panel-level ablation = flag as jury risk (§5.1 was already 4/5 in PSR).

### Step 6: Threshold Analysis
For the global decision threshold θ=0.8415 (canonical; derived F1-optimal on the %20-patojenik held-out calibration set, raw probability):
- PR curve must show this is near the F1-optimal point
- Compare: F1@0.8415 vs F1@0.50 — is 0.8415 actually better on the calibration prior?
- Panel opt-in thresholds (General 0.3990 | Hereditary_Cancer 0.4532 | PAH 0.4434 | CFTR 0.1922) are opt-in only — jury uses global θ=0.8415

### Step 7: PDR-Ready Output
Produce a table directly usable in PDR §3 (Results):

```
Table X: Panel-Based Performance (Competition Data, group-aware test hold-out @ global θ=0.8415)

Panel    | F1 ± std   | MCC   | PR-AUC | ROC-AUC | Brier | Threshold
---------|------------|-------|--------|---------|-------|----------
MASTER   | 0.8185     | 0.4951|        |         |       | 0.8415
KANSER   | 0.9060     | 0.7135|        |         |       | 0.8415
PAH      | 0.9120     | 0.5053|        |         |       | 0.8415
CFTR     | 0.7143     | n/a   |        |         |       | 0.8415
```

## Interpretation Rules

Do NOT say "the model performs well." Say (using canonical competition values):
- "CFTR F1=0.7143 @ θ=0.8415 on a test hold-out of only n=18 means a single misclassification produces a large F1 swing; MCC/ROC-AUC are undefined (degenerate), so only F1/precision/recall are interpretable."
- "KANSER MCC=0.7135 (F1=0.9060) is the most balanced panel; its higher MCC reflects the least class imbalance, better than F1 alone for small imbalanced sets."

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

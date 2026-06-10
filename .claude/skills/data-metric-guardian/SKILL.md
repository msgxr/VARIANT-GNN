---
name: data-metric-guardian
description: Use when verifying metric correctness, data split integrity, preprocessing leakage detection, label compliance, and evaluation logic in VARIANT-GNN. Guards primary metric (Binary F1, pos_label=1), prevents test-data leakage, validates SMOTE placement, confirms panel-specific thresholds. Activate on any metric claim, data processing question, code pipeline review, or when evaluation results are reported.
---

# Skill: data-metric-guardian — Metric & Data Integrity

## Official Source Boundary

Primary metric rule: TEKNOFEST 2026 Şartname §7.3 (Binary F1, pos_label=1=Patojenik)  
PDR metric requirement: PDR Şablonu §3 (F1 + MCC + PR-AUC mandatory per panel)  
Data NDA: Kurumsal Gizlilik Taahhütü (competition data cannot be in repo)  
KVKK: T.C. Kişisel Verilerin Korunması Kanunu  
Rejection: Metric claims without code/file read evidence

---

## Primary Metric — Immutable Rule

**Binary F1 Score (pos_label=1=Patojenik) is the ONLY final ranking metric. Source: Şartname §7.3**

```
F1 = TP / (TP + 0.5×FP + 0.5×FN)

pos_label = 1 → Patojenik is the POSITIVE class
TP = correctly predicted Patojenik
FP = predicted Patojenik, actually Benign
FN = predicted Benign, actually Patojenik
```

**⚠️ IMPORTANT RISK:** TÜSEB reserves the right to change evaluation metrics (Şartname §7.5).
Monitor official announcements for any metric change before PDR submission.

---

## Metric Hierarchy

| Metric | Status | Rule |
|---|---|---|
| Binary F1 (pos_label=1) | **PRIMARY — §7.3 MANDATORY** | Lead all result reports with this |
| MCC | Mandatory in PDR (template §3) | Report per panel alongside F1 |
| PR-AUC | Mandatory in PDR (template §3) | Report per panel alongside F1 |
| ROC-AUC | Supportive | Report, not mandatory |
| Accuracy | **FORBIDDEN as standalone primary** | Misleading on class-imbalanced data |
| Macro F1 | Supportive | Not the competition ranking metric |
| Weighted F1 | Supportive | Not the competition ranking metric |
| Brier Score | Supportive | Calibration quality |

**Any report where Accuracy appears as the main success metric = Şartname violation → FAIL**

---

## Verified Model Performance (Real Competition Data — 2026-06-02, CANONICAL: RESULTS_CANONICAL.json)

### Overall Test Set

⭐ **Jüri beklentisi (%20-patojenik — ✅ resmi Q&A-II ile DOĞRULANDI 2026-06-03) = balanced Binary F1 = 0.6042 ± 0.0324 (havuzlanmış); RESMİ headline = 0.6202.** Aşağıdaki test sayıları %75-poz iç hold-out ayrım gücüdür (jüri skoru değil).

| Metric | Value | Status |
|---|---|---|
| CV F1 (OOF-stacking nested) | **0.8936 ± 0.0004** | Primary declared metric |
| CV F1 (fold-CV component) | 0.8812 ± 0.0113 | Auxiliary |
| CV std (5 seeds) | ±0.0034 | Stability confirmed (0.8738 mean) |
| Test F1 (hold-out) | **0.8367** | Internal discrimination |
| MCC | **0.5112** | Overall (P/R ile tutarlı) |
| PR-AUC | **0.9267** | |
| ROC-AUC | **0.8538** | |
| Precision / Recall | **0.9241 / 0.7644** | Balanced |
| Brier / ECE | **0.1115 / 0.0291** | |
| Global threshold | **0.8415** | balanced-OOF F1-optimal (canonical) |

> ⚠️ WITHDRAWN: 0.8980/0.9269, MCC 0.5356, θ=0.241 were leakage-inflated → retracted (reports/leakage_quantification.json).

### Panel-Specific Results (Test Set, global θ=0.8415)

| Panel | Label in Data | F1 | MCC | PR-AUC | Opt-in θ | Risk Notes |
|---|---|---|---|---|---|---|
| MASTER | General | **0.8185** | 0.4951 | — | 0.3990 | Class imbalance 2.75:1 |
| KANSER | Hereditary_Cancer | **0.9060** | 0.7135 | — | 0.4532 | En dengeli, en iyi MCC |
| PAH | PAH | **0.9120** | 0.5053 | — | 0.4434 | n_benign=62 → MCC küçük-n etkisi |
| CFTR | CFTR | **0.7143** | tanımsız | — | 0.1922 | n=18 test → MCC/ROC tanımsız |

**⚠️ If any claimed metric value differs from this table → READ RESULTS_CANONICAL.json FIRST before proceeding.**  
Source: `reports/cv_report.json` → `RESULTS_CANONICAL.json` (2026-06-02). Panel eşikleri opt-in; jüri global θ=0.8415 kullanır. Opt-in panel eşikleri: General 0.3990 | Hereditary_Cancer 0.4532 | PAH 0.4434 | CFTR 0.1922.

> **NOT:** Panel MCC değerleri: MASTER 0.4951 | KANSER 0.7135 | PAH 0.5053 | CFTR tanımsız (n=18, degenerate). Overall MCC=0.5112.

### CFTR Sensitivity Warning
Test hold-out n=18 → 1 prediction error = large F1 swing; MCC/ROC-AUC tanımsız (degenerate). F1/precision/recall anlamlıdır.

---

## Data Leakage Detection Protocol

**Data leakage = preprocessing fit on test data. Consequence: all results invalid, disqualification risk.**

### SMOTE Placement Rule

```
CORRECT:
  Split data → train_fold / val_fold / test_set
  smote.fit_resample(X_train_fold, y_train_fold)   ← SMOTE inside CV loop, train only
  Train model on augmented train_fold

WRONG:
  smote.fit_resample(X_all, y_all)                  ← BEFORE split → test contaminated
  smote.fit_resample(X_train+X_val, y_train+y_val)  ← validation included → leakage
```

### Scaler / Imputer / Encoder Rule

```
CORRECT:
  scaler.fit(X_train_fold)              ← fit on TRAIN FOLD only
  X_train_scaled = scaler.transform(X_train_fold)
  X_test_scaled  = scaler.transform(X_test)   ← transform only, NO refit

WRONG (any of these = leakage):
  scaler.fit(X_full_dataset)            ← test statistics bleed into normalization
  scaler.fit(X_train + X_test)          ← explicit leakage
  scaler.fit_transform(X_test)          ← test-set fit
```

### Components to Verify

| Component | File | Expected Fit Point |
|---|---|---|
| `ColumnAligner` | `src/data/column_aligner.py` | Train schema reference |
| `CategoricalBioFeaturizer` | `src/features/categorical_bio_features.py` | Stateless (deterministic, leakage-free) |
| `RobustScaler` | `src/features/preprocessing.py` | Inside CV fold, train fold only |
| `SimpleImputer` (median) | `src/features/preprocessing.py` | Inside CV fold, train fold only |
| `SMOTE` | `src/features/preprocessing.py` | Inside CV fold, AFTER split, train only |
| ~~SelectKBest / AutoEncoder~~ | — | **REMOVED** (cost ~5.3pp; full 343 features kept) |

### Calibration Set Rule

```
Calibration set = held-out 15% of TRAINING DATA (not test set)
Purpose: threshold optimization (maximize F1 on the %20-patojenik prior)
Result: global θ=0.8415 applied to test set (jury default); panel-specific thresholds opt-in only
Isolation: calibration samples NEVER appear in test evaluation
```

### Test Label Contamination

```
Test labels must NOT appear in:
  - Training features (target leakage)
  - Feature selection fit
  - Any preprocessing fit
  - Calibration threshold optimization (calibration uses train-sourced set)

Violation = automatic disqualification — treat as CRITICAL FAIL
```

---

## Leakage Verification Procedure (Step-by-Step)

Activate when: code is reviewed, pipeline changes, results are disputed, or before submission.

**Step 1 — List all preprocessing components**
```
Read: src/core/pipeline.py, src/core/gnn.py, main.py
Identify every: fit(), fit_transform(), fit_resample() call
List each with its location in code
```

**Step 2 — Verify each component's fit point**
```
For each component:
  □ fit() called inside CV fold on X_train_fold only? → PASS
  □ fit() called on full X before split? → FAIL — LEAKAGE
  □ fit() called on X_test? → CRITICAL FAIL — DISQUALIFICATION
```

**Step 3 — Verify SMOTE placement**
```
□ SMOTE: after train/test split, inside CV loop → PASS
□ SMOTE: before split → FAIL
□ SMOTE: applied to validation data → FAIL
```

**Step 4 — Verify calibration isolation**
```
□ Calibration set sourced exclusively from training data → PASS
□ Calibration set contains any test samples → CRITICAL FAIL
□ Threshold optimized on calibration, applied to test → PASS
```

**Step 5 — Adversarial validation check**
```
Expected: train vs test distribution ROC-AUC ≈ 0.50
PSR documented: General=0.512, Hereditary=0.505, PAH=0.498, CFTR=0.521 ← acceptable
If AUC >> 0.55 after code changes: distribution shift detected → investigate before reporting
```

**Step 6 — Label contamination check**
```
□ Test set labels not present in X_train features → verify in code
□ No label-derived features constructed from test labels
□ Feature column names anonymous (confirmed by competition setup)
```

---

## Metric Computation Verification

When a metric value is claimed or reported:

**Step 1:** Identify the source file → read it (never accept from memory alone)  
**Step 2:** Confirm `pos_label=1` in F1 computation  
**Step 3:** Confirm threshold applied before metric: `y_pred = (probas >= threshold).astype(int)`  
**Step 4:** Confirm correct panel isolation (not mixed across panels)  
**Step 5:** Match to verified values table above  

```python
# CORRECT F1 computation
from sklearn.metrics import f1_score
y_pred = (y_proba >= threshold).astype(int)          # threshold applied first
f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')

# WRONG — reject these
f1 = f1_score(y_true, y_pred, average='macro')        # not competition metric
f1 = f1_score(y_true, y_pred, average='weighted')     # not competition metric
f1 = f1_score(y_true, y_pred)                         # verify pos_label default
accuracy_score(y_true, y_pred)                        # forbidden as primary
```

---

## Data Privacy & Repo Security

| Requirement | Rule | Verification |
|---|---|---|
| Competition data NOT in repo | NDA — zero tolerance | `git ls-files \| grep -E "\\.(csv\|tsv\|vcf\|xlsx)$"` → must be empty |
| Genomic address NOT as feature | Şartname — anonymous columns | Column names verified anonymous by committee |
| No patient identifiers | KVKK | Data anonymized by competition committee |
| KVKK compliance stated | Ethics declaration required | Check PDR ethics section |
| Gizlilik Taahhütü signed | Required before data access | Confirmed submitted |

**Data repo check command:**
```bash
git ls-files | grep -E "\.(csv|tsv|vcf|xlsx)$"
# Expected output: empty (no data files in repo)
```

---

## Panel Data Reference

**Fiili veri (data/train_variants.csv, 3802 satır / 3224 tekil varyant):**

| Panel | Toplam (Pat/Ben) | Test hold-out (n) | Risk Level |
|---|---|---|---|
| MASTER / General | 2931 (2149/782) | 582 | Class imbalance 2.75:1 |
| KANSER / Hereditary_Cancer | 388 (268/120) | 86 | Medium — small N |
| PAH | 372 (310/62) | 76 | n_benign=62 → MCC küçük-n |
| CFTR | 111 (90/21) | 18 | HIGH — MCC/ROC tanımsız |

*(Şartname-nominal tasarım: General 3000/2000, KANSER 400/200, PAH 400/200, CFTR 140/60 — fiili veriyle karıştırma.)*

---

## Validation Checklist (Quick Run)

```
□ Binary F1 pos_label=1 confirmed?
□ Threshold applied before metric computation?
□ All 4 panels reported separately?
□ Scaler fit on train fold only?
□ Imputer fit on train fold only?
□ CategoricalBioFeaturizer deterministic (no test fit)?
□ Group-aware split (Variant_ID) — 0 straddle?
□ SMOTE after split, inside CV loop only?
□ Calibration set sourced from training data only?
□ Test labels not in training?
□ Adversarial validation AUC ≈ 0.50?
□ Competition data not in git repo?
□ Genomic address not used as feature?
□ Accuracy NOT presented as primary metric?
```

---

## Hard Rules

1. Binary F1 pos_label=1 = primary metric. No exceptions. No alternate primary.
2. Accuracy as standalone main metric = Şartname violation. Flag and correct immediately.
3. Any preprocessing fit on test data = CRITICAL FAIL. All results must be considered suspect.
4. SMOTE before split = CRITICAL FAIL. Stop all analysis, verify pipeline, rerun if needed.
5. Competition data committed to repo = NDA violation. Remove from all git history.
6. Genomic address (chr/pos) as feature = Şartname violation. Remove immediately.
7. Never assert a metric value without reading the source file that produced it.
8. F1 with wrong pos_label or average = incorrect value. Reject and recompute.

---

## Escalation Map

| Trigger | Escalate To |
|---|---|
| Data leakage confirmed — any level | `error-checker` → CRITICAL, results suspect until verified |
| Metric formula wrong in PDR | `pdr-editor` + `report-template-checker` |
| Pipeline code changed | `code-change-verifier` |
| Result differs from verified table | Read source file first, then `experiment-review` |
| Full compliance audit | `competition-compliance-auditor` |
| Pre-submission metric check | `pre-submission-gate` |

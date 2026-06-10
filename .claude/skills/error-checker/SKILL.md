---
name: error-checker
description: Use when scanning VARIANT-GNN code, reports, or pipeline for errors at four priority levels: CRITICAL (disqualification risk), HIGH (major score loss), MODERATE (jury impression), LOW (quality). Covers data leakage, F1 metric violations, clinical claims, GNN mask errors, and specification violations.
---

# VARIANT-GNN ERROR CHECKER — Error Detection and Correction Skill

## Skill Purpose

This skill is used to systematically detect, prioritize, and produce directly actionable corrections for all types of errors in the VARIANT-GNN project, evaluated against the competition specification.

When this skill is active, Claude does not merely state errors. It writes concretely: what the error is, why it is an error, which clause of the specification it violates, and how to fix it.

---

## Absolute Reference Framework

This skill operates exclusively within the following framework:

- Competition: TEKNOFEST 2026 Health AI Competition
- Level: University and Above
- Project: VARIANT-GNN
- Task: Pathogenic / Benign binary classification
- Primary metric: F1 Score (based on TP, FP, FN)
- Final score: Task performance 90%, presentation 10%
- Jury authority: Re-running code and verifying declared results

---

## Error Priority Levels

Every error must be labeled with one of the following levels. Claude always writes the level next to the error title.

### LEVEL 1 — CRITICAL
Risk of disqualification or major score loss in the competition. Must be fixed immediately.

- Using test labels during training
- Data leakage (preprocessing/scaler/imputer/encoder fitted on entire dataset)
- Code not running at all
- F1 score calculated incorrectly or not reported at all
- Clinical diagnosis/treatment claim
- ECG, cardiology, high-school-level content
- Finding labels from external sources using genomic addresses
- Binary classification not set up

### LEVEL 2 — HIGH
Errors that can cause serious score loss. Must be fixed within 24 hours.

- No baseline model comparison
- GNN justification missing
- Missing confusion matrix, precision, recall
- Random seed not fixed
- SMOTE / oversampling applied before split
- Overlap in train/validation/test masks
- Prediction file cannot be produced
- No model checkpoint saved

### LEVEL 3 — MODERATE
Errors that reduce quality and negatively affect jury impression. Must be fixed before submission.

- Missing overfitting analysis
- Cross-validation or validation strategy insufficiently explained
- Threshold tuning not performed
- Panel-based evaluation incomplete
- Class imbalance not interpreted
- Insufficient code documentation
- Missing run commands in README

### LEVEL 4 — LOW
Suggestions that do not cause score loss but will improve quality.

- Missing additional visualizations
- Code style cleanup
- Improving log messages

---

## Data Leakage Alarm Rules

When the following patterns appear in code, Claude marks them as CRITICAL and provides a fix.

### Preprocessing Leakage
```
WRONG:   scaler.fit(X)              # X = entire dataset
WRONG:   scaler.fit_transform(X)    # X = entire dataset
CORRECT: scaler.fit(X_train)
         scaler.transform(X_val)
         scaler.transform(X_test)

WRONG:   imputer.fit(X)             # X = entire dataset
CORRECT: imputer.fit(X_train)
         imputer.transform(X_val)
         imputer.transform(X_test)

WRONG:   encoder.fit(X)             # X = entire dataset
CORRECT: encoder.fit(X_train)
         encoder.transform(X_val)
         encoder.transform(X_test)
```

### Feature Selection Leakage
```
WRONG:   selector = SelectKBest().fit(X, y)        # X, y = entire dataset
CORRECT: selector = SelectKBest().fit(X_train, y_train)
         X_train_sel = selector.transform(X_train)
         X_val_sel   = selector.transform(X_val)
         X_test_sel  = selector.transform(X_test)
```

### SMOTE / Oversampling Leakage
```
WRONG:   X_res, y_res = SMOTE().fit_resample(X, y)  # before split
         X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)

CORRECT: X_train, X_test, y_train, y_test = train_test_split(X, y)
         X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)
```

### Test Label Leakage
```
WRONG:   model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
WRONG:   if f1_score(y_test, pred) > best: model = current_model  # in model selection
CORRECT: model selection is performed only on the validation set
```

### Genomic Address Leakage (Specification Section 3.2)
```
WRONG:   Label query via ClinVar API using chr + pos
WRONG:   Pulling variant labels from gnomAD
WRONG:   Fetching ready-made class labels from external databases
CORRECT: Only variant profiles provided by the competition committee are used
```

### GNN Mask Leakage
```
WRONG:   data.train_mask = torch.ones(N, dtype=torch.bool)  # all nodes as train
WRONG:   loss = criterion(out, data.y)                      # loss over all nodes
CORRECT: loss = criterion(out[data.train_mask], data.y[data.train_mask])
         val_pred  = out[data.val_mask]
         test_pred = out[data.test_mask]
```

---

## Specification Violation Scanning Rules

### Violation 1: Out-of-Scope Content
Per Specification Sections 2 and 3.1, the following belong only to the HIGH SCHOOL level:

- ECG / 12-lead ECG
- Cardiology / arrhythmia / conduction disorder
- ECG Arrhythmia Dataset / PhysioNet signals
- Macro F1-score (high school metric)
- ECG subgroup classification

If this content appears in code, reports, or presentations, Claude marks it as CRITICAL and removes it.

### Violation 2: Wrong Metric
Specification Section 7.3: The primary metric is F1 Score (based on TP, FP, FN).

```
WRONG:   Reporting only accuracy
WRONG:   Presenting AUC-ROC as the sole metric
WRONG:   Not computing F1 at all
CORRECT: F1 = 2*TP / (2*TP + FP + FN)
         Precision = TP / (TP + FP)
         Recall = TP / (TP + FN)
         Confusion matrix
```

### Violation 3: Clinical Claim
Specification Section 10: Model outputs cannot be used for clinical diagnosis, treatment, or medical decision support.

Prohibited phrases:
- "Makes a definitive diagnosis"
- "Gives 100% accurate results"
- "Makes decisions instead of a doctor"
- "Can be used directly in the clinic"
- "Recommends treatment"
- "Detects disease"

Correct phrases:
- "Prediction model"
- "For research, education, and competition evaluation purposes"
- "Model outputs do not replace medical decisions"
- "Additional validation is required for clinical use"

### Violation 4: Class Merging Error
Specification Section 3.2:

```
CORRECT: Pathogenic ← Pathogenic + Likely Pathogenic
         Benign     ← Benign + Likely Benign

WRONG:   4-class model (Pathogenic / Likely_Path / Likely_Benign / Benign)
WRONG:   Keeping Likely Pathogenic as a separate class
WRONG:   Including VUS (Variant of Uncertain Significance)
```

### Violation 5: Genomic Address Usage
Specification Section 3.2: Genomic address (chromosome and position) information of variants is hidden.

```
WRONG:   Using chromosome, position, genomic_address columns as model features
WRONG:   Searching for labels in ClinVar using this information
CORRECT: Only the feature groups specified by the specification are used
```

---

## Code Error Scanning Procedure

When the user provides code, Claude inspects in the following order:

### Step 1: Syntax and Imports
- Are there import errors?
- Are there undefined variables?
- Are there unclosed parentheses/brackets?
- Are there indentation errors?

### Step 2: Data Reading and Labels
- Are file paths correct? Are they hard-coded?
- Is label encoding consistent? (Pathogenic=1 / Benign=0 or vice versa, defined in one place)
- Is NaN handling correct?
- Is class merging (Likely classes) applied?

### Step 3: Preprocessing Pipeline
- Are scaler, imputer, encoder fitted only on train data?
- Is SMOTE applied only on train data after the split?
- Is the full 343-feature set preserved? (SelectKBest + AutoEncoder were REMOVED — flag any reintroduced dimensionality reduction fitted on full data as leakage risk)

### Step 4: Train/Validation/Test Split
- Has a split been made?
- Is validation separate from test?
- Does the same variant appear in multiple splits?
- Is random_state / seed fixed?

### Step 5: Model Architecture
- Is binary classification set up correctly?
- Output size: 1 (sigmoid) or 2 (softmax)?
- Loss function: BCEWithLogitsLoss / BCELoss / CrossEntropyLoss — is it correct?
- Is there a Sigmoid + BCELoss conflict?
- Is there a Sigmoid + CrossEntropyLoss conflict?

### Step 6: Metric Calculation
- Is f1_score computed?
- Is average='binary' used? (for binary classification)
- Are precision_score and recall_score present?
- Is confusion_matrix produced?
- Is accuracy presented as the sole metric?

### Step 7: GNN Specific
- Is data.x shape: [num_nodes, num_features] correct?
- Is data.edge_index shape: [2, num_edges] correct?
- Is data.y shape: [num_nodes] correct?
- Do train_mask, val_mask, test_mask overlap?
- Is loss computed only over train_mask?
- Is the graph built over the entire dataset (train+test)?

### Step 8: Reproducibility
- Is torch.manual_seed(seed) present?
- Is np.random.seed(seed) present?
- Is random.seed(seed) present?
- Is torch.backends.cudnn.deterministic = True present?
- Is the model checkpoint saved?
- Can the prediction file (submission.csv / predictions.csv) be produced?

---

## Report Error Scanning Procedure

When the user provides a report, Claude checks the following headings:

### Project Presentation Report (PSR) Check
Specification Section 4 (University and Above):

- Is the general problem defined?
- Is there a literature review?
- Are the details of the proposed solution method present?
- Is the data approach explained?
- Is there an explicit connection to the specification scope?
- If preliminary results exist, are they justified? Or is "planned method" language used?
- Is there out-of-scope content (ECG, high school)?

### Project Detail Report (PDR) Check
Specification Section 4 (University and Above):

- Is the developed model architecture described in detail?
- Are the training processes explained?
- Are internal test (validation) results presented?
- Is the F1 score given in a table?
- Is there a confusion matrix?
- Is there a baseline comparison?
- Is the GNN justification scientific?
- Is there panel-based evaluation (General + Cancer + PAH + CFTR)?
- Are the data preprocessing steps explained?
- Are data leakage prevention measures stated?
- Is there an ethics statement and KVKK/GDPR section?
- Is there a clinical use disclaimer?
- Is code runnability explained?

---

## GNN Error Audit

When GNN code or description is provided, Claude answers the following questions:

**Graph Structure:**
- What do nodes represent? (Each variant should be one node)
- How are edges constructed? (kNN feature similarity, biological similarity, etc.)
- Does the edge construction connect train and test nodes? (Leakage risk)
- Are edge weights NaN or negative?

**GNN Justification:**
"Because biological, structural, and computational similarity relationships between variants can be represented as a graph structure, the GNN architecture has the potential to learn relational patterns compared to models that learn from independent tabular samples."

If this justification is missing from the report, it is flagged as a HIGH error.

**Baseline Requirement (per the spirit of the specification):**
If baseline performance without GNN is unknown, the contribution of GNN cannot be defended.

Recommended baseline models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- MLP / Tabular DNN

Using GNN without a baseline is not CRITICAL but is a HIGH error.

---

## F1 Score Audit

Specification Section 7.3: The primary metric is F1 Score.

```python
# CORRECT F1 Calculation (Binary Classification)
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

f1  = f1_score(y_true, y_pred, average='binary')
pre = precision_score(y_true, y_pred, average='binary')
rec = recall_score(y_true, y_pred, average='binary')
cm  = confusion_matrix(y_true, y_pred)

# Manual verification:
# F1 = 2*TP / (2*TP + FP + FN)
```

Claude performs the following checks:
- Is average='binary' used?
- Is threshold fixed at 0.5? Has it been optimized on validation?
- If class imbalance exists, is its effect interpreted?
- Is panel-based F1 computed separately?
- Is accuracy presented as the sole metric?

---

## Ethics and KVKK/GDPR Check

Specification Section 10:

Items to check:
- Is the phrase "for research, education, and competition evaluation purposes only" present?
- Is "cannot be used for clinical diagnosis, treatment, or medical decision support" present?
- Is a KVKK / GDPR section added?
- Is it stated that removing genomic addresses reduces re-identification risk?
- Is it explained that the TEKNOFEST organization is the data controller?

Correct ethics statement:
"The model developed within the scope of this project is designed solely for research, education, and competition evaluation purposes. Model outputs cannot be used directly in clinical diagnosis, treatment, or medical decision-making processes. Independent validation, regulatory compliance, and expert physician assessment are required for clinical use. The data used in the competition is publicly available and anonymized, in compliance with KVKK and GDPR standards."

---

## Error Output Format

When error detection is requested, Claude uses the following format:

```
## Error Scan Result

### Summary
[Total X errors: Y critical, Z high, W moderate]

---

### LEVEL 1 — CRITICAL ERRORS

#### [ERROR TITLE]
- Location: [file/line/function]
- Specification clause: [relevant section]
- Description: [what is wrong and why it is critical]
- Current:
  [faulty code or statement]
- Corrected:
  [correct code or statement]

---

### LEVEL 2 — HIGH ERRORS
[Same format]

### LEVEL 3 — MODERATE ERRORS
[Same format]

---

### Pre-Submission Checklist
[ ] All critical errors fixed
[ ] F1 score computed correctly
[ ] No data leakage
[ ] Code runs without errors
[ ] Prediction file can be produced
[ ] Ethics statement added
[ ] Clinical use disclaimer present
[ ] Baseline comparison present
[ ] Seed fixed
[ ] Model checkpoint saved
```

---

## Quick Command Table

When the user uses one of the commands below, Claude initiates the corresponding scan:

| User Command | Action |
|---|---|
| "find errors" | Full scan across all categories |
| "check code" | Steps 1–8 code scan |
| "leakage check" | Data leakage alarm rules |
| "spec compliant?" | Specification violation scan |
| "fix report" | PSR / PDR check procedure |
| "GNN check" | GNN-specific audit |
| "F1 check" | F1 and metric audit |
| "ethics check" | KVKK/GDPR and clinical risk audit |
| "pre-submission check" | Master checklist |

---

## Pre-Submission Master Checklist

### A. Code
- [ ] All files run without errors
- [ ] File paths are portable (not hard-coded)
- [ ] Random seed fixed
- [ ] Model checkpoint saved and loadable
- [ ] Prediction file produced and saved
- [ ] requirements.txt up to date
- [ ] Commands in README work

### B. Data
- [ ] Pathogenic + Likely Pathogenic → Pathogenic
- [ ] Benign + Likely Benign → Benign
- [ ] Train/val/test split correct
- [ ] Preprocessing fitted only on train
- [ ] SMOTE applied only on train
- [ ] Feature selection applied only on train
- [ ] No risk of finding labels via genomic address
- [ ] Gene/protein-level leakage checked

### C. Model and Metrics
- [ ] Binary classification set up correctly
- [ ] Loss function appropriate
- [ ] F1, Precision, Recall reported
- [ ] Confusion matrix produced
- [ ] Baseline models present
- [ ] Overfitting analysis performed

### D. Report
- [ ] PSR compliant with specification scope
- [ ] PDR contains all required sections
- [ ] No out-of-scope content (ECG, etc.)
- [ ] Ethics statement added
- [ ] KVKK/GDPR section present
- [ ] Clinical use disclaimer present
- [ ] Panel-based evaluation present

### E. GNN
- [ ] Graph structure justified
- [ ] Train/val/test masks correct and non-overlapping
- [ ] Test labels not used in training
- [ ] GNN vs Baseline comparison present

### F. Final Presentation
- [ ] Problem → Method → Result structure
- [ ] F1 results shown
- [ ] Baseline comparison present
- [ ] Ethical compliance stated
- [ ] Specification connection explicit

---

## Correction Rule

Claude does not merely state errors. It provides a directly actionable correction for every error.

Wrong: "This section is missing."
Correct: "This section is missing. The following text can be added: ..."

---

## Prohibitions

When this skill is active, Claude:

- Does not present non-existent errors as real
- Does not soften genuinely critical errors
- Does not use vague phrases like "probably no issue"
- Does not state an error without providing a correction
- Does not add or suggest ECG, cardiology, or high-school content
- Does not present untested code as "definitely works"
- Does not state anything as certain when uncertain

---

## Final Internal Control Question

Before producing any error report, Claude asks itself internally:

"Does this error detection and correction make the VARIANT-GNN project lose fewer points in the TEKNOFEST 2026 Health AI Competition University and Above Level final evaluation?"

If the answer is no, the output is revised.

---
name: pre-submission-gate
description: Use immediately before any major submission (PDR, final code package, jury presentation materials). Runs a hard GO/NO-GO gate with zero tolerance for critical failures. Does not produce encouragement — produces a binary decision with evidence. Activate when user says "son kontrol", "gönderelim mi?", "hazır mı?", "submit", or any variant of pre-submission validation.
---

# Pre-Submission Gate — VARIANT-GNN

This skill runs a hard, binary GO/NO-GO evaluation before any major submission. There is no "mostly ready" verdict. Either the submission is ready or it isn't.

## Submission Types

Identify what type of submission is being checked:
- **PDR Submission** (deadline: 29.06.2026)
- **Code Package Submission** (finals)
- **Final Presentation Package**
- **Competition Day Submission**

## Hard Stop Conditions (Any = NO-GO)

These are automatic NO-GO triggers. No exceptions.

```
HARD STOP 1: Code doesn't run
- Cannot complete: python main.py → predictions.csv → F1 score
- Status: NO-GO

HARD STOP 2: Data leakage confirmed
- Scaler/imputer/selector fitted on full dataset including test
- Status: NO-GO

HARD STOP 3: Wrong metric reported
- F1 not computed at all, or computed incorrectly (wrong average)
- Status: NO-GO

HARD STOP 4: Competition data in repository
- Actual competition data committed to git
- Status: NO-GO

HARD STOP 5: Clinical claim present
- Any statement implying clinical diagnosis capability
- Status: NO-GO

HARD STOP 6: PDR exceeds 10 pages
- PDR evaluated section is over 10 pages
- Status: NO-GO (reports exceeding limit are not evaluated)

HARD STOP 7: PDR in wrong font
- Font is not Aptos
- Status: NO-GO (format violations = disqualification risk)

HARD STOP 8: Seed not fixed
- Cannot reproduce same results with seed=42
- Status: NO-GO

HARD STOP 9: Model weights missing
- No trained model checkpoint available for inference
- Status: NO-GO

HARD STOP 10: requirements.txt broken
- `pip install -r requirements.txt` fails on clean environment
- Status: NO-GO
```

## Soft Stop Conditions (3+ = NO-GO, any = Warning)

```
SOFT STOP 1: MCC gap unexplained
- PSR MCC 0.892 vs actual 0.5863 not addressed in PDR
- Risk: Jury question without answer

SOFT STOP 2: GNN name still inconsistent
- PDR still says "VariantSAGEGNN" instead of "VariantGATv2GNN"
- Risk: Technical credibility damage

SOFT STOP 3: Experiment evolution table missing
- PDR §2 has no experiment history table
- Risk: §4.5 score remains 3.33/5

SOFT STOP 4: No individual SHAP example
- No specific pathogenic/benign SHAP explanation in PDR
- Risk: §4.4 score remains 3.33/5

SOFT STOP 5: PAH/CFTR MCC not discussed
- PDR doesn't explain PAH=0.1466, CFTR=0.2435 MCC
- Risk: Jury attacks without prepared defense

SOFT STOP 6: Baseline comparison missing
- No XGBoost-only vs ensemble comparison
- Risk: Cannot justify ensemble complexity

SOFT STOP 7: Smoke test fails on CPU
- Pipeline doesn't complete on CPU in reasonable time
- Risk: Jury re-run fails

SOFT STOP 8: README run instructions broken
- Commands in README produce errors
- Risk: Reproducibility failure

SOFT STOP 9: No confusion matrix in PDR
- PDR results section lacks confusion matrix
- Risk: PDR template requirement not met

SOFT STOP 10: References insufficient
- Fewer than 8 references, or broken references
- Risk: §5 deduction
```

## Gate Execution Protocol

```
Step 1: Identify submission type
Step 2: Check all 10 HARD STOP conditions
   → Any HARD STOP found = immediate NO-GO + detailed report
Step 3: Check all 10 SOFT STOP conditions
   → Count soft stops: 0-2 = GO | 3+ = CONDITIONAL or NO-GO
Step 4: Assess remaining timeline
   → 0 days: GO even with soft stops if no hard stops
   → 1-3 days: Fix all soft stops found
   → 3+ days: Fix everything before rerunning gate
Step 5: Issue final verdict with evidence
```

## Gate Output Format
```
## PRE-SUBMISSION GATE REPORT

Submission Type: [PDR / Code / Presentation]
Gate Run Date: [date]
Days Until Deadline: [N]

### VERDICT: [GO ✓ / CONDITIONAL GO ⚠ / NO-GO ✗]

---

### HARD STOP RESULTS
| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Code runs | ✓/✗ | [where checked] |
...

HARD STOPS FAILED: [N]
→ Any failure = NO-GO regardless of soft stops

---

### SOFT STOP RESULTS
| # | Condition | Status | Risk |
|---|---|---|---|
| 1 | MCC gap explained | ✓/✗ | [risk level] |
...

SOFT STOPS FAILED: [N]
→ 0-2: Acceptable | 3+: Should delay submission

---

### VERDICT RATIONALE
[Why GO/CONDITIONAL/NO-GO was decided]

### If NO-GO: Minimum Required Fixes
1. [Fix] — [Estimated time]
2. ...
Estimated time to GO: [X hours/days]

### If CONDITIONAL GO: Accept-Risk Items
[Items being accepted as-is with justification]

### If GO: Final Checklist
[ ] All hard stops: PASS
[ ] Soft stops < 3
[ ] Submission package assembled
[ ] Backup copy made
[ ] Submission system accessible
[ ] Deadline confirmed: [date/time]
```

---
name: code-change-verifier
description: Use after any code change to VARIANT-GNN source files — verifies the change didn't break anything, assesses cross-file impact, and identifies verification steps needed. Activate when code has just been modified, when "does this break anything else?" is asked, or when a bug fix needs post-fix verification. Do NOT use for non-code documentation changes.
---

# Code Change Verifier — VARIANT-GNN

When this skill is active, systematically verify that a code change achieves its stated intent, hasn't introduced regressions, and is consistent with the rest of the codebase.

## Verification Protocol

### Phase 1: Change Scope Assessment (always run)
```
1. What file(s) were changed?
2. What functions/classes were modified?
3. What was the stated intent of the change?
4. What problem does this fix or feature does this add?
```

### Phase 2: Dependency Impact Analysis (always run)
```
For each changed file:
1. Who imports this file? (find all importers in the repo)
2. What behavior does the changed function expose to callers?
3. Did the function signature change? (breaking change risk)
4. Did the return type/shape change? (silent incompatibility)
5. Did any default parameter change? (changes caller behavior silently)
```

### Phase 3: Competition-Critical Check (for pipeline-touching changes)
```
If the change touches: data loading, preprocessing, training, inference, or metrics:

[ ] Does the change maintain seed=42 determinism?
[ ] Does the change preserve SMOTE-only-on-train guarantee?
[ ] Does the change preserve scaler-fit-on-train-only?
[ ] Does the change maintain correct F1 computation (binary, GLOBAL threshold θ=0.8415)?
[ ] Does the change preserve group-aware split (Variant_ID; 0 straddle)?
[ ] Does the change maintain 4-panel evaluation structure?
[ ] Does the change affect submission.csv output format?
[ ] Does the change affect the threshold (global=0.8415 canonical; panel opt-in: General 0.3990, KANSER 0.4532, PAH 0.4434, CFTR 0.1922)?
```

### Phase 4: Regression Risk Assessment (for any non-trivial change)
```
What worked before that might not work now?

Score by risk:
- CRITICAL: Prediction output changes → results change → declared F1 no longer reproducible
- HIGH: Training behavior changes → model checkpoint different → inference fails
- MEDIUM: Utility function changes → downstream processing different
- LOW: Logging/display changes → no functional impact
```

### Phase 5: Verification Steps
Based on the above, determine minimum verification:
```
ROUTINE CHANGE (logging, comment, minor refactor):
→ Manual code review sufficient

IMPORTANT CHANGE (function logic, parameter change):
→ Identify and run affected unit tests
→ Check function behavior with sample inputs
→ Verify importers still work

CRITICAL CHANGE (pipeline, preprocessing, GNN, metrics):
→ Run full smoke test: python main.py --config configs/final.yaml --smoke-test
→ Verify F1 output matches expected (≈0.8367)
→ Verify all 4 panels produce results
→ Verify submission.csv format correct
```

## VARIANT-GNN Specific Verification Rules

### If preprocessing.py was changed:
```
MUST VERIFY:
- RobustScaler still fit on train only
- SelectKBest still fit on train/labels only  
- AutoEncoder still trained on train only
- SMOTE still applied after split
- Output feature dimensions consistent with GNN input expectations
```

### If gnn.py was changed:
```
MUST VERIFY:
- GATv2Conv still used (not replaced with SAGEConv)
- 4 attention heads maintained
- hidden_dim=128 maintained (or change documented)
- MC Dropout still uses 30 forward passes
- train/val/test masks still non-overlapping
- Loss computed only on train_mask
```

### If ensemble.py was changed:
```
MUST VERIFY:
- Weights still sum to 1.0 (XGB=0.3, LGBM=0.3, GNN=0.25, DNN=0.15)
- No division by zero in weight normalization
- Weight indexing consistent (not dict-order dependent)
- Stacking meta-learner still receives correct input
```

### If metrics.py was changed:
```
MUST VERIFY:
- f1_score called with average='binary'
- global threshold θ=0.8415 (canonical) still applied; panel thresholds opt-in from panel_thresholds.json
- confusion_matrix still computed
- Panel-level metrics computed separately
```

### If trainer.py was changed:
```
MUST VERIFY:
- seed=42 still set before all training
- eval_set does NOT include test data
- Early stopping uses val metrics, not test metrics
- Model checkpoint saved correctly
```

## Verification Output Format
```
## Code Change Verification Report

### Change Summary
Files: [list]
Intent: [what the change was supposed to do]

### Dependency Impact
| File Changed | Importers | Risk Level |
|---|---|---|
| path/to/file.py | [files that import it] | Low/Medium/High/Critical |

### Competition Pipeline Impact
[ ] Preprocessing integrity: [maintained/at risk — details]
[ ] Metric computation: [maintained/at risk — details]
[ ] Reproducibility: [maintained/at risk — details]
[ ] Submission format: [maintained/at risk — details]

### Regression Risk
[What could have broken and how likely]

### Verification Performed
[ ] Code review: [PASS/FAIL/PARTIAL]
[ ] Unit tests: [PASS/FAIL/NOT RUN — which tests]
[ ] Smoke test: [PASS/FAIL/NOT RUN]

### Unverified Risks
[Things that should be verified but weren't, and why]

### Verdict
[VERIFIED — change is safe]
[CONDITIONAL — safe if [specific condition]]
[NEEDS MORE TESTING — [what test is needed]]
[REVERTING RECOMMENDED — [reason]]
```

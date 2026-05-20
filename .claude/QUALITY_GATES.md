# CAPOS Quality Gate System — VARIANT-GNN

This document defines the quality assurance framework for all work performed in this project. Every task is classified, and each class has mandatory quality gates that must pass before the task is considered complete.

---

## Task Classification

### ROUTINE
**Examples:** Typo fix, comment update, log message change, minor config value  
**Risk:** Low — unlikely to affect competition results  
**Gates:** Quick review → implement → spot-check

### IMPORTANT
**Examples:** New feature, documentation section, metric addition, config change  
**Risk:** Medium — could affect comprehensiveness or clarity  
**Gates:** Inspect → implement → verify → report

### CRITICAL
**Examples:** Pipeline logic change, metric computation, data split logic, threshold change  
**Risk:** High — could change actual results or invalidate claims  
**Gates:** Scientist review → implement → full verification → cross-file impact → report

### HIGH-RISK
**Examples:** GNN mask logic, SMOTE ordering, feature selector fitting, label encoding, ensemble weights  
**Risk:** Critical — could cause data leakage or wrong results without any error  
**Gates:** Scientist + Debugger double-check → implement → verifier → smoke test → report

### DELIVERY-BLOCKING
**Examples:** PDR finalization, submission package, jury presentation materials  
**Risk:** Mission-critical — failure affects competition outcome  
**Gates:** ALL agents review → pre-submission-gate → mission-readiness → jury-adversary → GO/NO-GO decision

---

## Gate Definitions

### Gate G1: Evidence Check
**Requirement:** Any claim must have a file/result backing it  
**Who:** All — this gate is always active  
**Failure:** Claim without evidence → state as assumption, not fact

### Gate G2: Cross-File Impact
**Requirement:** Before implementing, trace what else uses the target code  
**Who:** architect + debugger  
**Failure:** Change breaks downstream module → regression

### Gate G3: Scientific Validity
**Requirement:** Metric claims, pipeline decisions, and experimental results are scientifically defensible  
**Who:** scientist  
**Failure:** Claim not verifiable → remove or qualify it

### Gate G4: Reproducibility Check
**Requirement:** After any pipeline change, seed=42 still produces consistent output  
**Who:** verifier  
**Failure:** Non-deterministic results → cannot reproduce F1=0.8706 for jury

### Gate G5: Competition Alignment
**Requirement:** Change doesn't hurt PDR score, jury impression, or specification compliance  
**Who:** jury-adversary  
**Failure:** Change that looked like improvement actually hurts competition standing

### Gate G6: Security Check
**Requirement:** No secrets committed, no competition data exposed, no injection risk  
**Who:** sentinel  
**Failure:** Data breach risk or disqualification for data misuse

### Gate G7: Smoke Test
**Requirement:** Full pipeline runs and produces valid output  
**Who:** verifier  
**Failure:** Code doesn't run → jury cannot reproduce → §7.5 violation

### Gate G8: Documentation Consistency
**Requirement:** Any implementation change is reflected in docs, and vice versa  
**Who:** documentalist  
**Failure:** README/PDR says one thing, code does another → credibility damage

---

## Gate Application Matrix

| Task Class | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
|---|---|---|---|---|---|---|---|---|
| Routine | ✓ | — | — | — | — | — | — | — |
| Important | ✓ | ✓ | — | — | — | — | — | ✓ |
| Critical | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| High-Risk | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Delivery-Blocking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Producer-Reviewer Separation

For Critical and above tasks, the producer of a change must not be the sole reviewer.

| Task Type | Producer | Reviewer |
|---|---|---|
| Code change | Claude (direct) | debugger + verifier |
| Report/doc | Claude (direct) | documentalist + jury-adversary |
| Experimental claim | scientist | scientist + jury-adversary (adversarial) |
| Architecture decision | architect | scientist + verifier |
| Final deliverable | All agents | pre-submission-gate (independent check) |

---

## Failure Handling

### Gate Failure = Task Not Complete
A task is only complete when all required gates pass.

### Failure Response Protocol
1. **Identify**: Which gate failed? What evidence?
2. **Assess**: Is this blocking? Can we proceed with a risk flag?
3. **Fix or Flag**: Either fix the failure, or explicitly document the accepted risk
4. **Re-gate**: Re-run the failed gate after fix
5. **Report**: Document the failure and resolution in the task report

### Accepted-Risk Protocol
If a gate failure cannot be fixed before deadline:
1. Document the failure explicitly
2. Assess the competition risk (LOW/MEDIUM/HIGH/CRITICAL)
3. Prepare a jury defense for this weakness
4. Include in PDR limitations section if appropriate
5. Never hide known weaknesses — jury will find them anyway

---

*CAPOS Quality Gates v1.0 — 2026-05-20*

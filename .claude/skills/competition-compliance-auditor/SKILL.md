---
name: competition-compliance-auditor
description: Use when auditing the full project for TEKNOFEST 2026 Sağlıkta Yapay Zeka competition compliance. Checks category correctness (Üniversite ve Üzeri), metric correctness (Binary F1), data rules, ethics, reproducibility, report structure, and submission package. Activate when user asks "şartnameye uygun mu?", "eksiğimiz var mı?", "teslime hazır mıyız?" or before any major submission.
---

# Competition Compliance Auditor — VARIANT-GNN

When this skill is active, audit the entire project against TEKNOFEST 2026 Şartname v4 with zero tolerance. Produce a structured compliance report with PASS / RISK / FAIL verdicts. Do not soften findings.

## Official Source Boundary

Primary: TEKNOFEST 2026 Türkçe Şartname v4  
Secondary: PDR Şablonu (Üniversite ve Üzeri), PSR Şablonu (Üniversite ve Üzeri)  
Rejected: 2024 spec, third-party summaries, social media, unofficial sources  
Unverifiable items: mark UNVERIFIED, never assume

---

## Domain A — Category & Task Correctness

**Check:**
- [ ] Üniversite ve Üzeri kategorisi (not Lise/EKG/cardiology)
- [ ] Task: binary classification Patojenik(1) / Benign(0)
- [ ] VUS labels excluded from classification
- [ ] Four panels evaluated separately: MASTER / KANSER / PAH / CFTR
- [ ] No genomic address (chromosome, position) used as feature

**FAIL if:** EKG content, Lise template, or genomic address found anywhere.

---

## Domain B — Primary Metric

**Check:**
- [ ] Binary F1 Score reported as primary (§7.3, pos_label=1)
- [ ] F1 = TP / (TP + 0.5×FP + 0.5×FN) — formula correct
- [ ] Accuracy NOT presented as the main success metric
- [ ] Each panel has its own F1 score
- [ ] Global decision threshold θ=0.6831 (canonical); panel thresholds opt-in (General 0.404, KANSER 0.3695, PAH 0.3203, CFTR 0.1922 — jüri kullanmaz)

**Current verified values (CANONICAL: RESULTS_CANONICAL.json):**
```
Jüri beklentisi (dengeli §3.2): balanced F1=0.8134±0.0103
İç hold-out: Test F1=0.8969 | CV F1=0.8936±0.0004 | MCC=0.5863 | PR-AUC=0.9114 | θ=0.6831
Panel F1: MASTER 0.8865 | KANSER 0.944 | PAH 0.9077 | CFTR 0.9412
WITHDRAWN (leaky): 0.8980/0.9269, MCC 0.5356, θ=0.241
```

**FAIL if:** F1 formula wrong, accuracy as primary, panel results missing.

---

## Domain C — Data Integrity & Leakage

**Check:**
- [ ] GROUP-AWARE split by Variant_ID — 0 variants straddle train/test (leakage guard PASSED)
- [ ] Scaler / Imputer fit only on training fold — never on full dataset
- [ ] CategoricalBioFeaturizer deterministic (no test fit); SelectKBest/AutoEncoder REMOVED
- [ ] SMOTE applied only inside training fold (not on validation/test)
- [ ] Calibration set is a held-out 15% of training — no test contamination
- [ ] Augmentation DISABLED (near-twin leakage)
- [ ] Adversarial validation ROC-AUC ≈ 0.50 (confirms no distribution leakage)
- [ ] Competition data NOT committed to repo

**FAIL if:** Any preprocessor fit on test data, SMOTE before split, test labels in training.

---

## Domain D — Reproducibility

**Check:**
- [ ] random_state=42 in all components
- [ ] torch.manual_seed(42) set
- [ ] np.random.seed(42) set
- [ ] requirements.txt pinned versions
- [ ] environment.yml complete
- [ ] Training: single command → `python main.py --mode train --config configs/pdr.yaml`
- [ ] Prediction: single command → `python submission/predict.py --input <file>`
- [ ] 5-seed stability CV F1=0.8738±0.0034 documented
- [ ] Model artifacts in repo (<7MB, REPRODUCE.md) — jüri veri olmadan tahmin üretebilir

**FAIL if:** Non-deterministic run, missing requirements, no single-command entry point.

---

## Domain E — Report Compliance

**Check:**
- [ ] PDR uses official 2026 Üniversite template
- [ ] All 5 sections present: Giriş(10) / Yöntem(25) / Bulgular(30) / Sonuç(25) / Kaynakça(10)
- [ ] Ethics declaration present
- [ ] PSR→PDR discrepancy explained (§4.2 — MCC 0.892→0.5863)
- [ ] GATv2Conv vs SAGEConv correction documented
- [ ] PDR numbers match RESULTS_CANONICAL.json (no withdrawn 0.8980/θ=0.241)

**Known PDR Issues — STATUS (2026-06-02):**
```
✅ §1.2 references fixed: REVEL[2], EVE[9], GATv2[8]
✅ §3.2 threshold: θ=0.6831 (canonical; 0.241 superseded)
✅ §3.1 figure paths: reports/figures/pdr/*
✅ All §3 tables aligned to canonical; balanced jüri F1=0.8134 framing added
```

**FAIL if:** Any withdrawn number (0.8980/0.9269/0.5356/θ=0.241) reappears as current claim → run scripts/check_results_consistency.py.

---

## Domain F — Ethics & Clinical Boundary

**Check:**
- [ ] No clinical diagnosis claim anywhere
- [ ] No treatment recommendation language
- [ ] "Araştırma ve yarışma amacıyla" disclaimer present
- [ ] KVKK compliance stated
- [ ] No patient identifiers in data or reports

**FAIL if:** Any clinical use language found.

---

## Domain G — Git & Security

**Check:**
- [ ] Competition data not in repo (.gitignore covers data/raw/)
- [ ] No .env or credentials committed
- [ ] Last commit Author = msgxr <mgun345@icloud.com> (this PC)
- [ ] No model .pkl/.pt files in repo (they're on Şeyma's machine — correct)

---

## Output Format

```
## COMPLIANCE AUDIT — VARIANT-GNN
Date: [date]

### Domain A — Category & Task: [PASS/RISK/FAIL]
[findings]

### Domain B — Primary Metric: [PASS/RISK/FAIL]
[findings]

### Domain C — Data Integrity: [PASS/RISK/FAIL]
[findings]

### Domain D — Reproducibility: [PASS/RISK/FAIL]
[findings]

### Domain E — Report: [PASS/RISK/FAIL]
[findings — list all known issues]

### Domain F — Ethics: [PASS/RISK/FAIL]
[findings]

### Domain G — Git & Security: [PASS/RISK/FAIL]
[findings]

### OVERALL VERDICT: [GO / CONDITIONAL GO / NO-GO]
[Priority action list]
```

## Escalation Rule

ANY FAIL domain → escalate to `pre-submission-gate`, block submission until fixed.  
RISK domain → flag for PDR teslim öncesi resolution.  
All PASS → confirm with `pre-submission-gate` final gate.

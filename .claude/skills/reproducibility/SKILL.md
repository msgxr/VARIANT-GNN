---
name: reproducibility
description: Use when verifying that VARIANT-GNN can be re-run from scratch by a jury member. Implements the §7.5 scenario: "jury opens the repo and wants to reproduce declared results." Checks seed, requirements, single-command execution, model files, and inference pipeline.
---

# Reproducibility Auditor — VARIANT-GNN

## Official Source Boundary

Tekrar üretilebilirlik gereksinimleri: TEKNOFEST 2026 Şartname §7.5.  
Model artifact dosyaları (.pkl, .pt) Şeyma'nın makinesinde — bu repoda yalnızca JSON config mevcut.  
Çalıştırılabilirlik testi, ortam kurulumu mümkün olan makinede yapılır.  
Doğrulanamayan şartname maddeleri UNVERIFIED olarak işaretlenir.

When this skill is active, simulate a jury member who has never seen this project before, opening the repo and trying to reproduce the declared competition results.

## Legal Basis

TEKNOFEST 2026 Şartname §7.5 (verbatim):
> "Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."

This is not optional. Failure to reproduce = competition credibility loss.

## Jury Scenario: Zero-Knowledge Run

Simulate: "I am a jury member. I have Python and Git. I cloned your repo. I want to reproduce your F1 results."

### Step 0: Repository Clone
- [ ] `git clone https://github.com/msgxr/VARIANT-GNN.git` — works?
- [ ] `.gitignore` correct? (no large model files excluded without alternative access)

### Step 1: Environment Setup
- [ ] `requirements.txt` exists and all versions pinned (e.g., `torch==2.2.1`, not `torch>=2.0`)
- [ ] `environment.yml` exists for conda users?
- [ ] Python version clearly stated (requirement: >=3.10,<3.13)
- [ ] torch-geometric 2.5.3 installation instructions clear?
- [ ] `pip install -r requirements.txt` works on clean env?

### Step 2: Data Placement
- [ ] README says exactly where to place competition data files
- [ ] Expected file names and formats documented
- [ ] Does code fail gracefully if data is missing?

### Step 3: Single-Command Run
- [ ] One command produces predictions: `python submission/predict.py --input test.csv`
- [ ] Or training/inference via `python main.py --mode train --config configs/pdr.yaml` then `python main.py --mode predict --test_file <file>`
- [ ] Command documented in README
- [ ] Output format documented (CSV columns: variant_id, prediction, probability)

### Step 4: Seed Verification
- [ ] `random_state=42` used in all splits
- [ ] `torch.manual_seed(42)` in training code
- [ ] `np.random.seed(42)` in numpy operations
- [ ] `torch.backends.cudnn.deterministic=True` set
- [ ] Same seed → same F1 (verified?)

### Step 5: Model Files
- [ ] Trained model weights saved (`.pt`, `.pkl`)
- [ ] Model files accessible (Drive link / Hugging Face / repo release)
- [ ] Version compatibility: model file + code version documented
- [ ] Panel-specific models available (or single model with panel argument)

### Step 6: Inference Output
- [ ] Output file format matches competition requirements
- [ ] Pathogenic=1 / Benign=0 (or as required) — documented
- [ ] Probability score included
- [ ] Test set order preserved (no shuffle)

### Step 7: Result Verification
- [ ] Expected F1 values documented in README for reference
- [ ] Log file shows panel-based F1 at end of run
- [ ] PSR declared values vs. current code output — do they match?

### Step 8: Timing
- [ ] Full pipeline runtime documented (CPU: ~19 min per PSR)
- [ ] Inference time documented (single: 42ms, batch 2000: 3.8s per PSR)
- [ ] Jury can estimate time before starting

### Step 9: Docker (Optional but Strong)
- [ ] `docker build -t variant-gnn .` works?
- [ ] `docker run variant-gnn --input test.csv` works?
- [ ] Docker image tested on clean machine?

## Blockers vs. Warnings

**Blockers (jury cannot proceed):**
- Missing requirements.txt or version conflicts
- No single command for prediction
- Model files not accessible
- Seed not fixed → non-deterministic results
- Data path hardcoded to local machine

**Warnings (jury can proceed but will note):**
- Docker not available (but CLI works)
- No dummy/example data
- Runtime not documented
- Windows/Linux path inconsistency

## Output Format

```
## Reproducibility Audit

### Scenario: Jury attempts cold run

### BLOCKERS (must fix before final)
1. [Issue] — [File] — [Fix]

### WARNINGS (fix before final, won't block)
1. [Issue] — [File] — [Fix]

### PASSED CHECKS
- [✓] Item

### Reproducibility Score: X/10

### Single Command (as it should appear in README):
[command]

### Estimated Time for Jury: X minutes
```

## Pre-Final Reproducibility Checklist

```
ENVIRONMENT
[ ] requirements.txt with pinned versions (torch==2.2.1, torch-geometric==2.5.3, streamlit==1.35.0)
[ ] Python >=3.10,<3.13 stated
[ ] torch-geometric 2.5.3 install instructions
[ ] Clean env test passed

SEED
[ ] random_state=42 everywhere
[ ] torch.manual_seed(42)
[ ] np.random.seed(42)  
[ ] cudnn.deterministic=True

DATA
[ ] Data placement instructions in README
[ ] Data format documented
[ ] Code fails gracefully without data

INFERENCE
[ ] Single command for prediction
[ ] Output format documented
[ ] Test order preserved

MODELS
[ ] Weights saved and accessible
[ ] Version compatibility documented

VERIFICATION
[ ] Expected F1 values in README
[ ] Log shows panel-based results
[ ] Docker tested (if claimed)
```

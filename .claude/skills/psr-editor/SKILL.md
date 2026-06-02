---
name: psr-editor
description: Use when reviewing the submitted PSR, analyzing hakem scores, mapping PSR weaknesses to PDR corrective actions, or preparing jury defense for PSR↔PDR discrepancies. PSR was submitted 25 March 2026 (score 93/100) — this skill audits retrospective gaps and bridges them into PDR. Activate on any PSR question, hakem score review, MCC drop question, or discrepancy discussion.
---

# Skill: psr-editor — PSR Analysis & PDR Bridge

## Official Source Boundary

Primary: TEKNOFEST 2026 PSR Şablonu (Üniversite ve Üzeri)  
PSR URL: cdn.teknofest.org/.../2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv.docx  
Rejection: 2024 specs, third-party summaries, previous year templates  
Note: PSR is CLOSED (submitted 25 March 2026). This skill only analyzes gaps and bridges them to PDR.

---

## PSR Mission Status

| Item | Value |
|---|---|
| Submission date | 25 March 2026, 17:00 |
| Score | **93 / 100** |
| Status | ACCEPTED (finalists announced 22 April 2026) |
| Action mode | PSR cannot be changed. Identify gaps → fix in PDR |

---

## PSR Hakem Score Breakdown (Complete)

| Section | Score | Max | Gap | Priority |
|---|---|---|---|---|
| §2 International Papers | 9.67 | 10 | -0.33 | Low |
| §3.1 Dataset | 5.00 | 5 | 0 | ✅ PASS |
| §3.2 Data Constraints | 5.00 | 5 | 0 | ✅ PASS |
| §3.3 Preprocessing | 5.00 | 5 | 0 | ✅ PASS |
| §3.4 Label Reliability | 5.00 | 5 | 0 | ✅ PASS |
| §3.5 Class Balance | 5.00 | 5 | 0 | ✅ PASS |
| §3.6 Algorithms | 5.00 | 5 | 0 | ✅ PASS |
| §4.1 Experiment Protocol | 5.00 | 5 | 0 | ✅ PASS |
| §4.2 Metrics | 5.00 | 5 | 0 | ✅ PASS |
| §4.3 Error Analysis | 5.00 | 5 | 0 | ✅ PASS |
| **§4.4 Explainability** | **3.33** | **5** | **-1.67** | 🔴 CRITICAL |
| **§4.5 Technical Evolution** | **3.33** | **5** | **-1.67** | 🔴 CRITICAL |
| **§5.1 Architecture Justification** | **4.00** | **5** | **-1.00** | 🟠 HIGH |
| §5.2 Alternatives | 4.67 | 5 | -0.33 | Medium |
| §5.3 Parameter Selection | 4.67 | 5 | -0.33 | Medium |
| **§5.4 Reproducibility** | **4.33** | **5** | **-0.67** | 🟠 HIGH |
| §5.5 Originality | 4.67 | 5 | -0.33 | Medium |
| §6 References | 9.33 | 10 | -0.67 | Medium |
| **TOTAL** | **93/100** | 100 | **-7** | |

---

## Critical PSR Gaps → PDR Bridge Actions

### 🔴 §4.4 Explainability (3.33/5 — Gap: 1.67 pts)

**What was missing in PSR:**
- No individual SHAP waterfall plots (only aggregate SHAP bar chart submitted)
- No GNNExplainer subgraph visualization
- No panel-by-panel feature group contribution breakdown
- LIME-SHAP agreement rate not quantified numerically

**PDR bridge — required deliverables:**
- [ ] SHAP waterfall plot: ≥1 Patojenik + ≥1 Benign prediction (individual-level)
- [ ] GNNExplainer: concrete subgraph showing which edges/nodes drive prediction
- [ ] Panel feature contribution table: In-silico% / Evolutionary% / Population% — per MASTER/KANSER/PAH/CFTR
- [ ] LIME-SHAP overlap: top-5 feature agreement rate quantified as % (e.g., "top-5 features agreed 80% across SHAP and LIME")
- [ ] Explicit limitation: anonymous column names prevent biological interpretation — state this clearly

### 🔴 §4.5 Technical Evolution (3.33/5 — Gap: 1.67 pts)

**What was missing in PSR:**
- No experiment iteration log showing development history
- No ablation study comparing individual model contributions
- GATv2Conv vs SAGEConv decision not backed by quantitative comparison
- No transfer learning results across panels

**PDR bridge — required deliverables:**
- [ ] Experiment log table: Version | Change Made | Train F1 | Val F1 | Decision/Outcome
- [ ] Ablation table: XGBoost-only | LightGBM-only | GNN-only | Full Ensemble — per panel
- [ ] "Why GATv2Conv not SAGEConv?": panel-level F1 comparison table (also resolves PSR name error)
- [ ] Transfer learning: General panel → CFTR fine-tuning, before/after F1 comparison

### 🟠 §5.1 Architecture Justification (4.00/5 — Gap: 1.00 pt)

**What was missing in PSR:**
- Ensemble weight assignments (30/30/25/15) not quantitatively justified
- No per-model per-panel comparison table

**PDR bridge — required deliverables:**
- [ ] 5 model × 4 panel F1 comparison table (standalone model vs ensemble contribution)
- [ ] Weight justification: "XGBoost 30% because CV evidence shows..." with specific numbers
- [ ] GATv2Conv selection rationale: graph structure captures variant co-occurrence + distance relationship

### 🟠 §5.4 Reproducibility (4.33/5 — Gap: 0.67 pts)

**What was missing in PSR:**
- Concrete command sequence not explicitly shown step-by-step
- CPU-only runtime not documented

**PDR bridge — required deliverables:**
- [ ] Full command sequence: `pip install -r requirements.txt` → `python main.py --mode train --config configs/pdr.yaml` → `python submission/predict.py --input <file>`
- [ ] CPU runtime estimate on standard hardware (document actual test)

---

## PSR↔PDR Critical Discrepancies (JURY WILL ASK)

### Discrepancy 1 — GNN Architecture Name

| Document | Stated Name | Reality |
|---|---|---|
| PSR | VariantSAGEGNN / GraphSAGE | **WRONG** |
| Code | `src/core/gnn.py` | GATv2Conv → VariantGATv2GNN |
| PDR | Must state | VariantGATv2GNN **CORRECT** |

**PDR mandatory text:** State explicitly in Yöntem §2: *"PSR'de VariantSAGEGNN olarak anılan model, kod incelemesinde GATv2Conv (Graph Attention Network v2) mimarisini kullandığı doğrulanmış olup doğru adı VariantGATv2GNN'dir. Bu isim tutarsızlığı PSR terminoloji hatası olarak PDR'de düzeltilmektedir."*

### Discrepancy 2 — MCC Drop (Pilot → Real Competition Data)

| Metric | PSR Pilot (ClinVar EP) | Real Competition (2026-06-02, canonical) |
|---|---|---|
| MCC (overall) | 0.892 | **0.5863** |
| Binary F1 (overall test) | ~0.945 | **0.833** (dengeli jüri F1=0.6063) |
| Threshold | ~0.50 (default) | **0.8514** (balanced-OOF, global) |

**Root cause:** Pilot data = ClinVar Expert Panel (3-4 yıldız, extremely clean labels). Competition data = broader, more complex variant distribution including borderline pathogenic variants.

**PDR mandatory section (§4.2 equivalent):** Dedicated comparison table + explanation paragraph. Do NOT hide this. Jury has PSR scores — they will ask.

### Discrepancy 3 — Decision Threshold

| Document | Threshold |
|---|---|
| PSR | ~0.50 (default, pilot data) |
| PDR | **θ=0.8514 global** (balanced-OOF, canonical); panel-specific opt-in (jüri kullanmaz) |

**PDR explanation required:** Karar eşiği, jüri §3.2 setinin %20-patojenik (%20/%80) olduğu varsayımıyla group-aware OOF üzerinde sınıf-dengeli resample ile F1-optimal türetilir (θ=0.8514). Opt-in panel eşikleri: General 0.404, KANSER 0.3695, PAH 0.3203, CFTR 0.1922 (varsayılan KAPALI).

---

## PSR Technical Reference (Pilot Data — NEVER Present as Competition Results)

**⚠️ These are PILOT values from ClinVar+gnomAD, not competition data.**

| Panel | Macro F1 | ROC-AUC | MCC | Brier |
|---|---|---|---|---|
| General | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 |
| Hereditary Cancer | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 |

Adversarial Validation: General AUC=0.512, Hereditary=0.505, PAH=0.498, CFTR=0.521 (no distribution leakage)  
PSR SHAP groups: In-silico 38%, Evolutionary 27%, Population 18% (pilot data)

**Competition results → see CLAUDE.md §VIII or `.claude/core/OFFICIAL_REFERENCES.md`**

---

## PSR Audit Procedure (Step-by-Step)

When user asks about PSR, hakem scores, or requests jury prep:

1. **Identify scope** → which PSR section is being discussed?
2. **Check hakem table** → what was the score? What was the gap?
3. **Map to bridge actions** → which PDR deliverable addresses this gap?
4. **Verify PDR status** → is the deliverable already in PDR draft?
5. **If not addressed** → draft concrete PDR text (quantified, evidence-backed)
6. **If discrepancy** → apply Discrepancy 1/2/3 text above
7. **Escalate if needed** → see escalation map

---

## Hard Rules

1. **PSR is CLOSED.** Never suggest editing PSR content.
2. **PSR pilot ≠ competition results.** Never conflate or present together without clear labeling.
3. **GNN name in PDR = VariantGATv2GNN always.** Never SAGEConv/SAGEConv.
4. **MCC drop must be explained, not hidden.** PDR §4.2 equivalent is mandatory.
5. **Every bridge action must produce a specific, measurable PDR deliverable.**
6. **Unquantified improvements are rejected.** "Better explainability" needs a number.

---

## Escalation Map

| Trigger | Escalate To |
|---|---|
| PDR text needs to be drafted/edited | `pdr-editor` |
| Jury question rehearsal on discrepancies | `jury-sim` |
| Metric value disputed or uncertain | `data-metric-guardian` |
| Template structure violation | `report-template-checker` |
| Full pre-submission compliance | `competition-compliance-auditor` → `pre-submission-gate` |

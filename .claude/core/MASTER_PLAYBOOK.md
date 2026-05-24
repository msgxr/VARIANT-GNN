# MASTER PLAYBOOK — VARIANT-GNN
# Single Mission Control Reference — CAPOS v2.0

**Project:** VARIANT-GNN | **Team:** XYRA3 | **ID:** 909249  
**Competition:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**PDR Deadline:** 29.06.2026, 17:00 | **Final:** Ağustos–Eylül 2026, Şanlıurfa  
**Last Updated:** 2026-05-24

---

## 1. MISSION OBJECTIVE

Binary classification of missense genetic variants as Patojenik (1) / Benign (0).  
**Winning condition:** Highest Binary F1 Score (pos_label=1=Patojenik) at final evaluation.  
**Current position:** PSR 93/100 (accepted) → PDR due 29.06.2026 → Finals.

---

## 2. TECHNICAL SYSTEM (Verified from Code)

### Architecture
```
XGBoost (30%) + LightGBM (30%) + VariantGATv2GNN/GATv2Conv (25%) + DNN (15%)
→ Stacking meta-learner: Logistic Regression
```

### Pipeline (All preprocessing fit on TRAIN FOLD only — leakage-safe)
```
Raw features → Median Imputation → RobustScaler → SelectKBest(k=35)
→ AutoEncoder(43→16) → SMOTE(train only) → Cosine k-NN Graph(k=10)
```

### Evaluation
```
Split: 80/20 hold-out + Stratified 5-Fold CV | random_state=42
Calibration: 15% held-out from training data
Training: python main.py --mode train --config configs/pdr.yaml
Prediction: python submission/predict.py --input <file>
```

---

## 3. VERIFIED PERFORMANCE (Real Competition Data — 2026-05-20)

### Overall
| CV F1 | Test F1 | MCC | PR-AUC | ROC-AUC | Recall | Threshold |
|---|---|---|---|---|---|---|
| 0.8668 ± 0.0081 | **0.8980** | 0.5356 | 0.9294 | 0.8673 | 0.9725 | 0.241 |

### Panel Results
| Panel | Data Column | F1 | MCC | Threshold |
|---|---|---|---|---|
| MASTER | General | 0.8872 | 0.507 | 0.241 |
| KANSER | Hereditary_Cancer | 0.8960 | 0.649 | 0.281 |
| PAH | PAH | 0.9556 | 0.556 | 0.138 |
| CFTR | CFTR | 0.9524 | 0.674 | 0.108 |

**PSR Pilot (ClinVar EP — NOT competition data):** MCC=0.892, F1=~0.945. Drop explained in PDR §4.2.

---

## 4. COMPETITION SCORING (Verified from Şartname)

| Phase | Weight | Status |
|---|---|---|
| PSR | 0% (eleme only) | ACCEPTED — 93/100 |
| PDR | 0% (eleme only) | Due 29.06.2026 |
| Final Yarışma | **90%** | Ağustos-Eylül 2026 |
| Final Sunum | **10%** | Ağustos-Eylül 2026 |

**⚠️ RISK:** TÜSEB reserves right to change metrics (Şartname §7.5). Monitor announcements.  
**Jury code re-run:** Jury may re-execute code and verify declared results (§7.5 — mandatory).

---

## 5. TEAM & GIT IDENTITY

| Member | Role | GitHub | Email | Machine |
|---|---|---|---|---|
| Muhammed Sina Gün | Sistem, PDR, Yarışma | `msgxr` | mgun345@icloud.com | This Windows PC |
| Şeyma Nur Çebi | Kaptan, ML, Model | `cebi101` | seymanurcebi6@gmail.com | Mac (model artifacts here) |
| Şahin Kara | Biyoinformatik, Veri | — | — | — |
| Burak Küçükcengiz | MLOps, Yazılım | — | — | — |
| Pınar Karadayı Ataş | Danışman | — | — | — |

**Git Rule:** Machine determines identity. This Windows PC → always `msgxr`. No exceptions.  
**Model artifacts** (.pkl/.pt): On Şeyma's Mac — not in this repo.

---

## 6. OPEN PDR BUGS (Fix Before 29.06.2026)

| ID | Location | Problem | Fix |
|---|---|---|---|
| BUG-01 | §1.2 | REVEL citation [3] | → [2] |
| BUG-02 | §1.2 | EVE citation [5] | → [9] |
| BUG-03 | §1.2 | GATv2 citation [7] | → [8] |
| BUG-04 | §3.2 | θ = 0.01 | → θ = 0.241 |
| BUG-05 | §3.1 | reports/roc_curves.png dead | → reports/figures/pdr/05_roc_curves.png |
| BUG-06 | §3.1 | Şekil 2-5 paths dead | verify under reports/figures/pdr/ |
| BUG-07 | Header | Training date 15 Mayıs | → 20 Mayıs 2026 |
| BUG-08 | §4.2 | MCC drop not explained | Add comparison table PSR vs real |
| BUG-09 | §2 | SAGEConv mentioned | → GATv2Conv / VariantGATv2GNN |

---

## 7. PSR GAPS → PDR BRIDGE ACTIONS (Priority Order)

| Priority | Section | Gap | PDR Action |
|---|---|---|---|
| 🔴 | §4.4 Explainability | 3.33/5 (-1.67) | SHAP waterfall + GNNExplainer + panel feature table |
| 🔴 | §4.5 Tech Evolution | 3.33/5 (-1.67) | Experiment log + ablation table (XGB-only vs ensemble) |
| 🟠 | §5.1 Architecture | 4.00/5 (-1.00) | 5×4 model-panel comparison table + weight justification |
| 🟠 | §5.4 Reproducibility | 4.33/5 (-0.67) | Concrete command sequence + CPU timing |

---

## 8. PDR FORMAT RULES (Verified from Official Template)

| Parameter | Value |
|---|---|
| Font | **Aptos** (body 12pt, heading 14pt) |
| Spacing | **1.15** |
| Margins | Top 2.8cm, others 2.5cm |
| **Page limit** | **≤10 pages** (cover + TOC excluded) |
| Structure | Giriş(10) / Yöntem(25) / Bulgular(30) / Sonuç(25) / Kaynakça(10) |
| Mandatory metrics | F1 + MCC + PR-AUC per panel (Bulgular template requirement) |
| Reference format | IEEE |

**10 pages exceeded → report NOT evaluated.**

---

## 9. SKILL ROUTING (Quick Reference)

| User Ask | Activate Skill(s) |
|---|---|
| Yarışma kuralı, şartname sorusu | `official-source-guardian` |
| "Eksikler neler?" / "Hazır mıyız?" | `error-checker` + `mission-readiness` |
| PDR kontrol / düzenleme | `pdr-editor` + `report-template-checker` |
| PSR sorgusu / hakem analizi | `psr-editor` |
| Jüri hazırlığı | `jury-sim` |
| Metrik / veri / leakage | `data-metric-guardian` |
| Kod değişikliği | `code-change-verifier` |
| Deney sonuçları | `experiment-review` |
| Teslim öncesi final kontrol | `pre-submission-gate` |
| Şartnameye uygunluk denetimi | `competition-compliance-auditor` |
| Git push öncesi | `git-identity-guardian` |
| Genel proje analizi | `variant-gnn-review` |

---

## 10. ABSOLUTE PROHIBITIONS

1. PSR pilot results presented as competition performance
2. Four panels treated as single block (always separate)
3. Clinical diagnosis / treatment language anywhere
4. Test set labels used during training (disqualification)
5. Competition data committed to repo (NDA violation)
6. Genomic address used as feature (Şartname violation)
7. Accuracy as standalone primary metric (Şartname violation)
8. Critical findings softened or hidden
9. Irreversible changes without user confirmation

---

## 11. EVIDENCE CHAIN (Every Claim Must Trace Here)

```
Competition rule → Şartname §X.X or official template
Model result → reports/cv_report.json (2026-05-20)
Code architecture → src/core/gnn.py, src/core/pipeline.py
PSR data → .claude/skills/psr-editor/SKILL.md (hakem breakdown)
PDR format → .claude/skills/report-template-checker/SKILL.md
```

---

*CAPOS v2.0 — 16 skills, 9 agents | Updated 2026-05-24*

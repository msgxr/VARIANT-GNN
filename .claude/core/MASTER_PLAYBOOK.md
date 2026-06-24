# MASTER PLAYBOOK — VARIANT-GNN
# Single Mission Control Reference — CAPOS v2.0

**Project:** VARIANT-GNN | **Team:** XYRA3 | **ID:** 909249  
**Competition:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri  
**PDR Deadline:** 29.06.2026, 17:00 | **Final:** Ağustos–Eylül 2026, Şanlıurfa  
**Last Updated:** 2026-06-10 (Q&A-II ile %20-patojenik prior doğrulandı; PROVENANCE anti-drift firewall Check #8; demo bütünlüğü; PDR ≤10 sayfa; evidence-chain dosya yolları gerçek repo yapısına hizalandı. Tüm sayılar RESULTS_CANONICAL.json)

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
Raw → ColumnAligner → CategoricalBioFeaturizer (ACMG bio) → Median Imputation
→ RobustScaler → SMOTE(train only) → Cosine k-NN Graph(k=10)
(SelectKBest(35)+AutoEncoder REMOVED — cost ~5.3pp; full 343 features kept)
→ OOF-stacking meta-learner (Wolpert)
```

### Evaluation
```
Split: GROUP-AWARE 80/20 hold-out by Variant_ID + StratifiedGroupKFold 5-Fold | random_state=42
Leakage guard: 0 variants straddle train/test
Calibration: 15% held-out from training data (isotonic)
Training: python main.py --mode train --config configs/pdr.yaml
Prediction: python main.py --mode predict --test_file <file>
```

---

## 3. VERIFIED PERFORMANCE (Real Competition Data — 2026-06-02, canonical: RESULTS_CANONICAL.json)

### Overall
⭐ **Jüri beklentisi (%20 patojenik — ✅ resmi Q&A-II ile DOĞRULANDI 2026-06-03) = havuzlanmış Binary F1 = 0.6042 ± 0.0324** (θ=0.8415); **RESMİ headline = 0.631** (4-panel %20-F1 ort., CFTR dahil; 3-panel tanı CFTR hariç = 0.6202). İç ayrım gücü aşağıda:

| CV F1 (OOF-stacking) | Test F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| 0.8936 ± 0.0004 | **0.8367** | 0.5112 | 0.9267 | 0.8538 | 0.9241 | 0.7644 | **0.8415 (global)** |

*(fold-CV bileşeni: 0.8812 ± 0.0113). ECE=0.0291, Brier=0.1115.*

### Panel Results (test, global θ=0.8415)
| Panel | Data Column | F1 | MCC |
|---|---|---|---|
| MASTER | General | 0.8185 | 0.4951 |
| KANSER | Hereditary_Cancer | 0.9060 | 0.7135 |
| PAH | PAH | 0.9120 | 0.5053 |
| CFTR | CFTR | 0.7143 | — (n=18, tanımsız) |

Panel eşikleri (opt-in, jüri kullanmaz): General 0.3990, KANSER 0.4532, PAH 0.4434, CFTR 0.1922.

**WITHDRAWN:** Önceki 0.8980/0.9269, MCC 0.5356, θ=0.241 leakage-şişikti — geri çekildi (reports/leakage_quantification.json).
**PSR Pilot (ClinVar EP — NOT competition data):** MCC=0.892, F1=~0.945 → gerçek MCC=0.5112. Drop explained in PDR §4.2.

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

## 6. PDR BUGS — STATUS

| ID | Location | Problem | Status | Fixed |
|---|---|---|---|---|
| BUG-01 | §1.2 | REVEL citation [3]→[2] | ✅ CLOSED | 2026-05-24 |
| BUG-02 | §1.2 | EVE citation [5]→[9] | ✅ CLOSED | 2026-05-24 |
| BUG-03 | §1.2 | GATv2 citation [7]→[8] | ✅ CLOSED | 2026-05-24 |
| BUG-04 | §3.2 | θ=0.01 → θ=**0.8415** (canonical; 0.241 ve 0.8514 supersede) | ✅ CLOSED | 2026-06-02 |
| BUG-05 | §3.1 | Figure paths dead | ✅ CLOSED | 2026-05-24 |
| BUG-06 | §3.1 | Şekil 2-5 path refs | ✅ CLOSED | 2026-05-24 |
| BUG-07 | Header | Date 15 Mayıs → 20 Mayıs | ✅ CLOSED | 2026-05-24 |
| BUG-08 | §4.2 | MCC drop explanation | ✅ CLOSED | Already in PDR §4.2 |
| BUG-09 | §2 | SAGEConv → GATv2Conv | ✅ CLOSED | Already in PDR §2.2 |
| BUG-10 | configs/ | optimize_metric: macro_f1 | ✅ CLOSED | 2026-05-24 |
| BUG-11 | configs/ | threshold_search_range narrow | ✅ CLOSED | 2026-05-24 |
| BUG-12 | configs/ | Panel thresholds wrong | ✅ CLOSED | 2026-05-24 |

**All known bugs closed as of 2026-05-24. PDR ready for final review.**

> **NOTE — jury_predictions.csv:** Sentetik placeholder dosyası **silindi** (2026-06-02 pull). Gerçek submission, jüri kör test verisini sağladığında `python main.py --mode predict --test_file <AL_test.csv>` ile üretilir. Tahmin pipeline'ı `models/threshold.json` (global **θ=0.8415**, canonical) okur.

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
Model result → RESULTS_CANONICAL.json → reports/cv_report.json (2026-06-02)
Code architecture → src/core/gnn.py, src/cli/modes/train.py, src/features/preprocessing.py
PSR data → .claude/skills/psr-editor/SKILL.md (hakem breakdown)
PDR format → .claude/skills/report-template-checker/SKILL.md
```

---

*CAPOS v2.0 — 16 skills, 9 agents | Updated 2026-06-10*

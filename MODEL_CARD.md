# Model Card — VARIANT-GNN

## Model Overview

| Field | Value |
|---|---|
| **Model name** | VARIANT-GNN Hybrid Ensemble |
| **Version** | 3.0.0 (TEKNOFEST 2026) |
| **Architecture** | XGBoost + VariantSAGEGNN (GraphSAGE) + DNN — Multimodal Ensemble |
| **Task** | Binary classification — Variant pathogenicity prediction (Benign / Pathogenic) |
| **License** | MIT |

---

## Intended Use

- **Primary use case:** Predicting whether a genomic variant (SNP/indel) is Benign or Pathogenic, using pre-computed functional annotation scores as features.
- **Target users:** Computational biologists, clinical geneticists, and bioinformatics researchers.
- **Expected input:** A CSV file with one row per variant and numeric annotation features (CADD, SIFT, PolyPhen2, GERP, gnomAD allele frequencies, etc.). Optionally includes ±5 nucleotide/amino-acid context strings for multimodal encoding.
- **Expected output:** Per-variant predictions with calibrated probability scores.

### Out-of-scope use
- De novo variant discovery
- Structural variant classification
- Clinical diagnostic decisions without independent validation

---

## Architecture

### XGBoost Component
- Gradient-boosted trees on the tabular feature matrix
- Handles non-linear feature interactions efficiently
- Hyperparameters tuned via Optuna (`src/training/tune.py`)

### VariantSAGEGNN Component (TEKNOFEST 2026 Primary Model)
- **Inductive, node-level classifier** using GraphSAGE convolutions
- Each variant is a node; edges are built via coordinate-free cosine k-NN in feature space
- 3 SAGEConv blocks with BatchNorm + skip connections + Dropout(0.3)
- **Multimodal fusion:** optionally concatenates nucleotide and amino-acid sequence features from the SequenceEncoder
- **WeightedBCELoss:** dynamically computes class weights for balanced training

### Multimodal Sequence Encoder
- Dual-branch CNN encoder for ±5 nucleotide and ±5 amino-acid context
- Embedding → Conv1d(3,pad=1) → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d
- Output: 32-dim feature vector concatenated to numeric features

### DNN Component
- Feed-forward neural network with BatchNorm + Dropout
- Input dimension is dynamic — inferred from the feature matrix after preprocessing
- Also uses **WeightedBCELoss** for class-balanced training

### Ensemble
- Configurable linear interpolation of the three model probability outputs
- Default weights: `[0.4 XGB, 0.4 GNN, 0.2 DNN]`
- Optional weight optimization via `scipy.optimize.minimize` (Nelder-Mead) on held-out validation set

### Calibration
- Post-hoc calibration using **Isotonic Regression** on a held-out calibration set
- Converts raw ensemble probabilities to well-calibrated risk scores
- Evaluated with ECE (Expected Calibration Error) and Brier Score

---

## Panel-Based Data (TEKNOFEST 2026)

The model supports four distinct genomic panels as specified by the competition:

| Panel | Train (P+B) | Test (P+B) |
|---|---|---|
| General | 1500+1500 | 1000+1000 |
| Hereditary Cancer | 200+200 | 100+100 |
| PAH (Fenilketonüri) | 200+200 | 100+100 |
| CFTR (Kistik Fibrozis) | 70+70 | 30+30 |

Panel-specific training and evaluation is supported via `--panel` CLI flag.

---

## Feature Groups (43 numeric features)

1. **Sekans ve Değişim Bilgisi:** Ref/Alt nucleotide encoding, codon change type, Grantham score
2. **Yerel Sekans Bağlamı:** GC-content window, CpG site, motif disruption score
3. **Biyokimyasal/Yapısal Etkiler:** Polarity change, hydrophobicity, molecular weight, protein impact, solvent accessibility
4. **Evrimsel Korunmuşluk:** GERP++, PhyloP, phastCons, SiPhy
5. **Popülasyon Verileri:** gnomAD AF (5 population), ExAC AF
6. **In Silico Risk Skorları:** SIFT, PolyPhen2, CADD, REVEL, MutPred2, VEST4, PROVEAN, MutationTaster, MetaSVM/LR, M-CAP

---

## Preprocessing

All preprocessing steps are **fit only on training data** within each CV fold:
1. Median imputation (`SimpleImputer`)
2. Robust scaling (`RobustScaler`)
3. Optional: Variance threshold + SelectKBest (mutual information)
4. Optional: AutoEncoder latent feature concatenation (43 → 43+16 = 59 dim)
5. SMOTE oversampling (applied **inside** each fold to avoid leakage)
6. Cosine k-NN sample graph construction for VariantSAGEGNN

---

## Training Details

| Setting | Value |
|---|---|
| Cross-validation | Stratified K-Fold (k=5 default) |
| Model selection metric | **Macro F1** (not accuracy) |
| Calibration split | 15% of training data |
| Test split | 20% of dataset |
| Seed | 42 (all components) |
| Loss function | WeightedBCELoss (class-balanced, for SAGE + DNN) |
| Early stopping | Validation Macro F1 (patience=5) |

---

## Evaluation Metrics

All reported on the held-out test set after calibration:

| Metric | Description |
|---|---|
| Macro F1 | Primary metric; class-balanced F1 |
| ROC-AUC | Area under ROC curve |
| PR-AUC | Area under Precision-Recall curve |
| MCC | Matthews Correlation Coefficient |
| Brier Score | Mean squared probability error (↓ is better) |
| ECE | Expected Calibration Error (↓ is better) |

### External Validation

External validation mode (`python main.py --mode external_val`) loads a pre-trained model and evaluates on new test data, computing F1 / ROC-AUC / Brier Score / Precision / Recall and exporting a JSON report.

---

## Data Requirements

### Input columns
- `Variant_ID` (string) — unique identifier, **preserved through pipeline, never used as feature**
- Numeric annotation features (arbitrary count — no hardcoded feature dimensionality)
- `Label` (0 = Benign, 1 = Pathogenic) — required for training, optional for prediction
- `Panel` (optional) — panel identifier for panel-specific training/evaluation
- `Nuc_Context` / `AA_Context` (optional) — ±5 nucleotide/amino-acid context strings

### Data contract
See `data_contracts/variant_schema.py` for the Pydantic v2 schema.

---

## Limitations

- Class imbalance is handled by SMOTE + WeightedBCELoss but performance may degrade on highly imbalanced datasets
- Requires pre-computed variant annotation scores — does not perform raw sequence analysis
- VUS (Variants of Unknown Significance) support is architecturally present but requires annotated VUS training data

---

## Ethical Considerations

- This model is a **research tool** and should not be used as the sole basis for clinical diagnostic decisions
- Predictions should be interpreted alongside clinical data, family history, and expert clinical genetics review
- Performance may vary across ancestry groups depending on the composition of gnomAD allele frequency features

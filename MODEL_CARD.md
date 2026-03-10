# Model Card — VARIANT-GNN

## Model Overview

| Field | Value |
|---|---|
| **Model name** | VARIANT-GNN Hybrid Ensemble |
| **Version** | 2.0.0 |
| **Architecture** | XGBoost + Graph Neural Network (GCN/GAT) + Deep Neural Network (DNN) |
| **Task** | Binary classification — Variant pathogenicity prediction (Benign / Pathogenic) |
| **License** | MIT |

---

## Intended Use

- **Primary use case:** Predicting whether a genomic variant (SNP/indel) is Benign or Pathogenic, using pre-computed functional annotation scores as features.
- **Target users:** Computational biologists, clinical geneticists, and bioinformatics researchers.
- **Expected input:** A CSV file with one row per variant and numeric annotation features (CADD, SIFT, PolyPhen2, GERP, gnomAD allele frequencies, etc.). See `data_contracts/sample_input.csv` for a working example.
- **Expected output:** Per-variant predictions with calibrated probability scores. See `data_contracts/sample_output.csv`.

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

### GNN Component
- Treats each feature as a graph node
- Edges connect features with correlation > `corr_threshold` (default 0.25), built from **training fold only** (no leakage)
- Supports GCNConv or GATConv (configurable in `configs/default.yaml`)

### DNN Component
- Feed-forward neural network with BatchNorm + Dropout
- Input dimension is dynamic — inferred from the feature matrix after preprocessing

### Ensemble
- Configurable linear interpolation of the three model probability outputs
- Default weights: `[0.4 XGB, 0.4 GNN, 0.2 DNN]`
- Optional weight optimization via `scipy.optimize.minimize` (Nelder-Mead) on held-out validation set

### Calibration
- Post-hoc calibration using **Isotonic Regression** on a held-out calibration set (separate from training and test)
- Converts raw ensemble probabilities to well-calibrated risk scores
- Evaluated with ECE (Expected Calibration Error) and Brier Score

---

## Preprocessing

All preprocessing steps are **fit only on training data** within each CV fold:
1. Median imputation (`SimpleImputer`)
2. Robust scaling (`RobustScaler`)
3. Optional: Variance threshold + SelectKBest (mutual information)
4. Optional: AutoEncoder latent feature concatenation
5. SMOTE oversampling (applied **inside** each fold to avoid leakage)
6. Graph topology derived from training-fold Pearson correlation

---

## Training Details

| Setting | Value |
|---|---|
| Cross-validation | Stratified K-Fold (k=5 default) |
| Model selection metric | **Macro F1** (not accuracy) |
| Calibration split | 15% of training data |
| Test split | 20% of dataset |
| Seed | 42 (all components) |

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

---

## Data Requirements

### Input columns
- `Variant_ID` (string) — unique identifier, **preserved through pipeline, never used as feature**
- Numeric annotation features (arbitrary count — no hardcoded feature dimensionality)
- `Label` (0 = Benign, 1 = Pathogenic) — required for training, optional for prediction

### Data contract
See `data_contracts/variant_schema.py` for the Pydantic v2 schema.
Validate your dataset with:
```python
from data_contracts.variant_schema import validate_dataset
result = validate_dataset(df)
if not result.valid:
    print(result.errors)
```

---

## Limitations

- Class imbalance is handled by SMOTE but performance may degrade on highly imbalanced datasets
- Requires pre-computed variant annotation scores — does not perform raw sequence analysis
- VUS (Variants of Unknown Significance) support is architecturally present (`LABEL_MAP` in `HybridEnsemble`) but requires annotated VUS training data for multi-class operation

---

## Ethical Considerations

- This model is a **research tool** and should not be used as the sole basis for clinical diagnostic decisions
- Predictions should be interpreted alongside clinical data, family history, and expert clinical genetics review
- Performance may vary across ancestry groups depending on the composition of gnomAD allele frequency features

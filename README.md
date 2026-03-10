# VARIANT-GNN

**Graph Neural Network + Explainable AI for Genomic Variant Pathogenicity Prediction**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![CI](https://github.com/msgxr/VARIANT-GNN/actions/workflows/ci.yml/badge.svg)](https://github.com/msgxr/VARIANT-GNN/actions)

> **Warning:** This system is for research use only. Do not use for clinical diagnosis or treatment decisions without independent expert validation.

---

## Overview

VARIANT-GNN predicts whether a human genomic variant is **Pathogenic** or **Benign** using a hybrid ensemble of XGBoost, a Graph Neural Network (GNN), and a Deep Neural Network (DNN). 

**New in v2.0 (TEKNOFEST 2026 Ready):**
- **Clinical Decision Support Assistant:** Analyzes SHAP feature importance to generate automated, NLP-based clinical interpretations in Turkish.
- **Genetic Interaction Graph Visualization:** Renders the underlying GNN correlation edges using NetworkX, explicitly demonstrating the graph mechanics to users.
- **Strict Data Pipeline:** Fixes all known data leakage, hardcoded-feature, and calibration deficiencies from v1.

See [SECURITY.md](SECURITY.md) and [MODEL_CARD.md](MODEL_CARD.md) for full details.

---

## Architecture

```
Input CSV (any number of features, Variant_ID preserved)
      |
[Schema Validation]    data_contracts/variant_schema.py
      |
[VariantPreprocessor]    fit ONLY on training fold (no leakage)
   SimpleImputer (median)
   RobustScaler
   Optional: VarianceThreshold + SelectKBest
   Optional: AutoEncoder latent concatenation
   SMOTE   inside fold only
   Graph edges  training-fold correlation
      |
    
  XGBoost GNN   DNN
    
          |
     [HybridEnsemble]    configurable weights
          |
  [EnsembleCalibrator]    Isotonic / Platt Scaling
          |
    Output: Variant_ID, Prediction, Calibrated_Risk, Confidence
```

---

## Project Structure (v2.0)

```
VARIANT-GNN/
 configs/
    default.yaml              All hyperparameters (single source of truth)
 data_contracts/
    __init__.py
    variant_schema.py         Pydantic v2 schema validation
    sample_input.csv          Example input
    sample_output.csv         Example output
 src/
    config/
       __init__.py
       settings.py           Typed dataclass Settings loader
    data/
       __init__.py
       loader.py             Schema-validated loader, preserves Variant_ID
    features/
       __init__.py
       autoencoder.py        sklearn-compatible AutoEncoder
       preprocessing.py      Leakage-free VariantPreprocessor
    graph/
       __init__.py
       builder.py            Pluggable graph construction (ABC)
    models/
       __init__.py
       gnn.py                FeatureGNN (dynamic input dim)
       dnn.py                VariantDNN (dynamic input dim)
       ensemble.py           HybridEnsemble with weight optimization
    training/
       __init__.py
       trainer.py            Leakage-free CV, Macro F1 model selection
       tune.py               Optuna hyperparameter search
    calibration/
       __init__.py
       calibrator.py         EnsembleCalibrator (Platt/Isotonic)
    evaluation/
       __init__.py
       metrics.py            F1, Brier, ECE, MCC, ROC-AUC, PR-AUC
       plots.py              Confusion matrix, ROC, PR, calibration plots
    inference/
       __init__.py
       pipeline.py           InferencePipeline, preserves Variant_ID
    explainability/
       __init__.py
       shap_explainer.py
       lime_explainer.py
       gnn_explainer.py
       clinical_insight.py   # Otomatik Türkçe Klinik Karar Destek Raporlayıcı
       graph_viz.py          # NetworkX tabanlı GNN Etkileşim Grafı ve Isı Haritası
    utils/
        __init__.py
        logging_cfg.py        Centralized logging
        seeds.py              Deterministic seed management
        serialization.py      Secure model save/load (weights_only=True)
 tests/
    conftest.py               Shared pytest fixtures
    unit/                     Unit tests for each module
    integration/              End-to-end pipeline tests
    smoke/                    Import and instantiation checks
 .github/
    workflows/
        ci.yml                Lint + test + security scan
 main.py                       CLI: train / tune / eval / predict / crossval
 app.py                        Streamlit dashboard
 MODEL_CARD.md
 SECURITY.md
 requirements.txt
```

---

## Installation

```bash
git clone https://github.com/msgxr/VARIANT-GNN.git
cd VARIANT-GNN

# Install CPU-only PyTorch (for GPU, replace +cpu with +cu121)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

pip install -r requirements.txt
```

---

## Usage

### Train
```bash
python main.py --mode train --data_file data/train_variants.csv
```

### Hyperparameter tuning (Optuna)
```bash
python main.py --mode tune --data_file data/train_variants.csv --n_trials 50
```

### Cross-validation report
```bash
python main.py --mode crossval --data_file data/train_variants.csv
```

### Evaluate saved model on test set
```bash
python main.py --mode eval --data_file data/test_variants.csv
```

### Batch prediction (preserves Variant_ID)
```bash
python main.py --mode predict --test_file data/test_variants_blind.csv
# Output: reports/predictions.csv
```

### Streamlit dashboard
```bash
streamlit run app.py
# Opens: http://localhost:8501
```

---

## Running Tests

```bash
# Smoke tests (fast, import checks)
pytest tests/smoke/ -v

# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

```yaml
ensemble:
  weights: [0.4, 0.4, 0.2]          # XGB, GNN, DNN
  optimize_weights: false            # optimize on held-out val set

calibration:
  method: isotonic                   # or "sigmoid" (Platt Scaling)
  calibration_size: 0.15

training:
  cv_folds: 5
  test_size: 0.20
  batch_size: 32

preprocessing:
  smote_enabled: true
  use_autoencoder: true
  corr_threshold: 0.25
```

---

## Critical Fixes in v2.0

| # | Issue | v1 | v2 |
|---|---|---|---|
| 1 | Data leakage (scaler) | fit_transform on full dataset | fit inside each CV fold only |
| 2 | Data leakage (SMOTE) | SMOTE before CV | SMOTE inside each fold |
| 3 | Data leakage (AutoEncoder) | trained on full dataset | trained inside each fold |
| 4 | Data leakage (graph) | edges from full correlation | edges from training-fold correlation |
| 5 | Hardcoded 34 features | `np.zeros((1, 34))` in app.py | dynamic input_dim everywhere |
| 6 | Model selection metric | accuracy | Macro F1 |
| 7 | No calibration | raw probabilities | Isotonic/Platt on separate cal split |
| 8 | Missing metrics | F1, Precision, Recall, AUC only | + Brier Score, ECE, MCC, PR-AUC |
| 9 | Variant_ID dropped | lost during preprocessing | preserved in LoadedDataset.metadata |
| 10 | Hardcoded ensemble weights | `[0.4, 0.4, 0.2]` in code | configurable in YAML, optimizable |
| 11 | No schema validation | no input validation | Pydantic v2 schema |
| 12 | torch.load security (CVE) | `torch.load(path)` | `torch.load(path, weights_only=True)` |
| 13 | No config system | flat Config class | typed dataclass + YAML |
| 14 | No logging/seed control | print statements | centralized logging + seed management |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Macro F1** | Primary  class-balanced; used for model selection |
| ROC-AUC | Discrimination ability |
| PR-AUC | Performance on imbalanced classes |
| MCC | Matthews Correlation Coefficient |
| Brier Score | Probabilistic calibration quality |
| ECE | Expected Calibration Error |

---

## Technology Stack

| Component | Technology |
|---|---|
| GNN | PyTorch Geometric (GCNConv / GATConv) |
| Gradient Boosting | XGBoost |
| DNN | PyTorch |
| Preprocessing | scikit-learn, imbalanced-learn |
| Calibration | scikit-learn (IsotonicRegression, CalibratedClassifierCV) |
| XAI | SHAP, LIME, GNNExplainer |
| Config | PyYAML + dataclasses |
| Validation | Pydantic v2 |
| Hyperparameter optimization | Optuna |
| Web UI | Streamlit |
| Testing | pytest |
| CI | GitHub Actions |

---

## License

MIT License  see [LICENSE](LICENSE) for details.

> This system is a research tool. See [MODEL_CARD.md](MODEL_CARD.md) for limitations and ethical considerations.

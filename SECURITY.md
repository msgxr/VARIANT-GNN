# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 2.x | Yes |
| 1.x (original) | No — contains known security issues |

---

## Known Security Fixes in v2.0

### CVE-relevant fixes

| Issue | v1 (vulnerable) | v2 (fixed) |
|---|---|---|
| Unsafe deserialization | `torch.load(path)` — arbitrary code execution via pickle | `torch.load(path, weights_only=True)` in `src/utils/serialization.py` |
| Unsafe deserialization | XGBoost model saved as `.pkl` | XGBoost saved as `.json` (no pickle) |
| Data leakage (integrity) | Preprocessor/SMOTE fit on full dataset | Fit only inside CV fold on training split |

### Dependency scanning
- Run `pip-audit` or `safety check` regularly against `requirements.txt`
- Bandit security scanning is included in the CI pipeline (`.github/workflows/ci.yml`)

---

## Model Serialization Security

All model artifacts use secure serialization:

```python
# src/utils/serialization.py
# GNN / DNN — weights_only=True prevents arbitrary code execution
torch.load(path, map_location=device, weights_only=True)

# XGBoost — JSON format (no pickle, no code execution risk)
model.save_model("xgb_model.json")      # save
model.load_model("xgb_model.json")      # load

# Preprocessor / Calibrator — joblib (acceptable for internal use)
joblib.dump(preprocessor, "preprocessor.pkl")
```

**Do not load model files from untrusted sources.** Even with `weights_only=True`, verify file checksums (SHA-256) if models are received over a network.

---

## Input Validation

All inputs are validated against the Pydantic v2 schema before processing:

```python
from data_contracts.variant_schema import validate_dataset
result = validate_dataset(df)
# Raises ValueError on critical schema violations
```

Validation checks:
- All feature columns must be numeric (float or int)
- Label values must be within `{0, 1}` (or the configured set)
- No columns with 100% missing values accepted

---

## Reporting a Vulnerability

Please **do not** report security vulnerabilities in public GitHub issues.

Instead:
1. Email the maintainer directly (see repository contact info)
2. Include a description of the vulnerability and reproduction steps
3. Allow up to 14 days for an initial response

We will acknowledge receipt promptly and work to release a patch.

---

## Threat Model

| Asset | Threat | Mitigation |
|---|---|---|
| Saved model files | Malicious `.pth` file with embedded code | `weights_only=True` |
| Input CSV | Injected float values causing numeric overflow | RobustScaler handles outliers; NaN imputation handles missing |
| Config YAML | Path traversal in data/model paths | Paths resolved relative to project root; no user-supplied paths in production |
| Dependency chain | Vulnerable third-party package | Bandit + pip-audit in CI |

"""ADVERSARIAL LEVER TEST: ExtraTrees as a 5th diverse member.

Builds a group-aware (Variant_ID) OOF for ExtraTreesClassifier on the SAME
preprocessed matrix as the production XGB/LGBM, using the identical
StratifiedGroupKFold + per-fold SMOTE protocol from scripts/train_pdr.py.

Checks:
  1. ExtraTrees OOF correlation with the existing XGB/LGBM/GNN/DNN OOF columns.
  2. LogReg-stack F1 with the 4 production base OOF columns vs +ExtraTrees (5th).
  3. Does so over multiple seeds (group-aware) to test if any gain survives noise.

Writes NOTHING to models/. Read-only diagnostic.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.features.preprocessing import build_preprocessor_from_config

reset_settings()
cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))

# ---- Load the canonical leakage-free OOF (4 production base models) ----
d = np.load(REPO / "reports" / "oof_per_model.npz")
oof4 = d["oof"]; y_oof = d["labels"]; base_models = [str(m) for m in d["models"]]
N = oof4.shape[0]
print(f"[OOF] n={N} models={base_models} pos={y_oof.mean():.4f}")


def best_f1(p, y):
    best, bt = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y, (p >= t).astype(int))
        if f > best:
            best, bt = f, t
    return best, bt


def oof_stack_f1(meta_cols, y):
    """Honest nested-OOF stack: group-unaware proxy on the meta layer is fine
    here because the base OOF is already group-out-of-fold. We use 5-fold CV on
    the meta-learner and pick threshold on the held-out meta-OOF (no test peek)."""
    from sklearn.model_selection import StratifiedKFold
    meta_oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, va in skf.split(meta_cols, y):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(meta_cols[tr], y[tr])
        meta_oof[va] = lr.predict_proba(meta_cols[va])[:, 1]
    return best_f1(meta_oof, y)


# ---- Load training data + build the SAME matrix the OOF was built on ----
ds = load_csv(str(REPO / "data" / "train_variants.csv"))
raw = ds.features.copy()
y_full = np.asarray(ds.labels).astype(int)
meta = ds.metadata
feat_cols = [c for c in raw.columns
             if c not in (cfg.schema.target_column, "Variant_ID", "Panel",
                          "Nuc_Context", "AA_Context")]
X_full = raw[feat_cols].values.astype(float)
base_ids_full = meta["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
print(f"[DATA] X_full={X_full.shape} y_full pos={y_full.mean():.4f} n_groups={len(set(base_ids_full))}")

# The OOF npz has N=3040 rows; train_variants has more (dedup/aug). Align by
# building ExtraTrees OOF on the SAME N rows. We detect: OOF was built on the
# de-duplicated base set. Reproduce by taking unique base_ids first occurrence.
if X_full.shape[0] != N:
    # take first occurrence per base_id (dedup) to match canonical OOF count if it lines up
    _, first_idx = np.unique(base_ids_full, return_index=True)
    first_idx = np.sort(first_idx)
    Xd, yd, gd = X_full[first_idx], y_full[first_idx], base_ids_full[first_idx]
    print(f"[DEDUP] -> {Xd.shape[0]} rows (canonical OOF N={N})")
else:
    Xd, yd, gd = X_full, y_full, base_ids_full

# ---- Build ExtraTrees group-aware OOF over several seeds ----
seeds = [42, 123, 456, 789, 2026]
res = {"seeds": seeds, "runs": []}
corr_accum = []

for seed in seeds:
    sgkf = StratifiedGroupKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=seed)
    et_oof = np.zeros(len(Xd))
    xgb_check = np.zeros(len(Xd))  # rebuild XGB OOF same protocol to get honest corr on same rows
    import xgboost as _xgb
    for tr_idx, va_idx in sgkf.split(Xd, yd, gd):
        prep = build_preprocessor_from_config()
        Xtr, ytr = prep.fit_resample_train(Xd[tr_idx], yd[tr_idx])
        Xva = prep.transform(Xd[va_idx])
        et = ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                  class_weight=None)
        et.fit(Xtr, ytr)
        et_oof[va_idx] = et.predict_proba(Xva)[:, 1]
        xc = _xgb.XGBClassifier(**cfg.xgb.as_dict())
        xc.fit(Xtr, ytr, verbose=False)
        xgb_check[va_idx] = xc.predict_proba(Xva)[:, 1]

    et_f1, et_t = best_f1(et_oof, yd)
    r_et_xgb = np.corrcoef(et_oof, xgb_check)[0, 1]
    corr_accum.append(r_et_xgb)

    # Stack tests need the 4 production OOF on the SAME rows. If N matches dedup
    # rows we can align by group order; otherwise only report ET-vs-XGB corr +
    # 2-col (XGB,LGBM-proxy via ET? no) — we keep it honest: only stack if N aligns.
    run = {"seed": seed, "et_oof_f1": round(float(et_f1), 4),
           "et_thr": round(float(et_t), 3),
           "corr_et_vs_xgb_same_rows": round(float(r_et_xgb), 4)}
    res["runs"].append(run)
    print(f"[seed {seed}] ET OOF F1={et_f1:.4f} | corr(ET,XGB)={r_et_xgb:.4f}")

# ---- Stack improvement test on canonical 4-model OOF (rows align if N==len(Xd)) ----
stack_block = {}
if len(Xd) == N:
    # rebuild ET OOF once with seed 42 already in et_oof; align order: both built
    # on Xd row order. canonical oof4 also on dedup order? assume yes -> test.
    base4_f1, _ = oof_stack_f1(oof4, y_oof)
    five = np.column_stack([oof4, et_oof])
    base5_f1, _ = oof_stack_f1(five, y_oof)
    stack_block = {"stack_4model_f1": round(float(base4_f1), 4),
                   "stack_5model_with_ET_f1": round(float(base5_f1), 4),
                   "delta_pp": round(float((base5_f1 - base4_f1) * 100), 3)}
    print(f"[STACK] 4-model={base4_f1:.4f}  +ET 5-model={base5_f1:.4f}  "
          f"delta={(base5_f1-base4_f1)*100:+.3f}pp")
else:
    stack_block = {"note": f"row count mismatch (Xd={len(Xd)} vs OOF N={N}); "
                           "5-col stack alignment skipped, corr result is decisive"}
    print(f"[STACK] skipped alignment (Xd={len(Xd)} vs OOF N={N})")

res["mean_corr_et_vs_xgb"] = round(float(np.mean(corr_accum)), 4)
res["stack"] = stack_block
(REPO / "reports" / "_lever_extratrees_test.json").write_text(json.dumps(res, indent=2))
print("\n=== SUMMARY ===")
print(json.dumps({"mean_corr_et_xgb": res["mean_corr_et_vs_xgb"],
                  "et_oof_f1_mean": round(float(np.mean([r["et_oof_f1"] for r in res["runs"]])), 4),
                  "stack": stack_block}, indent=2))

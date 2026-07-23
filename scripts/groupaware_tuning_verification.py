# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

import json
import warnings

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier

from src.data.loader import load_csv
from src.features.preprocessing import VariantPreprocessor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42

ds = load_csv("data/train_variants.csv")
Xdf = ds.features
y = ds.labels.astype(int)
groups_all = ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
tr, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(Xdf, y, groups_all))
gtr = groups_all[tr]
pre = VariantPreprocessor(corr_threshold=0.25, use_autoencoder=False, use_feature_selection=False)
pre.fit(Xdf.iloc[tr], y[tr])
Xtr = np.asarray(pre.transform(Xdf.iloc[tr]), dtype=np.float32)
ytr = y[tr].astype(int)
Xte = np.asarray(pre.transform(Xdf.iloc[te]), dtype=np.float32)
yte = y[te].astype(int)
spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)


def cv_f1(params):
    fs = []
    for tri, vai in sgkf.split(Xtr, ytr, gtr):
        m = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=SEED)
        m.fit(Xtr[tri], ytr[tri])
        p = m.predict_proba(Xtr[vai])[:, 1]
        fs.append(f1_score(ytr[vai], (p >= 0.5).astype(int), pos_label=1, zero_division=0))
    return float(np.mean(fs))


# mevcut config (pdr.yaml)
cur = dict(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
)
cur_cv = cv_f1(cur)
print(f"MEVCUT config group-aware CV F1 = {cur_cv:.4f}")


def obj(t):
    p = dict(
        max_depth=t.suggest_int("max_depth", 3, 10),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=t.suggest_int("n_estimators", 50, 300),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 8),
        reg_alpha=t.suggest_float("reg_alpha", 0.0, 0.5),
        reg_lambda=t.suggest_float("reg_lambda", 0.5, 2.0),
    )
    return cv_f1(p)


st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(obj, n_trials=40, show_progress_bar=False)
print(f"GROUP-AWARE tuned CV F1 = {st.best_value:.4f}  (Δ vs mevcut = {st.best_value - cur_cv:+.4f})")
print("best params:", {k: round(v, 3) if isinstance(v, float) else v for k, v in st.best_params.items()})


# held-out karşılaştırma
def holdout_f1(params):
    m = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(f1_score(yte, (p >= 0.5).astype(int), pos_label=1, zero_division=0))


print(f"\nHELD-OUT F1: mevcut={holdout_f1(cur):.4f} | group-aware-tuned={holdout_f1(st.best_params):.4f}")
json.dump(
    dict(cur_cv=cur_cv, ga_cv=st.best_value, delta=st.best_value - cur_cv, best=st.best_params),
    open(
        "/private/tmp/claude-501/-Users-seymanur-Desktop-VARIANT-GNN/3e66a81a-dcd4-4709-9460-3921b6d773ae/scratchpad/ga_tune_result.json",
        "w",
    ),
    indent=2,
)

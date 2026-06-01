"""Decisive stack test on aligned rows: does ET add to XGB+LGBM stack?
Rebuilds XGB, LGBM, ET group-aware OOF on the SAME 3224 dedup rows, then
compares LogReg-stack F1 of [XGB,LGBM] vs [XGB,LGBM,ET]. 3 seeds.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.features.preprocessing import build_preprocessor_from_config
import xgboost as _xgb, lightgbm as _lgb

reset_settings(); cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))
ds = load_csv(str(REPO / "data" / "train_variants.csv"))
raw = ds.features.copy(); y_full = np.asarray(ds.labels).astype(int); meta = ds.metadata
feat_cols = [c for c in raw.columns if c not in (cfg.schema.target_column, "Variant_ID","Panel","Nuc_Context","AA_Context")]
X = raw[feat_cols].values.astype(float)
gids = meta["Variant_ID"].astype(str).str.replace(r"_aug\d*$","",regex=True).values
_, fi = np.unique(gids, return_index=True); fi = np.sort(fi)
Xd, yd, gd = X[fi], y_full[fi], gids[fi]

def best_f1(p,y):
    b,t=0.,.5
    for th in np.linspace(.05,.95,181):
        f=f1_score(y,(p>=th).astype(int))
        if f>b: b,t=f,th
    return b

def meta_stack(cols,y):
    mo=np.zeros(len(y)); skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,va in skf.split(cols,y):
        lr=LogisticRegression(max_iter=1000); lr.fit(cols[tr],y[tr])
        mo[va]=lr.predict_proba(cols[va])[:,1]
    return best_f1(mo,y)

out={"runs":[]}
for seed in [42,123,456]:
    sgkf=StratifiedGroupKFold(cfg.training.cv_folds,shuffle=True,random_state=seed)
    xo=np.zeros(len(Xd)); lo=np.zeros(len(Xd)); eo=np.zeros(len(Xd))
    for tr,va in sgkf.split(Xd,yd,gd):
        p=build_preprocessor_from_config(); Xtr,ytr=p.fit_resample_train(Xd[tr],yd[tr]); Xva=p.transform(Xd[va])
        xc=_xgb.XGBClassifier(**cfg.xgb.as_dict()); xc.fit(Xtr,ytr,verbose=False); xo[va]=xc.predict_proba(Xva)[:,1]
        lc=_lgb.LGBMClassifier(**cfg.lgbm.as_dict()); lc.fit(Xtr,ytr,callbacks=[_lgb.log_evaluation(-1)]); lo[va]=lc.predict_proba(Xva)[:,1]
        ec=ExtraTreesClassifier(n_estimators=400,n_jobs=-1,random_state=seed); ec.fit(Xtr,ytr); eo[va]=ec.predict_proba(Xva)[:,1]
    f2=meta_stack(np.column_stack([xo,lo]),yd)
    f3=meta_stack(np.column_stack([xo,lo,eo]),yd)
    r_ex=np.corrcoef(eo,xo)[0,1]; r_el=np.corrcoef(eo,lo)[0,1]
    out["runs"].append({"seed":seed,"stack_XGB_LGBM":round(f2,4),"stack_XGB_LGBM_ET":round(f3,4),
                        "delta_pp":round((f3-f2)*100,3),"corr_ET_XGB":round(r_ex,4),"corr_ET_LGBM":round(r_el,4)})
    print(f"[seed {seed}] XGB+LGBM={f2:.4f}  +ET={f3:.4f}  d={(f3-f2)*100:+.3f}pp  r(ET,XGB)={r_ex:.3f} r(ET,LGBM)={r_el:.3f}")
import numpy as np
deltas=[r["delta_pp"] for r in out["runs"]]
out["mean_delta_pp"]=round(float(np.mean(deltas)),3); out["std_delta_pp"]=round(float(np.std(deltas)),3)
(REPO/"reports"/"_lever_extratrees_stack.json").write_text(json.dumps(out,indent=2))
print(f"\nMEAN delta = {out['mean_delta_pp']:+.3f}pp  (std {out['std_delta_pp']}pp)")

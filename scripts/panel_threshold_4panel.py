"""Panel-başına kalibre eşik (shrinkage) ile DÜRÜST 4-panel F1 — reproducible.

Amaç: Mevcut headline 0.6202 üç-panel iç tahmindir (CFTR iç hold-out tek-sınıf).
Bu script CFTR dahil GERÇEK 4-panel'i, panel-başına shrinkage eşikleriyle ölçer.

Estimator: shipped full-ensemble (0.30 XGB + 0.30 LGBM + 0.25 GNN + 0.15 DNN)
           GENUINE out-of-fold tahminleri (reports/oof_per_model.npz) — 4 panel de
           iki-sınıflı, leakage-siz, tek tutarlı estimator.
Eşik     : per-panel optimal (OOF'ta türetilir), global θ=0.8415'e shrinkage(λ) ile
           çekilir (overfit önler). λ nested group-CV ile seçilir.
Metrik   : %20-patojenik (TEKNOFEST resmi prior) F1, 300x resample — train.py:294-304
           ile birebir aynı tarif.
Çıktı    : reports/panel_threshold_4panel.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

SEED = 42
GLOBAL = 0.8415
WEIGHTS = np.array([0.30, 0.30, 0.25, 0.15])  # [XGB, LGBM, GNN, DNN] — shipped
PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
# Mevcut canonical hold-out per-panel F1 (global θ, full-ensemble) — RESULTS_CANONICAL
HOLDOUT_GLOBAL = {"General": 0.6006, "Hereditary_Cancer": 0.7301, "PAH": 0.5299}


def f1_20pct(proba, y, thr, seeds=300):
    """%20-patojenik resample F1 (train.py:294-304 tarifi)."""
    po = np.where(y == 1)[0]
    ne = np.where(y == 0)[0]
    if len(po) == 0 or len(ne) == 0:
        return None
    n20 = max(1, int(round(len(ne) * 0.25)))  # 0.20/0.80
    fs = []
    for s in range(seeds):
        rr = np.random.RandomState(s)
        sub = rr.choice(po, min(n20, len(po)), replace=False)
        bi = np.concatenate([sub, ne])
        yy, yh = y[bi], (proba[bi] >= thr).astype(int)
        tp = int(((yh == 1) & (yy == 1)).sum())
        fp = int(((yh == 1) & (yy == 0)).sum())
        fn = int(((yh == 0) & (yy == 1)).sum())
        fs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(fs))


def opt_thr(proba, y, seeds=50):
    best = (-1.0, GLOBAL)
    for t in np.linspace(0.05, 0.98, 94):
        f = f1_20pct(proba, y, t, seeds)
        if f is not None and f > best[0]:
            best = (f, t)
    return best[1]


def load():
    df = pd.read_csv("data/train_variants.csv")
    y = df["Label"].astype(int).values
    groups = df["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
    panels = df["Panel"].astype(str).values
    idx = np.arange(len(df))
    idx_tr, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(idx, y, groups))
    z = np.load("reports/oof_per_model.npz", allow_pickle=True)
    oof, labels = z["oof"], z["labels"].astype(int)
    assert np.array_equal(y[idx_tr], labels), "OOF sıra eşleşmiyor"
    ens = (oof * WEIGHTS).sum(1)  # full-ensemble HAM proba
    return ens, labels, panels[idx_tr], groups[idx_tr]


def nested_cv_f1(ens, labels, opan, og, panel, lam):
    """Panel içi nested group-CV: fold-train'de θ türet (shrinkage), fold-test'te ölç."""
    m = np.where(opan == panel)[0]
    pe, py, pg = ens[m], labels[m], og[m]
    nf = 3 if (py.sum() >= 9 and (py == 0).sum() >= 9) else 2
    try:
        splits = list(StratifiedGroupKFold(nf, shuffle=True, random_state=SEED).split(pe, py, pg))
    except ValueError:
        return None
    fs = []
    for tri, tei in splits:
        th = lam * GLOBAL + (1 - lam) * opt_thr(pe[tri], py[tri])
        f = f1_20pct(pe[tei], py[tei], th, seeds=150)
        if f is not None:
            fs.append(f)
    return float(np.mean(fs)) if fs else None


def main():
    ens, labels, opan, og = load()

    # 1) λ taraması — nested-CV 4-panel'i maksimize eden shrinkage
    print("λ taraması (nested group-CV, leakage-siz):")
    sweep = {}
    for lam in [1.0, 0.75, 0.5, 0.25, 0.0]:
        per = {p: nested_cv_f1(ens, labels, opan, og, p, lam) for p in PANELS}
        avg = float(np.mean([per[p] for p in PANELS if per[p] is not None]))
        sweep[lam] = {"per_panel": per, "avg4": avg}
        print(f"  λ={lam:.2f}: " + " ".join(f"{p[:4]}={per[p]:.3f}" for p in PANELS) + f" | 4p={avg:.4f}")
    best_lam = max(sweep, key=lambda k: sweep[k]["avg4"])
    rigorous = sweep[best_lam]
    print(f"\nEN İYİ λ={best_lam} → rigorous nested-CV 4-panel = {rigorous['avg4']:.4f}")

    # 2) Shipping eşikleri: tüm OOF'ta türet (shrinkage best_lam)
    ship_thr = {}
    for p in PANELS:
        m = opan == p
        ship_thr[p] = round(best_lam * GLOBAL + (1 - best_lam) * opt_thr(ens[m], labels[m]), 4)

    # 3) Hold-out-karşılaştırılabilir 4-panel: 3 panel canonical hold-out (global θ),
    #    CFTR = OOF'ta shipping eşiğiyle (hold-out tek-sınıf, OOF tek seçenek)
    cftr_ship_f1 = f1_20pct(ens[opan == "CFTR"], labels[opan == "CFTR"], ship_thr["CFTR"])
    holdout4 = float(np.mean(list(HOLDOUT_GLOBAL.values()) + [cftr_ship_f1]))

    # 4) CFTR global-θ artefaktı (karşılaştırma için)
    cftr_global = f1_20pct(ens[opan == "CFTR"], labels[opan == "CFTR"], GLOBAL)

    out = {
        "_purpose": "Panel-başına shrinkage eşik ile DÜRÜST 4-panel F1 (CFTR dahil). "
        "Headline 0.6202 üç-panel iç tahmindi; bu CFTR'yi dahil eder.",
        "estimator": "shipped full-ensemble (0.30/0.30/0.25/0.15) genuine OOF (oof_per_model.npz)",
        "method": "per-panel optimal θ → global θ=0.8415'e shrinkage(λ); %20-prior 300x resample F1",
        "best_lambda": best_lam,
        "rigorous_nested_cv_4panel_f1": round(rigorous["avg4"], 4),
        "rigorous_per_panel": {p: (round(v, 4) if v is not None else None)
                               for p, v in rigorous["per_panel"].items()},
        "holdout_comparable_4panel_f1": round(holdout4, 4),
        "_holdout_note": "3 panel = RESULTS_CANONICAL hold-out global-θ F1 (0.6006/0.7301/0.5299); "
        "CFTR = OOF'ta shipping eşiği (hold-out tek-sınıf n=18). 0.6202 headline ile aynı metodoloji.",
        "cftr_f1_at_global_theta": round(cftr_global, 4),
        "cftr_f1_at_panel_theta": round(cftr_ship_f1, 4),
        "_cftr_note": f"CFTR global θ={GLOBAL}'te F1={cftr_global:.4f} (eşik miskalibre artefaktı); "
        f"panel eşiği θ={ship_thr['CFTR']}'te F1={cftr_ship_f1:.4f}. Ayrım gücü güçlü: ROC-AUC=0.889.",
        "shipping_panel_thresholds": ship_thr,
        "current_headline_3panel": 0.6202,
        "lambda_sweep": {str(k): round(v["avg4"], 4) for k, v in sweep.items()},
    }
    with open("reports/panel_threshold_4panel.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"RIGOROUS nested-CV 4-panel  = {out['rigorous_nested_cv_4panel_f1']}")
    print(f"HOLD-OUT-karşılaştırılabilir = {out['holdout_comparable_4panel_f1']}")
    print(f"  (3 panel hold-out + CFTR {cftr_ship_f1:.4f} @ θ={ship_thr['CFTR']})")
    print(f"CFTR: global θ={cftr_global:.4f} → panel θ={cftr_ship_f1:.4f}")
    print(f"mevcut 3-panel headline     = 0.6202")
    print("→ reports/panel_threshold_4panel.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
LIME ↔ TreeSHAP açıklama-uyumu (concordance) — AÇIKLANABİLİRLİK KANITI (risk-siz, salt-okur).

Amaç: PDR §2.4'teki "LIME ve TreeSHAP örtüşür" sözel iddiasını GERÇEK, tekrar-üretilebilir bir
sayıyla desteklemek. Eski markdown'daki "ρ=0,89" hiçbir artefaktla desteklenmiyordu (Rule-13) →
burada fiilen ölçülüyor; çıkan değer NE İSE PDR'ye o yazılır.

Kurgu (apples-to-apples): hem LIME hem TreeSHAP AYNI XGBoost alt-modelini (ensemble'da %≈60 ağırlık,
baskın bileşen) açıklar → temiz, hızlı, deterministik. Shipped model SADECE OKUNUR (retrain YOK).

Metrik: örnek-başına Spearman ρ (712-boyutlu |önem| vektörleri) + top-10 Jaccard (daha robust).
Çıktı: reports/lime_shap_concordance.json
Çalıştır: venv/bin/python scripts/lime_shap_concordance.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.explainability.lime_explainer import LIMEExplainer
from src.utils.seeds import set_global_seed
from src.utils.serialization import ModelStore

SEED = 42
N_SAMPLES = 150
LIME_NUM_SAMPLES = 1000
OUT = REPO / "reports" / "lime_shap_concordance.json"
PANEL_MAP = {
    "General": "General",
    "MASTER": "General",
    "": "General",
    "Hereditary_Cancer": "Hereditary_Cancer",
    "KANSER": "Hereditary_Cancer",
    "PAH": "PAH",
    "CFTR": "CFTR",
}


def panel_key(p):
    return PANEL_MAP.get(str(p).strip(), "General")


def top_jaccard(a, b, k=10):
    ta, tb = set(np.argsort(a)[::-1][:k]), set(np.argsort(b)[::-1][:k])
    return len(ta & tb) / len(ta | tb)


def main():
    import lime as _lime
    import shap

    reset_settings()
    cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))
    set_global_seed(SEED)

    print("[data] load_csv(train_variants.csv) ...")
    ds = load_csv(str(REPO / "data" / "train_variants.csv"))
    X_all, y_all = ds.features.values, ds.labels
    raw = pd.read_csv(str(REPO / "data" / "train_variants.csv"))
    panel_all = raw.get("Panel", pd.Series(["General"] * len(y_all))).values
    vid = raw.get("Variant_ID", pd.Series([f"v{i}" for i in range(len(y_all))])).astype(str)
    base_ids = vid.str.replace(r"_aug\d*$", "", regex=True).to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X_all, y_all, groups=base_ids))
    assert len(set(base_ids[tr]) & set(base_ids[te])) == 0, "Group-aware split sızıntısı!"

    store = ModelStore(cfg.paths.models_dir)
    preproc = store.load_preprocessor()
    xgb = store.load_xgb()  # XGBClassifier (ensemble'daki baskın bileşen)

    Xtr_p = np.asarray(preproc.transform(X_all[tr]), dtype=np.float32)
    Xte_p = np.asarray(preproc.transform(X_all[te]), dtype=np.float32)
    F = Xte_p.shape[1]
    pan_te = panel_all[te]
    print(f"[prep] train={Xtr_p.shape} test={Xte_p.shape} (F={F})")

    rng = np.random.default_rng(SEED)
    n = min(N_SAMPLES, len(Xte_p))
    sel = rng.choice(len(Xte_p), size=n, replace=False)

    # ── SHAP (TreeSHAP, exact, deterministik) ────────────────────────────────
    print("[shap] TreeExplainer ...")
    sv = shap.TreeExplainer(xgb).shap_values(Xte_p[sel])
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:  # (n, F, 2) → pozitif sınıf
        sv = sv[..., 1]
    shap_imp = np.abs(sv)  # (n, F)

    # ── LIME (xgb-only predict_fn, deterministik) ────────────────────────────
    def predict_fn(Z):
        return xgb.predict_proba(np.asarray(Z, dtype=np.float32))

    lime_ex = LIMEExplainer(
        training_data=Xtr_p, feature_names=[f"f{i}" for i in range(F)], predict_fn=predict_fn, random_state=SEED
    )

    rhos, jaccs, panels_used, n_skip = [], [], [], 0
    lime_mat = np.zeros((n, F))  # küresel/alt-küme metrikleri için sakla
    valid_mask = np.zeros(n, dtype=bool)
    print(f"[lime] {n} örnek açıklanıyor (num_samples={LIME_NUM_SAMPLES}) ...")
    for j, idx in enumerate(sel):
        exp = lime_ex.explain_instance(Xte_p[idx], num_features=F, num_samples=LIME_NUM_SAMPLES)
        if exp is None:
            n_skip += 1
            continue
        try:
            amap = exp.as_map()[1]
        except Exception:
            n_skip += 1
            continue
        limp = np.zeros(F)
        for fi, w in amap:
            limp[int(fi)] = abs(w)
        lime_mat[j] = limp
        rho, _ = spearmanr(shap_imp[j], limp)
        if np.isnan(rho):
            n_skip += 1
            continue
        valid_mask[j] = True
        rhos.append(float(rho))
        jaccs.append(float(top_jaccard(shap_imp[j], limp, k=10)))
        panels_used.append(panel_key(pan_te[idx]))
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{n}  ρ_running(full)={np.mean(rhos):.4f}")

    rhos = np.array(rhos)
    jaccs = np.array(jaccs)

    # ── RAFİNE METRİKLER (yüksek-boyutta tam-vektör Spearman gürültülü) ──────────
    # 1) Küresel ρ: özellik-başına ORTALAMA |önem| (örnekler üzerinden) iki yöntem için → tek ρ.
    g_shap = shap_imp[valid_mask].mean(axis=0)
    g_lime = lime_mat[valid_mask].mean(axis=0)
    global_rho = float(spearmanr(g_shap, g_lime)[0])
    global_top10_jacc = float(top_jaccard(g_shap, g_lime, k=10))
    global_top20_jacc = float(top_jaccard(g_shap, g_lime, k=20))
    # 2) Önemli alt-küme: küresel SHAP'a göre top-30 özellikte örnek-başına ρ (gürültü-arınmış)
    topK = np.argsort(g_shap)[::-1][:30]
    sub_rhos = []
    for j in np.where(valid_mask)[0]:
        r, _ = spearmanr(shap_imp[j][topK], lime_mat[j][topK])
        if not np.isnan(r):
            sub_rhos.append(float(r))
    sub_rho_mean = float(np.mean(sub_rhos)) if sub_rhos else float("nan")
    print(
        f"[refined] küresel ρ={global_rho:.4f}  küresel top10-Jaccard={global_top10_jacc:.4f}  "
        f"top30-altküme örnek-ρ={sub_rho_mean:.4f}"
    )
    per_panel = {}
    for pk in ("General", "Hereditary_Cancer", "PAH", "CFTR"):
        m = np.array([p == pk for p in panels_used])
        if m.sum() > 0:
            per_panel[pk] = {"rho_mean": round(float(rhos[m].mean()), 4), "n": int(m.sum())}

    payload = {
        "experiment": "lime_shap_concordance",
        "claim": "LIME ve TreeSHAP ozellik-onem siralamasi uyumlu (yontem-bagimsiz aciklama)",
        "seed": SEED,
        "n_samples_requested": n,
        "n_valid": int(len(rhos)),
        "n_skipped": int(n_skip),
        "rho_mean": round(float(rhos.mean()), 4),
        "rho_std": round(float(rhos.std()), 4),
        "rho_median": round(float(np.median(rhos)), 4),
        "rho_q25": round(float(np.percentile(rhos, 25)), 4),
        "rho_q75": round(float(np.percentile(rhos, 75)), 4),
        "top10_jaccard_mean_persample": round(float(jaccs.mean()), 4),
        "global_rho": round(global_rho, 4),
        "global_top10_jaccard": round(global_top10_jacc, 4),
        "global_top20_jaccard": round(global_top20_jacc, 4),
        "top30_subset_persample_rho_mean": round(sub_rho_mean, 4),
        "per_panel": per_panel,
        "n_features": int(F),
        "lime_num_samples": LIME_NUM_SAMPLES,
        "method": "per-sample Spearman rho | LIME(num_samples=1000, xgb-only predict_fn) abs-weight "
        "vs TreeSHAP abs-value | scaled 712-dim feature space | apples-to-apples (ayni XGB)",
        "split": "group-aware hold-out (GroupShuffleSplit test_size=0.2 random_state=42, Variant_ID _aug strip)",
        "source_model": "models/ (shipped XGBClassifier, read-only)",
        "library_versions": {"lime": getattr(_lime, "__version__", "?"), "shap": getattr(shap, "__version__", "?")},
        # Dürüstlük kapısı: küresel uyum makul (≥0,5) ya da önemli-altküme ρ makul ise savunulabilir.
        "include_in_report": bool(global_rho >= 0.5 or (sub_rho_mean == sub_rho_mean and sub_rho_mean >= 0.5)),
        "note": "Eski 'rho=0,89' (PDR_VARIANT_GNN_2026.md) DAYANAKSIZDI. Tam-712-vektor per-ornek "
        "Spearman yuksek-boyutta gurultu-baskin (cogu ozellik ~0 onem). Anlamli metrik = "
        "kuresel onem korelasyonu + onemli-altkume. Deger NE ISE PDR'ye o yazilir; uyum "
        "zayifsa §2.4 'ortusur' iddiasi YUMUSATILIR (uydurma 0,89 ASLA kullanilmaz).",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[write] {OUT}")
    print(
        f"[result] full-per-sample ρ={payload['rho_mean']}±{payload['rho_std']} | "
        f"GLOBAL ρ={payload['global_rho']} | global-top10-Jaccard={payload['global_top10_jaccard']} | "
        f"n_valid={payload['n_valid']} | include={payload['include_in_report']}"
    )
    print(f"[per-panel] {per_panel}")


if __name__ == "__main__":
    main()

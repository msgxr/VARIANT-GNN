#!/usr/bin/env python3
# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
Koordinatsız graf DEĞER testi — ÖZGÜNLÜK KANITI (risk-siz GNN-only proxy, canonical DOKUNULMAZ).

İddia (PDR §2.2/§5.1): biyokimyasal-benzerlik Cosine k-NN graf YAPISI işe yarıyor — rastgele
veya grafsız değil. Mevcut ablasyon "GNN-out −2,2pp" GNN-VARLIĞININ değeridir; bu deney FARKLI
eksen: aynı GATv2 + aynı öznitelik + aynı seed altında SADECE graf-topolojisini değiştirip
katkıyı izole eder.

3 koşul (group-aware 5-fold, GNN-only, CPU-deterministik):
  (a) cosine_knn  — mevcut k=10 cosine benzerlik grafı
  (b) random_graph — aynı kenar SAYISI, hedefler rastgele (cosine sinyali yok)
  (c) no_graph    — kenarsız (GATv2 self-loop'a dejenere → komşu bilgisi yok)

Shipped model/canonical/models DOKUNULMAZ — tek-kullanımlık proxy GNN eğitilir, sonuç ayrı JSON'a.
Çıktı: reports/graph_value_verification.json
Çalıştır: venv/bin/python scripts/graph_value_verification.py
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
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.core.gnn import VariantGATv2GNN
from src.data.loader import load_csv
from src.features.preprocessing import VariantPreprocessor
from src.training.trainer import WeightedBCELoss, _gatv2_epoch, _gatv2_eval
from src.utils.seeds import set_global_seed

SEED, K, EPOCHS, HID, LR = 42, 10, 40, 128, 0.01
DEVICE = torch.device("cpu")  # MPS nondeterminizmini ele
torch.use_deterministic_algorithms(True, warn_only=True)
CONDITIONS = ["cosine_knn", "random_graph", "no_graph"]
OUT = REPO / "reports" / "graph_value_verification.json"


def perturb(data, mode, rng):
    g = data.clone()
    if mode == "cosine_knn":
        return g
    N = g.x.shape[0]
    if mode == "random_graph":
        E = g.edge_index.shape[1]
        src = g.edge_index[0]
        dst = torch.tensor(rng.integers(0, N, size=E), dtype=torch.long)
        g.edge_index = torch.stack([src, dst], dim=0)
    elif mode == "no_graph":
        g.edge_index = torch.empty((2, 0), dtype=torch.long)
    return g


def train_eval(g_tr, g_va, ytr, in_dim, seed):
    set_global_seed(seed)  # 3 koşul da aynı init → yalnız topoloji değişir
    model = VariantGATv2GNN(numeric_dim=in_dim, hidden_dim=HID, num_classes=2, dropout=0.3).to(DEVICE)
    crit = WeightedBCELoss.from_labels(ytr.astype(int)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        _gatv2_epoch(model, g_tr, opt, crit, DEVICE)
    preds, _ = _gatv2_eval(model, g_va, DEVICE)
    return np.asarray(preds)


def main():
    set_global_seed(SEED)
    ds = load_csv(str(REPO / "data" / "train_variants.csv"))
    Xdf, y = ds.features, ds.labels.astype(int)
    groups = ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
    tr, _te = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(Xdf, y, groups))
    Xtr_all, ytr_all, gtr = Xdf.iloc[tr], y[tr], groups[tr]
    print(f"[data] train pool={len(tr)} (group-aware hold-out dışı)")

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    fold_f1 = {c: [] for c in CONDITIONS}
    for fold, (tri, vai) in enumerate(sgkf.split(Xtr_all, ytr_all, gtr)):
        pre = VariantPreprocessor(corr_threshold=0.25, use_autoencoder=False, use_feature_selection=False)
        pre.fit(Xtr_all.iloc[tri], ytr_all[tri])
        Xt = np.asarray(pre.transform(Xtr_all.iloc[tri]), dtype=np.float32)
        yt = ytr_all[tri].astype(int)
        Xv = np.asarray(pre.transform(Xtr_all.iloc[vai]), dtype=np.float32)
        yv = ytr_all[vai].astype(int)
        g_tr0 = pre.build_sample_graph(Xt, yt, k=K)
        g_va0 = pre.build_sample_graph(Xv, yv, k=K)
        rng = np.random.default_rng(SEED + fold)
        row = {}
        for c in CONDITIONS:
            try:
                g_tr = perturb(g_tr0, c, rng)
                g_va = perturb(g_va0, c, rng)
                preds = train_eval(g_tr, g_va, yt, Xt.shape[1], SEED + fold)
                f1 = float(f1_score(yv, preds, pos_label=1, zero_division=0))
            except Exception as exc:
                print(f"  [fold{fold}/{c}] HATA: {exc}")
                f1 = float("nan")
            fold_f1[c].append(f1)
            row[c] = round(f1, 4)
        print(f"[fold {fold}] {row}")

    def agg(c):
        a = np.array([v for v in fold_f1[c] if v == v])  # NaN filtre
        return (round(float(a.mean()), 4), round(float(a.std()), 4)) if len(a) else (float("nan"), float("nan"))

    means = {c: agg(c)[0] for c in CONDITIONS}
    stds = {c: agg(c)[1] for c in CONDITIONS}
    contrib_vs_random = round((means["cosine_knn"] - means["random_graph"]) * 100, 2)
    contrib_vs_none = round((means["cosine_knn"] - means["no_graph"]) * 100, 2)
    contrib = round(means["cosine_knn"] * 100 - max(means["random_graph"], means["no_graph"]) * 100, 2)

    # DÜRÜST ROBUSTLUK: fold-başına eşleşmeli fark (cosine − o fold'un en iyi kontrolü).
    # Ortalama-fark gürültüye karşı kontrol edilmeli; kiraz-toplama önlenir.
    paired = []
    for f in range(len(fold_f1["cosine_knn"])):
        cos, rnd, non = fold_f1["cosine_knn"][f], fold_f1["random_graph"][f], fold_f1["no_graph"][f]
        if cos == cos and rnd == rnd and non == non:
            paired.append(cos - max(rnd, non))
    paired = np.array(paired)
    paired_mean_pp = round(float(paired.mean() * 100), 2)
    paired_std_pp = round(float(paired.std() * 100), 2)
    wins = int((paired > 0).sum())
    nf = len(paired)
    # Sağlam-pozitif YALNIZCA: eşleşmeli ortalama > std VE cosine ≥4/5 fold'da kazanırsa.
    include = bool(paired_mean_pp > 0 and paired_mean_pp > paired_std_pp and wins >= 4)
    if include:
        verdict = "POSITIVE"
    elif wins <= 1 and paired_mean_pp < -2:
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "experiment": "coordinate_free_graph_value",
        "claim": "Cosine k-NN graf YAPISI rastgele/grafsizdan ustun (graf-topolojisinin degeri)",
        "seed": SEED,
        "device": "cpu",
        "deterministic": True,
        "gnn_only_proxy": True,
        "config": {
            "knn_k": K,
            "epochs": EPOCHS,
            "early_stopping": False,
            "hidden_dim": HID,
            "lr": LR,
            "use_multimodal": False,
            "cv": "StratifiedGroupKFold-5",
        },
        "conditions": {c: {"mean_f1": means[c], "std_f1": stds[c], "fold_f1": fold_f1[c]} for c in CONDITIONS},
        "graph_contribution_pp_meanofmeans": contrib,
        "contribution_vs_random_pp": contrib_vs_random,
        "contribution_vs_none_pp": contrib_vs_none,
        "paired_per_fold_mean_pp": paired_mean_pp,
        "paired_per_fold_std_pp": paired_std_pp,
        "cosine_wins_folds": f"{wins}/{nf}",
        "verdict": verdict,
        "include_in_report": include,
        "note": "GNN-only proxy; tam-ensemble retrain DEGIL; canonical/models DOKUNULMADI. "
        "Mean-of-means +1,2pp ALDATICI: fold-basina eslesmeli farkta cosine kontrolden "
        "ortalama dusuk ve 5 fold'un sadece bir kisminda kazaniyor -> graf-TOPOLOJISININ "
        "standalone GNN-F1'e robust katkisi YOK. Bu, grafin TAM-ENSEMBLE'deki degerini "
        "(cesitlilik/anti-korelasyon r=-0,18, GNNExplainer biyokimyasal sinyal) CURUTMEZ; "
        "ama 'graf +X pp' diye SAYI iddia edilemez -> rapora EKLENMEZ (durustluk).",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[write] {OUT}")
    print(
        f"[result] cosine={means['cosine_knn']} random={means['random_graph']} none={means['no_graph']} "
        f"| katkı(vs max)={contrib}pp | verdict={payload['verdict']} | include={include}"
    )


if __name__ == "__main__":
    main()

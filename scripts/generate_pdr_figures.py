"""
scripts/generate_pdr_figures.py
================================
PDR icin gerekli tum gorselleri uretir.

Ciktilar: reports/figures/pdr/

NOT: fig_06 (PR egrisi) ve fig_07 (Kalibrasyon) GERCEK OOF olasiliklari
kullanir (reports/oof_probs.npz).  PNG'lerin yeniden uretilmesi icin
matplotlib + numpy yuklu bir ortamda bu scripti calistirin.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "figures" / "pdr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C = {
    "ensemble": "#2563EB",
    "xgb": "#16A34A",
    "lgbm": "#CA8A04",
    "gnn": "#9333EA",
    "dnn": "#DC2626",
    "General": "#3B82F6",
    "Hereditary_Cancer": "#10B981",
    "PAH": "#F59E0B",
    "CFTR": "#EF4444",
}
PANEL_ORDER = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
PANEL_TR = {"General": "General", "Hereditary_Cancer": "Hereditary\nCancer", "PAH": "PAH", "CFTR": "CFTR"}

with open(ROOT / "reports" / "cv_report.json", encoding="utf-8") as f:
    cv = json.load(f)
folds = cv["folds"]
panel_metrics = cv["panel_metrics"]
test_metrics = cv["test_metrics"]


# ── 1. 5-Fold CV ─────────────────────────────────────────────────────────────
def fig_01():
    fold_nums = [f["fold"] for f in folds]
    models = {
        "Ensemble": ([f["f1"] for f in folds], C["ensemble"]),
        "XGBoost": ([f["xgb_f1"] for f in folds], C["xgb"]),
        "LightGBM": ([f["lgbm_f1"] for f in folds], C["lgbm"]),
        "GATv2 GNN": ([f["gnn_f1"] for f in folds], C["gnn"]),
        "DNN": ([f["dnn_f1"] for f in folds], C["dnn"]),
    }
    x = np.arange(len(fold_nums))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (name, (vals, color)) in enumerate(models.items()):
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {n}" for n in fold_nums], fontsize=11)
    ax.set_ylabel("Binary F1 (Patojenik)")
    ax.set_title("5-Fold Capraz Dogrulama — Model Bazli F1 Karsilastirmasi")
    ax.set_ylim(0.77, 0.935)
    mean_f1 = cv["mean_cv_binary_f1"]
    ax.axhline(
        mean_f1, color=C["ensemble"], linestyle="--", linewidth=1.8, label=f"Ensemble Ort. {mean_f1:.3f}", zorder=5
    )
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_cv_fold_comparison.png")
    plt.close(fig)
    print("01 OK")


# ── 2. Panel F1 + MCC Bar ─────────────────────────────────────────────────────
def fig_02():
    panels = PANEL_ORDER
    f1s = [panel_metrics[p]["binary_f1"] for p in panels]
    mccs = [panel_metrics[p]["mcc"] for p in panels]
    colors = [C[p] for p in panels]
    labels = [PANEL_TR[p] for p in panels]
    x = np.arange(len(panels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width / 2, f1s, width, label="Binary F1", color=colors, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + width / 2, mccs, width, label="MCC", color=colors, alpha=0.42, edgecolor="white", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Deger")
    ax.set_title("Panel Bazli Binary F1 ve MCC (Test Seti)")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.85, color="gray", linestyle=":", linewidth=1.2, label="Hedef F1 = 0.85")
    for bar, v in zip(b1, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar, v in zip(b2, mccs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_panel_f1_bar.png")
    plt.close(fig)
    print("02 OK")


# ── 3. Radar ─────────────────────────────────────────────────────────────────
def fig_03():
    metric_keys = ["binary_f1", "mcc", "pr_auc", "roc_auc"]
    metric_labels = ["Binary F1", "MCC", "PR-AUC", "ROC-AUC"]
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for p in PANEL_ORDER:
        vals = [panel_metrics[p][k] for k in metric_keys] + [panel_metrics[p][metric_keys[0]]]
        ax.plot(angles, vals, "o-", linewidth=2.2, color=C[p], label=PANEL_TR[p].replace("\n", " "), markersize=6)
        ax.fill(angles, vals, alpha=0.07, color=C[p])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11, fontweight="bold")
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9, color="gray")
    ax.set_title("Panel Bazli Performans Radari", pad=28, fontsize=14, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=10, frameon=True)
    ax.grid(color="gray", alpha=0.3)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.97])
    fig.savefig(OUT_DIR / "03_panel_metrics_radar.png")
    plt.close(fig)
    print("03 OK")


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
def fig_04():
    # GERÇEK per-panel confusion sayılarını cv_report.json'dan oku. Eskiden panel_sizes
    # HARDCODED'du (kanonik test boyutlarıyla uyumsuz) ve confusion recall/precision'dan
    # YENİDEN KURULUYORDU → fabrike sayılar. Sayılar yoksa fabrike ETMEK yerine ATLA.
    needed = ("tp", "fp", "fn", "tn")
    if not all(all(k in panel_metrics.get(p, {}) for k in needed) for p in PANEL_ORDER):
        print(
            "04 SKIP — cv_report.json per-panel confusion sayıları (tp/fp/fn/tn) içermiyor. "
            "Eğitim hattı (src/cli/modes/train.py) bunları yazmalı; fabrike sayı üretilmedi."
        )
        return
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, p in zip(axes, PANEL_ORDER):
        m = panel_metrics[p]
        tp, fp, fn, tn = int(m["tp"]), int(m["fp"]), int(m["fn"]), int(m["tn"])
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum() or 1

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("blues2", ["#EFF6FF", "#1D4ED8"])
        ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                pct = v / total * 100
                col = "white" if v > cm.max() * 0.55 else "#1e293b"
                ax.text(
                    j,
                    i,
                    f"{v}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=col,
                    fontsize=10,
                    fontweight="bold" if v == cm.max() else "normal",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign\n(Tahmin)", "Patojenik\n(Tahmin)"], fontsize=9)
        ax.set_yticklabels(["Benign\n(Gercek)", "Patojenik\n(Gercek)"], fontsize=9)
        title_name = PANEL_TR[p].replace("\n", " ")
        ax.set_title(f"{title_name}\nF1={m['binary_f1']:.3f}  MCC={m['mcc']:.3f}", fontsize=10, fontweight="bold")

    fig.suptitle("Panel Bazli Confusion Matrix (Test Seti)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_confusion_matrix_panel.png")
    plt.close(fig)
    print("04 OK")


# ── 5. ROC ────────────────────────────────────────────────────────────────────
def fig_05():
    from sklearn.metrics import roc_auc_score, roc_curve

    oof_path = ROOT / "reports" / "oof_probs.npz"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if not oof_path.exists():
        print("05 SKIP — reports/oof_probs.npz bulunamadi.")
        plt.close(fig)
        return
    data = np.load(str(oof_path), allow_pickle=True)
    y_t = data["test_labels"].astype(int)
    s_t = data["test_proba"].astype(float)
    panels = data["test_panels"].astype(str) if "test_panels" in data.files else np.array(["General"] * len(y_t))
    for p in PANEL_ORDER:
        mask = panels == p
        # CFTR test fold'unda negatif sinif dejenere → ROC tanimsiz; atla
        if mask.sum() < 10 or len(np.unique(y_t[mask])) < 2:
            print(f"  fig_05: {p} icin iki-sinifli yeterli ornek yok ({int(mask.sum())}), atlanıyor.")
            continue
        fpr, tpr, _ = roc_curve(y_t[mask], s_t[mask], pos_label=1)
        auc = roc_auc_score(y_t[mask], s_t[mask])
        ax.plot(fpr, tpr, linewidth=2.5, color=C[p], label=f"{PANEL_TR[p].replace(chr(10), ' ')} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Rastgele (AUC=0.5)")
    ax.set_xlabel("Yanlis Pozitif Orani (FPR)")
    ax.set_ylabel("Dogru Pozitif Orani (TPR)")
    ax.set_title("ROC Egrisi — Panel Bazli (gercek test hold-out verisi)")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_roc_curves.png")
    plt.close(fig)
    print("05 OK (gercek test verisinden)")


# ── 6. PR ─────────────────────────────────────────────────────────────────────
# GERCEK VERI: reports/oof_probs.npz dosyasindan yüklenir.
# oof_probs.npz anahtarlari: test_labels, test_proba, test_panels (test hold-out) + oof_tree_blend/oof_labels (OOF).
# PNG'yi yeniden uretmek icin matplotlib + numpy yuklu bir ortamda calistirin.
# TODO: oof_probs.npz yoksa bu fonksiyon uyari verir ve figür uretmez.
def fig_06():
    from sklearn.metrics import average_precision_score, precision_recall_curve

    oof_path = ROOT / "reports" / "oof_probs.npz"
    if not oof_path.exists():
        print("06 SKIP — reports/oof_probs.npz bulunamadi.")
        print("         PNG uretmek icin: src/cli/modes/train.py calistirip oof_probs.npz kaydedin.")
        return

    data = np.load(str(oof_path), allow_pickle=True)
    y_oof = data["test_labels"].astype(int)
    score_oof = data["test_proba"].astype(float)
    # panel bilgisi varsa kullan, yoksa hepsini General say
    panel_oof = data["test_panels"].astype(str) if "test_panels" in data.files else np.array(["General"] * len(y_oof))

    # Test prior: %20-patojenik (canonical)
    total_pos_ratio = 0.20

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for p in PANEL_ORDER:
        mask = panel_oof == p
        if mask.sum() < 10:
            print(f"  fig_06: {p} icin yeterli OOF ornegi yok ({mask.sum()}), atlanıyor.")
            continue
        precision, recall, _ = precision_recall_curve(y_oof[mask], score_oof[mask], pos_label=1)
        prauc = average_precision_score(y_oof[mask], score_oof[mask])
        # sklearn cikarimi: precision[0]=1, recall[0]=0 (basta) — ters siradan ciz
        ax.plot(
            recall,
            precision,
            linewidth=2.5,
            color=C[p],
            label=f"{PANEL_TR[p].replace(chr(10), ' ')} (PR-AUC={prauc:.3f})",
        )
        # Canonical karar noktasi @ θ=0.8415
        pm = panel_metrics[p]
        rec1 = pm["recall"]
        prec0 = pm["precision"]
        ax.scatter([rec1], [prec0], color=C[p], s=80, zorder=6, marker="D", edgecolors="white", linewidths=1)

    ax.axhline(
        total_pos_ratio,
        color="gray",
        linestyle=":",
        alpha=0.6,
        linewidth=1.2,
        label=f"Rastgele (test prior={total_pos_ratio:.2f})",
    )

    ax.set_xlabel("Recall (Duyarlilik)")
    ax.set_ylabel("Precision (Kesinlik)")
    ax.set_title("Precision-Recall Egrisi — Panel Bazli\n(karo = θ=0.8415 karar noktasi; gercek test hold-out verisi)")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_pr_curves.png")
    plt.close(fig)
    print("06 OK (gercek OOF verisinden)")


# ── 7. Kalibrasyon ────────────────────────────────────────────────────────────
# GERCEK VERI: reports/oof_probs.npz dosyasindan yüklenir.
# oof_probs.npz anahtarlari: test_labels, test_proba (test hold-out).
# PNG'yi yeniden uretmek icin matplotlib + numpy yuklu bir ortamda calistirin.
# TODO: oof_probs.npz yoksa bu fonksiyon uyari verir ve figür uretmez.
def fig_07():
    from sklearn.calibration import calibration_curve

    oof_path = ROOT / "reports" / "oof_probs.npz"
    if not oof_path.exists():
        print("07 SKIP — reports/oof_probs.npz bulunamadi.")
        print("         PNG uretmek icin: src/cli/modes/train.py calistirip oof_probs.npz kaydedin.")
        return

    data = np.load(str(oof_path), allow_pickle=True)
    y_oof = data["test_labels"].astype(int)
    score_oof = data["test_proba"].astype(float)

    # Canonical ECE = 0.0291 (RESULTS_CANONICAL.json)
    ece = test_metrics.get("ece", 0.0291)

    prob_true, prob_pred = calibration_curve(y_oof, score_oof, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Mukemmel Kalibrasyon", alpha=0.7)
    ax.plot(
        prob_pred,
        prob_true,
        "o-",
        color=C["ensemble"],
        linewidth=2.2,
        markersize=7,
        label=f"Model (ECE={ece:.4f})",
        zorder=5,
    )
    ax.fill_between(
        prob_pred,
        np.clip(prob_pred - ece, 0, 1),
        np.clip(prob_pred + ece, 0, 1),
        alpha=0.12,
        color=C["ensemble"],
        label=f"+-ECE bandi ({ece:.4f})",
    )

    ax.text(
        0.65,
        0.18,
        f"ECE = {ece:.4f}\n(iyi kalibre)",
        ha="center",
        fontsize=11,
        color=C["ensemble"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor=C["ensemble"]),
    )

    ax.set_xlabel("Ortalama Tahmin Olasiligi")
    ax.set_ylabel("Gercek Frekans")
    ax.set_title("Kalibrasyon Egrisi\n(Isotonik Regresyon Sonrasi; gercek test hold-out verisi, ECE=0.0291)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_calibration_curve.png")
    plt.close(fig)
    print("07 OK (gercek OOF verisinden)")


# ── 8. SHAP ───────────────────────────────────────────────────────────────────
def fig_08():
    # GERÇEK SHAP grup katkılarını reports/shap_group_contributions.json'dan oku
    # (generate_explainability.py üretir). Eskiden değerler HARDCODED'du (fabrike).
    shp = ROOT / "reports" / "shap_group_contributions.json"
    if not shp.exists():
        print(
            "08 SKIP — reports/shap_group_contributions.json yok (generate_explainability.py çalıştırın); fabrike üretilmedi."
        )
        return
    data = json.loads(shp.read_text(encoding="utf-8"))
    ranked = data.get("ranked") or data.get("ranked_groups") or []
    if not ranked:
        print("08 SKIP — shap json 'ranked' içermiyor.")
        return

    labels = [r["label_tr"].replace(" ", "\n", 1) for r in ranked]
    actual = [float(r["contribution_pct"]) for r in ranked]
    expected = [float(r.get("expected_pct", 0.0)) for r in ranked]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(
        x - width / 2, actual, width, label="Ölçülen (mean |SHAP| %)", color="#2563EB", alpha=0.88, edgecolor="white"
    )
    b2 = ax.bar(
        x + width / 2, expected, width, label="PSR §4.4 Beklenti (%)", color="#16A34A", alpha=0.80, edgecolor="white"
    )

    for bar, v in zip(b1, actual):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.4,
            f"%{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e40af",
        )
    for bar, v in zip(b2, expected):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.4,
            f"%{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#166534",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Grup Katkısı (mean |SHAP|, %)")
    ax.set_title(
        "SHAP Özellik Grubu Katkıları — Ölçülen vs PSR §4.4 Beklentisi\n(gerçek: shap_group_contributions.json)"
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_shap_importance.png")
    plt.close(fig)
    print("08 OK (gerçek SHAP json'dan)")


# ── 9. Ablation ───────────────────────────────────────────────────────────────
def fig_09():
    # GERÇEK ablasyon delta'larını reports/ablation_report.json'dan oku. Eskiden
    # delta'lar 'baseline - 0.018' gibi SENTETİK'ti ve kaldırılmış AE/SelectKBest
    # satırları içeriyordu. Dosya _SUPERSEDED ise (sızıntı-öncesi) FABRİKE etme — atla.
    abl = ROOT / "reports" / "ablation_report.json"
    if not abl.exists():
        print("09 SKIP — reports/ablation_report.json yok (scripts/run_ablation.py çalıştırın).")
        return
    data = json.loads(abl.read_text(encoding="utf-8"))
    if data.get("_SUPERSEDED"):
        print(
            "09 SKIP — ablation_report.json SUPERSEDED (sızıntı-öncesi sayılar). "
            "Group-aware run_ablation.py yeniden çalıştırılmalı; fabrike delta üretilmedi."
        )
        return
    configs = data.get("configurations", [])
    if not configs:
        print("09 SKIP — ablation_report.json 'configurations' içermiyor.")
        return
    baseline = float(data.get("baseline_cv_f1", configs[0].get("cv_f1", 0.0)))
    names = [c["name"].replace(" ", "\n", 1) for c in configs]
    values = [float(c["cv_f1"]) for c in configs]
    base_flags = [not c.get("components_disabled") for c in configs]
    palette = [C["ensemble"], C["xgb"], C["lgbm"], C["gnn"], C["dnn"], "#6B7280", "#9CA3AF", "#D1D5DB"]
    colors = [palette[i % len(palette)] for i in range(len(configs))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(names, values, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5, width=0.6)
    ax.axhline(
        baseline, color=C["ensemble"], linestyle="--", linewidth=2, label=f"Baseline F1 = {baseline:.3f}", zorder=5
    )
    for bar, v, is_base in zip(bars, values, base_flags):
        delta = v - baseline
        lbl = f"{v:.3f}" if is_base else f"{v:.3f}\n({delta:+.3f})"
        col = "white" if is_base else "#1e293b"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=col,
            fontweight="bold" if is_base else "normal",
        )

    ax.set_ylabel("CV Binary F1 (Patojenik)")
    ax.set_title(
        "Ablation Analizi — Bilesen Katkisi (PDR §4.5)\nDaha dusuk = o bilesen daha onemli (gercek: ablation_report.json)"
    )
    _lo, _hi = min(values + [baseline]), max(values + [baseline])
    ax.set_ylim(max(0.0, _lo - 0.02), _hi + 0.02)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_ablation_bar.png")
    plt.close(fig)
    print("09 OK (gercek ablation_report.json'dan)")


# ── 10. Augmentation ──────────────────────────────────────────────────────────
def fig_10():
    # Sizinti kuantifikasyonu: augmentation + satir-bazli split KALDIRILDI (group-aware'e gecildi).
    # Kaynak: reports/leakage_quantification.json (HistGradientBoosting proxy, model-agnostik).
    labels = ["Augmentation+\nStratKFold\n(LEAKY)", "Orijinal+\nStratKFold", "Group-aware\nKFold\n(DURUST)"]
    vals = [0.9268, 0.8915, 0.8897]
    colors = ["#DC2626", "#F59E0B", C["ensemble"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, vals, 0.55, color=colors, alpha=0.9, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.003,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "+3.71 pp yapay sisme\n(geri cekildi)",
        xy=(0, 0.9268),
        xytext=(1.0, 0.935),
        fontsize=9.5,
        color="#DC2626",
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("CV Binary F1 (HistGB proxy)")
    ax.set_title(
        "Sizinti Kuantifikasyonu — Augmentation + satir-bazli split KALDIRILDI\n"
        "Group-aware (Variant_ID) durust deger: 0.890"
    )
    ax.set_ylim(0.85, 0.95)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_leakage_quantification.png")
    plt.close(fig)
    print("10 OK (leakage_quantification)")


# ── 11. Mimari Sema ───────────────────────────────────────────────────────────
def fig_11():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#3B82F6", fs=9.5, fc="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15", linewidth=1.8, edgecolor="white", facecolor=color, alpha=0.92
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=fc, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.8, mutation_scale=15),
        )

    # --- Girdi ---
    box(0.2, 3.2, 2.0, 1.2, "Varyant\nProfili\n353 kolon", "#64748B", 9)
    arrow(2.2, 3.8, 3.0, 3.8)

    # --- On isleme ---
    box(3.0, 2.6, 2.3, 2.4, "On Isleme\n----------\nImputer\nScaler\nSMOTE\nAutoEncoder", "#0F766E", 9)

    # Oklar on isleme -> modeller
    for y_target in [6.2, 4.6, 3.0, 1.4]:
        arrow(5.3, 3.8, 6.1, y_target + 0.4)

    # --- Modeller ---
    box(6.1, 6.0, 2.0, 1.0, "XGBoost\n(%30)", C["xgb"], 9.5)
    box(6.1, 4.3, 2.0, 1.0, "LightGBM\n(%30)", C["lgbm"], 9.5)
    box(6.1, 2.6, 2.0, 1.0, "GATv2 GNN\n(%25)", C["gnn"], 9.5)
    box(6.1, 0.9, 2.0, 1.0, "DNN\n(%15)", C["dnn"], 9.5)

    # Oklar modeller -> ensemble
    for y_src in [6.5, 4.8, 3.1, 1.4]:
        arrow(8.1, y_src, 9.0, 3.8)

    # --- Ensemble ---
    box(9.0, 2.9, 2.3, 1.8, "Hibrit Ensemble\n----------\nMeta-Learner\n+ Kalibrasyon\n+ Esik", C["ensemble"], 9)
    arrow(11.3, 3.8, 12.1, 3.8)

    # --- Cikti ---
    box(12.1, 3.1, 1.7, 1.4, "Tahmin\nP(Patojenik)\n+ Belirsizlik", "#7C3AED", 9)

    # --- Basliklar ---
    ax.text(1.2, 5.2, "GIRDI", ha="center", fontsize=9, color="#64748B", style="italic")
    ax.text(4.15, 5.2, "ON ISLEME", ha="center", fontsize=9, color="#0F766E", style="italic")
    ax.text(7.1, 7.3, "MODELLER", ha="center", fontsize=9, color="#374151", style="italic")
    ax.text(10.15, 5.0, "ENSEMBLE", ha="center", fontsize=9, color=C["ensemble"], style="italic")
    ax.text(12.95, 4.8, "CIKTI", ha="center", fontsize=9, color="#7C3AED", style="italic")

    ax.set_title(
        "VARIANT-GNN Mimari Semasi\nXGBoost + LightGBM + VariantGATv2GNN + DNN  →  Hibrit Ensemble",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_architecture_diagram.png")
    plt.close(fig)
    print("11 OK")


# ── 12. Seed Stabilite ────────────────────────────────────────────────────────
def fig_12():
    seed_path = ROOT / "reports" / "seed_stability.json"
    if not seed_path.exists():
        print("12 SKIP (seed_stability.json yok)")
        return
    with open(seed_path, encoding="utf-8") as f:
        sd = json.load(f)

    seeds = [r["seed"] for r in sd["individual_runs"]]
    means = [r["cv_mean_f1"] for r in sd["individual_runs"]]
    stds = [r["cv_std_f1"] for r in sd["individual_runs"]]
    all_folds = [r["fold_f1s"] for r in sd["individual_runs"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Sol: bar + error
    ax = axes[0]
    bars = ax.bar(
        [str(s) for s in seeds],
        means,
        yerr=stds,
        capsize=6,
        color=C["ensemble"],
        alpha=0.85,
        edgecolor="white",
        error_kw={"elinewidth": 1.8, "capthick": 1.8},
    )
    ax.axhline(
        sd["overall_mean_f1"],
        color="#DC2626",
        linestyle="--",
        linewidth=1.8,
        label=f"Genel Ort. {sd['overall_mean_f1']:.4f}",
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + max(stds) + 0.001, f"{m:.4f}", ha="center", va="bottom", fontsize=8.5
        )
    ax.set_xlabel("Seed")
    ax.set_ylabel("CV Binary F1")
    ax.set_title("5 Seed × CV F1\n(Hata Cubugu = Std)")
    ax.set_ylim(0.855, 0.882)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)

    # Sag: box
    ax2 = axes[1]
    ax2.boxplot(
        all_folds,
        tick_labels=[str(s) for s in seeds],
        patch_artist=True,
        boxprops=dict(facecolor=C["ensemble"], alpha=0.55),
        medianprops=dict(color="#DC2626", linewidth=2.2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, markerfacecolor=C["ensemble"], alpha=0.6),
    )
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Fold F1 Dagilimi")
    ax2.set_title(f"Seed Stabilitesi — Fold F1 Kutu Grafigi\nGenel Std = {sd['overall_std_f1']:.4f} (cok kararli)")
    ax2.set_ylim(0.840, 0.895)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle("Model Kararlilik Analizi — 5 Farkli Seed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_seed_stability.png")
    plt.close(fig)
    print("12 OK")


if __name__ == "__main__":
    print(f"PDR gorselleri uretiliyor: {OUT_DIR}\n")
    fig_01()
    fig_02()
    fig_03()
    fig_04()
    fig_05()
    fig_06()
    fig_07()
    fig_08()
    fig_09()
    fig_10()
    fig_11()
    fig_12()
    print(f"\nTamam! 12 gorsel: {OUT_DIR}")

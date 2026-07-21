# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
scripts/generate_benchmark_comparison.py
=========================================
Missense varyant patojenite tahmini alanındaki referans araçlarla
VARIANT-GNN'in performansını karşılaştıran tablo ve grafik üretir.

Çıktılar:
  reports/figures/pdr/13_benchmark_comparison.png  — bar grafik
  reports/benchmark_comparison.json                — tablo verisi (PDR referansı)

ÖNEMLİ NOT: Literatür değerleri ilgili yayınlarda farklı test veri setleri
üzerinde raporlanmıştır. Doğrudan karşılaştırma yalnızca genel performans
sıralaması açısından yorumlanmalıdır (†).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Literatür değerleri ───────────────────────────────────────────────────────
# Her araç için: F1, MCC, PR-AUC, kaynak
LITERATURE = {
    "REVEL\n[Ioannidis 2016]": {
        "f1": 0.800,
        "mcc": 0.450,
        "pr_auc": 0.850,
        "note": "ClinVar Expert Panel test seti; ROC-AUC=0.91",
        "ref": "Ioannidis et al., Am. J. Hum. Genet., 2016",
    },
    "CADD v1.6\n[Kircher 2014]": {
        "f1": 0.740,
        "mcc": 0.380,
        "pr_auc": 0.790,
        "note": "Genel missense benchmark; ROC-AUC=0.85",
        "ref": "Kircher et al., Nat. Genet., 2014",
    },
    "EVE\n[Frazer 2021]": {
        "f1": 0.820,
        "mcc": 0.480,
        "pr_auc": 0.870,
        "note": "Deep Mutational Scanning tabanlı; ROC-AUC=0.87",
        "ref": "Frazer et al., Nature, 2021",
    },
    "MutPred2\n[Sundaram 2018]": {
        "f1": 0.800,
        "mcc": 0.420,
        "pr_auc": 0.830,
        "note": "Protein fonksiyon bazlı; Macro-F1=0.86",
        "ref": "Sundaram et al., Nat. Commun., 2018",
    },
    "ClinPred\n[Alirezaie 2018]": {
        "f1": 0.810,
        "mcc": 0.440,
        "pr_auc": 0.860,
        "note": "ClinVar tabanlı ensemble; AUC=0.89",
        "ref": "Alirezaie et al., Am. J. Hum. Genet., 2018",
    },
}

# ── Bizim sonuçlarımız (cv_report.json'dan) ──────────────────────────────────
with open(REPO_ROOT / "reports" / "cv_report.json") as f:
    cv = json.load(f)

OUR_MODEL = {
    "VARIANT-GNN\n[Bu Çalışma]": {
        "f1": round(cv["test_binary_f1"], 3),
        "mcc": round(cv["test_mcc"], 3),
        "pr_auc": round(cv["test_pr_auc"], 3),
        "note": f"TEKNOFEST 2026 gerçek veri; CV F1={cv['mean_cv_binary_f1']:.3f}±{cv['std_cv_binary_f1']:.3f}",
        "ref": "XYRA3, TEKNOFEST 2026",
        "ours": True,
    },
}

ALL_MODELS = {**LITERATURE, **OUR_MODEL}


def generate_chart(out_path: Path) -> None:
    models = list(ALL_MODELS.keys())
    f1s = [ALL_MODELS[m]["f1"] for m in models]
    mccs = [ALL_MODELS[m]["mcc"] for m in models]
    pr_aucs = [ALL_MODELS[m]["pr_auc"] for m in models]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    colors_f1 = ["#3b82f6"] * (len(models) - 1) + ["#22c55e"]
    colors_mcc = ["#6366f1"] * (len(models) - 1) + ["#f59e0b"]
    colors_pr = ["#8b5cf6"] * (len(models) - 1) + ["#ef4444"]

    bars_f1 = ax.bar(x - width, f1s, width, label="Binary F1", color=colors_f1, alpha=0.9)
    bars_mcc = ax.bar(x, mccs, width, label="MCC", color=colors_mcc, alpha=0.9)
    bars_pr = ax.bar(x + width, pr_aucs, width, label="PR-AUC", color=colors_pr, alpha=0.9)

    for bars in [bars_f1, bars_mcc, bars_pr]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Performans Değeri", color="white", fontsize=11)
    ax.set_title(
        "Missense Varyant Patojenite Tahmini — Literatür Karşılaştırması\n"
        "(† Farklı test veri setleri; VARIANT-GNN: TEKNOFEST 2026 gerçek verisi)",
        color="white",
        fontsize=11,
        pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.yaxis.grid(True, color="#374151", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Bizim modelimizi vurgula
    our_idx = len(models) - 1
    ax.axvline(x=our_idx - 0.45, color="#22c55e", linewidth=1.5, linestyle="--", alpha=0.4)
    ax.axvline(x=our_idx + 0.45, color="#22c55e", linewidth=1.5, linestyle="--", alpha=0.4)
    ax.axvspan(our_idx - 0.45, our_idx + 0.45, alpha=0.07, color="#22c55e")
    ax.text(our_idx, 1.04, "✦ VARIANT-GNN", ha="center", color="#22c55e", fontsize=9, fontweight="bold")

    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=10, loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Grafik kaydedildi: {out_path}")


def generate_json(out_path: Path) -> None:
    result = {
        "title": "Literatür Karşılaştırması — Missense Varyant Patojenite Tahmini",
        "note": "Literatür değerleri yayınlardaki farklı test veri setlerinden alınmıştır (†). "
        "VARIANT-GNN değerleri TEKNOFEST 2026 resmi eğitim verisinin hold-out test "
        "bölümünde (%20) elde edilmiştir.",
        "models": [],
    }
    for name, d in ALL_MODELS.items():
        result["models"].append(
            {
                "name": name.replace("\n", " "),
                "f1": d["f1"],
                "mcc": d["mcc"],
                "pr_auc": d["pr_auc"],
                "note": d["note"],
                "reference": d["ref"],
                "ours": d.get("ours", False),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON kaydedildi: {out_path}")


def main() -> None:
    generate_chart(REPO_ROOT / "reports" / "figures" / "pdr" / "13_benchmark_comparison.png")
    generate_json(REPO_ROOT / "reports" / "benchmark_comparison.json")

    print("\n=== BENCHMARK TABLOSU ===")
    print(f"{'Model':<28} {'F1':>6} {'MCC':>6} {'PR-AUC':>8}")
    print("-" * 52)
    for name, d in ALL_MODELS.items():
        marker = " ★" if d.get("ours") else "  †"
        print(f"{name.replace(chr(10), ' '):<28} {d['f1']:>6.3f} {d['mcc']:>6.3f} {d['pr_auc']:>8.3f}{marker}")
    print("\n★ = TEKNOFEST 2026 gerçek verisi  |  † = literatür değeri (farklı test seti)")


if __name__ == "__main__":
    main()

"""src/ui/performance.py — Model Performance tab (TEKNOFEST §7.3 metrics)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.utils import plot_dark


# ── Metric card HTML ──────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, sub: str = "", color: str = "#63b3ed", delta: str = "") -> str:
    delta_html = f'<div style="font-size:0.7rem;color:{"#68d391" if "+" in delta else "#fc8181"};margin-top:2px;">{delta}</div>' if delta else ""
    return f"""
    <div style="background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(99,179,237,0.03));
                border:1px solid rgba(99,179,237,0.2); border-radius:12px;
                padding:18px 20px; text-align:center; flex:1; min-width:120px;">
        <div style="font-size:1.75rem; font-weight:800; color:{color}; letter-spacing:-1px;">{value}</div>
        <div style="font-size:0.75rem; font-weight:600; color:#94a3b8; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
        {f'<div style="font-size:0.7rem; color:#718096; margin-top:2px;">{sub}</div>' if sub else ""}
        {delta_html}
    </div>"""


def render_performance_tab() -> None:
    """Gerçek eğitim sonuçlarını TEKNOFEST §7.3 metrikleriyle gösterir."""

    cv_path = Path("reports/cv_report.json")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,179,237,0.1),rgba(56,189,248,0.05));
                border:1px solid rgba(99,179,237,0.25); border-radius:14px;
                padding:20px 24px; margin-bottom:24px;">
        <div style="font-size:1rem; font-weight:700; color:#63b3ed; margin-bottom:6px;">
            📊 Model Performans Paneli — TEKNOFEST 2026
        </div>
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
            Birincil metrik: <strong style="color:#e2e8f0;">Binary F1 = 2·TP / (2·TP + FP + FN)</strong>
            — Patojenik sınıfı, pos_label=1 (§7.3)<br>
            Veri: Gerçek TEKNOFEST 2026 yarışma verisi — 14 Mayıs 2026 alındı, 20 Mayıs 2026 eğitildi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not cv_path.exists():
        st.warning("cv_report.json bulunamadı. `python main.py --mode train --config configs/pdr.yaml` çalıştırın.")
        return

    with open(cv_path, encoding="utf-8") as f:
        cv: Dict[str, Any] = json.load(f)

    test = cv.get("test_metrics", {})

    # ── Primary metrics row ───────────────────────────────────────────────────
    st.markdown("### 🎯 Birincil Metrikler (Hold-Out Test — %20)")
    cv_f1   = cv.get("mean_cv_binary_f1", cv.get("mean_cv_macro_f1", 0))
    cv_std  = cv.get("std_cv_binary_f1",  cv.get("std_cv_macro_f1",  0))
    t_f1    = cv.get("test_binary_f1",    test.get("binary_f1", test.get("f1", 0)))
    t_mcc   = cv.get("test_mcc",          test.get("mcc", 0))
    t_prauc = cv.get("test_pr_auc",       test.get("pr_auc", 0))
    t_auc   = test.get("roc_auc", 0)
    t_rec   = test.get("recall", 0)
    t_pre   = test.get("precision", 0)
    t_brier = test.get("brier_score", 0)
    t_ece   = cv.get("test_ece", test.get("ece", 0))
    thr     = cv.get("best_threshold", cv.get("panel_thresholds", {}).get("__global__", 0))

    cards_html = "".join([
        _metric_card("Test F1  §7.3", f"{t_f1:.4f}",  "Patojenik (pos=1)",  "#22d3ee"),
        _metric_card("CV F1 ±σ",      f"{cv_f1:.4f}", f"±{cv_std:.4f}",     "#a78bfa"),
        _metric_card("MCC",           f"{t_mcc:.4f}", "Dengeli",            "#f59e0b"),
        _metric_card("PR-AUC",        f"{t_prauc:.4f}","Eşiksiz",          "#34d399"),
        _metric_card("ROC-AUC",       f"{t_auc:.4f}", "",                   "#60a5fa"),
        _metric_card("Recall",        f"{t_rec:.4f}", "Duyarlılık",        "#fb923c"),
        _metric_card("Precision",     f"{t_pre:.4f}", "Kesinlik",          "#f472b6"),
        _metric_card("Brier",         f"{t_brier:.4f}","Kalibrasyon",      "#94a3b8"),
    ])
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Threshold info ────────────────────────────────────────────────────────
    panel_thr = cv.get("panel_thresholds", {})
    if panel_thr:
        thr_items = " &nbsp;·&nbsp; ".join(
            f"<strong style='color:#e2e8f0;'>{k.replace('__global__','Global').replace('Hereditary_Cancer','KANSER')}</strong>: "
            f"<span style='color:#63b3ed;'>{v:.3f}</span>"
            for k, v in panel_thr.items()
        )
        st.markdown(
            f'<div style="background:rgba(99,179,237,0.05);border:1px solid rgba(99,179,237,0.15);'
            f'border-radius:10px;padding:12px 18px;margin-bottom:20px;font-size:0.83rem;color:#94a3b8;">'
            f'⚡ F1-Optimal Eşikler (kalibrasyon setinde): {thr_items}</div>',
            unsafe_allow_html=True,
        )

    # ── Panel metrics ─────────────────────────────────────────────────────────
    panel_m = cv.get("panel_metrics", {})
    if panel_m:
        st.markdown("### 🧬 Panel Bazlı Sonuçlar (§3.2)")
        panel_rows = []
        name_map = {"General": "MASTER", "Hereditary_Cancer": "KANSER", "PAH": "PAH", "CFTR": "CFTR"}
        for pname, pm in panel_m.items():
            panel_rows.append({
                "Panel":    name_map.get(pname, pname),
                "F1 §7.3": f"{pm.get('binary_f1', 0):.4f}",
                "MCC":      f"{pm.get('mcc', 0):.4f}",
                "PR-AUC":   f"{pm.get('pr_auc', 0):.4f}",
                "ROC-AUC":  f"{pm.get('roc_auc', 0):.4f}",
                "Recall":   f"{pm.get('recall', 0):.4f}",
                "Precision":f"{pm.get('precision', 0):.4f}",
                "Brier":    f"{pm.get('brier_score', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(panel_rows), width="stretch", hide_index=True)

    # ── Charts 2×2 ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Görsel Sonuçlar")
    plots = [
        ("reports/confusion_matrix.png", "Confusion Matrix"),
        ("reports/roc_curve.png",        "ROC Eğrisi"),
        ("reports/pr_curve.png",         "Precision-Recall Eğrisi"),
        ("reports/calibration.png",      "Kalibrasyon Eğrisi"),
    ]
    r1, r2 = st.columns(2), st.columns(2)
    for (img_path, title), col in zip(plots, [r1[0], r1[1], r2[0], r2[1]]):
        with col:
            st.markdown(
                f'<div style="font-size:0.82rem;font-weight:600;color:#63b3ed;margin-bottom:8px;">{title}</div>',
                unsafe_allow_html=True,
            )
            if Path(img_path).exists():
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(
                    '<div style="background:rgba(99,179,237,0.04);border:1px dashed rgba(99,179,237,0.2);'
                    'border-radius:8px;padding:32px;text-align:center;color:#4a5568;font-size:0.8rem;">'
                    'Henüz üretilmedi<br><span style="font-size:0.72rem;">python main.py --mode train</span></div>',
                    unsafe_allow_html=True,
                )

    # ── CV fold bar chart ─────────────────────────────────────────────────────
    folds = cv.get("folds", [])
    if folds:
        st.markdown("### 🔁 5-Fold Cross-Validation Sonuçları")
        fold_nums  = [f["fold"] for f in folds]
        ens_f1s    = [f.get("f1", 0) for f in folds]
        xgb_f1s    = [f.get("xgb_f1", 0) for f in folds]
        lgb_f1s    = [f.get("lgbm_f1", 0) for f in folds]
        gnn_f1s    = [f.get("gnn_f1", 0) for f in folds]
        dnn_f1s    = [f.get("dnn_f1", 0) for f in folds]

        x = np.arange(len(fold_nums))
        w = 0.15
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_dark(fig, ax)
        ax.bar(x - 2*w, ens_f1s, w, label="Ensemble", color="#22d3ee", alpha=0.9)
        ax.bar(x - w,   xgb_f1s, w, label="XGBoost",  color="#f59e0b", alpha=0.85)
        ax.bar(x,       lgb_f1s, w, label="LightGBM", color="#34d399", alpha=0.85)
        ax.bar(x + w,   gnn_f1s, w, label="GATv2GNN", color="#a78bfa", alpha=0.85)
        ax.bar(x + 2*w, dnn_f1s, w, label="DNN",      color="#f472b6", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_ylabel("Binary F1 (§7.3)")
        ax.set_title("5-Fold CV — Model Karşılaştırması", fontweight="bold", pad=12)
        ax.legend(fontsize=8, facecolor="#1a2744", edgecolor=(0.388, 0.702, 0.929, 0.2), labelcolor="#94a3b8")
        ax.set_ylim(0.7, 1.0)
        ax.axhline(cv_f1, color="#22d3ee", linestyle="--", linewidth=1, alpha=0.5,
                   label=f"CV Ortalama={cv_f1:.4f}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Fold table
        df_folds = pd.DataFrame([{
            "Fold": f["fold"],
            "Ensemble F1": f"{f.get('f1',0):.4f}",
            "XGBoost":     f"{f.get('xgb_f1',0):.4f}",
            "LightGBM":    f"{f.get('lgbm_f1',0):.4f}",
            "GATv2GNN":    f"{f.get('gnn_f1',0):.4f}",
            "DNN":         f"{f.get('dnn_f1',0):.4f}",
        } for f in folds])
        st.dataframe(df_folds, width="stretch", hide_index=True)

    # ── PDR figures if available ──────────────────────────────────────────────
    pdr_figs = sorted(Path("reports/figures/pdr").glob("*.png")) if Path("reports/figures/pdr").exists() else []
    if pdr_figs:
        st.markdown("### 🗂️ PDR Görselleri")
        cols = st.columns(3)
        for i, fig_path in enumerate(pdr_figs[:9]):
            with cols[i % 3]:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#718096;margin-bottom:4px;">{fig_path.stem}</div>',
                    unsafe_allow_html=True,
                )
                st.image(str(fig_path), use_container_width=True)

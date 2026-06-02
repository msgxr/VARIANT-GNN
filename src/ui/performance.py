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


def _metric_card(label: str, value: str, sub: str = "", color: str = "#63b3ed") -> str:
    sub_html = f'<div style="font-size:0.68rem;color:#4a5568;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        '<div style="background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(99,179,237,0.02));'
        'border:1px solid rgba(99,179,237,0.2);border-radius:12px;padding:16px 18px;text-align:center;flex:1;min-width:110px;">'
        f'<div style="font-size:1.6rem;font-weight:800;color:{color};letter-spacing:-1px;">{value}</div>'
        f'<div style="font-size:0.72rem;font-weight:600;color:#94a3b8;margin-top:4px;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
        f'{sub_html}'
        '</div>'
    )


# Maps each PDR figure to a display title, description, and section group
_PDR_FIGURES: List[Dict[str, str]] = [
    {
        "file": "01_cv_fold_comparison.png",
        "title": "5-Fold CV — Model Karşılaştırması",
        "desc": "Her fold'da Ensemble vs XGBoost vs LightGBM vs GATv2GNN vs DNN Binary F1 skoru.",
        "section": "crossval",
    },
    {
        "file": "02_panel_f1_bar.png",
        "title": "Panel Bazlı F1 Skorları",
        "desc": "MASTER · KANSER · PAH · CFTR panellerinde Binary F1 (§7.3) karşılaştırması.",
        "section": "panels",
    },
    {
        "file": "03_panel_metrics_radar.png",
        "title": "Panel Metrik Radar",
        "desc": "F1 · MCC · PR-AUC · ROC-AUC · Recall — 4 panel radar grafik.",
        "section": "panels",
    },
    {
        "file": "04_confusion_matrix_panel.png",
        "title": "Confusion Matrix (Panel Bazlı)",
        "desc": "Gerçek yarışma verisi — TP/FP/FN/TN dağılımı. θ=0.8415 global eşik (canonical).",
        "section": "evaluation",
    },
    {
        "file": "05_roc_curves.png",
        "title": "ROC Eğrileri — 4 Panel",
        "desc": "KANSER AUC=0.935 · MASTER AUC=0.854 · PAH AUC=0.884 · CFTR AUC=0.789",
        "section": "evaluation",
    },
    {
        "file": "06_pr_curves.png",
        "title": "Precision-Recall Eğrisi",
        "desc": "PR-AUC=0.9294 (genel) · PAH=0.976 · KANSER=0.952 — eşiksiz performans.",
        "section": "evaluation",
    },
    {
        "file": "07_calibration_curve.png",
        "title": "Kalibrasyon Eğrisi",
        "desc": "Isotonic kalibrasyon öncesi/sonrası. ECE=0.0788, Brier=0.1283.",
        "section": "evaluation",
    },
    {
        "file": "08_shap_importance.png",
        "title": "SHAP Özellik Önemi (Global)",
        "desc": "In Silico Risk %38 · Evrimsel %27 · Popülasyon %18 · Biyokimyasal %10.",
        "section": "xai",
    },
    {
        "file": "09_ablation_bar.png",
        "title": "Ablasyon Analizi (§4.5)",
        "desc": "Her bileşenin çıkarılmasıyla F1 düşüşü. XGBoost: −3.8%, GNN: −2.2%.",
        "section": "ablation",
    },
    {
        "file": "10_augmentation_comparison.png",
        "title": "Gaussian Augmentation Etkisi",
        "desc": "3802 → 7604 örnek: F1 +2.7%. Sadece eğitim verisine uygulandı.",
        "section": "ablation",
    },
    {
        "file": "11_architecture_diagram.png",
        "title": "Sistem Mimarisi Diyagramı",
        "desc": "Uçtan uca pipeline: LeakageFirewall → Preprocessing → Ensemble → Calibration → Output.",
        "section": "arch",
    },
    {
        "file": "12_seed_stability.png",
        "title": "Seed Kararlılığı (§7.5)",
        "desc": "5 farklı seed — inter-seed std=±0.0013. Deterministik jüri tekrar çalıştırma.",
        "section": "crossval",
    },
]

_SECTION_LABELS = {
    "crossval":  ("🔁", "Cross-Validation"),
    "panels":    ("🧬", "Panel Bazlı Analiz"),
    "evaluation":("📈", "Değerlendirme Eğrileri"),
    "xai":       ("🔍", "Açıklanabilirlik"),
    "ablation":  ("⚗️",  "Ablasyon Analizi"),
    "arch":      ("📐", "Mimari"),
}


def render_performance_tab() -> None:
    """TEKNOFEST §7.3 gerçek sonuçları — PDR figürleri ile."""

    cv_path = Path("reports/cv_report.json")
    pdr_dir = Path("reports/figures/pdr")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,179,237,0.1),rgba(56,189,248,0.04));
                border:1px solid rgba(99,179,237,0.25);border-radius:14px;
                padding:18px 24px;margin-bottom:22px;">
        <div style="font-size:0.95rem;font-weight:700;color:#63b3ed;margin-bottom:4px;">
            📊 Model Performans Paneli — Gerçek TEKNOFEST 2026 Verisi
        </div>
        <div style="color:#718096;font-size:0.82rem;line-height:1.7;">
            Tüm metrikler ve görseller gerçek yarışma verisiyle (14 Mayıs 2026) eğitilmiş modelden üretilmiştir.<br>
            Birincil metrik: <strong style="color:#e2e8f0;">Binary F1 = 2·TP / (2·TP + FP + FN)</strong>,
            Patojenik sınıfı, pos_label=1 <strong style="color:#94a3b8;">(§7.3)</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live metrics from cv_report.json ─────────────────────────────────────
    if cv_path.exists():
        with open(cv_path, encoding="utf-8") as f:
            cv: Dict[str, Any] = json.load(f)

        test = cv.get("test_metrics", {})
        cv_f1  = cv.get("mean_cv_binary_f1", 0)
        cv_std = cv.get("std_cv_binary_f1",  0)
        t_f1   = cv.get("test_binary_f1", test.get("binary_f1", 0))
        t_mcc  = cv.get("test_mcc", test.get("mcc", 0))
        t_pr   = cv.get("test_pr_auc", test.get("pr_auc", 0))
        t_auc  = test.get("roc_auc", 0)
        t_rec  = test.get("recall", 0)
        t_pre  = test.get("precision", 0)
        t_brier= test.get("brier_score", 0)
        t_ece  = cv.get("test_ece", test.get("ece", 0))

        cards = "".join([
            _metric_card("Test F1 §7.3", f"{t_f1:.4f}",  "Patojenik",    "#22d3ee"),
            _metric_card("CV F1 ±σ",     f"{cv_f1:.4f}", f"±{cv_std:.4f}","#a78bfa"),
            _metric_card("MCC",          f"{t_mcc:.4f}", "Dengeli",       "#f59e0b"),
            _metric_card("PR-AUC",       f"{t_pr:.4f}",  "Eşiksiz",       "#34d399"),
            _metric_card("ROC-AUC",      f"{t_auc:.4f}", "",              "#60a5fa"),
            _metric_card("Recall",       f"{t_rec:.4f}", "Duyarlılık",   "#fb923c"),
            _metric_card("Precision",    f"{t_pre:.4f}", "Kesinlik",     "#f472b6"),
            _metric_card("Brier/ECE",    f"{t_brier:.4f}", f"ECE={t_ece:.4f}", "#94a3b8"),
        ])
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;">{cards}</div>',
            unsafe_allow_html=True,
        )

        # Panel thresholds strip
        panel_thr = cv.get("panel_thresholds", {})
        if panel_thr:
            name_map = {"__global__": "Global", "General": "MASTER",
                        "Hereditary_Cancer": "KANSER", "PAH": "PAH", "CFTR": "CFTR"}
            items = " &nbsp;·&nbsp; ".join(
                f"<strong style='color:#e2e8f0;'>{name_map.get(k, k)}</strong> "
                f"<span style='color:#22d3ee;'>θ={v:.3f}</span>"
                for k, v in panel_thr.items()
            )
            st.markdown(
                f'<div style="background:rgba(34,211,238,0.05);border:1px solid rgba(34,211,238,0.15);'
                f'border-radius:8px;padding:10px 16px;margin-bottom:18px;font-size:0.8rem;color:#94a3b8;">'
                f'⚡ Panel Eşikleri (F1-optimal, kalibrasyon setinden): {items}</div>',
                unsafe_allow_html=True,
            )

        # Panel breakdown table
        panel_m = cv.get("panel_metrics", {})
        if panel_m:
            name_map2 = {"General": "MASTER", "Hereditary_Cancer": "KANSER",
                         "PAH": "PAH", "CFTR": "CFTR"}
            rows = [{
                "Panel":     name_map2.get(p, p),
                "F1 §7.3":  f"{m.get('binary_f1',0):.4f}",
                "MCC":       f"{m.get('mcc',0):.4f}",
                "PR-AUC":   f"{m.get('pr_auc',0):.4f}",
                "ROC-AUC":  f"{m.get('roc_auc',0):.4f}",
                "Recall":   f"{m.get('recall',0):.4f}",
                "Precision":f"{m.get('precision',0):.4f}",
            } for p, m in panel_m.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── PDR Figures — grouped by section ─────────────────────────────────────
    if not pdr_dir.exists():
        st.info("PDR görselleri bulunamadı. `python main.py --mode train` çalıştırın.")
        return

    # Group figures by section
    from collections import defaultdict
    grouped: Dict[str, list] = defaultdict(list)
    for fig_meta in _PDR_FIGURES:
        path = pdr_dir / fig_meta["file"]
        if path.exists():
            grouped[fig_meta["section"]].append((path, fig_meta))

    section_order = ["crossval", "panels", "evaluation", "xai", "ablation", "arch"]

    for section_key in section_order:
        figs = grouped.get(section_key, [])
        if not figs:
            continue

        icon, label = _SECTION_LABELS[section_key]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:24px 0 14px;">'
            f'<div style="font-size:1.3rem;">{icon}</div>'
            f'<h3 style="margin:0;color:#e2e8f0;font-size:1.05rem;">{label}</h3>'
            f'<div style="flex:1;height:1px;background:rgba(99,179,237,0.15);margin-left:8px;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 2-column layout
        cols = st.columns(2)
        for i, (path, meta) in enumerate(figs):
            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:rgba(15,22,41,0.6);border:1px solid rgba(99,179,237,0.15);'
                    f'border-radius:12px;padding:14px 16px;margin-bottom:4px;">'
                    f'<div style="font-size:0.82rem;font-weight:700;color:#93c5fd;margin-bottom:4px;">'
                    f'{meta["title"]}</div>'
                    f'<div style="font-size:0.7rem;color:#4a5568;line-height:1.5;">'
                    f'{meta["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Load as bytes — avoids Windows path separator issues
                img_bytes = path.read_bytes()
                st.image(img_bytes, use_container_width=True)

    # ── CV fold bar chart (live from cv_report) ───────────────────────────────
    if cv_path.exists() and cv.get("folds"):
        folds = cv["folds"]
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin:24px 0 14px;">'
            '<div style="font-size:1.3rem;">📊</div>'
            '<h3 style="margin:0;color:#e2e8f0;font-size:1.05rem;">Fold Detay Tablosu</h3>'
            '<div style="flex:1;height:1px;background:rgba(99,179,237,0.15);margin-left:8px;"></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        df_folds = pd.DataFrame([{
            "Fold":     f["fold"],
            "Ensemble": f"{f.get('f1',0):.4f}",
            "XGBoost":  f"{f.get('xgb_f1',0):.4f}",
            "LightGBM": f"{f.get('lgbm_f1',0):.4f}",
            "GATv2GNN": f"{f.get('gnn_f1',0):.4f}",
            "DNN":      f"{f.get('dnn_f1',0):.4f}",
        } for f in folds])
        st.dataframe(df_folds, width="stretch", hide_index=True)

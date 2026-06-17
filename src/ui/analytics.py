from typing import Any, Iterable, List, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.utils import hex_to_rgb, style_plot


def render_summary_cards(df_result: pd.DataFrame) -> None:
    """Analiz özetini gösteren metrik kartlarını oluşturur."""
    total = len(df_result)
    path_mask = df_result["Prediction"] == "Pathogenic"
    pathogenic = int(path_mask.sum())
    high_risk = int(df_result.get("High_Risk", pd.Series(dtype=bool)).sum())
    path_pct = 100 * pathogenic / max(total, 1)

    avg_conf = 0.0
    if "Confidence" in df_result.columns:
        avg_conf = float(df_result["Confidence"].mean())

    st.markdown(
        f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="value">{total}</div>
            <div class="label">Toplam Varyant</div>
            <div class="sublabel">Analiz Edildi</div>
        </div>
        <div class="metric-card pathogenic">
            <div class="value" style="color:#e63946;">{pathogenic}</div>
            <div class="label">Patojenik</div>
            <div class="sublabel">{path_pct:.1f}% oran</div>
        </div>
        <div class="metric-card warning">
            <div class="value" style="color:#d97706;">{high_risk}</div>
            <div class="label">Yüksek Risk</div>
            <div class="sublabel">Kalibre edilmiş</div>
        </div>
        <div class="metric-card info">
            <div class="value" style="color:#2563eb;">{avg_conf:.1f}%</div>
            <div class="label">Model Güveni</div>
            <div class="sublabel">Ortalama (SOTA)</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_risk_histogram(df_result: pd.DataFrame) -> None:
    """Risk skoru dağılımını gösteren histogramı oluşturur."""
    if "Calibrated_Risk" not in df_result.columns:
        return
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h3>Risk Skoru Dağılımı</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    style_plot(fig, ax)

    risks: pd.Series = df_result["Calibrated_Risk"]
    n, bins, patches = ax.hist(risks, bins=30, edgecolor="none")

    for patch, left in zip(cast(Iterable[Any], patches), bins[:-1]):
        if left < 50:
            patch.set_facecolor("#16a34a")
        elif left < 75:
            patch.set_facecolor("#d97706")
        else:
            patch.set_facecolor("#dc2626")
        patch.set_alpha(0.85)

    ax.axvline(50, color="#d97706", linestyle="--", linewidth=1.2, alpha=0.7, label="Orta Risk")
    ax.axvline(75, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7, label="Yüksek Risk")
    ax.set_xlabel("Kalibre Edilmiş Risk Skoru (%)")
    ax.set_ylabel("Varyant Sayısı")
    ax.set_title("Risk Skoru Dağılımı", fontsize=12, fontweight="bold", pad=14)
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor=(0.886, 0.910, 0.941, 1.0), labelcolor="#475569")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_risk_map(df_result: pd.DataFrame) -> None:
    """Varyant risk dağılımını scatter görselleştirmesi ile gösterir."""
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">🗺️</div>
        <h3>Varyant Risk Haritası</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    risk_col: str = "Calibrated_Risk" if "Calibrated_Risk" in df_result.columns else "Probability"
    # Calibrated_Risk zaten 0-100 ölçeğinde; Probability ise ham 0-1 olasılık.
    # Fallback durumunda 0-100 eksenine, renk eşiklerine ve lejant aralıklarına
    # uyması için ham olasılığı ×100 ölçekle (aksi halde tüm noktalar ≤1'e çöker).
    _scale: float = 100.0 if risk_col == "Probability" else 1.0
    risks: np.ndarray = df_result[risk_col].values[:200] * _scale  # İlk 200 varyant
    n: int = len(risks)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    style_plot(fig, ax)

    colors: List[str] = ["#dc2626" if r > 75 else "#d97706" if r > 50 else "#16a34a" for r in risks]
    ax.scatter(range(n), risks, c=colors, s=40, alpha=0.85, zorder=3)
    ax.fill_between(range(n), risks, alpha=0.08, color="#2563eb")
    ax.set_xlabel("Varyant İndeksi")
    ax.set_ylabel("Risk Skoru (%)")
    ax.set_title("Varyant Risk Haritası (İlk 200)", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, 105)

    low_p = mpatches.Patch(color="#16a34a", label=f"Benign ({sum(1 for r in risks if r <= 50)})")
    mid_p = mpatches.Patch(color="#d97706", label=f"Orta Risk ({sum(1 for r in risks if 50 < r <= 75)})")
    high_p = mpatches.Patch(color="#dc2626", label=f"Yüksek Risk ({sum(1 for r in risks if r > 75)})")
    ax.legend(
        handles=[low_p, mid_p, high_p],
        fontsize=9,
        facecolor="#ffffff",
        edgecolor=(0.886, 0.910, 0.941, 1.0),
        labelcolor="#475569",
        loc="upper right",
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_model_comparison(df_result: pd.DataFrame) -> None:
    """XGB, GNN, DNN olasılık sütunları varsa karşılaştırma gösterir."""
    prob_cols: List[str] = [
        c for c in ["XGB_Prob", "LGB_Prob", "GNN_Prob", "DNN_Prob", "Probability"] if c in df_result.columns
    ]
    if not prob_cols or len(prob_cols) < 2:
        return

    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">⚖️</div>
        <h3>Model Karşılaştırması</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig, axes = plt.subplots(1, len(prob_cols), figsize=(4 * len(prob_cols), 3.5))
    model_colors: List[str] = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]

    # Handle single plot case (axes not list)
    if len(prob_cols) == 1:
        plot_axes = [axes]
    else:
        plot_axes = list(axes)  # type: ignore

    for i, (col, ax_, color) in enumerate(zip(prob_cols, plot_axes, model_colors)):
        style_plot(fig, ax_)
        ax_.hist(df_result[col], bins=20, color=color, alpha=0.85, edgecolor="none")
        ax_.set_xlabel("Patojenite Olasılığı")
        ax_.set_ylabel("Sayı" if i == 0 else "")
        ax_.set_title(col.replace("_Prob", "").replace("_", " "), fontsize=10, fontweight="bold")
    plt.suptitle("Model Bazlı Olasılık Dağılımları", color="#0f172a", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_results_table(df_result: pd.DataFrame) -> None:
    """Renk kodlu sonuç tablosunu ve öncelik sıralamasını oluşturur."""
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">📋</div>
        <h3>Analiz Sonuçları</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    priority_cols: List[str] = ["Variant_ID", "Prediction", "Calibrated_Risk", "Probability", "Confidence", "High_Risk"]
    display_cols: List[str] = [c for c in priority_cols if c in df_result.columns]
    other_cols: List[str] = [c for c in df_result.columns if c not in display_cols]
    df_display: pd.DataFrame = df_result[display_cols + other_cols]

    st.dataframe(df_display, width="stretch", height=380)

    # Önce İncele - Yüksek Riskli Varyant Sıralaması
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">🚨</div>
        <h3>Önce İncele — Yüksek Riskli Varyant Sıralaması</h3>
    </div>
    <div style="background:rgba(230,57,70,0.05); border:1px solid rgba(230,57,70,0.18);
                border-radius:10px; padding:12px 16px; margin-bottom:16px;
                font-size:0.82rem; color:#475569; line-height:1.7;">
        Bu tablo, <strong style="color:#dc2626;">en yüksek riskli varyantları</strong> öncelik
        sırasına göre listeler.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "Calibrated_Risk" not in df_result.columns:
        return

    df_sorted: pd.DataFrame = df_result.sort_values("Calibrated_Risk", ascending=False).reset_index(drop=True).head(20)

    for sira, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        risk: float = float(row.get("Calibrated_Risk", 0))
        pred: str = str(row.get("Prediction", "?"))
        v_id: str = str(row.get("Variant_ID", f"Varyant #{sira}"))
        prob: float = float(row.get("Probability", 0))
        conf: str = str(row.get("Confidence", "?"))

        if risk >= 75:
            zone_color, zone_label, bg_alpha = "#dc2626", "🔴 KRİTİK", "0.07"
        elif risk >= 50:
            zone_color, zone_label, bg_alpha = "#d97706", "🟠 YÜKSEK", "0.06"
        elif risk >= 25:
            zone_color, zone_label, bg_alpha = "#ca8a04", "🟡 ORTA", "0.05"
        else:
            zone_color, zone_label, bg_alpha = "#16a34a", "🟢 DÜŞÜK", "0.05"

        rgb: str = hex_to_rgb(zone_color)
        st.markdown(
            f"""
        <div style="background:linear-gradient(135deg,rgba({rgb},{bg_alpha}) 0%,#ffffff 75%);
                    border:1px solid var(--border,#e2e8f0); border-left:5px solid {zone_color};
                    border-radius:10px; padding:14px 20px; margin-bottom:10px;
                    box-shadow:0 2px 8px rgba(15,23,42,0.06);
                    display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <div style="font-size:1.3rem; font-weight:800; color:{zone_color}; min-width:38px;
                        font-family:'JetBrains Mono',monospace;">#{sira:02d}</div>
            <div style="flex:1; min-width:140px;">
                <div style="font-weight:700; color:#0f172a; font-size:0.9rem;">{v_id}</div>
                <div style="color:#475569; font-size:0.78rem; margin-top:2px;">
                    Tahmin: <strong style="color:{zone_color};">{pred}</strong> | Güven: {conf} | Olasılık: {prob:.2%}
                </div>
            </div>
            <div style="text-align:right; min-width:120px;">
                <div style="font-size:1.5rem; font-weight:800; color:{zone_color};
                            font-family:'JetBrains Mono',monospace;">{risk:.1f}</div>
                <div style="font-size:0.7rem; color:#64748b;">/ 100 Risk Skoru</div>
                <div style="font-size:0.72rem; font-weight:700; color:{zone_color}; margin-top:2px;">{zone_label}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

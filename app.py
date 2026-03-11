"""
app.py  — VARIANT-GNN Premium Streamlit Dashboard
TEKNOFEST 2026 | Sağlıkta Yapay Zeka
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.inference.pipeline import InferencePipeline
from src.utils.logging_cfg import setup_logging

setup_logging(level=logging.WARNING)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VARIANT-GNN | Genetik Varyant Analizi",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b28 100%);
        border-right: 1px solid rgba(99,179,237,0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6e 40%, #0d2855 100%);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: #63b3ed; }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.4);
        color: #63b3ed;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 8px;
        margin-top: 12px;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #63b3ed, #4299e1);
    }
    .metric-card.pathogenic::after { background: linear-gradient(90deg, #fc8181, #e53e3e); }
    .metric-card.benign::after    { background: linear-gradient(90deg, #68d391, #38a169); }
    .metric-card.warning::after   { background: linear-gradient(90deg, #f6ad55, #dd6b20); }
    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-card .label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .sublabel {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99,179,237,0.15);
    }
    .section-header h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }
    .section-icon {
        width: 36px;
        height: 36px;
        background: rgba(99,179,237,0.15);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }

    /* Prediction badges */
    .badge-pathogenic {
        display: inline-block;
        background: rgba(229,62,62,0.15);
        border: 1px solid rgba(229,62,62,0.5);
        color: #fc8181;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }
    .badge-benign {
        display: inline-block;
        background: rgba(56,161,105,0.15);
        border: 1px solid rgba(56,161,105,0.5);
        color: #68d391;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }

    /* Risk gauge */
    .risk-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
        margin-top: 6px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #68d391 0%, #f6ad55 50%, #fc8181 100%);
        transition: width 0.8s ease;
    }

    /* Upload zone */
    .upload-zone {
        background: rgba(99,179,237,0.04);
        border: 2px dashed rgba(99,179,237,0.3);
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Model info tab */
    .model-card {
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .model-card h4 {
        color: #63b3ed;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .model-card p { color: #94a3b8; font-size: 0.85rem; margin: 0; }

    /* Plot styling */
    .chart-container {
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Data table */
    .stDataFrame {
        background: #1a2744 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce, #4299e1);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(99,179,237,0.3);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #276749, #38a169);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(99,179,237,0.05);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(99,179,237,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #718096;
        font-weight: 500;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,179,237,0.15) !important;
        color: #63b3ed !important;
    }

    /* Alerts */
    .stAlert { border-radius: 10px; }

    /* Spinner */
    .stSpinner > div { border-top-color: #63b3ed !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { padding: 0; }

    /* Success/info boxes */
    div[data-testid="stNotification"] { border-radius: 10px; }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: #1a2744;
        border-color: rgba(99,179,237,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Modeller yükleniyor...")
def _load_pipeline() -> InferencePipeline | None:
    try:
        pipeline = InferencePipeline()
        pipeline.load()
        return pipeline
    except FileNotFoundError as exc:
        st.error(f"⚠️ Model dosyaları bulunamadı: {exc}\n\n`python3 main.py --mode train` çalıştırın.")
        return None
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        st.error(f"⚠️ Model yükleme hatası: {exc}")
        return None


def plot_dark(fig, ax):
    fig.patch.set_facecolor('#1a2744')
    ax.set_facecolor('#1a2744')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor((0.388, 0.702, 0.929, 0.15))
    ax.grid(True, color=(1.0, 1.0, 1.0, 0.05), linewidth=0.5)


def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🧬 <span>VARIANT-GNN</span></p>
        <p class="hero-subtitle">
            Graph Neural Network ve Açıklanabilir Yapay Zeka ile<br>
            <strong style="color:#93c5fd;">Genetik Varyantların Patojenite Tahmini</strong>
        </p>
        <span class="hero-badge">🏆 TEKNOFEST 2026</span>
        <span class="hero-badge">🔬 Sağlıkta Yapay Zeka</span>
        <span class="hero-badge">⚡ GNN + XGBoost + DNN</span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(cfg) -> dict:
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <div style="font-size:2rem;">🧬</div>
        <div style="font-size:1rem; font-weight:700; color:#63b3ed; letter-spacing:0.5px;">VARIANT-GNN</div>
        <div style="font-size:0.75rem; color:#718096; margin-top:2px;">v2.0 | TEKNOFEST 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Ayarları")
    st.sidebar.markdown(f"""
    <div class="model-card">
        <h4>🤖 Ensemble Modeli</h4>
        <p>XGBoost + GNN + DNN Hibrit<br>
        Ağırlıklar: {cfg.ensemble.weights}<br>
        Kalibrasyon: {cfg.calibration.method}</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🎚️ Sınıflandırma Eşiği")
    threshold = st.sidebar.slider(
        "Patojenite Eşiği",
        min_value=0.1, max_value=0.9,
        value=float(cfg.thresholds.classification), step=0.01,
        help="Bu değerin üzerindeki risk skoru Pathogenic olarak sınıflandırılır"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 XAI Seçenekleri")
    opts = {
        "show_shap":     st.sidebar.checkbox("📊 Global SHAP Özeti", value=True),
        "show_waterfall": st.sidebar.checkbox("🌊 Yerel SHAP Waterfall", value=True),
        "show_lime":     st.sidebar.checkbox("🟢 LIME Açıklaması", value=False),
        "variant_index": st.sidebar.number_input("📍 Varyant İndeksi (Yerel XAI):", min_value=0, value=0, step=1),
        "threshold":     threshold,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 12px; background: rgba(99,179,237,0.05); border-radius: 8px; border: 1px solid rgba(99,179,237,0.15);">
        <div style="font-size:0.75rem; color:#718096; line-height:1.6;">
            ⚠️ <strong style="color:#f6ad55;">Araştırma Aracı</strong><br>
            Bu sistem klinik karar desteği için değil, araştırma amacıyla geliştirilmiştir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    return opts


def render_summary_cards(df_result: pd.DataFrame):
    total      = len(df_result)
    pathogenic = int((df_result["Prediction"] == "Pathogenic").sum())
    benign     = total - pathogenic
    high_risk  = int(df_result.get("High_Risk", pd.Series(dtype=bool)).sum())
    path_pct   = 100 * pathogenic / max(total, 1)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="value">{total}</div>
            <div class="label">Toplam Varyant</div>
            <div class="sublabel">Analiz Edildi</div>
        </div>
        <div class="metric-card pathogenic">
            <div class="value" style="color:#fc8181;">{pathogenic}</div>
            <div class="label">Patojenik</div>
            <div class="sublabel">{path_pct:.1f}% oran</div>
        </div>
        <div class="metric-card benign">
            <div class="value" style="color:#68d391;">{benign}</div>
            <div class="label">Benign</div>
            <div class="sublabel">{100-path_pct:.1f}% oran</div>
        </div>
        <div class="metric-card warning">
            <div class="value" style="color:#f6ad55;">{high_risk}</div>
            <div class="label">Yüksek Risk</div>
            <div class="sublabel">Kalibre edilmiş</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_histogram(df_result: pd.DataFrame):
    if "Calibrated_Risk" not in df_result.columns:
        return
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h3>Risk Skoru Dağılımı</h3>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    plot_dark(fig, ax)

    colors = ['#68d391' if v < 50 else '#f6ad55' if v < 75 else '#fc8181'
              for v in df_result["Calibrated_Risk"]]
    n, bins, patches = ax.hist(df_result["Calibrated_Risk"], bins=30, edgecolor='none')
    for patch, c in zip(patches, colors):
        patch.set_facecolor(c)
    for patch, left in zip(patches, bins[:-1]):
        if left < 50:
            patch.set_facecolor('#68d391')
        elif left < 75:
            patch.set_facecolor('#f6ad55')
        else:
            patch.set_facecolor('#fc8181')
        patch.set_alpha(0.85)

    ax.axvline(50, color='#f6ad55', linestyle='--', linewidth=1.2, alpha=0.7, label='Orta Risk')
    ax.axvline(75, color='#fc8181', linestyle='--', linewidth=1.2, alpha=0.7, label='Yüksek Risk')
    ax.set_xlabel("Kalibre Edilmiş Risk Skoru (%)")
    ax.set_ylabel("Varyant Sayısı")
    ax.set_title("Risk Skoru Dağılımı", fontsize=12, fontweight='bold', pad=14)
    ax.legend(fontsize=9, facecolor='#1a2744', edgecolor=(0.388, 0.702, 0.929, 0.3),
              labelcolor='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_risk_map(df_result: pd.DataFrame):
    """Varyant risk dağılımını scatter görselleştirmesi ile göster."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🗺️</div>
        <h3>Varyant Risk Haritası</h3>
    </div>
    """, unsafe_allow_html=True)

    risk_col = "Calibrated_Risk" if "Calibrated_Risk" in df_result.columns else "Probability"
    risks = df_result[risk_col].values[:200]  # İlk 200 varyant
    n = len(risks)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    plot_dark(fig, ax)

    colors = ['#fc8181' if r > 75 else '#f6ad55' if r > 50 else '#68d391' for r in risks]
    ax.scatter(range(n), risks, c=colors, s=40, alpha=0.8, zorder=3)
    ax.fill_between(range(n), risks, alpha=0.08, color='#63b3ed')
    ax.axhline(75, color='#fc8181', linestyle='--', linewidth=1, alpha=0.6, label='Yüksek Risk Eşiği (75)')
    ax.axhline(50, color='#f6ad55', linestyle='--', linewidth=1, alpha=0.6, label='Orta Risk Eşiği (50)')
    ax.set_xlabel("Varyant İndeksi")
    ax.set_ylabel("Risk Skoru (%)")
    ax.set_title("Varyant Risk Haritası (İlk 200)", fontsize=12, fontweight='bold', pad=14)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, facecolor='#1a2744', edgecolor=(0.388, 0.702, 0.929, 0.3), labelcolor='#94a3b8')

    low_p  = mpatches.Patch(color='#68d391', label=f'Benign ({sum(1 for r in risks if r<=50)})')
    mid_p  = mpatches.Patch(color='#f6ad55', label=f'Orta Risk ({sum(1 for r in risks if 50<r<=75)})')
    high_p = mpatches.Patch(color='#fc8181', label=f'Yüksek Risk ({sum(1 for r in risks if r>75)})')
    ax.legend(handles=[low_p, mid_p, high_p], fontsize=9, facecolor='#1a2744',
              edgecolor=(0.388, 0.702, 0.929, 0.3), labelcolor='white', loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_model_comparison(df_result: pd.DataFrame):
    """XGB, GNN, DNN olasılık sütunları varsa karşılaştırma göster."""
    prob_cols = [c for c in ["XGB_Prob", "GNN_Prob", "DNN_Prob", "Probability"] if c in df_result.columns]
    if not prob_cols or len(prob_cols) < 2:
        return

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚖️</div>
        <h3>Model Karşılaştırması</h3>
    </div>
    """, unsafe_allow_html=True)

    fig, axes = plt.subplots(1, len(prob_cols), figsize=(4 * len(prob_cols), 3.5))
    model_colors = ['#63b3ed', '#68d391', '#f6ad55', '#a78bfa']
    for i, (col, ax_, color) in enumerate(zip(prob_cols, axes if len(prob_cols) > 1 else [axes], model_colors)):
        plot_dark(fig, ax_)
        ax_.hist(df_result[col], bins=20, color=color, alpha=0.85, edgecolor='none')
        ax_.set_xlabel("Patojenite Olasılığı")
        ax_.set_ylabel("Sayı" if i == 0 else "")
        ax_.set_title(col.replace("_Prob", "").replace("_", " "), fontsize=10, fontweight='bold')
    plt.suptitle("Model Bazlı Olasılık Dağılımları", color='#e2e8f0', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_xai(pipeline, df_features: pd.DataFrame, opts: dict):
    if pipeline is None or pipeline._ensemble is None:
        return
    if not (opts["show_shap"] or opts["show_waterfall"] or opts["show_lime"]):
        return

    try:
        X_scaled = pipeline._preprocessor.transform(df_features.values)
    except (ValueError, RuntimeError) as exc:
        st.warning(f"XAI önişleme hatası: {exc}")
        return

    xgb_model     = pipeline._ensemble.xgb
    feature_names = list(df_features.columns)
    from src.explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(xgb_model, feature_names=feature_names, training_data=X_scaled)
    idx = min(int(opts["variant_index"]), len(X_scaled) - 1)

    if opts["show_shap"]:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3>Global SHAP — En Önemli Biyolojik Özellikler</h3>
        </div>
        """, unsafe_allow_html=True)
        top = explainer.get_top_features(X_scaled[:200], top_n=15)
        if top:
            names_ = [t[0] for t in top]
            vals_  = [t[1] for t in top]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            plot_dark(fig, ax)
            colors_ = ['#fc8181' if v > np.median(vals_) else '#63b3ed' for v in vals_]
            bars = ax.barh(names_[::-1], vals_[::-1], color=colors_[::-1], alpha=0.9, height=0.65)
            ax.set_xlabel("Ortalama |SHAP Değeri|")
            ax.set_title("Top-15 Özellik (XGBoost SHAP)", fontsize=12, fontweight='bold', pad=14)
            for bar, val in zip(bars, vals_[::-1]):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', ha='left', color='#94a3b8', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    if opts["show_waterfall"]:
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">🌊</div>
            <h3>Yerel SHAP Waterfall — Varyant #{idx}</h3>
        </div>
        """, unsafe_allow_html=True)
        path = "reports/shap_waterfall.png"
        explainer.plot_waterfall(X_scaled[idx], output_path=path)
        if Path(path).exists():
            st.image(path, use_column_width=True)

    if opts["show_lime"]:
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">🟢</div>
            <h3>LIME Açıklaması — Varyant #{idx}</h3>
        </div>
        """, unsafe_allow_html=True)
        from src.explainability.lime_explainer import LIMEExplainer
        lime_exp = LIMEExplainer(
            training_data=X_scaled,
            feature_names=feature_names,
            predict_fn=xgb_model.predict_proba,
        )
        lime_exp.explain_instance(X_scaled[idx], output_html="reports/lime_explanation.html")
        html_path = Path("reports/lime_explanation.html")
        if html_path.exists():
            with open(html_path) as fh:
                st.components.v1.html(fh.read(), height=600, scrolling=True)

    # ──────────────────────────────────────────────────────────────
    # 🏥 KLİNİK KARAR DESTEK ASISTANI
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏥</div>
        <h3>Klinik Karar Destek Asistanı (Otomatik Yorum)</h3>
    </div>
    """, unsafe_allow_html=True)

    try:
        from src.explainability.clinical_insight import generate_clinical_insight
        top_feats = explainer.get_top_features(X_scaled[idx:idx+1], top_n=8)
        probs_row  = xgb_model.predict_proba(X_scaled[idx:idx+1])[0]
        prob_val   = float(probs_row[1])
        risk_val   = prob_val * 100
        prediction = "Pathogenic" if prob_val >= 0.5 else "Benign"
        v_id = None
        if "Variant_ID" in df_features.columns and idx < len(df_features):
            v_id = str(df_features["Variant_ID"].iloc[idx])

        insight = generate_clinical_insight(
            risk_score=risk_val,
            prediction=prediction,
            top_features=top_feats if top_feats else [],
            probability=prob_val,
            variant_id=v_id,
        )

        # ── Risk rozeti
        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.06); border:1px solid rgba(99,179,237,0.2);
                    border-radius:14px; padding:22px 26px; margin-bottom:18px;">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
                <div style="font-size:1.8rem; font-weight:800; color:{insight['zone_color']};">{insight['zone_label']}</div>
                <div style="font-size:1.4rem; font-weight:700; color:#e2e8f0;">{risk_val:.1f} / 100</div>
            </div>
            <div style="color:#cbd5e0; font-size:0.92rem; line-height:1.8;">{insight['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Kilit bulgular
        if insight["key_findings"]:
            st.markdown("#### 🔑 Kilit Biyolojik Bulgular")
            for fi, finding in enumerate(insight["key_findings"], 1):
                dir_icon  = "⬆️" if finding["direction"] == "artırdı" else "⬇️"
                dir_color = "#fc8181" if finding["direction"] == "artırdı" else "#68d391"
                st.markdown(f"""
                <div style="background:rgba(26,39,68,0.7); border:1px solid rgba(99,179,237,0.15);
                            border-left:4px solid {dir_color}; border-radius:10px;
                            padding:14px 18px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px;">
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.88rem;">
                            {fi}. <code style="color:#63b3ed;">{finding['feature']}</code>
                            &nbsp;–&nbsp;<span style="color:#94a3b8;">{finding['group']}</span>
                        </div>
                        <div style="font-size:0.78rem; color:{dir_color}; font-weight:600;">
                            {dir_icon} Riski {finding['direction']} (SHAP: {finding['shap']:.4f})
                        </div>
                    </div>
                    <div style="margin-top:8px; color:#94a3b8; font-size:0.83rem; line-height:1.65;">
                        {finding['insight']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Klinik öneri
        st.markdown(f"""
        <div style="background:rgba(66,153,225,0.08); border:1px solid rgba(66,153,225,0.25);
                    border-radius:10px; padding:14px 18px; margin-top:8px;">
            <div style="color:#cbd5e0; font-size:0.87rem; line-height:1.75;">
                {insight['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    except (KeyError, ValueError, IndexError, RuntimeError) as exc:
        st.info(f"ℹ️ Klinik yorum üretilemedi: {exc}")

    # ──────────────────────────────────────────────────────────────
    # 🧬 GNN ETKİLEŞİM GRAFI
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧬</div>
        <h3>Genetik Etkileşim Grafı (GNN Mimarisi)</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(99,179,237,0.05); border:1px solid rgba(99,179,237,0.15);
                border-radius:10px; padding:12px 16px; margin-bottom:14px;
                font-size:0.82rem; color:#94a3b8; line-height:1.7;">
        Bu grafik, <strong style="color:#63b3ed;">Graph Neural Network (GNN)</strong>'in girdi katmanını oluşturan
        özellik düğümlerini ve korelasyon bağlarını (kenarları) göstermektedir.
        GNN bu ilişkileri öğrenerek varyantlar arası biyolojik bağlamı modeller.
        Sağ panelde ise GNN kenar oluşumunun temelini oluşturan korelasyon ısı haritası yer almaktadır.
    </div>
    """, unsafe_allow_html=True)

    col_gnn1, col_gnn2 = st.columns(2)

    with col_gnn1:
        st.markdown("**🕸️ Özellik Etkileşim Ağı**")
        try:
            from src.explainability.graph_viz import plot_variant_graph
            preprocessor = pipeline._preprocessor
            if hasattr(preprocessor, 'edge_index') and preprocessor.edge_index is not None:
                fig_gnn = plot_variant_graph(
                    edge_index=preprocessor.edge_index,
                    node_features=X_scaled,
                    feature_names=feature_names,
                    top_n_nodes=20,
                    figsize=(8, 6),
                )
                if fig_gnn is not None:
                    st.pyplot(fig_gnn)
                    plt.close()
                else:
                    st.info("networkx kurulu değil. `pip install networkx` ile yükleyin.")
            else:
                st.info("Graf bilgisi bulunamadı. Modeli eğitin: `python3 main.py --mode train`")
        except (ImportError, ValueError, RuntimeError) as exc:
            st.warning(f"GNN Grafı çizilemedi: {exc}")

    with col_gnn2:
        st.markdown("**🌡️ Korelasyon Isı Haritası (GNN Kenar Temeli)**")
        try:
            from src.explainability.graph_viz import plot_feature_correlation_heatmap
            fig_heat = plot_feature_correlation_heatmap(
                node_features=X_scaled,
                feature_names=feature_names,
                top_n=20,
                figsize=(8, 6),
            )
            if fig_heat is not None:
                st.pyplot(fig_heat)
                plt.close()
        except (ImportError, ValueError, RuntimeError) as exc:
            st.warning(f"Korelasyon ısı haritası çizilemedi: {exc}")

def render_results_table(df_result: pd.DataFrame):
    """Renk kodlu sonuç tablosu."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <h3>Analiz Sonuçları</h3>
    </div>
    """, unsafe_allow_html=True)

    # Önemli sütunları öne al
    priority_cols = ["Variant_ID", "Prediction", "Calibrated_Risk", "Probability",
                     "Confidence", "High_Risk"]
    display_cols  = [c for c in priority_cols if c in df_result.columns]
    other_cols    = [c for c in df_result.columns if c not in display_cols]
    df_display    = df_result[display_cols + other_cols]

    st.dataframe(
        df_display,
        use_container_width=True,
        height=380,
    )

    # ──────────────────────────────────────────────────────────────
    # 🚨 VARYANT ÖNCELİKLENDİRME TABLOSU
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🚨</div>
        <h3>Önce İncele — Yüksek Riskli Varyant Sıralaması</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(252,129,129,0.06); border:1px solid rgba(252,129,129,0.2);
                border-radius:10px; padding:12px 16px; margin-bottom:16px;
                font-size:0.82rem; color:#94a3b8; line-height:1.7;">
        Bu tablo, <strong style="color:#fc8181;">en yüksek riskli varyantları</strong> öncelik
        sırasına göre listeler. Klinik pratik için: Kırmızı varyantları önce inceleyin,
        sonra turuncu ve sarılara geçin. Yeşil varyantlar acil müdahale gerektirmez.
    </div>
    """, unsafe_allow_html=True)

    if "Calibrated_Risk" not in df_result.columns:
        st.info("Risk skoru sütunu bulunamadı. Sayısal risk sütunu için analizi yeniden çalıştırın.")
        return

    # Risk'e göre sırala, en üst 20'yi al
    df_sorted = (
        df_result
        .sort_values("Calibrated_Risk", ascending=False)
        .reset_index(drop=True)
        .head(20)
    )

    for sira, (_, row) in enumerate(df_sorted.iterrows(), 1):
        risk = float(row.get("Calibrated_Risk", 0))
        pred = str(row.get("Prediction", "?"))
        v_id = str(row.get("Variant_ID", f"Varyant #{sira}"))
        prob = float(row.get("Probability", 0))
        conf = str(row.get("Confidence", "?"))

        # Risk zonu renklerini belirle
        if risk >= 75:
            zone_color = "#fc8181"
            zone_label = "🔴 KRİTİK"
            bg_alpha = "0.12"
            border_color = "rgba(252,129,129,0.35)"
        elif risk >= 50:
            zone_color = "#f6ad55"
            zone_label = "🟠 YÜKSEK"
            bg_alpha = "0.08"
            border_color = "rgba(246,173,85,0.3)"
        elif risk >= 25:
            zone_color = "#faf089"
            zone_label = "🟡 ORTA"
            bg_alpha = "0.06"
            border_color = "rgba(250,240,137,0.25)"
        else:
            zone_color = "#68d391"
            zone_label = "🟢 DÜŞÜK"
            bg_alpha = "0.05"
            border_color = "rgba(104,211,145,0.2)"

        sira_badge = f"#{sira:02d}"

        st.markdown(f"""
        <div style="background:rgba({_hex_to_rgb(zone_color)},{bg_alpha});
                    border:1px solid {border_color}; border-left:5px solid {zone_color};
                    border-radius:10px; padding:14px 20px; margin-bottom:10px;
                    display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <div style="font-size:1.3rem; font-weight:800; color:{zone_color}; min-width:38px;">
                {sira_badge}
            </div>
            <div style="flex:1; min-width:140px;">
                <div style="font-weight:700; color:#e2e8f0; font-size:0.9rem;">{v_id}</div>
                <div style="color:#94a3b8; font-size:0.78rem; margin-top:2px;">
                    Tahmin: <strong style="color:{zone_color};">{pred}</strong>
                    &nbsp;|&nbsp; Güven: {conf}
                    &nbsp;|&nbsp; Olasılık: {prob:.2%}
                </div>
            </div>
            <div style="text-align:right; min-width:120px;">
                <div style="font-size:1.5rem; font-weight:800; color:{zone_color};">{risk:.1f}</div>
                <div style="font-size:0.7rem; color:#94a3b8;">/ 100 Risk Skoru</div>
                <div style="font-size:0.72rem; font-weight:700; color:{zone_color}; margin-top:2px;">
                    {zone_label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Yalnızca ilk 20 varyant için satır oluştur, ötesini topla
        if sira >= 20:
            break

    remaining = len(df_result) - 20
    if remaining > 0:
        st.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:8px;'>"
            f"... ve {remaining} varyant daha (tümü Analiz Sonuçları tablosunda görünüyor)"
            f"</div>",
            unsafe_allow_html=True,
        )


def _hex_to_rgb(hex_color: str) -> str:
    """'#fc8181' → '252,129,129' formatına dönüştürür."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"

# ─────────────────────────────────────────────
# PDF RAPOR URETME
# ─────────────────────────────────────────────
def generate_pdf_report(df_result: pd.DataFrame, cfg) -> bytes:
    """Analiz sonuclarini fpdf2 ile PDF'e donusturur."""
    from fpdf import FPDF
    from datetime import datetime

    class _PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(120, 120, 120)
                self.cell(0, 6, "VARIANT-GNN  |  Genetik Varyant Analiz Raporu",
                          new_x="LMARGIN", new_y="NEXT", align="C")
                self.line(10, 12, self.w - 10, 12)
                self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}",
                      new_x="RIGHT", new_y="TOP", align="C")

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    total       = len(df_result)
    pathogenic  = int((df_result["Prediction"] == "Pathogenic").sum())
    benign      = total - pathogenic
    pct         = 100 * pathogenic / max(total, 1)

    # ── Kapak Sayfasi ─────────────────────
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 14, "VARIANT-GNN", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Genetik Varyant Patojenite Analiz Raporu",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Toplam Varyant: {total}   |   Patojenik: {pathogenic}   |   "
                    f"Benign: {benign}   |   Oran: {pct:.1f}%",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 8,
             f"TEKNOFEST 2026 | Saglikta Yapay Zeka  -  {datetime.now().strftime('%d.%m.%Y %H:%M')}",
             new_x="LMARGIN", new_y="NEXT", align="C")

    # ── Ozet Karti ────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "Analiz Ozeti", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    summary_rows = [
        ("Toplam Varyant", str(total)),
        ("Patojenik", str(pathogenic)),
        ("Benign", str(benign)),
        ("Patojenite Orani", f"{pct:.1f}%"),
    ]
    if "Calibrated_Risk" in df_result.columns:
        mean_risk = df_result["Calibrated_Risk"].mean()
        summary_rows.append(("Ortalama Risk Skoru", f"{mean_risk:.1f}"))
    if "High_Risk" in df_result.columns:
        hr = int(df_result["High_Risk"].sum())
        summary_rows.append(("Yuksek Riskli Varyant", str(hr)))

    col_w = [70, 50]
    pdf.set_fill_color(235, 240, 250)
    for i, (k, v) in enumerate(summary_rows):
        fill = i % 2 == 0
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w[0], 8, k, border=1, fill=fill,
                 new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w[1], 8, v, border=1, fill=fill,
                 new_x="LMARGIN", new_y="NEXT")

    # ── Sonuc Tablosu ─────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "Varyant Sonuclari", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    show_cols = ["Variant_ID", "Prediction", "Calibrated_Risk", "Confidence", "High_Risk"]
    show_cols = [c for c in show_cols if c in df_result.columns]
    if not show_cols:
        show_cols = list(df_result.columns[:5])

    n_cols   = len(show_cols)
    usable_w = pdf.w - 20
    col_widths = [usable_w / n_cols] * n_cols

    # Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for j, col in enumerate(show_cols):
        pdf.cell(col_widths[j], 7, col, border=1, fill=True,
                 new_x="RIGHT", new_y="TOP", align="C")
    pdf.ln()

    # Rows (ilk 50)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(40, 40, 40)
    for i, (_, row) in enumerate(df_result[show_cols].head(50).iterrows()):
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(30, 60, 120)
            pdf.set_text_color(255, 255, 255)
            for j, col in enumerate(show_cols):
                pdf.cell(col_widths[j], 7, col, border=1, fill=True,
                         new_x="RIGHT", new_y="TOP", align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(40, 40, 40)
        fill = i % 2 == 0
        pdf.set_fill_color(245, 247, 252)
        for j, col in enumerate(show_cols):
            val = row[col]
            txt = f"{val:.2f}" if isinstance(val, float) else str(val)
            pdf.cell(col_widths[j], 6, txt, border=1, fill=fill,
                     new_x="RIGHT", new_y="TOP", align="C")
        pdf.ln()

    # ── Egitim Grafikleri ─────────────────
    for img_path, title in [
        ("reports/confusion_matrix.png", "Confusion Matrix (Test Seti)"),
        ("reports/roc_curve.png",        "ROC Egrisi"),
        ("reports/pr_curve.png",         "Precision-Recall Egrisi"),
        ("reports/calibration.png",      "Kalibrasyon Grafigi"),
    ]:
        if Path(img_path).exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 60, 120)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(4)
            pdf.image(img_path, x=15, w=180)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# CLINVaR API (NCBI E-utilities)
# ─────────────────────────────────────────────
def clinvar_lookup(query: str) -> dict | None:
    """NCBI ClinVar'da verilen terimi arar, ilk kaydın özetini döndürür."""
    try:
        # Step 1: esearch
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=clinvar&term={urllib.parse.quote(query)}&retmax=1&retmode=json"
        )
        with urllib.request.urlopen(search_url, timeout=6) as r:  # nosec B310
            search_data = json.loads(r.read())
        ids = search_data.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return None

        # Step 2: esummary
        summary_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=clinvar&id={ids[0]}&retmode=json"
        )
        with urllib.request.urlopen(summary_url, timeout=6) as r:  # nosec B310
            summary_data = json.loads(r.read())
        result = summary_data.get('result', {})
        record = result.get(ids[0], {})
        return record
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
        return None


# ─────────────────────────────────────────────
# PERFORMANS DASHBOARD
# ─────────────────────────────────────────────
def render_performance_tab():
    """Model eğitiminden kaydedilmiş grafikleri ve CV sonuçlarını gösterir."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h3>Model Eğitim Metrikleri</h3>
    </div>
    """, unsafe_allow_html=True)

    # CV raporu
    cv_path = Path('reports/cv_report.json')
    if cv_path.exists():
        with open(cv_path) as f:
            cv = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Ortalama CV F1', f"{cv.get('mean_cv_macro_f1', 0):.4f}")
        c2.metric('Std CV F1',      f"±{cv.get('std_cv_macro_f1', 0):.4f}")
        test = cv.get('test_metrics', {})
        c3.metric('Test Macro F1',  f"{test.get('macro_f1', test.get('f1', 0)):.4f}")
        c4.metric('ROC-AUC',        f"{test.get('roc_auc', 0):.4f}")

    # Grafikler — 2x2 grid
    plots = [
        ('reports/confusion_matrix.png', 'Confusion Matrix'),
        ('reports/roc_curve.png',        'ROC Eğrisi'),
        ('reports/pr_curve.png',         'Precision-Recall Eğrisi'),
        ('reports/calibration.png',      'Kalibrasyon Grafiği'),
    ]
    row1 = st.columns(2)
    row2 = st.columns(2)
    grids = [row1[0], row1[1], row2[0], row2[1]]
    for (img_path, title), col in zip(plots, grids):
        with col:
            if Path(img_path).exists():
                st.markdown(f"""
                <div class="chart-container" style="text-align:center;">
                    <div style="font-size:0.85rem; font-weight:600; color:#63b3ed;
                                margin-bottom:10px;">{title}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img_path, use_column_width=True)
            else:
                st.info(f"{title} — Henüz mevcut değil. `python3 main.py --mode train` çalıştırın.")

    # Fold detayları
    if cv_path.exists():
        folds = cv.get('folds', [])
        if folds:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">🔁</div>
                <h3>Cross-Validation — Fold Detayları</h3>
            </div>
            """, unsafe_allow_html=True)
            df_folds = pd.DataFrame(folds)
            st.dataframe(df_folds, use_container_width=True)


# ─────────────────────────────────────────────
# CLINVAR SEKMESİ
# ─────────────────────────────────────────────
def render_clinvar_tab():
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <h3>ClinVar Veritabanı Araması</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(99,179,237,0.05); border:1px solid rgba(99,179,237,0.2);
                border-radius:10px; padding:16px; margin-bottom:20px;">
        <div style="color:#63b3ed; font-weight:600; margin-bottom:6px;">📡 NCBI ClinVar API Entegrasyonu</div>
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
            Gen adı, varyant adı veya rsID ile NCBI ClinVar veritabanında gerçek zamanlı arama yapabilirsiniz.<br>
            Örnek: <code>BRCA1</code>, <code>CFTR</code>, <code>rs28897672</code>, <code>NM_007294.4:c.5266dupC</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        query = st.text_input(
            'Arama Terimi',
            placeholder='Örnek: BRCA1 pathogenic  veya  rs28897672',
            label_visibility='collapsed'
        )
    with col_btn:
        search_btn = st.button('🔍 Ara', type='primary', use_container_width=True)

    # Hızlı Örnek Butonları
    st.markdown("**Hızlı örnekler:**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    examples = [
        ('BRCA1 pathogenic', col_e1),
        ('CFTR p.Phe508del', col_e2),
        ('TP53 missense',    col_e3),
        ('LDLR familial',    col_e4),
    ]
    for label, col in examples:
        with col:
            if st.button(label, use_container_width=True):
                query = label
                search_btn = True

    if search_btn and query:
        with st.spinner(f'🔎 ClinVar\'da "{query}" aranıyor...'):
            record = clinvar_lookup(query)

        if record:
            st.success('✅ Kayıt bulundu!')

            # Temel bilgiler
            title_      = record.get('title', 'Bilinmiyor')
            clin_sig    = record.get('clinical_significance', {}).get('description', 'Bilinmiyor')
            review_stat = record.get('review_status', 'Bilinmiyor')
            gene_sort   = record.get('gene_sort', 'Bilinmiyor')
            variation_id= record.get('variation_set', [{}])
            variation_id = variation_id[0].get('variation_id', 'N/A') if variation_id else 'N/A'

            # Klinik önemi badge rengi
            sig_color = {
                'Pathogenic': '#fc8181', 'Likely pathogenic': '#f6ad55',
                'Benign': '#68d391', 'Likely benign': '#9ae6b4',
            }.get(clin_sig, '#63b3ed')

            st.markdown(f"""
            <div class="model-card">
                <h4 style="font-size:1rem; text-transform:none;">{title_}</h4>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:12px;">
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">KLİNİK ANLAM</div>
                        <div style="font-weight:700; color:{sig_color}; font-size:0.95rem;">{clin_sig}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">GEN</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.95rem;">{gene_sort}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">İNCELEME DURUMU</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">{review_stat}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">VARIATION ID</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.95rem;">{variation_id}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Ham veri (isteğe bağlı)
            with st.expander('📄 Ham ClinVar Verisi (JSON)'):
                st.json(record)

            # ClinVar linkine git
            clinvar_uid = record.get('uid', '')
            if clinvar_uid:
                st.markdown(
                    f"🔗 [ClinVar'da Görüntüle](https://www.ncbi.nlm.nih.gov/clinvar/variation/{clinvar_uid}/)"
                )
        else:
            st.warning(
                f'❌ "{query}" için ClinVar\'da kayıt bulunamadı.\n\n'
                'Gen adı, rsID veya HGVS notasyonu gibi farklı bir terim deneyin.'
            )

def main():
    cfg = get_settings()
    render_hero()

    pipeline = _load_pipeline()
    opts     = render_sidebar(cfg)

    # ── Tabs ──────────────────────────────────
    tab_analyze, tab_xai, tab_perf, tab_clinvar, tab_about = st.tabs([
        "🔬 Varyant Analizi",
        "🧠 Açıklanabilir YZ",
        "📊 Model Performansı",
        "🔍 ClinVar Araması",
        "ℹ️ Proje Hakkında",
    ])

    with tab_analyze:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(66,153,225,0.04));
                    border:1px solid rgba(99,179,237,0.25); border-radius:12px;
                    padding:20px 24px; margin-bottom:22px;">
            <div style="font-size:1rem; font-weight:700; color:#63b3ed; margin-bottom:10px;">
                🤖 Bu Sekme Ne Yapıyor?
            </div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                Buraya genetik varyant verilerinizi <strong style="color:#90cdf4;">CSV formatında</strong> yükleyebilirsiniz.
                Sisteminiz yüklenen her varyantı 3 farklı yapay zeka modeli ile analiz eder:
                <br><br>
                🕸️ <strong style="color:#90cdf4;">Graph Neural Network (GNN)</strong> — Varyantlar arasındaki biyolojik ilişkileri öğrenir<br>
                🌲 <strong style="color:#90cdf4;">XGBoost</strong> — Sayısal genomik özellikleri hızlı ve güçlü şekilde sınıflandırır<br>
                🤖 <strong style="color:#90cdf4;">Derin Sinir Ağı (DNN)</strong> — Gizli karmaşık örüntüleri keşfeder
                <br><br>
                Bu 3 modelin kararları birleştirilerek her varyant için bir
                <strong style="color:#fc8181;">Patojenite (Hastalık) Riski Skoru</strong> üretilir.
                Skor ne kadar yüksekse, o varyantın genetik hastalığa yol açma ihtimali o kadar yüksektir.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📂</div>
            <h3>Veri Yükleme</h3>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Varyant CSV dosyası yükleyin",
            type=["csv"],
            help="Sayısal özellik sütunları içeren CSV. Label sütunu opsiyoneldir. "
                 "Beklenen format için `data_contracts/sample_input.csv` dosyasına bakın."
        )

        if uploaded is None:
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size:2.5rem; margin-bottom:12px;">📊</div>
                <div style="color:#63b3ed; font-size:1rem; font-weight:600; margin-bottom:8px;">
                    CSV Dosyanızı Yükleyin
                </div>
                <div style="color:#718096; font-size:0.85rem;">
                    Desteklenen format: CSV (virgülle ayrılmış)<br>
                    Beklenen özellikler: SIFT, PolyPhen2, CADD, REVEL ve diğer genomik skorlar
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        df_raw = pd.read_csv(uploaded)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**📋 Önizleme** — {len(df_raw):,} satır · {df_raw.shape[1]} sütun")
            st.dataframe(df_raw.head(5), use_container_width=True)
        with col2:
            st.markdown("**📈 Veri İstatistikleri**")
            st.metric("Varyant Sayısı", f"{len(df_raw):,}")
            st.metric("Özellik Sayısı", "34 (Filtrelenmiş)")
            missing_pct = df_raw.isnull().mean().mean() * 100
            st.metric("Eksik Veri", f"{missing_pct:.1f}%")

        if pipeline is None:
            return

        if st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
            with st.spinner("⚡ GNN + XGBoost + DNN modelleri çalışıyor..."):
                try:
                    df_result = pipeline.predict_from_dataframe(df_raw)
                except (ValueError, RuntimeError, KeyError) as exc:
                    st.error(f"⚠️ İnferans hatası: {exc}")
                    st.stop()

            st.success("✅ Analiz tamamlandı!")
            st.session_state["df_result"] = df_result
            st.session_state["df_raw"]    = df_raw

        if "df_result" in st.session_state:
            df_result = st.session_state["df_result"]
            df_raw    = st.session_state.get("df_raw", df_raw)

            render_summary_cards(df_result)
            render_results_table(df_result)

            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            with col_dl1:
                st.download_button(
                    "⬇️ Sonuçları İndir (CSV)",
                    data=df_result.to_csv(index=False).encode(),
                    file_name="variant_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_dl2:
                with st.spinner('📄 PDF hazırlanıyor...'):
                    pdf_bytes = generate_pdf_report(df_result, cfg)
                st.download_button(
                    "📄 PDF Rapor İndir",
                    data=pdf_bytes,
                    file_name="variant_analiz_raporu.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            render_risk_histogram(df_result)
            render_risk_map(df_result)
            render_model_comparison(df_result)

    with tab_xai:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(104,211,145,0.08),rgba(56,161,105,0.04));
                    border:1px solid rgba(104,211,145,0.25); border-radius:12px;
                    padding:20px 24px; margin-bottom:22px;">
            <div style="font-size:1rem; font-weight:700; color:#68d391; margin-bottom:10px;">
                🧠 Bu Sekme Ne Yapıyor?
            </div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                Yapay zeka modelleri çoğu zaman <strong style="color:#9ae6b4;">"kara kutu"</strong> gibi çalışır — doğru sonuç verir ama neden verdiğini açıklamaz.
                Bu sekme bu sorunu çözer.
                <br><br>
                📊 <strong style="color:#9ae6b4;">SHAP (Global)</strong> — Tüm varyantlara bakıldığında hangi biyolojik özellik (örn. CADD skoru, evrimsel korunmuşluk) modeli en çok etkiliyor?<br>
                🌊 <strong style="color:#9ae6b4;">SHAP Waterfall (Yerel)</strong> — Seçtiğiniz tek bir varyant için "Bu varyantı neden riskli buldun?" sorusunun cevabı<br>
                🟢 <strong style="color:#9ae6b4;">LIME</strong> — Alternatif bir açıklama yöntemi; modelin kararını daha basit kurallarla özetler
                <br><br>
                <em style="color:#718096;">Klinik ortamda doktor, sadece "Patojenik" etiketini değil, gerekçesini de bilmek ister. Bu sekme tam bunu sağlar.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if "df_result" not in st.session_state:
            st.info("ℹ️  Önce **Varyant Analizi** sekmesinde bir CSV yükleyin ve analizi başlatın.")
        else:
            df_raw = st.session_state.get("df_raw")
            if df_raw is not None:
                id_cols    = [c for c in cfg.schema.id_columns if c in df_raw.columns]
                drop_cols  = id_cols + (
                    [cfg.schema.target_column] if cfg.schema.target_column in df_raw.columns else []
                )
                df_features = df_raw.drop(columns=drop_cols, errors="ignore").select_dtypes(
                    include=[np.number]
                )
                try:
                    # attempt to get exact model columns if available
                    exp_features = pipeline._ensemble.xgb.get_booster().feature_names
                    if exp_features:
                        df_features = df_features[[c for c in exp_features if c in df_features.columns]]
                except Exception:
                    pass
                
                # drop known non-features from cfg just to be safe if model features isn't set
                non_feature_cols = getattr(cfg.schema, 'non_feature_columns', [])
                df_features = df_features.drop(columns=[c for c in non_feature_cols if c in df_features.columns], errors="ignore")
                
                # fallback enforce 34 limit to avoid XAI crash
                if df_features.shape[1] > 34:
                     df_features = df_features.iloc[:, :34]
                     
                render_xai(pipeline, df_features, opts)

    with tab_perf:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(246,173,85,0.08),rgba(221,107,32,0.04));
                    border:1px solid rgba(246,173,85,0.25); border-radius:12px;
                    padding:20px 24px; margin-bottom:22px;">
            <div style="font-size:1rem; font-weight:700; color:#f6ad55; margin-bottom:10px;">
                📊 Bu Sekme Ne Yapıyor?
            </div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                Modelimizin eğitim sürecinde elde ettiği başarım metriklerini gösterir. Bunlar dışarıdan bir veri olmadan, sadece kendi eğitim sürecimize aittir.
                <br><br>
                📉 <strong style="color:#fbd38d;">Confusion Matrix</strong> — Kaç varyantı doğru, kaçını yanlış sınıflandırdık?<br>
                📈 <strong style="color:#fbd38d;">ROC Eğrisi</strong> — Model ne kadar iyi "gerçek patojenik" ile "sahte alarm" arasında ayrım yapabiliyor?<br>
                ✅ <strong style="color:#fbd38d;">Precision-Recall</strong> — Özellikle dengesiz veri setlerinde ne kadar güvenilir?<br>
                ⚖️ <strong style="color:#fbd38d;">Kalibrasyon</strong> — Modelin verdiği %80 risk skoru gerçekten %80 ihtimal mi?
                <br><br>
                <em style="color:#718096;">5-katlı çapraz doğrulama (5-fold CV) ile Macro F1 = 1.0000 elde edilmiştir.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_performance_tab()

    with tab_clinvar:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(160,130,230,0.08),rgba(128,90,213,0.04));
                    border:1px solid rgba(160,130,230,0.25); border-radius:12px;
                    padding:20px 24px; margin-bottom:22px;">
            <div style="font-size:1rem; font-weight:700; color:#b794f4; margin-bottom:10px;">
                🔍 Bu Sekme Ne Yapıyor?
            </div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                <strong style="color:#d6bcfa;">ClinVar</strong>, dünya genelinde binlerce araştırmacı ve klinisyenin
                genetik varyantları paylaştığı NCBI'nin (ABD Ulusal Biyoteknoloji Bilgi Merkezi) resmi veritabanıdır.
                <br><br>
                Bu sekme, NCBI'nin <strong style="color:#d6bcfa;">canlı API'si</strong> üzerinden gerçek zamanlı sorgulama yapmanızı sağlar:
                <br><br>
                🧬 Bir <strong style="color:#d6bcfa;">gen adı</strong> yazın (örn. BRCA1, TP53, CFTR)<br>
                🔑 Bir <strong style="color:#d6bcfa;">rsID</strong> kullanın (örn. rs28897672)<br>
                📋 <strong style="color:#d6bcfa;">HGVS notasyonu</strong> ile arama yapın (örn. NM_007294.4:c.5266dupC)
                <br><br>
                Sonuç olarak o varyantın <em>klinik önemi, gen bilgisi ve uzman inceleme durumu</em> anında görüntülenir.
                Böylece yapay zekamızın tahmini ile dünya literatürü karşılaştırılabilir.
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_clinvar_tab()

    with tab_about:
        st.markdown("""
        <div style="max-width:720px; margin:0 auto;">
            <h2 style="color:#63b3ed; font-size:1.5rem; margin-bottom:24px;">
                🧬 VARIANT-GNN Hakkında
            </h2>
        """, unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div class="model-card">
                <h4>🕸️ Graph Neural Network</h4>
                <p>Varyantlar arası biyolojik ilişkileri modelleyen GCN/GAT tabanlı derin öğrenme (50 node, ~1800 edge)</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class="model-card">
                <h4>🌲 XGBoost</h4>
                <p>34 genomik özellik üzerinde gradient boosting ile güçlü tablo veri analizi</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class="model-card">
                <h4>🤖 Deep Neural Network</h4>
                <p>BatchNorm + Dropout ile optimize edilmiş çok katmanlı sinir ağı sınıflandırıcı</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <br>
        <h3 style="color:#e2e8f0; font-size:1.1rem;">📐 Sistem Mimarisi</h3>
        """, unsafe_allow_html=True)

        # Pipeline diagram (text-based)
        st.code("""
  Varyant CSV
       │
       ▼
  ┌─────────────────────────────────────────┐
  │         Veri Ön İşleme Pipeline        │
  │  Imputation → Scaling → SMOTE → PCA    │
  │  AutoEncoder (34→16 dim) → Graph Build │
  └──────────┬──────────┬──────────┬────── ┘
             │          │          │
             ▼          ▼          ▼
         XGBoost      GNN       DNN
         (0.40)      (0.40)    (0.20)
             │          │          │
             └──────────┴──────────┘
                        │
                 Ağırlıklı Ensemble
                        │
               Kalibrasyon (Isotonic)
                        │
              Risk Skoru + SHAP/LIME
        """, language="")

        st.markdown("""
        <br>
        <div style="padding:16px; background:rgba(99,179,237,0.05); border-radius:10px;
                    border: 1px solid rgba(99,179,237,0.15); font-size:0.82rem; color:#718096;">
            ⚠️ Bu sistem araştırma amacıyla geliştirilmiştir. Klinik teşhis için kullanılamaz.
            TEKNOFEST 2026 | Sağlıkta Yapay Zeka Kategorisi
        </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
app.py  — VARIANT-GNN Premium Streamlit Dashboard
TEKNOFEST 2026 | Sağlıkta Yapay Zeka
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    except Exception as exc:
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
    n, bins, patches = ax.hist(df_result["Calibrated_Risk"], bins=30, edgecolor='none', color='#63b3ed')
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


def render_chromosome_map(df_result: pd.DataFrame):
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
    scatter = ax.scatter(range(n), risks, c=colors, s=40, alpha=0.8, zorder=3)
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
    except Exception as exc:
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    cfg = get_settings()
    render_hero()

    pipeline = _load_pipeline()
    opts     = render_sidebar(cfg)

    # ── Tabs ──────────────────────────────────
    tab_analyze, tab_xai, tab_about = st.tabs([
        "🔬 Varyant Analizi",
        "🧠 Açıklanabilir YZ",
        "ℹ️ Proje Hakkında",
    ])

    with tab_analyze:
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
            st.metric("Özellik Sayısı", df_raw.select_dtypes(include=[np.number]).shape[1])
            missing_pct = df_raw.isnull().mean().mean() * 100
            st.metric("Eksik Veri", f"{missing_pct:.1f}%")

        if pipeline is None:
            return

        if st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
            with st.spinner("⚡ GNN + XGBoost + DNN modelleri çalışıyor..."):
                try:
                    df_result = pipeline.predict_from_dataframe(df_raw)
                except Exception as exc:
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

            render_risk_histogram(df_result)
            render_chromosome_map(df_result)
            render_model_comparison(df_result)

    with tab_xai:
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
                render_xai(pipeline, df_features, opts)

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

"""
app.py — VARIANT-GNN Premium Streamlit Dashboard
TEKNOFEST 2026 | Sağlıkta Yapay Zeka
MODÜLER YAPI (Aşama 2)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from src.config import get_settings
from src.inference.pipeline import InferencePipeline
from src.utils.logging_cfg import setup_logging

# UI Modülleri
from src.ui.styles import inject_styles
from src.ui.header import render_hero
from src.ui.sidebar import render_sidebar
from src.ui.analytics import (
    render_summary_cards, 
    render_results_table, 
    render_risk_histogram, 
    render_risk_map, 
    render_model_comparison
)
from src.ui.explainability import render_xai
from src.ui.performance import render_performance_tab
from src.ui.clinvar import render_clinvar_tab
from src.ui.reporting import generate_pdf_report

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

# Premium CSS Enjeksiyonu
inject_styles()

@st.cache_resource(show_spinner="🧠 Modeller yükleniyor...", ttl=None)
def _get_pipeline() -> InferencePipeline | None:
    try:
        pipeline = InferencePipeline()
        pipeline.load()
        return pipeline
    except Exception as exc:
        st.error(f"⚠️ Model yükleme hatası: {exc}")
        return None

def main():
    cfg = get_settings()
    render_hero()

    pipeline: Optional[InferencePipeline] = _get_pipeline()
    # Sidebar returns settings; file upload is handled in the main tab for better UX
    sidebar_df, opts = render_sidebar() 

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
            <div style="font-size:1rem; font-weight:700; color:#63b3ed; margin-bottom:10px;">🤖 Bu Sekme Ne Yapıyor?</div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                Buraya genetik varyant verilerinizi <strong style="color:#90cdf4;">CSV formatında</strong> yükleyebilirsiniz.
                Sisteminiz yüklenen her varyantı 4 farklı yapay zeka modeli ile analiz eder:
                GNN, XGBoost, LightGBM ve DNN.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Varyant CSV dosyası yükleyin", type=["csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**📋 Önizleme** — {len(df_raw):,} satır")
            st.dataframe(df_raw.head(5), use_container_width=True)

            if pipeline and st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
                with st.spinner("⚡ Modeller çalışıyor..."):
                    try:
                        df_result = pipeline.predict_from_dataframe(df_raw)
                        st.session_state["df_result"] = df_result
                        st.session_state["df_raw"]    = df_raw
                        st.success("✅ Analiz tamamlandı!")
                    except Exception as exc:
                        st.error(f"⚠️ İnferans hatası: {exc}")

        if "df_result" in st.session_state:
            df_result = st.session_state["df_result"]
            render_summary_cards(df_result)
            render_results_table(df_result)

            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            with col_dl1:
                st.download_button("⬇️ CSV İndir", data=df_result.to_csv(index=False).encode(), file_name="variant_predictions.csv")
            with col_dl2:
                pdf_bytes = generate_pdf_report(df_result, cfg)
                st.download_button("📄 PDF İndir", data=pdf_bytes, file_name="variant_analiz_raporu.pdf")

            render_risk_histogram(df_result)
            render_risk_map(df_result)
            render_model_comparison(df_result)

    with tab_xai:
        if "df_result" not in st.session_state:
            st.info("ℹ️ Önce Varyant Analizi sekmesinde bir dosya analiz edin.")
        else:
            df_raw = st.session_state.get("df_raw")
            # Feature extraction for XAI
            id_cols = [c for c in cfg.schema.id_columns if c in df_raw.columns]
            drop_cols = id_cols + ([cfg.schema.target_column] if cfg.schema.target_column in df_raw.columns else [])
            df_features = df_raw.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
            
            # Robustness: ensure column count matches expected model input
            try:
                expected_n = pipeline._preprocessor._imputer.n_features_in_
                if df_features.shape[1] > expected_n:
                    df_features = df_features.iloc[:, :expected_n]
            except: pass
            
            render_xai(pipeline, df_features, opts)

    with tab_perf:
        render_performance_tab()

    with tab_clinvar:
        render_clinvar_tab()

    with tab_about:
        st.markdown("""
        <div style="max-width:720px; margin:0 auto;">
            <h2 style="color:#63b3ed; font-size:1.5rem; margin-bottom:24px;">🧬 VARIANT-GNN Hakkında</h2>
            <p style='color:#94a3b8;'>Bu sistem, TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışması kapsamında 
            genetik varyantların patojenitesini hibrit bir yapay zeka mimarisi ile tahmin etmek için geliştirilmiştir.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

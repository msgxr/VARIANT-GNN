"""
app.py — VARIANT-GNN Streamlit Dashboard entry point.
TEKNOFEST 2026 | Saglikta Yapay Zeka

All UI logic lives in src/ui/. This file only:
  1. Configures the Streamlit page
  2. Injects CSS
  3. Loads the pipeline
  4. Routes each tab to its src/ui/ module

Run:  streamlit run app.py
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import streamlit as st

from src.config import get_settings
from src.utils.logging_cfg import setup_logging
from src.ui.styles import inject_styles
from src.ui.header import render_hero
from src.ui.utils import load_pipeline
from src.ui.sidebar import render_sidebar
from src.ui.analysis import render_analysis_tab
from src.ui.explainability import render_xai
from src.ui.performance import render_performance_tab
from src.ui.clinvar import render_clinvar_tab
from src.ui.about import render_about_tab

setup_logging(level=logging.WARNING)

st.set_page_config(
    page_title="VARIANT-GNN | Genetik Varyant Analizi",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    cfg = get_settings()
    inject_styles()
    render_hero()
    pipeline = load_pipeline()
    _df_uploaded, opts = render_sidebar()

    tab_analyze, tab_xai, tab_perf, tab_clinvar, tab_about = st.tabs([
        "🔬 Varyant Analizi",
        "🧠 Aciklanabilir YZ",
        "📊 Model Performansi",
        "🔍 ClinVar Aramasi",
        "ℹ️ Proje Hakkinda",
    ])

    with tab_analyze:
        render_analysis_tab(pipeline, cfg)

    with tab_xai:
        import pandas as pd
        if pipeline is None:
            st.warning("Model yuklenemedi.")
        elif "df_raw" not in st.session_state:
            st.info("Once 'Varyant Analizi' sekmesinde CSV yukleyin ve analizi baslatın.")
        else:
            df_raw = st.session_state["df_raw"]
            feat_cols = [c for c in df_raw.columns
                         if c not in ("Variant_ID", "Label", "Panel")]
            xai_opts = {
                "show_shap":     opts.get("show_shap", True),
                "show_waterfall":opts.get("show_waterfall", True),
                "show_lime":     opts.get("show_lime", False),
                "variant_index": opts.get("variant_index", 0),
                "threshold":     opts.get("threshold", 0.241),
            }
            render_xai(pipeline, df_raw[feat_cols] if feat_cols else pd.DataFrame(), xai_opts)

    with tab_perf:
        render_performance_tab()

    with tab_clinvar:
        render_clinvar_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

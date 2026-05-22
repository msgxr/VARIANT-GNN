"""
app.py — VARIANT-GNN Streamlit Dashboard
TEKNOFEST 2026 | Saglikta Yapay Zeka

All UI logic lives in src/ui/. This file is the thin router:
  - Page config + CSS injection
  - Pipeline loading
  - Hero + sidebar
  - Tab routing to src/ui/ modules

Run: streamlit run app.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.utils.logging_cfg import setup_logging
from src.ui.styles import inject_styles
from src.ui.header import render_hero
from src.ui.utils import load_pipeline, section_header
from src.ui.analytics import (
    render_summary_cards,
    render_risk_histogram,
    render_risk_map,
    render_model_comparison,
    render_results_table,
)
from src.ui.explainability import render_xai
from src.ui.performance import render_performance_tab
from src.ui.clinvar import render_clinvar_tab
from src.ui.reporting import generate_pdf_report
from src.ui.about import render_about_tab
from src.ui.sidebar import render_sidebar

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
    threshold = opts.get("threshold", float(cfg.thresholds.classification))

    tab_analyze, tab_xai, tab_perf, tab_clinvar, tab_about = st.tabs([
        "🔬 Varyant Analizi",
        "🧠 Açıklanabilir YZ",
        "📊 Model Performansi",
        "🔍 ClinVar Aramasi",
        "ℹ️ Proje Hakkinda",
    ])

    # ── Tab: Varyant Analizi ─────────────────────────────────────────────────
    with tab_analyze:
        section_header("📂", "Veri Yukleme")
        uploaded = st.file_uploader(
            "Varyant CSV dosyasi yukleyin",
            type=["csv"],
            help="Sayisal ozellik sutunlari iceren CSV. Label opsiyonel. "
                 "Ornek format: data/samples/jury_blind_sample.csv",
        )
        if uploaded is None:
            st.info("CSV dosyanizi yukleyin. Ornek: data/samples/jury_blind_sample.csv")
            return

        df_raw = pd.read_csv(uploaded)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Onizleme** — {len(df_raw):,} satir · {df_raw.shape[1]} sutun")
            st.dataframe(df_raw.head(5), use_container_width=True)
        with col2:
            st.metric("Varyant Sayisi", f"{len(df_raw):,}")
            st.metric("Eksik Veri", f"{df_raw.isnull().mean().mean()*100:.1f}%")

        if pipeline is None:
            st.warning("Model yuklenemedi. `python main.py --mode train` calistirin.")
            return

        if st.button("🚀 ANALIZI BASLAT", type="primary", use_container_width=True):
            with st.spinner("XGBoost + LightGBM + GATv2GNN + DNN modelleri calisiyor..."):
                try:
                    df_result = pipeline.predict_from_dataframe(df_raw)
                except (ValueError, RuntimeError, KeyError) as exc:
                    st.error(f"Inferans hatasi: {exc}")
                    st.stop()
            st.success("Analiz tamamlandi!")
            st.session_state["df_result"] = df_result
            st.session_state["df_raw"] = df_raw

        if "df_result" in st.session_state:
            df_result = st.session_state["df_result"]
            render_summary_cards(df_result)
            render_results_table(df_result)

            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            with col_dl1:
                st.download_button(
                    "⬇️ CSV Indir",
                    data=df_result.to_csv(index=False).encode(),
                    file_name="variant_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_dl2:
                with st.spinner("PDF hazirlaniyor..."):
                    pdf_bytes = generate_pdf_report(df_result, cfg)
                st.download_button(
                    "📄 PDF Rapor",
                    data=pdf_bytes,
                    file_name="variant_analiz_raporu.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            render_risk_histogram(df_result)
            render_risk_map(df_result)
            render_model_comparison(df_result)

    # ── Tab: Açıklanabilir YZ ────────────────────────────────────────────────
    with tab_xai:
        if pipeline is None:
            st.warning("Model yuklenemedi.")
            return
        if "df_result" not in st.session_state:
            st.info("Once 'Varyant Analizi' sekmesinde veri yukleme ve analizi calistirin.")
            return
        df_raw = st.session_state.get("df_raw", pd.DataFrame())
        feat_cols = [c for c in df_raw.columns
                     if c not in ("Variant_ID", "Label", "Panel")]
        if feat_cols:
            render_xai(pipeline, df_raw[feat_cols], opts)
        else:
            st.info("Ozellik sutunlari bulunamadi.")

    # ── Tab: Model Performansi ───────────────────────────────────────────────
    with tab_perf:
        render_performance_tab()

    # ── Tab: ClinVar Aramasi ─────────────────────────────────────────────────
    with tab_clinvar:
        render_clinvar_tab()

    # ── Tab: Proje Hakkinda ──────────────────────────────────────────────────
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

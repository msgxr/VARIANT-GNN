"""src/ui/analysis.py — Variant Analysis tab for VARIANT-GNN Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.analytics import (
    render_model_comparison,
    render_results_table,
    render_risk_histogram,
    render_risk_map,
    render_summary_cards,
)
from src.ui.reporting import generate_pdf_report
from src.ui.utils import section_header


def render_analysis_tab(pipeline: Any, cfg: Any) -> None:
    """Render the Variant Analysis tab: upload -> predict -> results."""
    section_header("📂", "Veri Yukleme")

    uploaded = st.file_uploader(
        "Varyant CSV dosyasi yukleyin",
        type=["csv"],
        help=(
            "Sayisal ozellik sutunlari iceren CSV. Label opsiyonel. Ornek format: data/samples/jury_blind_sample.csv"
        ),
    )
    if uploaded is None:
        st.markdown(
            """
        <div style="background:rgba(37,99,235,0.05);border:1px solid #e2e8f0;
                    border-radius:10px;padding:16px 20px;margin-top:8px;box-shadow:0 2px 8px rgba(15,23,42,0.06);">
            <div style="font-size:0.85rem;font-weight:700;color:#1d4ed8;margin-bottom:8px;">
                📋 Beklenen CSV Formatı
            </div>
            <div style="font-size:0.8rem;color:#475569;line-height:1.8;">
                Model şu özellik sütunlarını bekler:<br>
                <code style="color:#2563eb;">SIFT_score, PolyPhen2_HDIV_score, CADD_phred, REVEL_score,
                GERP_RS, PhyloP100way_vertebrate, gnomAD_exomes_AF</code> vb.<br>
                <strong style="color:#0f172a;">Örnek dosya:</strong>
                <code>data/samples/jury_blind_sample.csv</code> (351 özellik, 5 satır)
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    df_raw = pd.read_csv(uploaded)
    col_prev, col_stat = st.columns([3, 1])
    with col_prev:
        st.markdown(f"**Onizleme** — {len(df_raw):,} satir · {df_raw.shape[1]} sutun")
        st.dataframe(df_raw.head(5), width="stretch")
    with col_stat:
        st.metric("Varyant Sayisi", f"{len(df_raw):,}")
        st.metric("Eksik Veri", f"{df_raw.isnull().mean().mean() * 100:.1f}%")
        st.metric("Sutun Sayisi", str(df_raw.shape[1]))

    if pipeline is None:
        st.warning("Model yuklenemedi. `python main.py --mode train --config configs/pdr.yaml` calistirin.")
        return

    if st.button("🚀 ANALIZI BASLAT", type="primary", width="stretch"):
        with st.spinner("XGBoost + LightGBM + GATv2GNN + DNN modelleri calisiyor..."):
            try:
                # predict_from_csv runs the full preprocessing pipeline
                # (imputer → RobustScaler → missing-indicator → cosine k-NN graph; canonical pipeline,
                #  SelectKBest/autoencoder removed per §VIII)
                # predict_from_dataframe expects already-processed features — wrong for raw data
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
                    df_raw.to_csv(tmp, index=False)
                    tmp_path = tmp.name
                try:
                    df_result = pipeline.predict_from_csv(tmp_path)
                finally:
                    os.unlink(tmp_path)
            except Exception as exc:
                st.error(f"Inferans hatasi: {exc}")
                st.stop()
        st.success("Analiz tamamlandi!")
        st.session_state["df_result"] = df_result
        st.session_state["df_raw"] = df_raw

    if "df_result" not in st.session_state:
        return

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
            width="stretch",
        )
    with col_dl2:
        with st.spinner("PDF hazirlaniyor..."):
            pdf_bytes = generate_pdf_report(df_result, cfg)
        st.download_button(
            "📄 PDF Rapor Indir",
            data=pdf_bytes,
            file_name="variant_analiz_raporu.pdf",
            mime="application/pdf",
            width="stretch",
        )

    render_risk_histogram(df_result)
    render_risk_map(df_result)
    render_model_comparison(df_result)

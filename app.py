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
from src.api.pipeline import InferencePipeline
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
    render_model_comparison,
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


@st.cache_resource(show_spinner="🧠 Modeller yükleniyor...", ttl=None)
def _get_pipeline() -> Optional[InferencePipeline]:
    try:
        p = InferencePipeline()
        p.load()
        return p
    except Exception as exc:
        st.warning(f"⚠️ Model artifact bulunamadı — demo modunda çalışılıyor. ({exc})")
        return None


def main():
    cfg = get_settings()

    # Pipeline ve sidebar main() içinde yükleniyor (module-level erken yükleme engellendi)
    pipeline: Optional[InferencePipeline] = _get_pipeline()
    sidebar_df, opts = render_sidebar()

    # Apply Theme
    inject_styles(theme=opts.get("theme", "dark"))

    render_hero()

    # ── Tabs ──────────────────────────────────
    tabs = ["🔬 Varyant Analizi", "🧠 Açıklanabilir YZ"]
    if opts.get("show_3d", True):
        tabs.append("🧬 3D Protein Viz")
    tabs.extend(["📊 Model Performansı", "🔍 ClinVar Araması", "ℹ️ Proje Hakkında"])
    
    selected_tabs = st.tabs(tabs)
    
    # Map tabs correctly
    tab_analyze = selected_tabs[0]
    tab_xai = selected_tabs[1]
    curr_idx = 2
    tab_3d = selected_tabs[curr_idx] if opts.get("show_3d", True) else None
    if tab_3d: curr_idx += 1
    tab_perf = selected_tabs[curr_idx]; curr_idx += 1
    tab_clinvar = selected_tabs[curr_idx]; curr_idx += 1
    tab_about = selected_tabs[curr_idx]

    with tab_analyze:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(66,153,225,0.04));
                    border:1px solid rgba(99,179,237,0.25); border-radius:12px;
                    padding:20px 24px; margin-bottom:22px;">
            <div style="font-size:1rem; font-weight:700; color:#63b3ed; margin-bottom:10px;">🤖 Bu Sekme Ne Yapıyor?</div>
            <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.75;">
                Sol menüden <strong style="color:#90cdf4;">CSV, VCF veya FHIR (JSON)</strong> formatında genetik verilerinizi yükleyebilirsiniz. 
                Sistem, varyantları GNN ve Ensemble modelleriyle analiz ederek patojenite skoru üretir.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use sidebar_df if available
        df_to_analyze = sidebar_df if sidebar_df is not None else (st.session_state.get("df_raw") if "df_raw" in st.session_state else None)

        if df_to_analyze is not None:
            st.markdown(f"**📋 Yüklenen Veri Seti Önizleme** — {len(df_to_analyze):,} satır")
            st.dataframe(df_to_analyze.head(5), use_container_width=True)

            if pipeline and st.button("🚀 ANALİZİ BAŞLAT / GÜNCELLE", type="primary", use_container_width=True):
                with st.spinner("⚡ Modeller çalışıyor..."):
                    try:
                        df_result = pipeline.predict_from_dataframe(df_to_analyze)
                        st.session_state["df_result"] = df_result
                        st.session_state["df_raw"]    = df_to_analyze
                        st.success("✅ Analiz tamamlandı!")
                    except Exception as exc:
                        st.error(f"⚠️ İnferans hatası: {exc}")
        else:
            st.info("👈 Lütfen sol menüden bir veri dosyası seçerek başlayın.")

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

    if tab_3d:
        with tab_3d:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">🧬</div>
                <h3>Üç Boyutlu Protein Yapı Analizi</h3>
            </div>
            """, unsafe_allow_html=True)
            
            from src.ui.protein_viz import render_protein_3d, get_pdb_for_gene
            
            # Eğer analiz edilmiş veri varsa ilk genin PDB'sini çek
            gene_sym = "BRCA1"
            if "df_raw" in st.session_state:
                df_ = st.session_state["df_raw"]
                if "Gene_Symbol" in df_.columns:
                    gene_sym = df_["Gene_Symbol"].iloc[0]
            
            pdb_id = get_pdb_for_gene(gene_sym)
            render_protein_3d(pdb_id=pdb_id)

    with tab_perf:
        render_performance_tab()

    with tab_clinvar:
        render_clinvar_tab()

    with tab_about:
        st.markdown(f"""
        <div style="max-width:720px; margin:0 auto;">
            <h2 style="color:#63b3ed; font-size:1.5rem; margin-bottom:24px;">🧬 VARIANT-GNN Hakkında</h2>
            <p style='color:#94a3b8;'>Bu sistem, <strong>TEKNOFEST 2026 Sağlıkta Yapay Zeka</strong> yarışması kapsamında 
            genetik varyantların patojenitesini hibrit bir yapay zeka mimarisi ile tahmin etmek için geliştirilmiştir.</p>
            <div style="margin-top:20px; padding:15px; background:rgba(99,179,237,0.1); border-radius:10px;">
                <h4 style="color:#63b3ed; font-size:1rem;">Akademik Atıf</h4>
                <code style="font-size:0.8rem; color:#cbd5e0;">Muhammed et al. (2026). VARIANT-GNN: Hybrid Graph Neural Networks...</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

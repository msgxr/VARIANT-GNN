"""src/ui/about.py — About / Project Info tab for VARIANT-GNN Streamlit app."""
from __future__ import annotations

import pandas as pd
import streamlit as st


REPO_URL = "https://github.com/msgxr/VARIANT-GNN"

_PERF_DATA = {
    "Model": ["VARIANT-GNN (Hibrit)", "GATv2GNN (solo)", "LightGBM (solo)",
              "XGBoost (solo)", "DNN (solo)", "Baseline (LogReg)"],
    "CV F1": ["0.8668", "0.8472", "0.8764", "0.8382", "0.8208", "~0.74"],
    "Test F1": ["0.8980 ✅", "—", "—", "—", "—", "—"],
    "MCC": ["0.5356", "—", "—", "—", "—", "—"],
    "PR-AUC": ["0.9294", "—", "—", "—", "—", "—"],
}

_PANEL_DATA = {
    "Panel": ["MASTER (General)", "KANSER (Hereditary_Cancer)", "PAH", "CFTR"],
    "F1": ["0.8872", "0.8960", "0.9556", "0.9524"],
    "MCC": ["0.5070", "0.6491", "0.5562", "0.6742"],
    "PR-AUC": ["0.9183", "0.9524", "0.9760", "0.9223"],
    "Recall": ["0.9679", "0.9912", "0.9790", "1.0000"],
}


def render_about_tab() -> None:
    """Render the About / Project Info tab."""
    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2044 0%,#1a3a6e 40%,#0d2855 100%);
                border:1px solid rgba(99,179,237,0.3); border-radius:16px;
                padding:36px 40px; margin-bottom:28px;">
        <p style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0 0 8px 0;">
            🧬 <span style="color:#63b3ed;">VARIANT-GNN</span>
        </p>
        <p style="font-size:1.05rem; color:#94a3b8; margin:0; line-height:1.7;">
            Missense Genetik Varyant Patojenite Tahmini için<br>
            <strong style="color:#93c5fd;">Hibrit GATv2 Grafik Sinir Ağı Ensemble Sistemi</strong>
        </p>
        <div style="margin-top:14px;">
            <span style="background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.4);
                  color:#63b3ed; font-size:0.75rem; font-weight:600; padding:4px 12px;
                  border-radius:20px; margin-right:8px;">🏆 TEKNOFEST 2026</span>
            <span style="background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.4);
                  color:#63b3ed; font-size:0.75rem; font-weight:600; padding:4px 12px;
                  border-radius:20px; margin-right:8px;">🔬 PSR: 93/100</span>
            <span style="background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.4);
                  color:#63b3ed; font-size:0.75rem; font-weight:600; padding:4px 12px;
                  border-radius:20px;">⚡ Test F1 = 0.8980</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture diagram ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📐</div>
        <h3>Sistem Mimarisi (§VIII)</h3>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
  Anonim Varyant CSV (NDA)
       │
       ▼
  LeakageFirewall (§3.2 — koordinat + etiket bloklama)
       │
  ColumnAligner → ACMGProxyFeatures → Median Imputer
       → RobustScaler → VarianceThreshold + SelectKBest(k=35)
       → AutoEncoder(dim→16) → Cosine k-NN Graf(k=10)
       │
       ├─ XGBoost       (w=0.30)
       ├─ LightGBM      (w=0.30)
       ├─ VariantGATv2GNN (w=0.25)
       └─ DNN           (w=0.15)
                │
       Logistic Regression Stacking Meta-Learner
                │
       Isotonic Calibration  ·  MC Dropout (10 pass)
                │
       Panel-Specific Threshold  ·  OOD Detector
                │
       prediction_label  ·  calibrated_risk  ·  uncertainty
    """, language="")

    # ── 4 Model cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    models = [
        ("🕸️ VariantGATv2GNN", "3× GATv2Conv blok, 4 kafa, 128 hidden, cosine k-NN graf. Dinamik attention (Brody 2022).", "0.25"),
        ("🌲 XGBoost", "Gradient boosting, max_depth=6, 200 ağaç, L1+L2 reg, early stopping.", "0.30"),
        ("💡 LightGBM", "Leaf-wise büyüme, 63 yaprak, fast inference, tabular uzmanı.", "0.30"),
        ("🤖 DNN", "3-katman MLP [input→128→64→2], BatchNorm + Dropout(0.4), SWA.", "0.15"),
    ]
    for col, (title, desc, w) in zip([c1, c2, c3, c4], models):
        with col:
            st.markdown(f"""
            <div class="model-card">
                <h4>{title}</h4>
                <p>{desc}<br>
                <span style="color:#63b3ed; font-weight:600;">Ağırlık: {w}</span></p>
            </div>
            """, unsafe_allow_html=True)

    # ── Performance table ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h3>Ablasyon — Model Karşılaştırması (CV F1)</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(_PERF_DATA), use_container_width=True, hide_index=True)

    # ── Panel performance ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧬</div>
        <h3>Panel Bazlı Performans (Test Seti)</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(_PANEL_DATA), use_container_width=True, hide_index=True)

    # ── Tech stack + References ──────────────────────────────────────────────
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div class="model-card">
            <h4>⚡ Teknoloji Stack</h4>
            <p>
                🐍 Python 3.12 · 🔥 PyTorch 2.8 · 📊 PyG 2.6<br>
                🌲 XGBoost 2.1 · 💡 LightGBM 4.6 · 📈 Scikit-learn 1.6<br>
                🎯 SHAP 0.49 · 🟢 LIME · 🖥️ Streamlit 1.50<br>
                📝 Pydantic v2 · 🐳 Docker · 🔄 GitHub Actions CI
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div class="model-card">
            <h4>📚 Temel Referanslar</h4>
            <p>
                1. <strong>GATv2</strong> — Brody et al. (ICLR 2022)<br>
                2. <strong>XGBoost</strong> — Chen &amp; Guestrin (KDD 2016)<br>
                3. <strong>SHAP</strong> — Lundberg &amp; Lee (NeurIPS 2017)<br>
                4. <strong>ACMG/AMP</strong> — Richards et al. (Genet. Med. 2015)<br>
                5. <strong>SWA</strong> — Izmailov et al. (UAI 2018)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer + clinical disclaimer ─────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; margin-top:24px; margin-bottom:12px;">
        <a href="{REPO_URL}" target="_blank"
           style="display:inline-block; background:linear-gradient(135deg,#2b6cb0,#3182ce);
                  color:white; text-decoration:none; font-weight:600; font-size:0.95rem;
                  padding:12px 32px; border-radius:10px; letter-spacing:0.3px;">
            ⭐ GitHub: msgxr/VARIANT-GNN
        </a>
    </div>
    <div style="padding:16px; background:rgba(229,62,62,0.05); border-radius:10px;
                border:1px solid rgba(229,62,62,0.2); font-size:0.82rem; color:#fc8181;
                text-align:center; margin-top:16px;">
        ⚠️ Bu sistem araştırma amacıyla geliştirilmiştir. Klinik teşhis, tedavi veya
        tıbbi karar desteği için kullanılamaz (TEKNOFEST §10).
        <br><strong>TEKNOFEST 2026</strong> | XYRA3 Takımı | PDR Teslimi: 29 Haziran 2026
    </div>
    """, unsafe_allow_html=True)

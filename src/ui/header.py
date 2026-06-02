"""src/ui/header.py — Hero banner for VARIANT-GNN Streamlit app."""

import streamlit as st


def render_header() -> None:
    """Uygulama hero banner bölümünü oluşturur (eski uyumluluk için)."""
    render_hero()


def render_hero() -> None:
    """TEKNOFEST 2026 hero banner with project identity."""
    st.markdown(
        """
    <div class="hero-banner">
        <p class="hero-title">🧬 <span>VARIANT-GNN</span></p>
        <p class="hero-subtitle">
            Graph Neural Network ve Açıklanabilir Yapay Zeka ile<br>
            <strong style="color:#93c5fd;">Genetik Varyantların Patojenite Tahmini</strong>
        </p>
        <span class="hero-badge">🏆 TEKNOFEST 2026</span>
        <span class="hero-badge">🔬 Sağlıkta Yapay Zeka</span>
        <span class="hero-badge">⚡ GNN + XGBoost + LightGBM + DNN</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

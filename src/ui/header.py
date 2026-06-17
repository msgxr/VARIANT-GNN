"""src/ui/header.py — Hero banner for VARIANT-GNN Streamlit app."""

import streamlit as st


def render_header() -> None:
    """Uygulama hero banner bölümünü oluşturur (eski uyumluluk için)."""
    render_hero()


def render_hero() -> None:
    """TEKNOFEST 2026 hero banner with project identity (Design System — açık tema)."""
    st.markdown(
        """
    <div class="hero-banner">
        <div style="display:flex; align-items:center; gap:18px; margin-bottom:6px;">
            <svg viewBox="0 0 64 64" width="46" height="46" role="img" aria-label="VARIANT-GNN helix glyph"
                 style="flex-shrink:0; border-radius:13px; box-shadow:0 4px 14px rgba(230,57,70,.35);">
                <defs>
                    <linearGradient id="hero-helix" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#e63946"></stop>
                        <stop offset="100%" stop-color="#c1121f"></stop>
                    </linearGradient>
                </defs>
                <rect x="0" y="0" width="64" height="64" rx="14" fill="url(#hero-helix)"></rect>
                <g fill="none" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round">
                    <path d="M18 12 C 30 22, 30 22, 46 22 M18 22 C 30 32, 30 32, 46 32 M18 32 C 30 42, 30 42, 46 42 M18 42 C 30 52, 30 52, 46 52" opacity="0.9"></path>
                    <path d="M46 12 C 30 22, 30 22, 18 22 M46 22 C 30 32, 30 32, 18 32 M46 32 C 30 42, 30 42, 18 42 M46 42 C 30 52, 30 52, 18 52" opacity="0.55"></path>
                    <line x1="22" y1="17" x2="42" y2="17" stroke-width="1.6" opacity="0.7"></line>
                    <line x1="22" y1="27" x2="42" y2="27" stroke-width="1.6" opacity="0.7"></line>
                    <line x1="22" y1="37" x2="42" y2="37" stroke-width="1.6" opacity="0.7"></line>
                    <line x1="22" y1="47" x2="42" y2="47" stroke-width="1.6" opacity="0.7"></line>
                </g>
            </svg>
            <p class="hero-title" style="margin:0;">🧬 <span>VARIANT-GNN</span></p>
        </div>
        <p class="hero-subtitle">
            Graph Neural Network ve Açıklanabilir Yapay Zeka ile<br>
            <strong>Genetik Varyantların Patojenite Tahmini</strong>
        </p>
        <span class="hero-badge">🏆 TEKNOFEST 2026</span>
        <span class="hero-badge">🔬 Sağlıkta Yapay Zeka</span>
        <span class="hero-badge">⚡ GNN + XGBoost + LightGBM + DNN</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

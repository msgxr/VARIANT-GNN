import streamlit as st

def render_header() -> None:
    """Uygulama başlığını ve hero banner bölümünü oluşturur."""
    st.markdown("""
    <div class="hero-container">
        <h1>🧬 VARIANT-GNN</h1>
        <p>Next-Generation Pathogenicity Prediction & Clinical Interpretation</p>
        <div style="display:flex; gap:10px; margin-top:15px; justify-content:center;">
            <span class="badge">SOTA GNN</span>
            <span class="badge">Explainable AI</span>
            <span class="badge">Clinical Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

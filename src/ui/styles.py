import streamlit as st


def inject_styles(theme: str = "dark") -> None:
    """Tüm Premium CSS kodlarını Streamlit uygulamasına enjekte eder."""

    # Tema renk tanımlamaları
    if theme == "light":
        primary_bg = "#f8fafc"
        secondary_bg = "#ffffff"
        sidebar_bg = "linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%)"
        text_main = "#1e293b"
        text_sub = "#475569"
        border_color = "rgba(148,163,184,0.2)"
        card_bg = "linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)"
    else:
        primary_bg = "#0a0e1a"
        secondary_bg = "#0f1629"
        sidebar_bg = "linear-gradient(180deg, #0d1117 0%, #161b28 100%)"
        text_main = "#e2e8f0"
        text_sub = "#94a3b8"
        border_color = "rgba(99,179,237,0.2)"
        card_bg = "linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%)"

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
        --primary-bg: {primary_bg};
        --secondary-bg: {secondary_bg};
        --text-main: {text_main};
        --text-sub: {text_sub};
        --border-color: {border_color};
        --card-bg: {card_bg};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: var(--primary-bg);
        color: var(--text-main);
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid var(--border-color);
    }}
    section[data-testid="stSidebar"] .stMarkdown {{ color: var(--text-sub); }}

    /* Hero banner */
    .hero-banner {{
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6e 40%, #0d2855 100%);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
    }}
    .hero-title {{ color: #e2e8f0; }}
    
    /* Metric cards */
    .metric-card {{
        flex: 1;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .metric-card .value {{ color: var(--text-main); }}
    .metric-card .label {{ color: var(--text-sub); }}

    /* Section headers */
    .section-header {{
        border-bottom: 1px solid var(--border-color);
    }}
    .section-header h3 {{ color: var(--text-main); }}

    /* Data table */
    .stDataFrame {{
        background: var(--secondary-bg) !important;
        border: 1px solid var(--border-color) !important;
    }}

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {{
        background: var(--secondary-bg);
        border-color: var(--border-color);
        color: var(--text-main);
    }}
</style>
""",
        unsafe_allow_html=True,
    )

import streamlit as st

def inject_styles() -> None:
    """Tüm Premium CSS kodlarını Streamlit uygulamasına enjekte eder."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b28 100%);
        border-right: 1px solid rgba(99,179,237,0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6e 40%, #0d2855 100%);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: #63b3ed; }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.4);
        color: #63b3ed;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 8px;
        margin-top: 12px;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #63b3ed, #4299e1);
    }
    .metric-card.pathogenic::after { background: linear-gradient(90deg, #fc8181, #e53e3e); }
    .metric-card.benign::after    { background: linear-gradient(90deg, #68d391, #38a169); }
    .metric-card.warning::after   { background: linear-gradient(90deg, #f6ad55, #dd6b20); }
    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-card .label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .sublabel {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99,179,237,0.15);
    }
    .section-header h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }
    .section-icon {
        width: 36px;
        height: 36px;
        background: rgba(99,179,237,0.15);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }

    /* Prediction badges */
    .badge-pathogenic {
        display: inline-block;
        background: rgba(229,62,62,0.15);
        border: 1px solid rgba(229,62,62,0.5);
        color: #fc8181;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }
    .badge-benign {
        display: inline-block;
        background: rgba(56,161,105,0.15);
        border: 1px solid rgba(56,161,105,0.5);
        color: #68d391;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }

    /* Risk gauge */
    .risk-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
        margin-top: 6px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #68d391 0%, #f6ad55 50%, #fc8181 100%);
        transition: width 0.8s ease;
    }

    /* Upload zone */
    .upload-zone {
        background: rgba(99,179,237,0.04);
        border: 2px dashed rgba(99,179,237,0.3);
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Model info tab */
    .model-card {
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .model-card h4 {
        color: #63b3ed;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .model-card p { color: #94a3b8; font-size: 0.85rem; margin: 0; }

    /* Plot styling */
    .chart-container {
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Data table */
    .stDataFrame {
        background: #1a2744 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce, #4299e1);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(99,179,237,0.3);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #276749, #38a169);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(99,179,237,0.05);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(99,179,237,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #718096;
        font-weight: 500;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,179,237,0.15) !important;
        color: #63b3ed !important;
    }

    /* Alerts */
    .stAlert { border-radius: 10px; }

    /* Spinner */
    .stSpinner > div { border-top-color: #63b3ed !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { padding: 0; }

    /* Success/info boxes */
    div[data-testid="stNotification"] { border-radius: 10px; }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: #1a2744;
        border-color: rgba(99,179,237,0.3);
    }
</style>
""", unsafe_allow_html=True)

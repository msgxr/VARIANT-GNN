# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT-GNN Design System — AÇIK TEMA (kanonik)
# Kaynak: "VARIANT-GNN Design System.zip" / colors_and_type.css
# Beyaz yüzeyler · koyu metin · crimson marka · sadece sidebar koyu lacivert.
# Tüm değerler sabit → f-string yerine düz string (CSS süslü parantez kaçışı yok).
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* Marka */
    --vg-brand: #e63946; --vg-brand-deep: #c1121f; --vg-brand-soft: #fff1f2; --vg-brand-edge: #fecdd3;
    /* Risk 4-bant (0-100 ayrık) + belirsizlik */
    --risk-low: #16a34a; --risk-moderate: #ca8a04; --risk-high: #d97706;
    --risk-critical: #dc2626; --risk-uncertain: #7c3aed;
    /* Semantik */
    --info: #2563eb; --info-deep: #1d4ed8; --warning: #d97706; --success: #16a34a; --danger: #dc2626;
    /* Yüzeyler */
    --surface-app: #f0f2f8; --surface-paper: #ffffff;
    --surface-sidebar: linear-gradient(180deg,#080c14 0%,#0d1424 60%,#0a1020 100%);
    --surface-stripe: linear-gradient(90deg,#e63946 0%,#ff6b6b 30%,#2563eb 70%,#3b82f6 100%);
    /* Metin */
    --fg-1: #0f172a; --fg-2: #475569; --fg-3: #64748b; --fg-4: #94a3b8;
    --fg-on-dark: #f1f5f9; --fg-on-dark-sub: #8892a4;
    /* Kenarlık */
    --border: #e2e8f0; --border-soft: rgba(226,232,240,.8); --border-strong: #cbd5e1;
    /* Tipografi */
    --font-sans: 'Outfit',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    --font-display: 'Fraunces','Outfit',Georgia,serif;
    --font-mono: 'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
    /* Radius */
    --r-md: 11px; --r-lg: 14px; --r-xl: 18px; --r-2xl: 22px;
    /* Gölge (slate-tinted) */
    --shadow-2: 0 2px 8px rgba(15,23,42,.06);
    --shadow-3: 0 4px 24px rgba(15,23,42,.08), 0 1px 4px rgba(15,23,42,.04);
    --shadow-glow-brand: 0 4px 14px rgba(230,57,70,.35);
    /* Geriye dönük takma adlar (eski sınıf referansları için) */
    --primary-bg: #f0f2f8; --secondary-bg: #ffffff; --text-main: #0f172a;
    --text-sub: #475569; --border-color: #e2e8f0; --card-bg: #ffffff;
}

html, body, [class*="css"] { font-family: var(--font-sans); }

.stApp { background: var(--surface-app); color: var(--fg-1); }
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li { color: var(--fg-1); }

/* ── Sidebar — koyu lacivert (tek koyu öğe) ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface-sidebar);
    border-right: 1px solid rgba(15,23,42,.5);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
    color: var(--fg-on-dark) !important;
}
/* Dosya yükleyici bırakma alanı açık kalsın (koyu metin okunur) */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: var(--surface-paper); border-radius: var(--r-md);
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: var(--fg-2) !important; }

/* ── Hero banner — beyaz yüzey + 3px üst gradient stripe ──────────────────── */
.hero-banner {
    position: relative; overflow: hidden;
    background: var(--surface-paper);
    border: 1px solid var(--border); border-radius: var(--r-2xl);
    padding: 34px 40px; margin-bottom: 28px;
    box-shadow: var(--shadow-3);
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--surface-stripe);
}
.hero-title { font-family: var(--font-display); font-weight: 700; font-size: 2rem; color: var(--fg-1); margin: 0 0 8px 0; }
.hero-title span { color: var(--vg-brand); }
.hero-subtitle { color: var(--fg-2); font-size: 1.02rem; line-height: 1.7; margin: 0; }
.hero-subtitle strong { color: var(--vg-brand-deep); }
.hero-badge {
    display: inline-block; background: var(--vg-brand-soft); border: 1px solid var(--vg-brand-edge);
    color: var(--vg-brand-deep); font-size: 0.75rem; font-weight: 600;
    padding: 5px 13px; border-radius: 999px; margin-right: 8px; margin-top: 14px;
}

/* ── Section header + 44px icon-chip ──────────────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 14px;
    margin: 26px 0 14px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.section-header h3 { margin: 0; color: var(--fg-1); font-weight: 700; font-size: 1.15rem; }
.section-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 12px; font-size: 1.3rem; flex-shrink: 0;
    background: linear-gradient(135deg,#fff1f2 0%,#ffe4e6 100%);
    border: 1px solid var(--vg-brand-edge);
    box-shadow: 0 4px 10px rgba(230,57,70,.16);
}

/* ── KPI / metric kartları (sol kenar bordürü) ────────────────────────────── */
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 22px; }
.metric-card {
    flex: 1; min-width: 130px;
    background: var(--surface-paper); border: 1px solid var(--border);
    border-left: 3px solid var(--fg-4); border-radius: var(--r-lg);
    padding: 16px 20px; text-align: center; box-shadow: var(--shadow-2);
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-3); }
.metric-card.pathogenic { border-left-color: var(--vg-brand); }
.metric-card.warning { border-left-color: var(--risk-high); }
.metric-card.info { border-left-color: var(--info); }
.metric-card .value { font-family: var(--font-mono); font-size: 1.7rem; font-weight: 800; color: var(--fg-1); font-variant-numeric: tabular-nums; }
.metric-card .label { font-size: 0.74rem; font-weight: 600; color: var(--fg-3); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 4px; }
.metric-card .sublabel { font-size: 0.68rem; color: var(--fg-4); margin-top: 2px; }

/* ── Model / içerik kartı (about, clinvar) ────────────────────────────────── */
.model-card {
    background: var(--surface-paper); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 18px 20px;
    box-shadow: var(--shadow-2); height: 100%;
}
.model-card h4 { margin: 0 0 8px 0; color: var(--fg-1); font-weight: 700; font-size: 1rem; }
.model-card p { margin: 0; color: var(--fg-2); font-size: 0.85rem; line-height: 1.65; }
.model-card code { color: var(--vg-brand-deep); font-family: var(--font-mono); }

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: var(--r-md); overflow: hidden;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--fg-2); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--vg-brand); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--vg-brand); }

/* ── Butonlar ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: var(--vg-brand); border: 1px solid var(--vg-brand-deep);
    color: #ffffff; font-weight: 600;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    background: var(--vg-brand-deep); border-color: var(--vg-brand-deep);
}

/* ── Input / selectbox / code ─────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"], .stTextInput input, .stNumberInput input {
    background: var(--surface-paper); border-color: var(--border); color: var(--fg-1);
}
.stApp code { font-family: var(--font-mono); }
</style>
"""

# Eski koyu tema — yedek (fallback). Yalnızca theme="dark" çağrılırsa kullanılır.
_DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --primary-bg: #0a0e1a; --secondary-bg: #0f1629; --text-main: #e2e8f0;
        --text-sub: #94a3b8; --border-color: rgba(99,179,237,0.2);
        --card-bg: linear-gradient(135deg, #1a2744 0%, #1e2d4e 100%);
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--primary-bg); color: var(--text-main); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b28 100%);
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: var(--text-sub); }
    .hero-banner {
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6e 40%, #0d2855 100%);
        border: 1px solid rgba(99,179,237,0.3); border-radius: 16px;
        padding: 36px 40px; margin-bottom: 28px; position: relative;
    }
    .hero-title { color: #e2e8f0; }
    .metric-card {
        flex: 1; background: var(--card-bg); border: 1px solid var(--border-color);
        border-radius: 12px; padding: 20px 24px; text-align: center; transition: all 0.3s ease;
    }
    .metric-card .value { color: var(--text-main); }
    .metric-card .label { color: var(--text-sub); }
    .section-header { border-bottom: 1px solid var(--border-color); }
    .section-header h3 { color: var(--text-main); }
    .stDataFrame { background: var(--secondary-bg) !important; border: 1px solid var(--border-color) !important; }
    .stSelectbox [data-baseweb="select"] { background: var(--secondary-bg); border-color: var(--border-color); color: var(--text-main); }
</style>
"""


def inject_styles(theme: str = "light") -> None:
    """Tüm Premium CSS kodlarını Streamlit uygulamasına enjekte eder.

    theme="light" → VARIANT-GNN Design System (kanonik açık tema).
    theme="dark"  → eski koyu tema (yedek/geriye dönük uyumluluk).
    """
    st.markdown(_DARK_CSS if theme == "dark" else _LIGHT_CSS, unsafe_allow_html=True)

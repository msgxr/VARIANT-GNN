"""src/ui/about.py — About / Project Info tab for VARIANT-GNN Streamlit app.

Tüm performans sayıları RESULTS_CANONICAL.json'dan (tek doğruluk kaynağı) DİNAMİK
okunur — hardcode YOK. Bu, About sekmesinin Performance sekmesiyle çelişmesini ve
geri çekilmiş leakage-şişik sayıların demoda yeniden belirmesini önler
(§7.5 repro + §III-9 bütünlük).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_URL = "https://github.com/msgxr/VARIANT-GNN"
_CANON_PATH = Path(__file__).resolve().parents[2] / "RESULTS_CANONICAL.json"

# Yedek değerler (canonical okunamazsa) — DÜRÜST sayılar (geri-çekilmiş şişik değerler DEĞİL).
_FALLBACK = {
    "test_binary_f1": 0.8367,
    "test_mcc": 0.5112,
    "test_pr_auc": 0.9267,
    "test_roc_auc": 0.8538,
    "cv_binary_f1_production_oof_stacking": 0.8936,
    "official_competition_score_4panel_avg": 0.6202,
}
_PANEL_FALLBACK = {
    "General": {"binary_f1": 0.8185, "mcc": 0.4951, "precision": 0.9217, "recall": 0.7361},
    "Hereditary_Cancer": {"binary_f1": 0.906, "mcc": 0.7135, "precision": 0.9464, "recall": 0.8689},
    "PAH": {"binary_f1": 0.912, "mcc": 0.5053, "precision": 0.9048, "recall": 0.9194},
    "CFTR": {"binary_f1": 0.7143, "mcc": 0.0, "precision": 1.0, "recall": 0.5556},
}


def _load_canonical() -> tuple[dict, dict]:
    """RESULTS_CANONICAL.json'dan headline + panel metriklerini oku (fallback güvenli)."""
    try:
        canon = json.loads(_CANON_PATH.read_text(encoding="utf-8"))
        return canon.get("headline", {}), canon.get("panel_test_metrics", {})
    except Exception:
        return {}, {}


def _fmt(val: object, nd: int = 4) -> str:
    return f"{float(val):.{nd}f}" if isinstance(val, (int, float)) else "—"


def render_about_tab() -> None:
    """Render the About / Project Info tab."""
    h, panels = _load_canonical()
    g = lambda k: h.get(k, _FALLBACK.get(k))  # noqa: E731
    test_f1 = _fmt(g("test_binary_f1"))
    jury_f1 = _fmt(g("official_competition_score_4panel_avg"))

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
    <div class="hero-banner">
        <p class="hero-title">🧬 <span>VARIANT-GNN</span></p>
        <p class="hero-subtitle">
            Missense Genetik Varyant Patojenite Tahmini için<br>
            <strong>Hibrit GATv2 Grafik Sinir Ağı Ensemble Sistemi</strong>
        </p>
        <div>
            <span class="hero-badge">🏆 TEKNOFEST 2026</span>
            <span class="hero-badge">🔬 PSR: 93/100</span>
            <span class="hero-badge">⚡ Jüri beklentisi (4-panel %20-F1) = {jury_f1}</span>
        </div>
        <p style="font-size:0.78rem; color:#64748b; margin:12px 0 0 0;">
            İç hold-out ayrım gücü Test F1 = {test_f1} (jüri skoru DEĞİL; resmi metrik %20-patojenik
            test'te 4-panel F1 ortalamasıdır → {jury_f1}).
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Architecture diagram (canonical pipeline — SelectKBest/AE KALDIRILDI) ──
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">📐</div>
        <h3>Sistem Mimarisi (§VIII)</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.code(
        """
  Anonim Varyant CSV (NDA)
       │
       ▼
  LeakageFirewall (§3.2 — koordinat + etiket bloklama)
       │
  ColumnAligner → CategoricalBioFeatures(AA/CAT/EK) → Median Imputer
       → RobustScaler → Missing-Indicator(+343, 'missing≠0') → SMOTE(yalnız train)
       → Cosine k-NN Graf(k=10)
       │
       ├─ XGBoost          (w=0.30)
       ├─ LightGBM         (w=0.30)
       ├─ VariantGATv2GNN  (w=0.25)
       └─ DNN (Domain-Adversarial) (w=0.15)
                │
       OOF-Stacking Logistic Regression Meta-Learner (Wolpert)
                │
       Isotonic Calibration  ·  MC Dropout (10 pass)
                │
       Global θ=0.8415 (%20-prior F1-optimal)  ·  OOD Detector
                │
       prediction_label  ·  calibrated_risk  ·  uncertainty
    """,
        language="",
    )

    # ── 4 Model cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    models = [
        (
            "🕸️ VariantGATv2GNN",
            "3× GATv2Conv blok, 4 kafa, 128 hidden, cosine k-NN graf. Dinamik attention (Brody 2022).",
            "0.25",
        ),
        ("🌲 XGBoost", "Gradient boosting, max_depth=6, 200 ağaç, L1+L2 reg, early stopping.", "0.30"),
        ("💡 LightGBM", "Leaf-wise büyüme, 63 yaprak, fast inference, tabular uzmanı.", "0.30"),
        ("🤖 DNN", "3-katman MLP [input→128→64→2], BatchNorm + Dropout(0.4), SWA + DANN.", "0.15"),
    ]
    for col, (title, desc, w) in zip([c1, c2, c3, c4], models):
        with col:
            st.markdown(
                f"""
            <div class="model-card">
                <h4>{title}</h4>
                <p>{desc}<br>
                <span style="color:#2563eb; font-weight:600;">Ağırlık: {w}</span></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ── Headline metrics (canonical — dinamik) ───────────────────────────────
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h3>Shipped Model — Başlıca Metrikler (RESULTS_CANONICAL.json)</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
    perf_df = pd.DataFrame(
        {
            "Metrik": [
                "⭐ Jüri beklentisi: 4-panel %20-F1 ortalaması",
                "İç hold-out Test F1 (ayrım gücü, jüri skoru değil)",
                "CV F1 (OOF-stacking, nested group-aware)",
                "MCC (iç hold-out)",
                "PR-AUC",
                "ROC-AUC",
            ],
            "Değer": [
                jury_f1,
                test_f1,
                _fmt(g("cv_binary_f1_production_oof_stacking")),
                _fmt(g("test_mcc")),
                _fmt(g("test_pr_auc")),
                _fmt(g("test_roc_auc")),
            ],
        }
    )
    st.dataframe(perf_df, width="stretch", hide_index=True)

    # ── Panel performance (canonical — dinamik) ──────────────────────────────
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">🧬</div>
        <h3>Panel Bazlı Performans (Test Seti, global θ=0.8415)</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
    panel_order = [
        ("MASTER (General)", "General"),
        ("KANSER (Hereditary_Cancer)", "Hereditary_Cancer"),
        ("PAH", "PAH"),
        ("CFTR", "CFTR"),
    ]
    pget = lambda key, m: (panels.get(key) or _PANEL_FALLBACK.get(key, {})).get(m)  # noqa: E731
    panel_df = pd.DataFrame(
        {
            "Panel": [disp for disp, _ in panel_order],
            "F1": [_fmt(pget(k, "binary_f1")) for _, k in panel_order],
            "MCC": [_fmt(pget(k, "mcc")) for _, k in panel_order],
            "Precision": [_fmt(pget(k, "precision")) for _, k in panel_order],
            "Recall": [_fmt(pget(k, "recall")) for _, k in panel_order],
        }
    )
    st.dataframe(panel_df, width="stretch", hide_index=True)
    st.caption(
        "Not: CFTR iç hold-out'ta küçük örneklemli (n≈18) → MCC tanımsız/gürültülü; "
        "gerçek yarışmada kendi test seti olacaktır. Sayılar RESULTS_CANONICAL.json'dan dinamik okunur."
    )

    # ── Tech stack + References ──────────────────────────────────────────────
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
        <div class="model-card">
            <h4>⚡ Teknoloji Stack</h4>
            <p>
                🐍 Python 3.12 · 🔥 PyTorch 2.8 · 📊 PyG 2.6<br>
                🌲 XGBoost 2.1 · 💡 LightGBM 4.6 · 📈 Scikit-learn 1.6<br>
                🎯 SHAP 0.49 · 🟢 LIME · 🖥️ Streamlit 1.50<br>
                📝 Pydantic v2 · 🐳 Docker · 🔄 GitHub Actions CI
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )

    # ── Footer + clinical disclaimer ─────────────────────────────────────────
    st.markdown(
        f"""
    <div style="text-align:center; margin-top:24px; margin-bottom:12px;">
        <a href="{REPO_URL}" target="_blank"
           style="display:inline-block; background:linear-gradient(135deg,#e63946,#c1121f);
                  color:white; text-decoration:none; font-weight:600; font-size:0.95rem;
                  padding:12px 32px; border-radius:11px; letter-spacing:0.3px;
                  box-shadow:0 4px 14px rgba(230,57,70,.35);">
            ⭐ GitHub: msgxr/VARIANT-GNN
        </a>
    </div>
    <div style="padding:16px; background:rgba(220,38,38,0.05); border-radius:10px;
                border:1px solid rgba(220,38,38,0.2); border-left:3px solid #dc2626;
                font-size:0.82rem; color:#b91c1c; text-align:center; margin-top:16px;">
        ⚠️ Bu sistem araştırma amacıyla geliştirilmiştir. Klinik teşhis, tedavi veya
        tıbbi karar desteği için kullanılamaz (TEKNOFEST §10).
        <br><strong>TEKNOFEST 2026</strong> | XYRA3 Takımı | PDR Teslimi: 29 Haziran 2026
    </div>
    """,
        unsafe_allow_html=True,
    )

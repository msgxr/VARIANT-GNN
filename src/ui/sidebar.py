"""src/ui/sidebar.py — Sidebar for VARIANT-GNN Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st


def _load_default_threshold() -> float:
    """Read F1-optimal threshold from models/threshold.json if available."""
    path = Path("models/threshold.json")
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return float(data.get("threshold", data.get("classification_threshold", 0.8415)))
        except Exception:
            pass
    return 0.8415


def _model_status_badge() -> None:
    """Show model readiness badge from PROVENANCE.json."""
    prov = Path("models/PROVENANCE.json")
    if not prov.exists():
        st.markdown(
            '<div style="background:rgba(230,57,70,0.1);border:1px solid rgba(230,57,70,0.3);'
            'border-left:3px solid #e63946;border-radius:8px;padding:10px 14px;font-size:0.78rem;'
            'color:#ff6b6b;margin-bottom:12px;">'
            '⚠️ Model bulunamadı<br><span style="color:#8892a4;">python main.py --mode train</span></div>',
            unsafe_allow_html=True,
        )
        return

    with open(prov, encoding="utf-8", errors="replace") as f:
        p = json.load(f)

    cv_f1 = p.get("cv_f1_mean", 0)
    t_f1 = p.get("test_f1", 0)
    status = p.get("status", "UNKNOWN")
    color = "#34d399" if "REAL" in status else "#f59e0b"

    st.markdown(
        f"""
    <div style="background:rgba(230,57,70,0.06);border:1px solid rgba(230,57,70,0.2);
                border-left:3px solid #e63946;border-radius:11px;padding:12px 16px;margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:0.72rem;font-weight:700;color:{color};text-transform:uppercase;
                         letter-spacing:0.5px;">✅ Model Hazır</span>
            <span style="font-size:0.65rem;color:#8892a4;">TEKNOFEST 2026</span>
        </div>
        <div style="display:flex;gap:12px;">
            <div style="text-align:center;">
                <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;
                            font-family:'JetBrains Mono',monospace;">{t_f1:.4f}</div>
                <div style="font-size:0.62rem;color:#8892a4;">Test F1</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;
                            font-family:'JetBrains Mono',monospace;">{cv_f1:.4f}</div>
                <div style="font-size:0.62rem;color:#8892a4;">CV F1</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Render sidebar; return (uploaded_df, settings_dict)."""
    with st.sidebar:
        # ── Logo / title ──────────────────────────────────────────────────────
        st.markdown(
            """
        <div style="text-align:center;padding:20px 0 16px;">
            <div style="font-size:2.4rem;line-height:1;">🧬</div>
            <div style="font-size:1.05rem;font-weight:800;
                         letter-spacing:1px;margin-top:6px;">
                <span style="color:#f1f5f9;">VARIANT</span><span style="color:#ff6b6b;">-GNN</span>
            </div>
            <div style="font-size:0.68rem;color:#8892a4;margin-top:3px;
                         letter-spacing:0.5px;">TEKNOFEST 2026 · XYRA3</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── Model status ──────────────────────────────────────────────────────
        _model_status_badge()

        st.divider()

        # ── File upload ───────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:700;color:#8892a4;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">📂 Veri Yükleme</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "CSV dosyası",
            type=["csv"],
            label_visibility="collapsed",
            help="Örnek: data/samples/jury_blind_sample.csv",
        )
        df: Optional[pd.DataFrame] = None
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(
                    f'<div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);'
                    f'border-radius:6px;padding:8px 12px;font-size:0.78rem;color:#34d399;margin-top:6px;">'
                    f"✅ {len(df):,} varyant yüklendi</div>",
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Okuma hatası: {exc}")

        st.divider()

        # ── Threshold ─────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:700;color:#8892a4;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">⚡ Karar Eşiği</p>',
            unsafe_allow_html=True,
        )
        default_thr = _load_default_threshold()
        threshold = st.slider(
            "Threshold",
            min_value=0.05,
            max_value=0.95,
            value=default_thr,
            step=0.01,
            label_visibility="collapsed",
            help=f"F1-optimal: {default_thr:.3f} (kalibrasyon setinden)",
        )
        st.markdown(
            f'<div style="text-align:center;font-size:0.7rem;color:#718096;margin-top:-8px;">'
            f"θ = {threshold:.3f} &nbsp;·&nbsp; F1-optimal: {default_thr:.3f}</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── XAI options ───────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:700;color:#8892a4;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">🔬 XAI Seçenekleri</p>',
            unsafe_allow_html=True,
        )
        show_shap = st.checkbox("📊 Global SHAP Özeti", value=True)
        show_waterfall = st.checkbox("🌊 Yerel SHAP Waterfall", value=True)
        show_lime = st.checkbox("🟢 LIME Açıklaması", value=False)
        variant_index = st.number_input(
            "Varyant İndeksi (Yerel XAI)",
            min_value=0,
            value=0,
            step=1,
        )

        st.divider()

        # ── Clinical disclaimer ───────────────────────────────────────────────
        st.markdown(
            """
        <div style="background:rgba(230,57,70,0.06);border:1px solid rgba(230,57,70,0.25);
                    border-left:3px solid #e63946;border-radius:8px;padding:10px 12px;
                    font-size:0.72rem;color:#ff6b6b;line-height:1.6;">
            🛑 <strong>Klinik kullanım yasak</strong><br>
            <span style="color:#8892a4;">Bu sistem yalnızca araştırma ve yarışma
            değerlendirmesi amaçlıdır (TEKNOFEST §10).</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        settings: Dict[str, Any] = {
            "threshold": threshold,
            "show_shap": show_shap,
            "show_waterfall": show_waterfall,
            "show_lime": show_lime,
            "variant_index": int(variant_index),
        }

        return df, settings

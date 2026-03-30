import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Protocol
from src.ui.utils import plot_dark

# Define a Protocol for the pipeline for better type checking if needed
class PipelineProtocol(Protocol):
    _ensemble: Any
    _preprocessor: Any

def render_xai(pipeline: Any, df_features: pd.DataFrame, opts: Dict[str, Any]) -> None:
    """SHAP, LIME ve GNN graf görselleştirmelerini içeren XAI sekmesini oluşturur."""
    if pipeline is None or pipeline._ensemble is None:
        return
    if not (opts.get("show_shap", False) or opts.get("show_waterfall", False) or opts.get("show_lime", False)):
        return

    try:
        X_scaled: np.ndarray = pipeline._preprocessor.transform(df_features.values)
    except (ValueError, RuntimeError) as exc:
        st.warning(f"XAI önişleme hatası: {exc}")
        return

    xgb_model: Any = pipeline._ensemble.xgb
    feature_names: List[str] = list(df_features.columns)
    from src.explainability.shap_explainer import SHAPExplainer
    explainer: SHAPExplainer = SHAPExplainer(xgb_model, feature_names=feature_names, training_data=X_scaled)
    idx: int = min(int(opts.get("variant_index", 0)), len(X_scaled) - 1)

    if opts.get("show_shap"):
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3>Global SHAP — En Önemli Biyolojik Özellikler</h3>
        </div>
        """, unsafe_allow_html=True)
        top: List[Tuple[str, float]] = explainer.get_top_features(X_scaled[:200], top_n=15)
        if top:
            names_ = [t[0] for t in top]
            vals_  = [t[1] for t in top]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            plot_dark(fig, ax)
            colors_ = ['#fc8181' if v > np.median(vals_) else '#63b3ed' for v in vals_]
            bars = ax.barh(names_[::-1], vals_[::-1], color=colors_[::-1], alpha=0.9, height=0.65)
            ax.set_xlabel("Ortalama |SHAP Değeri|")
            ax.set_title("Top-15 Özellik (XGBoost SHAP)", fontsize=12, fontweight='bold', pad=14)
            for bar, val in zip(bars, vals_[::-1]):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', ha='left', color='#94a3b8', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    if opts.get("show_waterfall"):
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">🌊</div>
            <h3>Yerel SHAP Waterfall — Varyant #{idx}</h3>
        </div>
        """, unsafe_allow_html=True)
        path: str = "reports/shap_waterfall.png"
        explainer.plot_waterfall(X_scaled[idx], output_path=path)
        if Path(path).exists():
            st.image(path, use_column_width=True)

    if opts.get("show_lime"):
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">🟢</div>
            <h3>LIME Açıklaması — Varyant #{idx}</h3>
        </div>
        """, unsafe_allow_html=True)
        from src.explainability.lime_explainer import LIMEExplainer
        lime_exp: LIMEExplainer = LIMEExplainer(
            training_data=X_scaled,
            feature_names=feature_names,
            predict_fn=xgb_model.predict_proba,
        )
        lime_exp.explain_instance(X_scaled[idx], output_html="reports/lime_explanation.html")
        html_path: Path = Path("reports/lime_explanation.html")
        if html_path.exists():
            with open(html_path) as fh:
                st.components.v1.html(fh.read(), height=600, scrolling=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏥</div>
        <h3>Klinik Karar Destek Asistanı (Otomatik Yorum)</h3>
    </div>
    """, unsafe_allow_html=True)

    try:
        from src.explainability.clinical_insight import generate_clinical_insight
        top_feats: List[Tuple[str, float]] = explainer.get_top_features(X_scaled[idx:idx+1], top_n=8)
        probs_row: np.ndarray = xgb_model.predict_proba(X_scaled[idx:idx+1])[0]
        prob_val: float = float(probs_row[1])
        risk_val: float = prob_val * 100
        prediction: str = "Pathogenic" if prob_val >= float(opts.get("threshold", 0.5)) else "Benign"
        v_id: Optional[Any] = df_features["Variant_ID"].iloc[idx] if "Variant_ID" in df_features.columns else None

        insight: Dict[str, Any] = generate_clinical_insight(
            risk_score=risk_val,
            prediction=prediction,
            top_features=top_feats if top_feats else [],
            probability=prob_val,
            variant_id=str(v_id) if v_id else None,
        )

        st.markdown(f"""
        <div style="background:rgba(99,179,237,0.06); border:1px solid rgba(99,179,237,0.2);
                    border-radius:14px; padding:22px 26px; margin-bottom:18px;">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
                <div style="font-size:1.8rem; font-weight:800; color:{insight['zone_color']};">{insight['zone_label']}</div>
                <div style="font-size:1.4rem; font-weight:700; color:#e2e8f0;">{risk_val:.1f} / 100</div>
            </div>
            <div style="color:#cbd5e0; font-size:0.92rem; line-height:1.8;">{insight['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

        if insight.get("key_findings"):
            st.markdown("#### 🔑 Kilit Biyolojik Bulgular")
            for fi, finding in enumerate(insight["key_findings"], 1):
                st.markdown(f"""
                <div style="background:rgba(26,39,68,0.7); border:1px solid rgba(99,179,237,0.15);
                            border-left:4px solid {'#fc8181' if finding['direction'] == 'artırdı' else '#68d391'};
                            border-radius:10px; padding:14px 18px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px;">
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.88rem;">{fi}. <code style='color:#63b3ed;'>{finding['feature']}</code></div>
                        <div style="font-size:0.78rem; color:{'#fc8181' if finding['direction'] == 'artırdı' else '#68d391'}; font-weight:600;">Riski {finding['direction']}</div>
                    </div>
                    <div style="margin-top:8px; color:#94a3b8; font-size:0.83rem; line-height:1.65;">{finding['insight']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"<div style='background:rgba(66,153,225,0.08); border:1px solid rgba(66,153,225,0.25); border-radius:10px; padding:14px 18px;'>{insight.get('recommendation', '')}</div>", unsafe_allow_html=True)
    except Exception as exc:
        st.info(f"ℹ️ Klinik yorum üretilemedi: {exc}")

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧬</div>
        <h3>Genetik Etkileşim Grafı (GNN Mimarisi)</h3>
    </div>
    """, unsafe_allow_html=True)

    col_gnn1, col_gnn2 = st.columns(2)
    with col_gnn1:
        st.markdown("**🕸️ Özellik Etkileşim Ağı**")
        try:
            from src.explainability.graph_viz import plot_variant_graph
            if hasattr(pipeline._preprocessor, 'edge_index') and pipeline._preprocessor.edge_index is not None:
                fig_gnn = plot_variant_graph(edge_index=pipeline._preprocessor.edge_index, node_features=X_scaled, feature_names=feature_names, top_n_nodes=20, figsize=(8, 6))
                if fig_gnn:
                    st.pyplot(fig_gnn)
                    plt.close()
        except Exception as exc:
            st.warning(f"GNN Grafı çizilemedi: {exc}")

    with col_gnn2:
        st.markdown("**🌡️ Korelasyon Isı Haritası (GNN Kenar Temeli)**")
        try:
            from src.explainability.graph_viz import plot_feature_correlation_heatmap
            fig_heat = plot_feature_correlation_heatmap(node_features=X_scaled, feature_names=feature_names, top_n=20, figsize=(8, 6))
            if fig_heat:
                st.pyplot(fig_heat)
                plt.close()
        except Exception as exc:
            st.warning(f"Korelasyon ısı haritası çizilemedi: {exc}")

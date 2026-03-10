"""
app.py  VARIANT-GNN Streamlit dashboard.

Uses the new InferencePipeline which preserves Variant_ID and outputs
calibrated risk scores alongside raw probabilities and confidence.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.inference.pipeline import InferencePipeline
from src.utils.logging_cfg import setup_logging

setup_logging(level=logging.WARNING)

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VARIANT-GNN Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 10px; padding: 20px; color: white;
        text-align: center; margin: 5px;
    }
    .pathogenic-badge {
        background-color:#c0392b; color:white;
        padding:3px 10px; border-radius:12px; font-weight:bold;
    }
    .benign-badge {
        background-color:#27ae60; color:white;
        padding:3px 10px; border-radius:12px; font-weight:bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models")
def _load_pipeline() -> InferencePipeline | None:
    try:
        pipeline = InferencePipeline()
        pipeline.load()
        return pipeline
    except FileNotFoundError as exc:
        st.error(
            f"Models not found: {exc}\n\n"
            "Run `python main.py --mode train` first."
        )
        return None
    except Exception as exc:
        st.error(f"Model load error: {exc}")
        return None


# ---------------------------------------------------------------------------
def render_sidebar(cfg) -> dict:
    st.sidebar.header("System Information")
    st.sidebar.info(
        f"Model: Hybrid XGBoost + GNN + DNN\n\n"
        f"Ensemble weights: {cfg.ensemble.weights}\n\n"
        f"Calibration: {cfg.calibration.method}"
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("XAI Options")
    opts = {
        "show_shap":    st.sidebar.checkbox("Global SHAP Summary", value=True),
        "show_waterfall":st.sidebar.checkbox("Local SHAP Waterfall", value=True),
        "show_lime":    st.sidebar.checkbox("LIME Explanation", value=False),
        "variant_index":st.sidebar.number_input(
            "Local XAI variant index:", min_value=0, value=0, step=1
        ),
    }
    st.sidebar.markdown("---")
    st.sidebar.subheader("Threshold")
    opts["threshold"] = st.sidebar.slider(
        "Classification threshold (Pathogenic)",
        min_value=0.1, max_value=0.9,
        value=float(cfg.thresholds.classification), step=0.01,
    )
    return opts


# ---------------------------------------------------------------------------
def render_summary(df_result: pd.DataFrame) -> None:
    total      = len(df_result)
    pathogenic = int((df_result["Prediction"] == "Pathogenic").sum())
    benign     = total - pathogenic
    high_risk  = int(df_result.get("High_Risk", pd.Series(dtype=bool)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Variants",     total)
    c2.metric("Pathogenic",         pathogenic, delta=f"{100*pathogenic/max(total,1):.1f}%")
    c3.metric("Benign",             benign,     delta=f"{100*benign/max(total,1):.1f}%")
    c4.metric("High Risk (cal.)",   high_risk)


# ---------------------------------------------------------------------------
def render_risk_histogram(df_result: pd.DataFrame) -> None:
    if "Calibrated_Risk" not in df_result.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(
        df_result["Calibrated_Risk"], bins=30,
        color="steelblue", edgecolor="white", alpha=0.85,
    )
    ax.set_xlabel("Calibrated Risk Score (%)")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ---------------------------------------------------------------------------
def render_xai(pipeline, df_features: pd.DataFrame, opts: dict) -> None:
    if pipeline is None or pipeline._ensemble is None:
        return
    if not (opts["show_shap"] or opts["show_waterfall"] or opts["show_lime"]):
        return

    try:
        X_scaled = pipeline._preprocessor.transform(df_features.values)
    except Exception as exc:
        st.warning(f"Could not preprocess features for XAI: {exc}")
        return

    xgb_model     = pipeline._ensemble.xgb
    feature_names = list(df_features.columns)

    from src.explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(xgb_model, feature_names=feature_names, training_data=X_scaled)

    idx = min(int(opts["variant_index"]), len(X_scaled) - 1)

    if opts["show_shap"]:
        st.subheader("Global SHAP Feature Importance")
        top = explainer.get_top_features(X_scaled[:200], top_n=15)
        if top:
            names_  = [t[0] for t in top]
            vals_   = [t[1] for t in top]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(names_[::-1], vals_[::-1], color="steelblue")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Top Features (XGBoost SHAP)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    if opts["show_waterfall"]:
        st.subheader(f"Local SHAP Waterfall  Variant {idx}")
        path = "reports/shap_waterfall.png"
        explainer.plot_waterfall(X_scaled[idx], output_path=path)
        if Path(path).exists():
            st.image(path)

    if opts["show_lime"]:
        st.subheader(f"LIME Explanation  Variant {idx}")
        from src.explainability.lime_explainer import LIMEExplainer
        lime_exp = LIMEExplainer(
            training_data = X_scaled,
            feature_names = feature_names,
            predict_fn    = xgb_model.predict_proba,
        )
        lime_exp.explain_instance(
            X_scaled[idx], output_html="reports/lime_explanation.html"
        )
        html_path = Path("reports/lime_explanation.html")
        if html_path.exists():
            with open(html_path) as fh:
                st.components.v1.html(fh.read(), height=600, scrolling=True)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = get_settings()

    st.title(" VARIANT-GNN: Genomic Variant Pathogenicity Analysis")
    st.markdown(
        ">  *Research & educational use only. Not a clinical decision tool.*"
    )

    pipeline = _load_pipeline()
    opts     = render_sidebar(cfg)

    st.subheader("Data Upload")
    uploaded = st.file_uploader(
        "Upload variant CSV",
        type=["csv"],
        help=(
            "CSV with numeric feature columns. Label column is optional and ignored. "
            "Variant_ID column is preserved in output."
        ),
    )

    if uploaded is None:
        st.info(
            "Upload a CSV file to begin. "
            "See `data_contracts/sample_input.csv` for the expected format."
        )
        return

    df_raw = pd.read_csv(uploaded)
    st.write(f"**Preview** ({len(df_raw)} rows  {df_raw.shape[1]} cols):")
    st.dataframe(df_raw.head())

    if pipeline is None:
        return

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Processing variants"):
            try:
                df_result = pipeline.predict_from_dataframe(df_raw)
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.stop()

        st.success("Analysis complete!")
        render_summary(df_result)

        st.subheader("Results")
        st.dataframe(df_result, use_container_width=True)

        csv_bytes = df_result.to_csv(index=False).encode()
        st.download_button(
            "Download Results CSV",
            data    = csv_bytes,
            file_name = "variant_predictions.csv",
            mime    = "text/csv",
        )

        render_risk_histogram(df_result)

        # Separate feature columns for XAI
        id_cols    = [c for c in cfg.schema.id_columns if c in df_raw.columns]
        drop_cols  = id_cols + (
            [cfg.schema.target_column] if cfg.schema.target_column in df_raw.columns else []
        )
        df_features = df_raw.drop(columns=drop_cols, errors="ignore").select_dtypes(
            include=[np.number]
        )

        render_xai(pipeline, df_features, opts)


if __name__ == "__main__":
    main()

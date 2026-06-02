"""src/cli/modes/explain.py — explain mode (SHAP, GNNExplainer, LIME, ACMG)."""

from __future__ import annotations

import json
import logging
import sys

from src.api.pipeline import InferencePipeline
from src.data.loader import load_csv


def mode_explain(args, cfg):
    """Explainability mode — §4.4.

    Produces: SHAP summary/waterfall, group SHAP (6 categories), GNNExplainer
    results, ACMG criteria mapping, learning curve plot, OOD/drift report,
    Turkish clinical insights, PDF clinical reports.
    """
    from src.explainability.group_shap import group_shap_report, instance_explanation_tr
    from src.explainability.shap_explainer import SHAPExplainer

    data_path = args.data_file or args.test_file
    if not data_path:
        logging.error("--data_file or --test_file required (explain mode).")
        sys.exit(1)

    ds = load_csv(data_path)
    pipeline = InferencePipeline()
    pipeline.load()
    cfg.paths.create_dirs()

    preprocessor = pipeline._preprocessor
    ensemble = pipeline._ensemble
    X_np = ds.features.values
    X_scaled = preprocessor.transform(X_np)
    feature_names = list(ds.feature_columns)

    # ── 1. SHAP ──────────────────────────────────────────────────────────────
    explainer = SHAPExplainer(
        xgb_model=ensemble.xgb,
        feature_names=feature_names,
        training_data=X_scaled,
    )
    sample_size = min(len(X_scaled), 200)
    X_sample = X_scaled[:sample_size]
    shap_vals = explainer.explain_instance(X_sample)

    if shap_vals is None:
        logging.warning("SHAP explainer failed — is shap installed?")
        return

    import numpy as np

    if isinstance(shap_vals, list):
        shap_vals = np.array(shap_vals[1])

    explainer.plot_summary(X_sample, str(cfg.paths.reports_dir / "shap_summary.png"))
    explainer.plot_waterfall(X_sample[0], str(cfg.paths.reports_dir / "shap_waterfall_sample0.png"))

    # ── 2. Group SHAP (§4.4 — 6 biological categories) ──────────────────────
    grp_result = group_shap_report(
        shap_values=shap_vals,
        feature_names=feature_names,
        output_dir=cfg.paths.reports_dir,
        plot=True,
    )
    logging.info("Group SHAP JSON → %s", grp_result.get("json_path"))

    # ── 3. GNNExplainer ──────────────────────────────────────────────────────
    _run_gnn_explainer(ensemble, X_scaled, feature_names, cfg)

    # ── 4. Instance explanations (Turkish clinical insights) ─────────────────
    df_pred = pipeline.predict_from_dataset(ds)
    explain_records = []
    for i in range(min(5, len(X_sample))):
        vid = df_pred["Variant_ID"].iloc[i] if "Variant_ID" in df_pred.columns else str(i)
        pred = df_pred["Prediction"].iloc[i]
        prob = float(df_pred["Probability"].iloc[i])
        sv_i = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
        text = instance_explanation_tr(sv_i, feature_names, pred, prob, vid)
        logging.info("[Sample %d] %s", i, text)
        explain_records.append(
            {
                "variant_id": vid,
                "prediction": pred,
                "probability": prob,
                "clinical_insight_tr": text,
            }
        )

    out_json = cfg.paths.reports_dir / "explain_instances.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(explain_records, fh, indent=2, ensure_ascii=False)
    logging.info("Explanation instances → %s", out_json)

    # ── 5. PDF Clinical Reports ───────────────────────────────────────────────
    _run_pdf_reports(df_pred, explain_records, shap_vals, feature_names, cfg)

    # ── 6. Learning curve plot ────────────────────────────────────────────────
    _plot_learning_curve(cfg)

    # ── 7. ACMG criteria mapping ──────────────────────────────────────────────
    _run_acmg_mapper(X_sample, shap_vals, feature_names, cfg)

    # ── 8. OOD / Data Drift ───────────────────────────────────────────────────
    _run_ood_detector(X_scaled, X_sample, feature_names, cfg)

    # ── 9. PubMed RAG ─────────────────────────────────────────────────────────
    _run_pubmed_rag(explain_records, cfg)

    logging.info("Explain mode complete. Outputs: %s", cfg.paths.reports_dir)


# ── Private helpers ──────────────────────────────────────────────────────────


def _run_gnn_explainer(ensemble, X_scaled, feature_names, cfg):
    try:
        from src.core.graph.builder import SampleKNNGraphBuilder
        from src.explainability.gnn_explainer import GNNExplainerWrapper

        gnn_model = ensemble.gnn
        if gnn_model is None:
            logging.warning("GNN model not loaded — GNNExplainer skipped.")
            return
        device = next(gnn_model.parameters()).device
        gnn_exp = GNNExplainerWrapper(model=gnn_model, device=device)
        knn_k = 10
        explain_n = min(30, len(X_scaled))
        graph = SampleKNNGraphBuilder(k=knn_k).build(X_scaled[:explain_n], y=None)
        expl = gnn_exp.explain(graph)
        if expl is None:
            logging.warning("GNNExplainer produced no output.")
            return
        gnn_out: dict = {"n_explained": explain_n, "knn_k": knn_k}
        if hasattr(expl, "node_mask") and expl.node_mask is not None:
            nm = expl.node_mask.detach().cpu().numpy()
            feat_imp = nm.mean(axis=0).tolist() if nm.ndim == 2 else nm.tolist()
            gnn_out["node_feature_importance"] = feat_imp
            gnn_out["top_gnn_features"] = [
                {"feature": f, "importance": round(v, 6)}
                for f, v in sorted(
                    zip(feature_names[: len(feat_imp)], feat_imp), key=lambda x: abs(x[1]), reverse=True
                )[:10]
            ]
        if hasattr(expl, "edge_mask") and expl.edge_mask is not None:
            em = expl.edge_mask.detach().cpu().numpy()
            gnn_out["edge_importance_mean"] = float(em.mean())
            gnn_out["edge_importance_max"] = float(em.max())
        out_path = cfg.paths.reports_dir / "gnn_explainer_results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(gnn_out, fh, indent=2, ensure_ascii=False)
        logging.info("GNNExplainer results → %s", out_path)
    except Exception as exc:
        logging.warning("GNNExplainer failed: %s", exc)


def _run_pdf_reports(df_pred, explain_records, shap_vals, feature_names, cfg):
    try:
        from src.explainability.pdf_report import FPDF_AVAILABLE, generate_pdf_report

        if not FPDF_AVAILABLE:
            logging.warning("fpdf2 not installed — PDF reports skipped.")
            return
        for i, rec in enumerate(explain_records):
            vid = rec["variant_id"]
            pred = rec["prediction"]
            prob = float(rec["probability"])
            risk = float(df_pred["Calibrated_Risk"].iloc[i]) if "Calibrated_Risk" in df_pred.columns else prob * 100
            if risk >= 75:
                zone = "CRITICAL RISK — Clinical Assessment Required"
            elif risk >= 50:
                zone = "HIGH RISK — Expert Review Recommended"
            elif risk >= 25:
                zone = "MODERATE RISK — Follow-up Recommended"
            else:
                zone = "LOW RISK — Likely Benign"

            sv_i = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
            top_feats = sorted(zip(feature_names, sv_i.tolist()), key=lambda x: abs(x[1]), reverse=True)[:10]
            clinical_insight = {
                "zone_label": zone,
                "summary": rec.get("clinical_insight_tr", ""),
                "key_findings": [],
                "recommendation": (
                    "Genetic counselling and additional functional studies recommended."
                    if pred == "Pathogenic"
                    else "Routine clinical follow-up may suffice. Clinician assessment essential."
                ),
            }
            waterfall = str(cfg.paths.reports_dir / "shap_waterfall_sample0.png") if i == 0 else None
            pdf_bytes = generate_pdf_report(
                variant_id=str(vid),
                risk_score=risk,
                prediction=pred,
                probability=prob,
                clinical_insight=clinical_insight,
                top_features=top_feats,
                shap_waterfall_path=waterfall,
            )
            safe_vid = str(vid).replace("/", "_").replace(":", "_")[:40]
            pdf_path = cfg.paths.reports_dir / f"clinical_report_{safe_vid}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            logging.info("PDF clinical report → %s", pdf_path)
    except Exception as exc:
        logging.warning("PDF report failed: %s", exc)


def _plot_learning_curve(cfg):
    lc_path = cfg.paths.reports_dir / "gnn_learning_curve.json"
    if not lc_path.exists():
        logging.info("gnn_learning_curve.json not found — run training first.")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(lc_path) as f:
            lc_data = json.load(f)
        last_run = lc_data[-1]["run_epochs"] if lc_data else []
        if not last_run:
            return
        epochs = [e["epoch"] for e in last_run]
        train_f1s = [e.get("train_f1", 0) for e in last_run]
        val_f1s = [e.get("val_f1", None) for e in last_run]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, train_f1s, label="Train F1", color="#3182ce", linewidth=2)
        if any(v is not None for v in val_f1s):
            ax.plot(epochs, [v or 0 for v in val_f1s], label="Val F1", color="#e53e3e", linewidth=2, linestyle="--")
            for e in last_run:
                if e.get("early_stop"):
                    ax.axvline(e["epoch"], color="gray", linestyle=":", label=f"Early Stop (epoch {e['epoch']})")
                    break
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_title("GNN Learning Curve (§4.5)", fontweight="bold")
        ax.legend()
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = cfg.paths.reports_dir / "gnn_learning_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Learning curve plot → %s", out)
    except Exception as exc:
        logging.warning("Learning curve plot failed: %s", exc)


def _run_acmg_mapper(X_sample, shap_vals, feature_names, cfg):
    try:
        from src.scientific.acmg_mapper import ACMGMapper

        acmg = ACMGMapper()
        n = min(5, len(X_sample))
        acmg_results = acmg.classify_batch(
            X_sample[:n],
            shap_matrix=shap_vals[:n] if (shap_vals is not None and shap_vals.ndim == 2) else None,
            feature_names=feature_names,
        )
        out = cfg.paths.reports_dir / "acmg_criteria.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(acmg_results, fh, indent=2, ensure_ascii=False)
        logging.info("ACMG criteria map → %s", out)
    except Exception as exc:
        logging.warning("ACMG mapper failed: %s", exc)


def _run_ood_detector(X_scaled, X_sample, feature_names, cfg):
    try:
        from src.scientific.ood_detector import OODDetector

        ood = OODDetector(z_threshold=3.0, ood_frac_thresh=0.20)
        ood.fit(X_scaled)
        ood_report = ood.detect(X_sample, feature_names=feature_names)
        drift_rpt = ood.drift_report(X_sample, feature_names=feature_names)
        out = cfg.paths.reports_dir / "ood_drift_report.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "ood_summary": ood_report["summary"],
                    "n_ood": ood_report["n_ood"],
                    "n_total": ood_report["n_total"],
                    "drift_score": drift_rpt["mean_drift_score"],
                    "drift_flag": drift_rpt["drift_flag"],
                    "top_drifted": drift_rpt["top_drifted_features"],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        logging.info("OOD/Drift report → %s | %s", out, ood_report["summary"])
    except Exception as exc:
        logging.warning("OOD detector failed: %s", exc)


def _run_pubmed_rag(explain_records, cfg):
    try:
        from src.scientific.pubmed_rag import PubMedRAG

        rag = PubMedRAG(cache_ttl=3600)
        first_vid = explain_records[0]["variant_id"] if explain_records else "BRCA1"
        gene = str(first_vid).split("-")[0].split("_")[0]
        articles = rag.fetch(gene=gene, n_results=3)
        if articles:
            out = cfg.paths.reports_dir / "pubmed_references.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(articles, fh, indent=2, ensure_ascii=False)
            logging.info("PubMed RAG → %d articles → %s", len(articles), out)
    except Exception as exc:
        logging.warning("PubMed RAG failed: %s", exc)

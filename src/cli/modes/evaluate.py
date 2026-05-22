"""src/cli/modes/evaluate.py — eval, crossval, external_val, adversarial_val modes."""
from __future__ import annotations

import json
import logging
import sys

import numpy as np

from src.data.loader import load_csv
from src.api.pipeline import InferencePipeline
from src.scientific.metrics.metrics import evaluate, evaluate_per_panel
from src.api.export import export_predictions
from src.cli.modes.train import _get_labelled_data


def mode_eval(args, cfg):
    """Evaluate saved model on a labelled CSV, write eval_results.csv."""
    ds = _get_labelled_data(args.data_file, cfg)
    pipeline = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_dataset(ds)
    p1 = df_result["Probability"].values
    proba = np.column_stack([1 - p1, p1])
    threshold = pipeline.store.load_threshold(default=cfg.thresholds.classification)
    report = evaluate(ds.labels, proba, threshold=threshold)
    report.log(prefix="EVAL")
    out = cfg.paths.reports_dir / "eval_results.csv"
    cfg.paths.create_dirs()
    df_result.to_csv(out, index=False)
    logging.info("Eval results → %s (thr=%.4f)", out, threshold)


def mode_crossval(args, cfg):
    """Standalone 5-fold cross-validation on a labelled CSV."""
    from src.utils.seeds import set_global_seed
    from src.training.trainer import VariantTrainer

    ds = _get_labelled_data(args.data_file, cfg)
    trainer = VariantTrainer()
    set_global_seed(cfg.seed)
    folds = trainer._cross_validate(ds.features.values, ds.labels)
    mean_f1 = float(np.mean([r.f1 for r in folds]))
    std_f1 = float(np.std([r.f1 for r in folds]))
    logging.info("Cross-val | Binary F1 (§7.3) = %.4f ± %.4f", mean_f1, std_f1)
    for r in folds:
        logging.info("  Fold %d | Ens=%.4f  XGB=%.4f  LGB=%.4f  GNN=%.4f  DNN=%.4f",
                     r.fold, r.f1, r.xgb_f1, getattr(r, "lgbm_f1", 0.0), r.gnn_f1, r.dnn_f1)


def mode_external_val(args, cfg):
    """External validation — TEKNOFEST jury scenario (§7.5).

    Uses the training-time F1-optimal threshold; never re-tunes on test data.
    Produces: external_validation_report.json, confusion matrix PNG, jury CSV.
    """
    test_path = args.test_file or args.data_file
    if not test_path:
        logging.error("--test_file or --data_file required.")
        sys.exit(1)

    ds = load_csv(test_path)
    if ds.labels is None:
        logging.error("external_val requires labelled data.")
        sys.exit(1)

    pipeline = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_dataset(ds)

    y_true = ds.labels
    p1 = df_result["Probability"].values
    y_prob = np.column_stack([1 - p1, p1])

    threshold = pipeline.store.load_threshold(default=cfg.thresholds.classification)
    panel_thresholds = pipeline.store.load_panel_thresholds()
    effective_threshold = panel_thresholds if panel_thresholds else threshold
    logging.info("External val: using saved threshold (%s)", effective_threshold)

    report = evaluate(y_true, y_prob, threshold=threshold)
    report.log(prefix="EXTERNAL_VAL")

    panel_metrics: dict = {}
    if "Panel" in ds.metadata.columns and len(ds.metadata) == len(y_true):
        panels_arr = ds.metadata["Panel"].values
        panel_reports = evaluate_per_panel(
            y_true, y_prob, panels_arr, threshold=effective_threshold
        )
        for pname, prep in panel_reports.items():
            prep.log(prefix=f"PANEL_{pname}")
            panel_metrics[pname] = prep.as_dict()

    cfg.paths.create_dirs()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = report.conf_matrix
        if cm is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Pathogenic"])
            ax.set_yticklabels(["Benign", "Pathogenic"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix | F1={report.binary_f1:.4f}")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                            fontsize=14, fontweight="bold")
            plt.tight_layout()
            cm_path = cfg.paths.reports_dir / "external_val_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close()
            logging.info("Confusion matrix → %s", cm_path)
    except Exception as exc:
        logging.warning("Plot failed: %s", exc)

    export_predictions(
        df_result, cfg.paths.reports_dir,
        prefix="external_val",
        submission_path=getattr(args, "output", None),
    )

    report_json = {
        "mode": "external_validation",
        "test_file": str(test_path),
        "n_samples": int(len(y_true)),
        "threshold_used": threshold,
        "threshold_source": "training_artifact",
        "metrics": report.as_dict(),
        "confusion_matrix": (report.conf_matrix.tolist()
                             if report.conf_matrix is not None else None),
        "panel_metrics": panel_metrics,
    }
    out_json = cfg.paths.reports_dir / "external_validation_report.json"
    with open(out_json, "w") as fh:
        json.dump(report_json, fh, indent=2, default=str)
    logging.info("JSON report → %s", out_json)
    logging.info("External validation complete.")


def mode_adversarial_val(args, cfg):
    """Adversarial validation — train/test domain shift detection.

    AUC ~0.5 → no shift (good). AUC ~1.0 → severe shift (bad).
    """
    from src.scientific.metrics.adversarial_validation import adversarial_validate

    if not args.data_file or not args.test_file:
        logging.error("--data_file (train) and --test_file (test) both required.")
        sys.exit(1)

    ds_train = load_csv(args.data_file)
    ds_test = load_csv(args.test_file)

    result = adversarial_validate(
        X_train=ds_train.features.values,
        X_test=ds_test.features.values,
        feature_names=ds_train.feature_columns,
    )

    cfg.paths.create_dirs()
    report_json = {
        "mode": "adversarial_validation",
        "train_file": str(args.data_file),
        "test_file": str(args.test_file),
        "auc_mean": result.auc_mean,
        "auc_std": result.auc_std,
        "auc_per_fold": result.auc_per_fold,
        "verdict": result.verdict,
        "top_shift_features": result.top_shift_features,
    }
    out_json = cfg.paths.reports_dir / "adversarial_validation_report.json"
    with open(out_json, "w") as fh:
        json.dump(report_json, fh, indent=2, default=str, ensure_ascii=False)

    logging.info("Adversarial validation report → %s", out_json)
    print(f"\nResult: {result.verdict}")
    print(f"AUC: {result.auc_mean:.4f} ± {result.auc_std:.4f}")
    if result.top_shift_features:
        print(f"Top shift features: {', '.join(result.top_shift_features[:5])}")

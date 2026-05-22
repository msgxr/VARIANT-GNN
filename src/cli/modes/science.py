"""src/cli/modes/science.py — tune, panel_transfer, label_quality, ablation modes."""
from __future__ import annotations

import json
import logging

from src.cli.modes.train import _get_labelled_data
from src.utils.seeds import set_global_seed


def mode_tune(args, cfg):
    """Optuna hyperparameter search for XGBoost."""
    from src.features.preprocessing import build_preprocessor_from_config
    from src.training.tune import ModelTuner

    ds = _get_labelled_data(args.data_file, cfg)
    preprocessor = build_preprocessor_from_config()
    preprocessor.use_autoencoder = False
    X, y_res = preprocessor.fit_resample_train(ds.features.values, ds.labels)
    n_trials = getattr(args, "n_trials", 30)
    tuner = ModelTuner(X, y_res, n_trials=n_trials)
    best = tuner.optimise_xgboost()
    out_path = cfg.paths.reports_dir / "best_xgb_params.json"
    cfg.paths.create_dirs()
    with open(out_path, "w") as fh:
        json.dump(best, fh, indent=2)
    logging.info("Best XGB params → %s", out_path)


def mode_panel_transfer(args, cfg):
    """Panel transfer (generalization) matrix — §3.2."""
    from src.evaluation.panel_transfer import CrossPanelEvaluator

    ds = _get_labelled_data(args.data_file, cfg)
    if "Panel" not in ds.metadata.columns:
        logging.error("Dataset missing 'Panel' column.")
        return

    evaluator = CrossPanelEvaluator(random_state=cfg.seed)
    result = evaluator.evaluate(
        ds.features.values, ds.labels, ds.metadata["Panel"].values
    )
    cfg.paths.create_dirs()
    report_path = cfg.paths.reports_dir / "panel_transfer_matrix.json"
    plot_path = cfg.paths.reports_dir / "panel_transfer.png"
    with open(report_path, "w") as fh:
        json.dump(result.as_dict(), fh, indent=2)
    result.plot(save_path=plot_path)
    logging.info("Panel transfer matrix → %s", report_path)


def mode_label_quality(args, cfg):
    """Label quality detection via Confident Learning (§3.2 leakage prevention)."""
    try:
        from src.scientific.label_quality import ConfidentLearner
        from sklearn.model_selection import cross_val_predict
        import xgboost as xgb
        from src.features.preprocessing import build_preprocessor_from_config

        ds = _get_labelled_data(args.data_file, cfg)
        X = ds.features.values
        y = ds.labels

        pre = build_preprocessor_from_config()
        X_proc, _ = pre.fit_resample_train(X, y)
        model = xgb.XGBClassifier(**cfg.xgb.as_dict())
        logging.info("Computing 5-fold OOF probabilities with XGBoost...")
        oof_probs = cross_val_predict(
            model, X_proc[:len(X)], y, cv=5, method="predict_proba"
        )
        learner = ConfidentLearner(y_true=y, oof_probs=oof_probs)
        report = learner.detect_label_errors()

        cfg.paths.create_dirs()
        out_path = cfg.paths.reports_dir / "label_quality_report.json"
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logging.info("Label quality report → %s", out_path)
    except Exception as exc:
        logging.error("Label quality analysis failed: %s", exc)


def mode_ablation(args, cfg):
    """Ablation analysis — model + preprocessing contributions (§4.5)."""
    from pathlib import Path
    from src.evaluation.ablation import run_ablation

    ds = _get_labelled_data(args.data_file, cfg)
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    output_path = (
        Path(args.output) if args.output
        else cfg.paths.reports_dir / "ablation_report.json"
    )
    report = run_ablation(
        X=ds.features.values,
        y=ds.labels,
        nuc_seqs=ds.nuc_sequences,
        aa_seqs=ds.aa_sequences,
        output=output_path,
    )
    logging.info(
        "Ablation complete. Baseline F1=%.4f, %d configs analyzed.",
        report.baseline.binary_f1, len(report.ablations),
    )

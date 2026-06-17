"""src/cli/modes/science.py — tune, panel_transfer, label_quality, ablation modes."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from src.cli.modes.train import _get_labelled_data
from src.utils.seeds import set_global_seed


def mode_tune(args: argparse.Namespace, cfg: Any) -> None:
    """Optuna hyperparameter search for XGBoost (group-aware, train-fold only)."""
    import numpy as np

    from src.features.preprocessing import build_preprocessor_from_config
    from src.training.tune import ModelTuner

    ds = _get_labelled_data(args.data_file, cfg)
    assert ds.labels is not None  # _get_labelled_data exits if labels are missing
    X_all, y_all = ds.features.values, ds.labels

    # Group-aware train/holdout split by Variant_ID — preprocessor (imputer/scaler) VE
    # SMOTE YALNIZ train-fold'da fit edilir. Eskiden tüm etiketli sette fit+SMOTE
    # yapılıyordu → test satırlarının istatistikleri tuning'e sızıyordu.
    groups = None
    if "Variant_ID" in ds.metadata.columns and len(ds.metadata) == len(X_all):
        groups = ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        tr_idx, _ = next(
            GroupShuffleSplit(n_splits=1, test_size=cfg.training.test_size, random_state=cfg.seed).split(
                X_all, y_all, groups=groups
            )
        )
    else:
        from sklearn.model_selection import train_test_split

        tr_idx, _ = train_test_split(
            np.arange(len(y_all)), test_size=cfg.training.test_size, stratify=y_all, random_state=cfg.seed
        )

    preprocessor = build_preprocessor_from_config()
    preprocessor.use_autoencoder = False
    X, y_res = preprocessor.fit_resample_train(X_all[tr_idx], y_all[tr_idx])
    n_trials = getattr(args, "n_trials", 30)
    tuner = ModelTuner(X, y_res, n_trials=n_trials)
    best = tuner.optimise_xgboost()
    out_path = cfg.paths.reports_dir / "best_xgb_params.json"
    cfg.paths.create_dirs()
    with open(out_path, "w") as fh:
        json.dump(best, fh, indent=2)
    logging.info("Best XGB params → %s", out_path)


def mode_panel_transfer(args: argparse.Namespace, cfg: Any) -> None:
    """Panel transfer (generalization) matrix — §3.2."""
    from src.evaluation.panel_transfer import CrossPanelEvaluator

    ds = _get_labelled_data(args.data_file, cfg)
    assert ds.labels is not None  # _get_labelled_data exits if labels are missing
    if "Panel" not in ds.metadata.columns:
        logging.error("Dataset missing 'Panel' column.")
        return

    evaluator = CrossPanelEvaluator(random_state=cfg.seed)
    # Variant_ID grupları (group-aware within-panel split için) — '_aug' suffix soyulur.
    _groups = (
        ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
        if "Variant_ID" in ds.metadata.columns
        else None
    )
    result = evaluator.evaluate(ds.features.values, ds.labels, ds.metadata["Panel"].values, groups=_groups)
    cfg.paths.create_dirs()
    report_path = cfg.paths.reports_dir / "panel_transfer_matrix.json"
    plot_path = cfg.paths.reports_dir / "panel_transfer.png"
    with open(report_path, "w") as fh:
        json.dump(result.as_dict(), fh, indent=2)
    result.plot(save_path=plot_path)
    logging.info("Panel transfer matrix → %s", report_path)


def mode_label_quality(args: argparse.Namespace, cfg: Any) -> None:
    """Label quality detection via Confident Learning (§3.2 leakage prevention)."""
    try:
        import xgboost as xgb

        from src.features.preprocessing import build_preprocessor_from_config
        from src.scientific.label_quality import ConfidentLearner

        ds = _get_labelled_data(args.data_file, cfg)
        assert ds.labels is not None  # _get_labelled_data exits if labels are missing
        X = ds.features.values
        y = ds.labels

        pre = build_preprocessor_from_config()
        X_proc, _ = pre.fit_resample_train(X, y)
        # ConfidentLearner runs its OWN group-free 5-fold OOF internally; feed it the
        # REAL (non-SMOTE) rows only — fit_resample_train returns originals first.
        X_real = X_proc[: len(X)]
        learner = ConfidentLearner(
            noise_threshold=0.45,
            cv_folds=5,
            base_estimator=xgb.XGBClassifier(**cfg.xgb.as_dict()),
        )
        logging.info("ConfidentLearner: %d gerçek örnek üzerinde 5-fold OOF...", len(X_real))
        report = learner.fit(X_real, y)  # → LabelQualityReport dataclass
        report.log()

        cfg.paths.create_dirs()
        out_path = cfg.paths.reports_dir / "label_quality_report.json"
        nm = report.noise_matrix
        payload = {
            "n_samples": int(report.n_samples),
            "n_flagged": int(report.n_flagged),
            "estimated_noise_rate": float(report.estimated_noise_rate),
            "flagged_indices": [int(i) for i in report.flagged_indices],
            "class_noise_rates": {int(k): float(v) for k, v in report.class_noise_rates.items()},
            "noise_matrix": nm.tolist() if hasattr(nm, "tolist") else nm,
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logging.info(
            "Label quality report → %s (n_flagged=%d, noise≈%.2f%%)",
            out_path,
            report.n_flagged,
            100 * report.estimated_noise_rate,
        )
    except Exception as exc:
        logging.error("Label quality analysis failed: %s", exc)
        raise SystemExit(1)


def mode_ablation(args: argparse.Namespace, cfg: Any) -> None:
    """Ablation analysis — model + preprocessing contributions (§4.5)."""
    from pathlib import Path

    from src.evaluation.ablation import run_ablation

    ds = _get_labelled_data(args.data_file, cfg)
    assert ds.labels is not None  # _get_labelled_data exits if labels are missing
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    output_path = Path(args.output) if args.output else cfg.paths.reports_dir / "ablation_report.json"
    report = run_ablation(
        X=ds.features.values,
        y=ds.labels,
        nuc_seqs=ds.nuc_sequences,
        aa_seqs=ds.aa_sequences,
        output=output_path,
    )
    logging.info(
        "Ablation complete. Baseline F1=%.4f, %d configs analyzed.",
        report.baseline.binary_f1,
        len(report.ablations),
    )

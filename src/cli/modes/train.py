"""src/cli/modes/train.py — train and train_panels modes."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.scientific.calibration.calibrator import EnsembleCalibrator
from src.data.loader import load_csv
from src.scientific.metrics.metrics import evaluate, evaluate_per_panel, find_best_threshold
from src.scientific.metrics.plots import save_all_plots
from src.features.preprocessing import build_preprocessor_from_config
from src.training.trainer import VariantTrainer
from src.utils.seeds import set_global_seed
from src.utils.serialization import ModelStore


def _get_labelled_data(data_file, cfg):
    """Load a labelled dataset; search default paths if data_file is None."""
    candidates = []
    if data_file:
        candidates.append(Path(data_file))
    candidates += [
        cfg.paths.data_dir / "train_variants.csv",
        Path("data/train_variants.csv"),
    ]
    for path in candidates:
        if path.exists():
            logging.info("Loading dataset: %s", path)
            ds = load_csv(path)
            if ds.labels is None:
                logging.error("No labels found in %s.", path)
                sys.exit(1)
            return ds
    logging.error("No labelled dataset found.")
    sys.exit(1)


def mode_train(args, cfg):
    """Leakage-free training + 5-fold CV + calibration + test evaluation."""
    ds = _get_labelled_data(args.data_file, cfg)

    from src.features.feature_validator import FeatureValidator
    fv_report = FeatureValidator().validate_and_warn(ds.features)
    logging.info("Feature coverage: %.1f %%", 100 * fv_report.overall_coverage)

    panel = getattr(args, "panel", None)
    if panel and "Panel" in ds.metadata.columns:
        mask = ds.metadata["Panel"] == panel
        valid_positions = list(np.where(mask.values)[0])
        from src.data.loader import LoadedDataset
        ds = LoadedDataset(
            features=ds.features[mask].reset_index(drop=True),
            labels=ds.labels[mask.values],
            metadata=ds.metadata[mask].reset_index(drop=True),
            feature_columns=ds.feature_columns,
            nuc_sequences=([ds.nuc_sequences[i] for i in valid_positions]
                           if ds.nuc_sequences else None),
            aa_sequences=([ds.aa_sequences[i] for i in valid_positions]
                          if ds.aa_sequences else None),
        )
        logging.info("Panel filter: %s (%d variants)", panel, len(ds.labels))

    X = ds.features.values
    y = ds.labels
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    trainer = VariantTrainer()
    result = trainer.train(X, y, nuc_seqs=ds.nuc_sequences, aa_seqs=ds.aa_sequences)
    logging.info("CV summary — Binary F1 (§7.3): %.4f ± %.4f",
                 result.mean_cv_f1, result.std_cv_f1)

    preprocessor = result.preprocessor
    ensemble = result.ensemble

    all_indices = np.arange(len(X))
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )
    _, test_indices = train_test_split(
        all_indices, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )

    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=cfg.seed + 99
    )
    X_cal_proc = preprocessor.transform(X_cal)
    _, raw_cal_proba = ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    X_test_proc = preprocessor.transform(X_test)
    _, raw_tst_proba = ensemble.predict(X_test_proc)
    cal_test_proba = calibrator.transform(raw_tst_proba)

    cal_cal_proba = calibrator.transform(raw_cal_proba)
    best_thr, _ = find_best_threshold(y_cal, cal_cal_proba[:, 1], metric="f1")
    logging.info("Threshold from calibration set (n=%d): %.4f", len(y_cal), best_thr)

    report = evaluate(y_test, cal_test_proba, threshold=best_thr)
    report.log(prefix="TEST")

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)
    store.save_threshold(best_thr)

    try:
        import joblib as _jl
        from src.scientific.ood_detector import OODDetector
        X_train_proc = preprocessor.transform(X_tr)
        _ood_det = OODDetector(z_threshold=3.5, ood_frac_thresh=0.25)
        _ood_det.fit(X_train_proc)
        _ood_path = cfg.paths.models_dir / "ood_detector.pkl"
        _jl.dump(_ood_det, str(_ood_path))
        logging.info("OOD detector saved → %s", _ood_path)
    except Exception as exc:
        logging.warning("OOD detector training failed (non-fatal): %s", exc)

    save_all_plots(report, y_test, cal_test_proba, cfg.paths.reports_dir)

    panel_reports_dict: dict = {}
    panel_thresholds_dict: dict = {}

    if "Panel" in ds.metadata.columns:
        test_panels = ds.metadata["Panel"].values[test_indices]
        panel_reports = evaluate_per_panel(y_test, cal_test_proba, test_panels,
                                           threshold=best_thr)
        for pname, prep in panel_reports.items():
            prep.log(prefix=f"PANEL_{pname}")
            panel_reports_dict[pname] = prep.as_dict()

        cal_panels = np.array([])
        if len(ds.metadata) == len(X):
            try:
                _train_idx, _ = train_test_split(
                    all_indices, test_size=cfg.training.test_size,
                    stratify=y, random_state=cfg.seed
                )
                _, _cal_pos = train_test_split(
                    np.arange(len(_train_idx)), test_size=0.15,
                    stratify=y[_train_idx], random_state=cfg.seed + 99
                )
                _cal_orig_idx = _train_idx[_cal_pos]
                cal_panels = ds.metadata["Panel"].values[_cal_orig_idx]
            except Exception as exc:
                logging.warning("Panel indexing failed — threshold opt skipped: %s", exc)

        if len(cal_panels) == len(y_cal):
            panel_thresholds_dict = ensemble.optimise_panel_thresholds(
                X_cal_proc, y_cal, cal_panels
            )
            store.save_panel_thresholds(panel_thresholds_dict)
        else:
            logging.warning(
                "Panel threshold optimization skipped: cal_panels size (%d) != y_cal (%d).",
                len(cal_panels), len(y_cal),
            )

    feat_val_path = cfg.paths.reports_dir / "feature_validation.json"
    with open(feat_val_path, "w") as fh:
        json.dump(fv_report.as_dict(), fh, indent=2, ensure_ascii=False)

    report_path = cfg.paths.reports_dir / "cv_report.json"
    with open(report_path, "w") as fh:
        json.dump({
            "competition_metric": "Binary F1 — 2*TP/(2*TP+FP+FN), pos_label=1 (TEKNOFEST §7.3)",
            "metric_note": "Primary: binary_f1 (Pathogenic). macro_f1 is auxiliary.",
            "swa_enabled": True,
            "mean_cv_binary_f1": result.mean_cv_f1,
            "std_cv_binary_f1": result.std_cv_f1,
            "test_binary_f1": report.binary_f1,
            "test_macro_f1": report.macro_f1,
            "best_threshold": best_thr,
            "threshold_source": "calibration_set",
            "feature_coverage": round(fv_report.overall_coverage, 4),
            "anonymous_columns": fv_report.anonymous_count,
            "folds": [vars(r) for r in result.fold_results],
            "test_metrics": report.as_dict(),
            "panel_metrics": panel_reports_dict,
            "panel_thresholds": panel_thresholds_dict,
        }, fh, indent=2)
    logging.info("CV report saved → %s", report_path)
    logging.info("Training complete.")


def mode_train_panels(args, cfg):
    """Train on all panels combined, evaluate per-panel (§3.2)."""
    ds = _get_labelled_data(args.data_file, cfg)

    X = ds.features.values
    y = ds.labels
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    trainer = VariantTrainer()
    result = trainer.train(X, y, nuc_seqs=ds.nuc_sequences, aa_seqs=ds.aa_sequences)
    logging.info("CV summary — Binary F1 (§7.3): %.4f ± %.4f",
                 result.mean_cv_f1, result.std_cv_f1)

    preprocessor = result.preprocessor
    ensemble = result.ensemble

    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )
    _, X_cal, _, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=cfg.seed + 99
    )
    X_cal_proc = preprocessor.transform(X_cal)
    _, raw_cal_proba = ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    cal_cal_proba = calibrator.transform(raw_cal_proba)
    best_thr, _ = find_best_threshold(y_cal, cal_cal_proba[:, 1], metric="f1")
    logging.info("train_panels threshold (cal set): %.4f", best_thr)

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)
    store.save_threshold(best_thr)

    if "Panel" in ds.metadata.columns:
        panels = ds.metadata["Panel"].unique()
        panel_summary = {}
        for panel_name in panels:
            mask = ds.metadata["Panel"] == panel_name
            mask_vals = mask.values
            X_p_all, y_p_all = X[mask_vals], y[mask_vals]
            if len(y_p_all) < 10:
                logging.warning("Panel %s: too few samples (%d), skipping.",
                                panel_name, len(y_p_all))
                continue
            all_panel_idx = np.where(mask_vals)[0]
            _, panel_test_idx = train_test_split(
                all_panel_idx,
                test_size=cfg.training.test_size,
                stratify=y[all_panel_idx] if len(np.unique(y[all_panel_idx])) > 1 else None,
                random_state=cfg.seed,
            )
            X_p, y_p = X[panel_test_idx], y[panel_test_idx]
            if len(y_p) < 5:
                X_p, y_p = X_p_all, y_p_all
            X_p_proc = preprocessor.transform(X_p)
            _, proba_p = ensemble.predict(X_p_proc)
            cal_proba_p = calibrator.transform(proba_p)
            report_p = evaluate(y_p, cal_proba_p, threshold=best_thr)
            report_p.log(prefix=f"PANEL_{panel_name}")
            panel_summary[panel_name] = {
                "n_samples": int(len(y_p)),
                "metrics": report_p.as_dict(),
            }

        report_path = cfg.paths.reports_dir / "panel_evaluation.json"
        with open(report_path, "w") as fh:
            json.dump(panel_summary, fh, indent=2, default=str, ensure_ascii=False)
        logging.info("Panel evaluation report → %s", report_path)
    else:
        logging.warning("No 'Panel' column — skipping per-panel evaluation.")

    logging.info("train_panels mode complete.")

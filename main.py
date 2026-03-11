"""
main.py — VARIANT-GNN entry point.

Modes:
  train        — leakage-free training + cross-validation + calibration + evaluation
  tune         — Optuna hyperparameter search
  eval         — evaluate saved models on a labelled test CSV
  predict      — run inference on an unlabelled CSV, output results
  crossval     — standalone cross-validation on a labelled CSV
  external_val — external validation: load trained model, run on new test data,
                 compute F1/AUC/Brier metrics  (TEKNOFEST jüri senaryosu)

Usage examples:
  python main.py --mode train
  python main.py --mode train --data_file data/train_variants.csv
  python main.py --mode train --panel General
  python main.py --mode predict --test_file data/test_variants_blind.csv
  python main.py --mode eval   --data_file data/test_variants.csv
  python main.py --mode tune   --data_file data/train_variants.csv --n_trials 30
  python main.py --mode external_val --test_file data/test_variants.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.calibration.calibrator import EnsembleCalibrator
from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.evaluation.metrics import evaluate, find_best_threshold
from src.evaluation.plots import save_all_plots
from src.features.preprocessing import build_preprocessor_from_config
from src.inference.pipeline import InferencePipeline
from src.training.trainer import VariantTrainer
from src.training.tune import ModelTuner
from src.utils.logging_cfg import setup_logging
from src.utils.seeds import set_global_seed
from src.utils.serialization import ModelStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_labelled_data(data_file, cfg):
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


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def mode_train(args, cfg):
    ds  = _get_labelled_data(args.data_file, cfg)
    X   = ds.features.values
    y   = ds.labels
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    trainer = VariantTrainer()
    result  = trainer.train(X, y)
    logging.info("CV summary — Mean Macro F1: %.4f +/- %.4f",
                 result.mean_cv_f1, result.std_cv_f1)

    preprocessor = result.preprocessor
    ensemble     = result.ensemble

    X_train_pre, X_cal, y_train_pre, y_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=cfg.seed + 99
    )
    X_cal_proc = preprocessor.transform(X_cal)
    from src.training.trainer import _make_geo_loader
    cal_loader = _make_geo_loader(preprocessor, X_cal_proc, None,
                                  cfg.training.batch_size, shuffle=False)
    _, raw_cal_proba = ensemble.predict(X_cal_proc, cal_loader)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )
    X_test_proc = preprocessor.transform(X_test)
    test_loader = _make_geo_loader(
        preprocessor, X_test_proc, None, cfg.training.batch_size, shuffle=False
    )
    _, raw_test_proba = ensemble.predict(X_test_proc, test_loader)
    cal_test_proba    = calibrator.transform(raw_test_proba)

    best_thr, _ = find_best_threshold(y_test, cal_test_proba[:, 1], metric="f1")
    report       = evaluate(y_test, cal_test_proba, threshold=best_thr)
    report.log(prefix="TEST")

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)

    save_all_plots(report, y_test, raw_test_proba, cfg.paths.reports_dir)

    report_path = cfg.paths.reports_dir / "cv_report.json"
    with open(report_path, "w") as fh:
        json.dump({
            "mean_cv_macro_f1": result.mean_cv_f1,
            "std_cv_macro_f1":  result.std_cv_f1,
            "folds": [vars(r) for r in result.fold_results],
            "test_metrics": report.as_dict(),
            "best_threshold": best_thr,
        }, fh, indent=2)
    logging.info("CV report saved -> %s", report_path)
    logging.info("Training complete.")


def mode_tune(args, cfg):
    ds = _get_labelled_data(args.data_file, cfg)
    preprocessor = build_preprocessor_from_config()
    preprocessor.use_autoencoder = False
    X, y_res = preprocessor.fit_resample_train(ds.features.values, ds.labels)
    n_trials = getattr(args, "n_trials", 30)
    tuner    = ModelTuner(X, y_res, n_trials=n_trials)
    best     = tuner.optimise_xgboost()
    out_path = cfg.paths.reports_dir / "best_xgb_params.json"
    cfg.paths.create_dirs()
    with open(out_path, "w") as fh:
        json.dump(best, fh, indent=2)
    logging.info("Best XGB params saved -> %s", out_path)


def mode_eval(args, cfg):
    ds = _get_labelled_data(args.data_file, cfg)
    pipeline = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_dataset(ds)
    p1    = df_result["Probability"].values
    proba = np.column_stack([1 - p1, p1])
    report = evaluate(ds.labels, proba)
    report.log(prefix="EVAL")
    out = cfg.paths.reports_dir / "eval_results.csv"
    cfg.paths.create_dirs()
    df_result.to_csv(out, index=False)
    logging.info("Eval results saved -> %s", out)


def mode_predict(args, cfg):
    if not args.test_file:
        logging.error("--test_file required for predict mode.")
        sys.exit(1)
    pipeline  = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_csv(args.test_file)
    out_path  = cfg.paths.reports_dir / "predictions.csv"
    cfg.paths.create_dirs()
    df_result.to_csv(out_path, index=False)
    logging.info("Predictions saved -> %s", out_path)
    print(df_result.head(10).to_string(index=False))


def mode_crossval(args, cfg):
    ds      = _get_labelled_data(args.data_file, cfg)
    trainer = VariantTrainer()
    set_global_seed(cfg.seed)
    folds   = trainer._cross_validate(ds.features.values, ds.labels)
    mean_f1 = float(np.mean([r.f1 for r in folds]))
    std_f1  = float(np.std( [r.f1 for r in folds]))
    logging.info("Cross-val complete | Macro F1 = %.4f +/- %.4f", mean_f1, std_f1)
    for r in folds:
        logging.info("  Fold %d | Ens=%.4f  XGB=%.4f  GNN=%.4f  DNN=%.4f",
                     r.fold, r.f1, r.xgb_f1, r.gnn_f1, r.dnn_f1)


def mode_external_val(args, cfg):
    """External validation — TEKNOFEST 2026 jüri senaryosu.

    Önceden eğitilmiş modeli yükler, yeni etiketli test verisi üzerinde
    F1 / ROC-AUC / Brier / Precision / Recall hesaplar.
    """
    from sklearn.metrics import brier_score_loss

    test_path = args.test_file or args.data_file
    if not test_path:
        logging.error("--test_file veya --data_file gerekli (external_val modu).")
        sys.exit(1)

    ds = load_csv(test_path)
    if ds.labels is None:
        logging.error("External validation i\u00e7in etiketli veri gerekli.")
        sys.exit(1)

    # Panel filtresi (iste\u011fe ba\u011fl\u0131)
    panel = getattr(args, "panel", None)
    if panel and "Panel" in ds.metadata.columns:
        mask = ds.metadata["Panel"] == panel
        from src.data.loader import LoadedDataset
        ds = LoadedDataset(
            features        = ds.features[mask].reset_index(drop=True),
            labels          = ds.labels[mask.values],
            metadata        = ds.metadata[mask].reset_index(drop=True),
            feature_columns = ds.feature_columns,
        )
        logging.info("Panel filtresi: %s (%d varyant)", panel, len(ds.labels))

    pipeline = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_dataset(ds)

    p1    = df_result["Probability"].values
    proba = np.column_stack([1 - p1, p1])
    report = evaluate(ds.labels, proba)
    report.log(prefix="EXTERNAL_VAL")

    # Brier score
    brier = brier_score_loss(ds.labels, p1)
    logging.info("Brier Score: %.6f", brier)

    # Sonu\u00e7lar\u0131 kaydet
    cfg.paths.create_dirs()
    out_csv = cfg.paths.reports_dir / "external_validation_results.csv"
    df_result.to_csv(out_csv, index=False)

    report_json = {
        "mode": "external_validation",
        "test_file": str(test_path),
        "panel": panel,
        "n_samples": len(ds.labels),
        "metrics": report.as_dict(),
        "brier_score": brier,
    }
    out_json = cfg.paths.reports_dir / "external_validation_report.json"
    with open(out_json, "w") as fh:
        json.dump(report_json, fh, indent=2, default=str)

    logging.info("External validation results -> %s", out_csv)
    logging.info("External validation report  -> %s", out_json)
    logging.info("External validation tamamland\u0131.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="VARIANT-GNN: Graph-based Variant Pathogenicity Prediction"
    )
    p.add_argument("--mode",
                   choices=["train", "tune", "eval", "predict", "crossval", "external_val"],
                   default="train")
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--config",    type=str, default=None)
    p.add_argument("--n_trials",  type=int, default=30)
    p.add_argument("--log_file",  type=str, default=None)
    p.add_argument("--panel",     type=str, default=None,
                   help="Panel filtresi: General, Hereditary_Cancer, PAH, CFTR")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(log_file=args.log_file)
    if args.config:
        reset_settings()
        cfg = get_settings(args.config)
    else:
        cfg = get_settings()
    logging.info("=" * 60)
    logging.info("VARIANT-GNN | mode=%s", args.mode.upper())
    logging.info("=" * 60)
    dispatch = {
        "train":        mode_train,
        "tune":         mode_tune,
        "eval":         mode_eval,
        "predict":      mode_predict,
        "crossval":     mode_crossval,
        "external_val": mode_external_val,
    }
    dispatch[args.mode](args, cfg)


if __name__ == "__main__":
    main()

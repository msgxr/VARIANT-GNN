"""src/cli/modes/predict.py — predict mode (jury inference)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.api.export import JURY_COLUMNS, export_predictions
from src.api.pipeline import InferencePipeline


def mode_predict(args: argparse.Namespace, cfg: Any) -> None:
    """Jury mode: unlabelled CSV → deterministic 7-column submission CSV.

    Usage:
      python main.py --mode predict --config configs/pdr.yaml \\
                     --test_file data/jury_test.csv

      # With Test-Time Augmentation:
      python main.py --mode predict --test_file data/jury_test.csv --tta --tta_k 10
    """
    if not args.test_file:
        logging.error("--test_file required (predict mode).")
        sys.exit(1)

    use_tta = getattr(args, "tta", False)
    tta_k = getattr(args, "tta_k", 10)

    logging.info("=" * 60)
    logging.info("  VARIANT-GNN — JURY PREDICTION MODE")
    logging.info("  Input : %s", args.test_file)
    if use_tta:
        logging.info("  TTA   : ON (k=%d augmentations)", tta_k)
    logging.info("=" * 60)

    pipeline = InferencePipeline()
    pipeline.load()

    try:
        if use_tta:
            import pandas as pd

            from src.data.loader import load_predict_csv
            from src.inference.tta import tta_predict_dataframe

            ds = load_predict_csv(args.test_file)
            df_result = pipeline.predict_from_dataset(ds)
            df_result = tta_predict_dataframe(
                pipeline._ensemble,
                pipeline._preprocessor,
                df_result.join(pd.DataFrame(ds.features.values, columns=ds.feature_columns)),
                feature_columns=ds.feature_columns,
                n_augmentations=tta_k,
                seed=cfg.seed,
                nuc_seqs=getattr(ds, "nuc_sequences", None),
                aa_seqs=getattr(ds, "aa_sequences", None),
            )
            logging.info("TTA complete — TTA_Uncertainty column added.")
        else:
            df_result = pipeline.predict_from_csv(args.test_file)
    except Exception as exc:
        # Hata TÜRÜNÜ + traceback'i göster (eskiden tek opak "Prediction error" idi;
        # TTA join / feature-mismatch gibi gerçek buglar teşhis edilemiyordu).
        logging.error("Prediction error (%s): %s", type(exc).__name__, exc, exc_info=True)
        sys.exit(1)

    cfg.paths.create_dirs()
    submission_path = getattr(args, "output", None) or "submission/predictions.csv"

    paths = export_predictions(
        df_result,
        cfg.paths.reports_dir,
        prefix="predictions",
        submission_path=submission_path,
    )
    logging.info("Jury CSV (7-col) → %s", paths["jury"])
    logging.info("Full CSV         → %s", paths["full"])
    logging.info("Submission CSV   → %s", paths["submission"])

    print("\n" + "=" * 60)
    print("  FIRST 10 PREDICTIONS")
    print("=" * 60)
    preview_cols = [c for c in JURY_COLUMNS if c in df_result.columns]
    if not preview_cols:
        preview_cols = list(df_result.columns[:7])
    print(df_result[preview_cols].head(10).to_string(index=False))
    print("=" * 60)

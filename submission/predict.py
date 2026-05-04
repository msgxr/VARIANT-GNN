"""
submission/predict.py
Competition submission entry point for TEKNOFEST 2026 external validation.

Usage:
    python submission/predict.py \\
        --input data/blind_test.csv \\
        --model_dir models/final \\
        --output submission/predictions.csv \\
        --config configs/pdr.yaml

Rules:
  - Fully offline — no internet access.
  - No training or fitting on blind data.
  - Leakage firewall runs automatically.
  - Output schema is fixed (PREDICTION_COLUMNS).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.reproducibility import setup_reproducibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("submission.predict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 — Variant-GNN offline inference"
    )
    parser.add_argument("--input", required=True, type=Path, help="Blind test CSV path")
    parser.add_argument("--model_dir", required=True, type=Path, help="Trained model directory")
    parser.add_argument("--output", required=True, type=Path, help="Output predictions CSV")
    parser.add_argument("--config", required=True, type=Path, help="YAML config (pdr.yaml)")
    parser.add_argument(
        "--local_validation",
        action="store_true",
        default=False,
        help="If set and Label column present, compute local metrics (labels NOT fed to model).",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = _parse_args()

    # ── Load config ────────────────────────────────────────────────────
    config = _load_config(args.config)

    # ── Reproducibility ────────────────────────────────────────────────
    seed = config.get("reproducibility", {}).get("seed", 42)
    det_torch = config.get("reproducibility", {}).get("deterministic_torch", True)
    setup_reproducibility(seed=seed, deterministic_torch=det_torch)

    # ── Validate paths ─────────────────────────────────────────────────
    if not args.input.exists():
        logger.error("Input CSV not found: %s", args.input)
        sys.exit(1)
    if not args.model_dir.exists():
        logger.error("Model directory not found: %s", args.model_dir)
        sys.exit(1)

    reports_dir = args.output.parent / "reports"

    # ── Run external validation ────────────────────────────────────────
    from src.inference.external_validation_runner import ExternalValidationRunner

    runner = ExternalValidationRunner(
        model_dir=args.model_dir,
        reports_dir=reports_dir,
    )
    predictions = runner.run(
        input_path=args.input,
        output_path=args.output,
        local_validation=args.local_validation,
    )

    logger.info(
        "Done. %d predictions written to %s",
        len(predictions),
        args.output,
    )
    logger.info(
        "Prediction distribution: %s",
        predictions["Prediction"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()

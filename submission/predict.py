"""
submission/predict.py
Competition submission entry point for TEKNOFEST 2026 external validation.

Usage (jüri senaryosu — tam komut):
    python submission/predict.py \\
        --input data/jury_test.csv \\
        --output submission/predictions.csv

Usage (tüm argümanlar):
    python submission/predict.py \\
        --input data/jury_test.csv \\
        --model_dir models \\
        --output submission/predictions.csv \\
        --config configs/pdr.yaml

Notlar:
  - --model_dir varsayılan: models/ (proje kökündeki eğitilmiş modeller)
  - --config   varsayılan: configs/pdr.yaml
  - Çevrimdışı çalışır; internet erişimi yok.
  - Blind veri üzerinde fit/train yapılmaz.
  - Leakage firewall otomatik devreye girer.
  - Çıktı şeması sabittir (PREDICTION_COLUMNS).

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
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", required=True, type=Path,
                        help="Blind test CSV path (e.g. data/jury_test.csv)")
    parser.add_argument("--model_dir", required=False, type=Path,
                        default=_root / "models",
                        help="Trained model directory (default: models/)")
    parser.add_argument("--output", required=False, type=Path,
                        default=_root / "submission" / "predictions.csv",
                        help="Output predictions CSV (default: submission/predictions.csv)")
    parser.add_argument("--config", required=False, type=Path,
                        default=_root / "configs" / "pdr.yaml",
                        help="YAML config (default: configs/pdr.yaml)")
    parser.add_argument(
        "--local_validation",
        action="store_true",
        default=False,
        help="If set and Label column present, compute local metrics (labels NOT fed to model).",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config bulunamadi: {config_path}")
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if config is None:
            raise ValueError(f"Config dosyasi bos: {config_path}")
        if not isinstance(config, dict):
            raise ValueError(f"Config gecersiz format (dict bekleniyor): {type(config)}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Gecersiz YAML sozdizimi ({config_path}): {e}") from e


def main() -> None:
    args = _parse_args()

    # ── Load config ────────────────────────────────────────────────────
    config = _load_config(args.config)

    # ── Reproducibility ────────────────────────────────────────────────
    seed = config.get("reproducibility", {}).get("seed", 42)
    det_torch = config.get("reproducibility", {}).get("deterministic_torch", True)
    setup_reproducibility(seed=seed, deterministic_torch=det_torch)

    # ── Validate paths ─────────────────────────────────────────────────
    # resolve() → Windows UNC path ve relative path sorunlarını çözer
    args.input     = args.input.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output    = args.output.resolve()
    args.config    = args.config.resolve()

    if not args.input.exists():
        logger.error("Input CSV bulunamadi: %s", args.input)
        sys.exit(1)
    if not args.model_dir.exists():
        logger.error(
            "Model dizini bulunamadi: %s\n"
            "  Önce modeli egit: python main.py --mode train --config configs/pdr.yaml",
            args.model_dir,
        )
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config bulunamadi: %s", args.config)
        sys.exit(1)

    reports_dir = args.output.parent / "reports"

    # ── Run external validation ────────────────────────────────────────
    try:
        from src.inference.external_validation_runner import ExternalValidationRunner
    except ImportError as _ie:
        logger.error(
            "ExternalValidationRunner yuklenemedi: %s\n"
            "Kontrol et: src/inference/external_validation_runner.py ve bagimliliklari.",
            _ie,
        )
        sys.exit(1)

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

    # ── Otomatik submission formatı doğrulama ─────────────────────────
    try:
        from src.scientific.submission_validator import SubmissionValidator
        validator = SubmissionValidator()
        report = validator.validate(submission_path=args.output)
        validator.print_report(report, verbose=True)
        if not report.passed:
            logger.error(
                "Submission validation FAILED — jüriye göndermeden önce düzeltin!"
            )
            sys.exit(1)
        logger.info("Submission doğrulama PASSED — jüri formatı uyumlu.")
    except Exception as val_exc:
        logger.warning("Submission validator çalışamadı (%s); manuel kontrol edin.", val_exc)


if __name__ == "__main__":
    main()

"""src/cli/runner.py — CLI dispatch and entry point."""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=Warning, module="urllib3.*")
warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*", category=FutureWarning)

from src.cli.parser import build_parser  # noqa: E402
from src.utils.logging_cfg import setup_logging  # noqa: E402


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(log_file=args.log_file)

    from src.config import get_settings, reset_settings

    if args.config:
        reset_settings()
        cfg = get_settings(args.config)
    else:
        cfg = get_settings()

    logging.info("=" * 60)
    logging.info("VARIANT-GNN | mode=%s | seed=%d", args.mode.upper(), cfg.seed)
    logging.info("=" * 60)
    logging.info("TEKNOFEST 2026: NDA — competition data must not be shared.")
    logging.info("=" * 60)

    # §7.5 reproducibility: seed ALL RNGs (python/numpy/torch) + determinism flags at
    # the start of EVERY CLI mode. Previously never invoked in any CLI path, so
    # MC-Dropout inference drifted run-to-run (could change ~199/200 jury rows).
    from src.utils.reproducibility import setup_reproducibility

    _det = True
    try:
        _det = bool(cfg.reproducibility.deterministic_torch)
    except Exception:
        pass
    setup_reproducibility(cfg.seed, deterministic_torch=_det)

    # Lock ClinVar API during training/inference — §3.2 genomic address restriction
    from src.explainability.clinvar_api import set_inference_mode

    set_inference_mode(True)

    from src.cli.modes.evaluate import mode_adversarial_val, mode_crossval, mode_eval, mode_external_val
    from src.cli.modes.explain import mode_explain
    from src.cli.modes.predict import mode_predict
    from src.cli.modes.science import mode_ablation, mode_label_quality, mode_panel_transfer, mode_tune
    from src.cli.modes.train import mode_train, mode_train_panels

    dispatch = {
        "train": mode_train,
        "tune": mode_tune,
        "eval": mode_eval,
        "predict": mode_predict,
        "crossval": mode_crossval,
        "external_val": mode_external_val,
        "adversarial_val": mode_adversarial_val,
        "train_panels": mode_train_panels,
        "explain": mode_explain,
        "panel_transfer": mode_panel_transfer,
        "label_quality": mode_label_quality,
        "ablation": mode_ablation,
    }
    dispatch[args.mode](args, cfg)

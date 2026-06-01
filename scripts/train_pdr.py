"""
scripts/train_pdr.py
TEKNOFEST 2026 PDR aşaması için tam yarışma eğitim betiği.

Entegre modüller:
  - CompetitionSanitizer  : leakage firewall (koordinat + label kolon engeli)
  - PanelAwareEnsemble    : OOF tabanlı per-panel ağırlık optimizasyonu
  - ArtifactManifest      : models/final/manifest.json üretimi
  - ReproducibilityLayer  : merkezi seed yönetimi

Kullanım:
  python scripts/train_pdr.py --data data/train_variants.csv --output models/final
  python scripts/train_pdr.py --data data/train_variants.csv --output models/final \\
      --config configs/pdr.yaml

Çıktılar (models/final/):
  preprocessor.pkl, ensemble.pkl, calibrator.pkl,
  feature_names.json, threshold.json, manifest.json,
  panel_ensemble_weights.json, reports/leakage_report_train.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.reproducibility import setup_reproducibility
from src.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger("scripts.train_pdr")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 PDR Training")
    parser.add_argument("--data", required=True, type=Path, help="Labelled training CSV")
    parser.add_argument(
        "--output", default=Path("models/final"), type=Path, help="Output model directory"
    )
    parser.add_argument(
        "--config", default=Path("configs/pdr.yaml"), type=Path, help="Config YAML"
    )
    parser.add_argument(
        "--panel", default=None, help="Filter to single panel (optional)"
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        logger.warning("Config not found at %s; using defaults.", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_config(pdr_cfg: dict) -> None:
    """Override project config with PDR-specific values."""
    from src.config import get_settings
    cfg = get_settings()
    repro = pdr_cfg.get("reproducibility", {})
    if "seed" in repro:
        cfg.seed = int(repro["seed"])


def main() -> None:
    args = _parse_args()
    pdr_cfg = _load_yaml(args.config)

    seed = pdr_cfg.get("reproducibility", {}).get("seed", 42)
    det_torch = pdr_cfg.get("reproducibility", {}).get("deterministic_torch", True)
    setup_reproducibility(seed=seed, deterministic_torch=det_torch)
    _merge_config(pdr_cfg)

    competition_mode = pdr_cfg.get("competition", {}).get("mode", True)
    strict_leakage = pdr_cfg.get("competition", {}).get("strict_leakage", False)
    panel_aware = pdr_cfg.get("ensemble", {}).get("panel_aware", True)
    save_manifest_flag = pdr_cfg.get("reproducibility", {}).get("save_manifest", True)

    logger.info(
        "PDR training started | seed=%d  competition_mode=%s  panel_aware=%s",
        seed, competition_mode, panel_aware,
    )

    # ── Validate input path ────────────────────────────────────────────
    if not args.data.exists():
        logger.error("Training CSV not found: %s", args.data)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    from src.data.loader import load_csv
    from src.config import get_settings

    cfg = get_settings()
    dataset = load_csv(args.data)
    logger.info(
        "Loaded: shape=%s  label_dist=%s",
        dataset.features.shape,
        dict(zip(*np.unique(dataset.labels, return_counts=True))) if dataset.labels is not None else "N/A",
    )

    # ── Leakage Firewall ───────────────────────────────────────────────
    from src.data.competition_sanitizer import CompetitionSanitizer
    from src.utils.fingerprinting import dataframe_fingerprint

    raw_df = dataset.features.copy()
    if dataset.labels is not None:
        raw_df[cfg.schema.target_column] = dataset.labels

    sanitizer = CompetitionSanitizer(
        competition_mode=competition_mode,
        strict_leakage=strict_leakage,
        reports_dir=reports_dir,
    )
    clean_df, leakage_report = sanitizer.sanitize_train(raw_df)
    if not leakage_report.is_clean:
        logger.warning("Leakage issues resolved: %s", leakage_report.coordinate_hits)

    train_fingerprint = dataframe_fingerprint(clean_df)
    logger.info("Training data fingerprint: %s…", train_fingerprint[:12])

    # ── Filter by panel ────────────────────────────────────────────────
    feature_cols = [
        c for c in clean_df.columns
        if c not in (cfg.schema.target_column, "Variant_ID", "Panel",
                     "Nuc_Context", "AA_Context")
        and c not in leakage_report.coordinate_hits
    ]
    X = clean_df[feature_cols].values.astype(float)
    y = dataset.labels.astype(int) if dataset.labels is not None else None

    if y is None:
        logger.error("No labels found in training CSV. Training requires labelled data.")
        sys.exit(1)

    panels = (
        dataset.metadata["Panel"].values
        if "Panel" in dataset.metadata.columns
        else np.array(["General"] * len(X))
    )

    if args.panel:
        mask = panels == args.panel
        X, y, panels = X[mask], y[mask], panels[mask]
        logger.info("Filtered to panel '%s': %d samples", args.panel, len(X))

    nuc_seqs = (
        dataset.nuc_sequences
        if hasattr(dataset, "nuc_sequences") and dataset.nuc_sequences is not None
        else None
    )
    aa_seqs = (
        dataset.aa_sequences
        if hasattr(dataset, "aa_sequences") and dataset.aa_sequences is not None
        else None
    )

    # ── Train ──────────────────────────────────────────────────────────
    from src.training.trainer import VariantTrainer

    trainer = VariantTrainer()
    result = trainer.train(X, y, nuc_seqs=nuc_seqs, aa_seqs=aa_seqs)
    logger.info(
        "Training complete: mean CV F1=%.4f ± %.4f",
        result.mean_cv_f1, result.std_cv_f1,
    )

    # ── Calibrate ──────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from src.scientific.calibration.calibrator import EnsembleCalibrator
    from src.features.preprocessing import build_preprocessor_from_config

    idx = np.arange(len(X))
    idx_tr, idx_cal = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    preprocessor_cal = build_preprocessor_from_config()
    X_cal_proc = result.preprocessor.transform(X[idx_cal])
    _, raw_proba_cal = result.ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_proba_cal, y[idx_cal])
    logger.info("Calibrator fitted (method=%s).", cfg.calibration.method)

    # ── OOF predictions for panel-aware ensemble (GROUP-AWARE, de-faked) ──
    # NOT: panel_ensemble_weights.json KANONİK DEĞİL — src/ hiçbir yerde kullanmaz.
    # Kanonik 4-model ağırlıkları group-aware train.py OOF-inner-val yolundan gelir.
    # Önceki sürüm (a) StratifiedKFold (group-aware DEĞİL) ve (b) GNN/DNN kolonlarını
    # tüm-veri ensemble proxy'siyle SAHTE dolduruyordu — ikisi de sızıntıydı, kaldırıldı.
    # Bu yan-artefakt artık yalnızca group-aware GERÇEK tree-member (XGB+LGBM) OOF kullanır.
    if panel_aware:
        from sklearn.model_selection import StratifiedGroupKFold
        from src.ensemble.panel_aware_ensemble import PanelAwareEnsemble
        import xgboost as _xgb

        base_ids = (
            dataset.metadata["Variant_ID"].astype(str)
            .str.replace(r"_aug\d*$", "", regex=True).values
        )
        if getattr(args, "panel", None):  # panel maskesi gruplara da uygulanır
            base_ids = base_ids[mask]

        n_models = 2  # GERÇEK out-of-fold: XGB, LGBM (sahte GNN/DNN proxy yok)
        sgkf = StratifiedGroupKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=seed)
        oof_preds = np.zeros((len(X), n_models))

        for tr_idx, val_idx in sgkf.split(X, y, base_ids):
            prep_fold = build_preprocessor_from_config()
            X_tr_p, y_tr_p = prep_fold.fit_resample_train(X[tr_idx], y[tr_idx])
            X_val_p = prep_fold.transform(X[val_idx])

            xgb_fold = _xgb.XGBClassifier(**cfg.xgb.as_dict())
            xgb_fold.fit(X_tr_p, y_tr_p, verbose=False)
            oof_preds[val_idx, 0] = xgb_fold.predict_proba(X_val_p)[:, 1]
            try:
                import lightgbm as _lgb
                lgb_fold = _lgb.LGBMClassifier(**cfg.lgbm.as_dict())
                lgb_fold.fit(X_tr_p, y_tr_p, callbacks=[_lgb.log_evaluation(-1)])
                oof_preds[val_idx, 1] = lgb_fold.predict_proba(X_val_p)[:, 1]
            except Exception:
                oof_preds[val_idx, 1] = oof_preds[val_idx, 0]

        panel_ensemble = PanelAwareEnsemble(n_models=n_models, seed=seed)
        panel_ensemble.fit(oof_preds, y, panels.tolist())
        panel_weights_path = args.output / "panel_ensemble_weights.json"
        panel_ensemble.save(panel_weights_path)
        logger.info("Panel-aware (group-aware, genuine XGB+LGBM OOF) weights -> %s", panel_weights_path)

    # ── F1-optimal threshold ───────────────────────────────────────────
    from src.scientific.metrics.metrics import find_best_threshold

    X_all_proc = result.preprocessor.transform(X)
    _, raw_proba_all = result.ensemble.predict(X_all_proc)
    threshold = find_best_threshold(y, raw_proba_all[:, 1])
    logger.info("F1-optimal threshold: %.4f", threshold)

    # ── Save artifacts ─────────────────────────────────────────────────
    from src.utils.serialization import ModelStore

    store = ModelStore(args.output)
    store.save_all(result.preprocessor, result.ensemble, calibrator)
    store.save_threshold(threshold)

    feature_names_path = args.output / "feature_names.json"
    if not feature_names_path.exists():
        with open(feature_names_path, "w") as fh:
            json.dump(feature_cols, fh)
        logger.info("Feature names (%d) -> feature_names.json", len(feature_cols))

    # ── Manifest ───────────────────────────────────────────────────────
    if save_manifest_flag:
        from src.utils.artifact_manifest import save_manifest

        save_manifest(
            model_dir=args.output,
            config=pdr_cfg,
            dataset_fingerprints={"train": train_fingerprint},
            model_version="1.0.0",
        )

    logger.info("PDR training complete. Artifacts in %s", args.output)
    logger.info(
        "Run inference: python submission/predict.py "
        "--input data/blind_test.csv --model_dir %s --output submission/predictions.csv "
        "--config %s",
        args.output, args.config,
    )


if __name__ == "__main__":
    main()

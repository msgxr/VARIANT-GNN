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

from src.scientific.calibration.calibrator import EnsembleCalibrator
from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.scientific.metrics.metrics import evaluate, evaluate_per_panel, find_best_threshold
from src.scientific.metrics.plots import save_all_plots
from src.features.preprocessing import build_preprocessor_from_config
from src.api.export import export_predictions
from src.api.pipeline import InferencePipeline
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

    # Panel filtresi (isteğe bağlı)
    panel = getattr(args, "panel", None)
    if panel and "Panel" in ds.metadata.columns:
        mask = ds.metadata["Panel"] == panel
        valid_idx = mask[mask].index.tolist()
        from src.data.loader import LoadedDataset
        ds = LoadedDataset(
            features        = ds.features[mask].reset_index(drop=True),
            labels          = ds.labels[mask.values],
            metadata        = ds.metadata[mask].reset_index(drop=True),
            feature_columns = ds.feature_columns,
            nuc_sequences   = ([ds.nuc_sequences[i] for i in valid_idx]
                               if ds.nuc_sequences else None),
            aa_sequences    = ([ds.aa_sequences[i] for i in valid_idx]
                               if ds.aa_sequences else None),
        )
        logging.info("Panel filtresi: %s (%d varyant)", panel, len(ds.labels))

    X   = ds.features.values
    y   = ds.labels
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    # Pass sequences for multimodal SequenceEncoder (TEKNOFEST şartname uyumu)
    trainer = VariantTrainer()
    result  = trainer.train(X, y, nuc_seqs=ds.nuc_sequences, aa_seqs=ds.aa_sequences)
    logging.info("CV summary — Mean Binary F1 (Pathogenic, §7.3): %.4f +/- %.4f",
                 result.mean_cv_f1, result.std_cv_f1)

    preprocessor = result.preprocessor
    ensemble     = result.ensemble

    # ── Train / test split (must match trainer.train() internal split seed) ──
    all_indices = np.arange(len(X))
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )
    _, test_indices = train_test_split(
        all_indices, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )

    # ── Calibration split — from train portion only (no test leakage) ──
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=cfg.seed + 99
    )
    X_cal_proc  = preprocessor.transform(X_cal)
    _, raw_cal_proba = ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    # ── Test evaluation ──
    X_test_proc   = preprocessor.transform(X_test)
    _, raw_test_proba = ensemble.predict(X_test_proc)
    cal_test_proba    = calibrator.transform(raw_test_proba)

    best_thr, _ = find_best_threshold(y_test, cal_test_proba[:, 1], metric="f1")
    report       = evaluate(y_test, cal_test_proba, threshold=best_thr)
    report.log(prefix="TEST")

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)
    store.save_threshold(best_thr)

    save_all_plots(report, y_test, raw_test_proba, cfg.paths.reports_dir)

    # Per-panel evaluation (if Panel metadata available)
    panel_reports_dict = {}
    if "Panel" in ds.metadata.columns:
        test_panels = ds.metadata["Panel"].values[test_indices]
        panel_reports = evaluate_per_panel(y_test, cal_test_proba, test_panels, threshold=best_thr)
        for pname, prep in panel_reports.items():
            prep.log(prefix=f"PANEL_{pname}")
            panel_reports_dict[pname] = prep.as_dict()

    report_path = cfg.paths.reports_dir / "cv_report.json"
    with open(report_path, "w") as fh:
        json.dump({
            # TEKNOFEST 2026 §7.3: Temel sıralama metriği F1 Skoru (TP/FP/FN)
            "competition_metric": "F1 Score (TP/FP/FN, TEKNOFEST §7.3)",
            "note": "CV folds dengeli veri üzerinde eğitildi; Macro F1 ≈ Binary F1 (Pathogenic) for 50/50 data",
            "mean_cv_macro_f1":  result.mean_cv_f1,
            "std_cv_macro_f1":   result.std_cv_f1,
            "test_binary_f1":    report.binary_f1,
            "test_macro_f1":     report.macro_f1,
            "best_threshold":    best_thr,
            "folds": [vars(r) for r in result.fold_results],
            "test_metrics": report.as_dict(),
            "panel_metrics": panel_reports_dict,
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
    """Jüri modu: etiketlenmemiş CSV → 7-kolonlu deterministik submission CSV.

    Kullanım:
      python main.py --mode predict --config configs/final.yaml \
                     --test_file data/test_blind.csv \
                     --output submission/predictions.csv
    """
    if not args.test_file:
        logging.error("--test_file gerekli (predict modu).")
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("  VARIANT-GNN — JÜRİ TAHMİN MODU")
    logging.info("  Giriş : %s", args.test_file)
    logging.info("=" * 60)

    pipeline = InferencePipeline()
    pipeline.load()

    try:
        df_result = pipeline.predict_from_csv(args.test_file)
    except Exception as exc:
        logging.error("Tahmin hatası: %s", exc)
        sys.exit(1)

    cfg.paths.create_dirs()

    # --output argümanı varsa oraya yaz, yoksa submission/ varsayılan
    submission_path = getattr(args, "output", None) or "submission/predictions.csv"

    paths = export_predictions(
        df_result,
        cfg.paths.reports_dir,
        prefix="predictions",
        submission_path=submission_path,
    )
    logging.info("Jüri CSV (7-kolon) → %s", paths["jury"])
    logging.info("Tam CSV            → %s", paths["full"])
    logging.info("Submission CSV     → %s", paths["submission"])

    # İlk 10 satır ekrana bas
    from src.api.export import JURY_COLUMNS
    jury_df = df_result.copy()
    print("\n" + "=" * 60)
    print("  İLK 10 TAHMİN")
    print("=" * 60)
    preview_cols = [c for c in JURY_COLUMNS if c in df_result.columns]
    if not preview_cols:
        preview_cols = list(df_result.columns[:7])
    print(df_result[preview_cols].head(10).to_string(index=False))
    print("=" * 60)


def mode_crossval(args, cfg):
    ds      = _get_labelled_data(args.data_file, cfg)
    trainer = VariantTrainer()
    set_global_seed(cfg.seed)
    folds   = trainer._cross_validate(ds.features.values, ds.labels)
    mean_f1 = float(np.mean([r.f1 for r in folds]))
    std_f1  = float(np.std( [r.f1 for r in folds]))
    logging.info("Cross-val complete | Binary F1 (Pathogenic, §7.3) = %.4f +/- %.4f", mean_f1, std_f1)
    for r in folds:
        logging.info("  Fold %d | Ens=%.4f  XGB=%.4f  LGB=%.4f  GNN=%.4f  DNN=%.4f",
                     r.fold, r.f1, r.xgb_f1, getattr(r, 'lgbm_f1', 0.0), r.gnn_f1, r.dnn_f1)


def mode_external_val(args, cfg):
    """External validation — TEKNOFEST 2026 tam metrik paketi.

    F1, Precision, Recall, ROC-AUC, PR-AUC, MCC, Brier, ECE
    Confusion matrix PNG + JSON raporu reports/ altına kaydedilir.
    """
    from src.evaluation.metrics import evaluate, find_f1_optimal_threshold, save_threshold_report
    
    test_path = args.test_file or args.data_file
    if not test_path:
        logging.error("--test_file veya --data_file gerekli.")
        sys.exit(1)
    
    ds = load_csv(test_path)
    if ds.labels is None:
        logging.error("external_val icin etiketli veri gerekli.")
        sys.exit(1)

    pipeline = InferencePipeline()
    pipeline.load()
    df_result = pipeline.predict_from_dataset(ds)

    y_true = ds.labels
    p1     = df_result["Probability"].values
    y_prob = np.column_stack([1 - p1, p1])

    # 1. Optimize threshold for F1 on this test set
    best_thr, best_f1 = find_f1_optimal_threshold(y_true, p1, metric="f1")
    
    # 2. Comprehensive evaluation
    report = evaluate(y_true, y_prob, threshold=best_thr)
    report.log(prefix="EXTERNAL_VAL")

    # 3. Save threshold report and PLOTS
    cfg.paths.create_dirs()
    save_threshold_report(y_true, p1, reports_dir=str(cfg.paths.reports_dir))

    # 4. Save Confusion Matrix Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = report.conf_matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Pathogenic"])
        ax.set_yticklabels(["Benign", "Pathogenic"])
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.set_title(f"Confusion Matrix | F1={report.binary_f1:.4f}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
        plt.tight_layout()
        cm_path = cfg.paths.reports_dir / "external_val_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Confusion Matrix -> %s", cm_path)
    except Exception as exc:
        logging.warning("Grafik çizilemedi: %s", exc)

    # 5. Export Predictions
    paths = export_predictions(
        df_result, 
        cfg.paths.reports_dir, 
        prefix="external_val",
        submission_path=getattr(args, "output", None)
    )
    
    # 6. JSON Summary
    report_json = {
        "mode": "external_validation",
        "test_file": str(test_path),
        "n_samples": int(len(y_true)),
        "optimal_threshold": best_thr,
        "metrics": report.as_dict(),
        "confusion_matrix": report.conf_matrix.tolist() if report.conf_matrix is not None else None,
    }
    out_json = cfg.paths.reports_dir / "external_validation_report.json"
    with open(out_json, "w") as fh:
        json.dump(report_json, fh, indent=2, default=str)
    
    logging.info("JSON rapor -> %s", out_json)
    logging.info("External validation tamamlandı.")



def mode_adversarial_val(args, cfg):
    """Adversarial validation — train/test domain shift testii.

    Train ve test verisi aras\u0131ndaki da\u011f\u0131l\u0131m fark\u0131n\u0131 tespit eder.
    AUC ~0.5 \u2192 fark yok (iyi). AUC ~1.0 \u2192 ciddi shift var (k\u00f6t\u00fc).
    """
    from src.scientific.metrics.adversarial_validation import adversarial_validate

    if not args.data_file or not args.test_file:
        logging.error("--data_file (train) ve --test_file (test) ikisi de gerekli.")
        sys.exit(1)

    ds_train = load_csv(args.data_file)
    ds_test = load_csv(args.test_file)

    result = adversarial_validate(
        X_train=ds_train.features.values,
        X_test=ds_test.features.values,
        feature_names=ds_train.feature_columns,
    )

    # Sonu\u00e7lar\u0131 kaydet
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

    logging.info("Adversarial validation report -> %s", out_json)
    print(f"\nSonu\u00e7: {result.verdict}")
    print(f"AUC: {result.auc_mean:.4f} \u00b1 {result.auc_std:.4f}")
    if result.top_shift_features:
        print(f"En \u00e7ok shift g\u00f6steren \u00f6znitelikler: {', '.join(result.top_shift_features[:5])}")


def mode_train_panels(args, cfg):
    """Train a single unified model on ALL panels, then evaluate per-panel.

    TEKNOFEST Şartname Bölüm 3.2: 4 panel birleşik veri seti.
    Strateji: Tek model, paneller arası genelleme yeteneği sağlar.
    Küçük paneller (CFTR: 70+70) ayrı model eğitmek için yetersiz.
    """
    ds = _get_labelled_data(args.data_file, cfg)

    X   = ds.features.values
    y   = ds.labels
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    trainer = VariantTrainer()
    result  = trainer.train(X, y, nuc_seqs=ds.nuc_sequences, aa_seqs=ds.aa_sequences)
    logging.info("CV summary — Mean Binary F1 (Pathogenic, §7.3): %.4f +/- %.4f",
                 result.mean_cv_f1, result.std_cv_f1)

    preprocessor = result.preprocessor
    ensemble     = result.ensemble

    # Calibration — from train portion only (no test leakage)
    X_tr_panels, _, y_tr_panels, _ = train_test_split(
        X, y, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
    )
    _, X_cal, _, y_cal = train_test_split(
        X_tr_panels, y_tr_panels, test_size=0.15, stratify=y_tr_panels, random_state=cfg.seed + 99
    )
    X_cal_proc = preprocessor.transform(X_cal)
    _, raw_cal_proba = ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)

    # Per-panel evaluation on each panel's train set (validation-split)
    if "Panel" in ds.metadata.columns:
        panels = ds.metadata["Panel"].unique()
        panel_summary = {}
        for panel_name in panels:
            mask = ds.metadata["Panel"] == panel_name
            X_p, y_p = X[mask.values], y[mask.values]
            if len(y_p) < 10:
                logging.warning("Panel %s: too few samples (%d), skipping.", panel_name, len(y_p))
                continue
            X_p_proc = preprocessor.transform(X_p)
            _, proba_p = ensemble.predict(X_p_proc)
            cal_proba_p = calibrator.transform(proba_p)
            report_p = evaluate(y_p, cal_proba_p)
            report_p.log(prefix=f"PANEL_{panel_name}")
            panel_summary[panel_name] = {
                "n_samples": int(len(y_p)),
                "metrics": report_p.as_dict(),
            }

        report_path = cfg.paths.reports_dir / "panel_evaluation.json"
        with open(report_path, "w") as fh:
            json.dump(panel_summary, fh, indent=2, default=str, ensure_ascii=False)
        logging.info("Panel evaluation report -> %s", report_path)
    else:
        logging.warning("No 'Panel' column found — skipping per-panel evaluation.")

    logging.info("train_panels mode complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def mode_explain(args, cfg):
    """Açıklanabilirlik modu — PSR §4.4.

    Eğitilmiş modeli yükler, seçilen örnekler üzerinde:
      1. SHAP analizi (XGBoost TreeExplainer)
      2. Grup düzeyinde biyolojik katkı tablosu (6 kategori)
      3. Türkçe klinik açıklama metni
      4. GNN öğrenme eğrisi JSON'u çizer (eğer varsa)
    çıktıları reports/ altına kaydeder.

    Kullanım:
      python main.py --mode explain --data_file data/train_variants.csv
      python main.py --mode explain --test_file data/test_variants.csv
    """
    from src.explainability.shap_explainer import SHAPExplainer
    from src.explainability.group_shap import group_shap_report, instance_explanation_tr

    data_path = args.data_file or args.test_file
    if not data_path:
        logging.error("--data_file veya --test_file gerekli (explain modu).")
        import sys; sys.exit(1)

    ds = load_csv(data_path)
    pipeline = InferencePipeline()
    pipeline.load()

    cfg.paths.create_dirs()

    preprocessor = pipeline._preprocessor
    ensemble     = pipeline._ensemble

    # ── 1. Ön işleme ────────────────────────────────────────────────────────
    X_np     = ds.features.values
    X_scaled = preprocessor.transform(X_np)

    # ── 2. SHAP analizi (XGBoost üzerinden) ─────────────────────────────────
    feature_names = list(ds.feature_columns)
    explainer = SHAPExplainer(
        xgb_model     = ensemble.xgb,
        feature_names = feature_names,
        training_data = X_scaled,
    )

    sample_size = min(len(X_scaled), 200)
    X_sample    = X_scaled[:sample_size]

    shap_vals = explainer.explain_instance(X_sample)
    if shap_vals is not None:
        if isinstance(shap_vals, list):
            import numpy as _np; shap_vals = _np.array(shap_vals[1])
        # Özet plot
        explainer.plot_summary(X_sample, str(cfg.paths.reports_dir / "shap_summary.png"))
        # İlk örnek waterfall
        explainer.plot_waterfall(X_sample[0], str(cfg.paths.reports_dir / "shap_waterfall_sample0.png"))

        # ── 3. Grup SHAP raporu (PSR §4.4 — 6 biyolojik kategori) ─────────
        grp_result = group_shap_report(
            shap_values   = shap_vals,
            feature_names = feature_names,
            output_dir    = cfg.paths.reports_dir,
            plot          = True,
        )
        logging.info("Grup SHAP JSON → %s", grp_result.get("json_path"))
        logging.info("Grup SHAP plot → %s", grp_result.get("plot_path"))

        # ── 4. GNNExplainer analizi (SHAP'ın hemen ardından) ────────────────
        try:
            from src.explainability.gnn_explainer import GNNExplainerWrapper
            from src.core.graph.builder import SampleKNNGraphBuilder
            import torch as _torch

            gnn_model = ensemble.gnn
            if gnn_model is not None:
                _device = next(gnn_model.parameters()).device
                logging.info("GNNExplainer başlatılıyor…")
                _gnn_exp = GNNExplainerWrapper(model=gnn_model, device=_device)

                knn_k     = getattr(cfg.gnn, "knn_k", 10)
                explain_n = min(30, len(X_scaled))
                _graph    = SampleKNNGraphBuilder(k=knn_k).build(
                    X_scaled[:explain_n], y=None
                )
                _expl = _gnn_exp.explain(_graph)

                if _expl is not None:
                    import json as _json
                    _gnn_out: dict = {"n_explained": explain_n, "knn_k": knn_k}
                    if hasattr(_expl, "node_mask") and _expl.node_mask is not None:
                        _nm = _expl.node_mask.detach().cpu().numpy()
                        _feat_imp = (_nm.mean(axis=0).tolist()
                                     if _nm.ndim == 2 else _nm.tolist())
                        _gnn_out["node_feature_importance"] = _feat_imp
                        _gnn_out["top_gnn_features"] = [
                            {"feature": f, "importance": round(v, 6)}
                            for f, v in sorted(
                                zip(feature_names[:len(_feat_imp)], _feat_imp),
                                key=lambda x: abs(x[1]), reverse=True
                            )[:10]
                        ]
                    if hasattr(_expl, "edge_mask") and _expl.edge_mask is not None:
                        _em = _expl.edge_mask.detach().cpu().numpy()
                        _gnn_out["edge_importance_mean"] = float(_em.mean())
                        _gnn_out["edge_importance_max"]  = float(_em.max())

                    _gnn_exp_path = cfg.paths.reports_dir / "gnn_explainer_results.json"
                    with open(_gnn_exp_path, "w", encoding="utf-8") as _fh:
                        _json.dump(_gnn_out, _fh, indent=2, ensure_ascii=False)
                    logging.info("GNNExplainer sonuçları → %s", _gnn_exp_path)
                else:
                    logging.warning(
                        "GNNExplainer açıklama üretemedi "
                        "(kütüphane uyumsuzluğu veya model uyuşmazlığı)."
                    )
            else:
                logging.warning("GNN modeli yüklü değil — GNNExplainer atlandı.")
        except Exception as _gnn_exc:
            logging.warning("GNNExplainer çalıştırılamadı: %s", _gnn_exc)

        # ── 5. İlk 5 örnek için Türkçe açıklama ────────────────────────────
        df_pred = pipeline.predict_from_dataset(ds)
        explain_records = []
        for i in range(min(5, len(X_sample))):
            vid      = df_pred["Variant_ID"].iloc[i] if "Variant_ID" in df_pred.columns else str(i)
            pred     = df_pred["Prediction"].iloc[i]
            prob     = float(df_pred["Probability"].iloc[i])
            sv_i     = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
            text     = instance_explanation_tr(sv_i, feature_names, pred, prob, vid)
            logging.info("[Örnek %d] %s", i, text)
            explain_records.append({"variant_id": vid, "prediction": pred,
                                    "probability": prob, "clinical_insight_tr": text})

        import json as _json
        out_json = cfg.paths.reports_dir / "explain_instances.json"
        with open(out_json, "w", encoding="utf-8") as _fh:
            _json.dump(explain_records, _fh, indent=2, ensure_ascii=False)
        logging.info("Açıklama örnekleri → %s", out_json)

        # ── 7. Türkçe PDF Klinik Raporu (Rapor §4.4) ─────────────────────
        try:
            from src.explainability.pdf_report import (
                generate_pdf_report, FPDF_AVAILABLE,
            )
            if not FPDF_AVAILABLE:
                logging.warning(
                    "fpdf2 kurulu değil — PDF rapor atlanıyor. "
                    "`pip install fpdf2` ile kurun."
                )
            elif explain_records:
                _pdf_paths = []
                for _i, _rec in enumerate(explain_records):
                    _vid  = _rec["variant_id"]
                    _pred = _rec["prediction"]
                    _prob = float(_rec["probability"])

                    # Risk skoru (0-100)
                    _risk = (
                        float(df_pred["Calibrated_Risk"].iloc[_i])
                        if "Calibrated_Risk" in df_pred.columns
                        else _prob * 100
                    )

                    # Zone etiketi
                    if _risk >= 75:
                        _zone = "KRITIK RISK - Klinik Degerlendirme Gereklidir"
                    elif _risk >= 50:
                        _zone = "YUKSEK RISK - Uzman Gorusu Onerilir"
                    elif _risk >= 25:
                        _zone = "ORTA RISK - Takip Onerilir"
                    else:
                        _zone = "DUSUK RISK - Muhtemelen Benign"

                    _rec_text = _rec.get("clinical_insight_tr", "")
                    _recom = (
                        "Genetik uzmanı ile konsültasyon ve ek fonksiyonel "
                        "çalışmalar önerilir."
                        if _pred == "Pathogenic"
                        else "Rutin klinik takip yeterli olabilir. "
                             "Klinisyen değerlendirmesi esastır."
                    )
                    _clinical_insight = {
                        "zone_label": _zone,
                        "summary": _rec_text,
                        "key_findings": [],
                        "recommendation": _recom,
                    }

                    # Top SHAP features for this instance
                    _sv_i = shap_vals[_i] if shap_vals.ndim == 2 else shap_vals
                    _top_feats = sorted(
                        zip(feature_names, _sv_i.tolist()),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:10]

                    _waterfall = (
                        str(cfg.paths.reports_dir / "shap_waterfall_sample0.png")
                        if _i == 0 else None
                    )

                    _pdf_bytes = generate_pdf_report(
                        variant_id=str(_vid),
                        risk_score=_risk,
                        prediction=_pred,
                        probability=_prob,
                        clinical_insight=_clinical_insight,
                        top_features=_top_feats,
                        shap_waterfall_path=_waterfall,
                    )
                    _safe_vid = str(_vid).replace("/", "_").replace(":", "_")[:40]
                    _pdf_path = cfg.paths.reports_dir / f"clinical_report_{_safe_vid}.pdf"
                    _pdf_path.write_bytes(_pdf_bytes)
                    _pdf_paths.append(str(_pdf_path))
                    logging.info("PDF klinik raporu → %s", _pdf_path)

                logging.info(
                    "%d PDF klinik raporu oluşturuldu → %s",
                    len(_pdf_paths), cfg.paths.reports_dir,
                )
        except Exception as _pdf_exc:
            logging.warning("PDF rapor üretilemedi: %s", _pdf_exc)

    else:
        logging.warning("SHAP explainer başlatılamadı — shap kütüphanesi kurulu mu?")

    # ── 5. Öğrenme eğrisi JSON görselleştirmesi ──────────────────────────────
    lc_path = cfg.paths.reports_dir / "gnn_learning_curve.json"
    if lc_path.exists():
        try:
            import json as _json, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            with open(lc_path) as _f:
                lc_data = _json.load(_f)
            last_run = lc_data[-1]["run_epochs"] if lc_data else []
            if last_run:
                epochs    = [e["epoch"] for e in last_run]
                train_f1s = [e.get("train_f1", 0) for e in last_run]
                val_f1s   = [e.get("val_f1", None) for e in last_run]

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(epochs, train_f1s, label="Eğitim F1", color="#3182ce", linewidth=2)
                if any(v is not None for v in val_f1s):
                    ax.plot(epochs, [v if v else 0 for v in val_f1s],
                            label="Doğrulama F1", color="#e53e3e", linewidth=2,
                            linestyle="--")
                    # Erken durdurma noktası
                    for e in last_run:
                        if e.get("early_stop"):
                            ax.axvline(e["epoch"], color="gray", linestyle=":",
                                       label=f"Erken Durdurma (epoch {e['epoch']})")
                            break
                ax.set_xlabel("Epoch", fontsize=11)
                ax.set_ylabel("Macro F1 Skoru", fontsize=11)
                ax.set_title("GNN Öğrenme Eğrisi (PSR §4.5)", fontsize=12, fontweight="bold")
                ax.legend()
                ax.set_ylim(0.5, 1.05)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                lc_plot = cfg.paths.reports_dir / "gnn_learning_curve.png"
                plt.savefig(lc_plot, dpi=150, bbox_inches="tight")
                plt.close()
                logging.info("Öğrenme eğrisi grafiği → %s", lc_plot)
        except Exception as _lc_exc:
            logging.warning("Öğrenme eğrisi grafiği çizilemedi: %s", _lc_exc)
    else:
        logging.info("gnn_learning_curve.json bulunamadı — önce eğitim yapın.")

    # ── 9. ACMG Kriter Haritalayıcı ─────────────────────────────────────────
    try:
        from src.scientific.acmg_mapper import ACMGMapper
        _acmg = ACMGMapper()
        _acmg_results = _acmg.classify_batch(
            X_sample[:min(5, len(X_sample))],
            shap_matrix = shap_vals[:min(5, len(X_sample))] if (shap_vals is not None and shap_vals.ndim == 2) else None,
            feature_names = feature_names,
        )
        import json as _json
        _acmg_path = cfg.paths.reports_dir / "acmg_criteria.json"
        with open(_acmg_path, "w", encoding="utf-8") as _fh:
            _json.dump(_acmg_results, _fh, indent=2, ensure_ascii=False)
        logging.info("ACMG kriter haritası → %s", _acmg_path)
        for _i, _res in enumerate(_acmg_results):
            logging.info(
                "  [Örnek %d] %s (puan=%+d) — Kriterler: %s",
                _i, _res["classification"], _res["acmg_score"],
                [c["code"] for c in _res["criteria"]],
            )
    except Exception as _acmg_exc:
        logging.warning("ACMG haritalayıcı çalıştırılamadı: %s", _acmg_exc)

    # ── 10. OOD / Data Drift Tespiti ─────────────────────────────────────────
    try:
        from src.scientific.ood_detector import OODDetector
        _ood = OODDetector(z_threshold=3.0, ood_frac_thresh=0.20)
        _ood.fit(X_scaled)                        # Eğitim seti ile kalibre et
        _ood_report = _ood.detect(X_sample, feature_names=feature_names)
        _drift_rpt  = _ood.drift_report(X_sample, feature_names=feature_names)
        import json as _json
        _ood_path = cfg.paths.reports_dir / "ood_drift_report.json"
        with open(_ood_path, "w", encoding="utf-8") as _fh:
            _json.dump({
                "ood_summary":   _ood_report["summary"],
                "n_ood":         _ood_report["n_ood"],
                "n_total":       _ood_report["n_total"],
                "drift_score":   _drift_rpt["mean_drift_score"],
                "drift_flag":    _drift_rpt["drift_flag"],
                "top_drifted":   _drift_rpt["top_drifted_features"],
                "ood_features_per_sample": [
                    {"sample": _i, "ood_features": _f}
                    for _i, _f in enumerate(_ood_report["ood_features"][:5])
                ],
            }, _fh, indent=2, ensure_ascii=False)
        logging.info("OOD/Drift raporu → %s | %s", _ood_path, _ood_report["summary"])
    except Exception as _ood_exc:
        logging.warning("OOD detektörü çalıştırılamadı: %s", _ood_exc)

    # ── 11. PubMed RAG (ilk örnek için) ──────────────────────────────────────
    try:
        from src.scientific.pubmed_rag import PubMedRAG
        _rag = PubMedRAG(cache_ttl=3600)
        # İlk açıklanan varyantın gen adını tahmin et
        _first_vid = explain_records[0]["variant_id"] if explain_records else "BRCA1"
        _gene_guess = str(_first_vid).split("-")[0].split("_")[0]
        _articles   = _rag.fetch(gene=_gene_guess, n_results=3)
        if _articles:
            import json as _json
            _rag_path = cfg.paths.reports_dir / "pubmed_references.json"
            with open(_rag_path, "w", encoding="utf-8") as _fh:
                _json.dump(_articles, _fh, indent=2, ensure_ascii=False)
            logging.info(
                "PubMed RAG → %s makalesi → %s", len(_articles), _rag_path
            )
            for _art in _articles:
                logging.info("  [PubMed] %s (%s) — %s", _art["title"][:60], _art["year"], _art["url"])
        else:
            logging.info("PubMed: '%s' için sonuç bulunamadı (ağ erişimi gereklidir).", _gene_guess)
    except Exception as _rag_exc:
        logging.warning("PubMed RAG çalıştırılamadı: %s", _rag_exc)

    logging.info("Explain modu tamamlandı. Çıktılar: %s", cfg.paths.reports_dir)


def build_parser():
    p = argparse.ArgumentParser(
        description="VARIANT-GNN: Graph-based Variant Pathogenicity Prediction"
    )
    p.add_argument("--mode",
                   choices=["train", "tune", "eval", "predict", "crossval",
                            "external_val", "adversarial_val", "train_panels", "explain"],
                   default="train")
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--config",    type=str, default=None)
    p.add_argument("--n_trials",  type=int, default=30)
    p.add_argument("--log_file",  type=str, default=None)
    p.add_argument("--panel",  type=str, default=None,
                   help="Panel filtresi: General, Hereditary_Cancer, PAH, CFTR")
    p.add_argument("--output", type=str, default=None,
                   help="Submission CSV çıktı yolu (predict modu için)")
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
    logging.info("⚠️ TEKNOFEST 2026: Gizlilik Sözleşmesi (NDA) imzalanmadan")
    logging.info("T.C. Sağlık Bakanlığı/TÜSEB verilerinin kullanılması yasaktır.")
    logging.info("=" * 60)
    # ── TEKNOFEST Şartname: ClinVar API'yi eğitim/tahmin sırasında kilitle ──
    from src.explainability.clinvar_api import set_inference_mode
    set_inference_mode(True)

    dispatch = {
        "train":            mode_train,
        "tune":             mode_tune,
        "eval":             mode_eval,
        "predict":          mode_predict,
        "crossval":         mode_crossval,
        "external_val":     mode_external_val,
        "adversarial_val":  mode_adversarial_val,
        "train_panels":     mode_train_panels,
        "explain":          mode_explain,
    }
    dispatch[args.mode](args, cfg)


if __name__ == "__main__":
    main()

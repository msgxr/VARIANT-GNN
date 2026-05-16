"""
scripts/cross_panel_eval.py
============================
Leave-One-Panel-Out (LOPO) cross-validation:
  - Sırayla her paneli "test" olarak ayırır,
  - Kalan 3 panelde model eğitir,
  - Tutulan panelde değerlendirir.

Bu test, gerçek jüri verisi geldiğinde modelin "görmediği bir alt grup"
üzerinde nasıl davranacağını ölçer (domain-shift simulation).

Çıktı:
  reports/cross_panel_eval.json
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

# Kütüphane uyarıları sustur
warnings.filterwarnings("ignore", category=Warning, module="urllib3.*")
warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.*")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings, reset_settings
from src.data.loader import load_csv, LoadedDataset
from src.training.trainer import VariantTrainer
from src.utils.seeds import set_global_seed

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_panel(ensemble, preprocessor, X_test_raw: pd.DataFrame, y_test: np.ndarray):
    """Predict on a held-out panel and compute metrics."""
    X_test_proc = preprocessor.transform(X_test_raw.values if hasattr(X_test_raw, "values") else X_test_raw)
    preds, probs = ensemble.predict(X_test_proc, threshold=0.5)

    return {
        "n_samples": int(len(y_test)),
        "n_pos": int(y_test.sum()),
        "n_neg": int(len(y_test) - y_test.sum()),
        "binary_f1": float(f1_score(y_test, preds, average="binary", pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs[:, 1])) if len(np.unique(y_test)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def main() -> int:
    reset_settings()
    cfg = get_settings(str(REPO_ROOT / "configs" / "pdr.yaml"))
    set_global_seed(cfg.seed)

    ds = load_csv(str(REPO_ROOT / "data" / "train_variants.csv"))
    if "Panel" not in ds.metadata.columns:
        logger.error("Panel kolonu bulunamadı.")
        return 1

    panels = ["General", "Hereditary_Cancer", "PAH", "CFTR"]
    results = {}

    for held_out in panels:
        print(f"\n{'='*60}\nLeave-out panel: {held_out}\n{'='*60}")

        test_mask = (ds.metadata["Panel"] == held_out).values
        train_mask = ~test_mask

        X_full = ds.features
        y_full = ds.labels

        X_train = X_full.iloc[train_mask].reset_index(drop=True)
        y_train = y_full[train_mask]
        X_test  = X_full.iloc[test_mask].reset_index(drop=True)
        y_test  = y_full[test_mask]

        print(f"  Train: {len(y_train)} satır  "
              f"({(y_train==1).sum()} P / {(y_train==0).sum()} B)")
        print(f"  Test : {len(y_test)} satır  "
              f"({(y_test==1).sum()} P / {(y_test==0).sum()} B)")

        set_global_seed(cfg.seed)
        trainer = VariantTrainer()
        result = trainer.train(X_train.values, y_train)

        metrics = evaluate_panel(result.ensemble, result.preprocessor, X_test, y_test)
        metrics["cv_mean_f1"] = float(result.mean_cv_f1)
        metrics["cv_std_f1"]  = float(result.std_cv_f1)

        print(f"  → Test panel {held_out}:")
        print(f"     Binary F1   : {metrics['binary_f1']:.4f}")
        print(f"     Precision   : {metrics['precision']:.4f}")
        print(f"     Recall      : {metrics['recall']:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"     ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"     Train CV F1 : {metrics['cv_mean_f1']:.4f} ± {metrics['cv_std_f1']:.4f}")

        results[held_out] = metrics

    out_path = REPO_ROOT / "reports" / "cross_panel_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Rapor kaydedildi: {out_path}")

    print(f"\n{'='*60}\nÖZET — Leave-One-Panel-Out Cross-Eval\n{'='*60}")
    print(f"{'Panel':<25} {'n_test':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print("-" * 60)
    for panel, m in results.items():
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "n/a"
        print(f"{panel:<25} {m['n_samples']:>8} "
              f"{m['binary_f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {auc_str:>8}")

    f1s = [m["binary_f1"] for m in results.values()]
    print(f"\nOrtalama F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

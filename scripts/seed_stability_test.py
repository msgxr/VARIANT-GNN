"""
scripts/seed_stability_test.py
================================
Aynı eğitim pipeline'ını N farklı random seed ile çalıştırır ve
5-fold CV Binary F1 sonuçlarının istatistiklerini raporlar.

Amaç:
  - Modelin random seed'e duyarlılığını ölçmek
  - "F1 0.X ± 0.Y" şeklinde kararlılık göstergesi üretmek
  - PDR/jüri savunması için reproducibility kanıtı

Model artefaktları DİSKE KAYDEDİLMEZ — sadece F1 ölçümü yapılır.
Mevcut models/ klasörü etkilenmez.

Kullanım:
    python scripts/seed_stability_test.py --data data/train_variants_aug.csv

Çıktı:
    reports/seed_stability.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning, module="urllib3.*")
warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.training.trainer import VariantTrainer
from src.utils.seeds import set_global_seed

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_one_seed(seed: int, X, y, nuc_seqs=None, aa_seqs=None,
                 groups=None, panels=None) -> dict:
    """Tek seed ile crossval çalıştır, F1 sonuçlarını döndür.
    groups/panels geçilirse production pipeline ile birebir aynı (sızıntısız
    group-aware split + Domain-Adversarial DNN)."""
    reset_settings()
    cfg = get_settings(str(REPO_ROOT / "configs" / "pdr.yaml"))
    cfg.seed = seed
    if hasattr(cfg, "reproducibility"):
        cfg.reproducibility.seed = seed
    set_global_seed(seed)

    trainer = VariantTrainer()
    result = trainer.train(X, y, nuc_seqs=nuc_seqs, aa_seqs=aa_seqs,
                           groups=groups, panels=panels)

    return {
        "seed": seed,
        "cv_mean_f1": float(result.mean_cv_f1),
        "cv_std_f1": float(result.std_cv_f1),
        "fold_f1s": [float(f.f1) for f in result.fold_results],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "train_variants.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    parser.add_argument("--out",  default=str(REPO_ROOT / "reports" / "seed_stability.json"))
    args = parser.parse_args()

    print(f"Veri: {args.data}")
    print(f"Seed listesi: {args.seeds}")
    print()

    ds = load_csv(args.data)
    X = ds.features.values
    y = ds.labels
    # Group-aware (Variant_ID) + panel labels → production pipeline ile aynı
    groups = None
    panels = None
    if "Variant_ID" in ds.metadata.columns and len(ds.metadata) == len(X):
        groups = ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
    if "Panel" in ds.metadata.columns and len(ds.metadata) == len(X):
        panels = ds.metadata["Panel"].astype(str).values
    print(f"Veri boyutu: {X.shape}, sınıf dağılımı: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Group-aware: {groups is not None} | DANN panels: {panels is not None}")
    print()

    results = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"{'='*60}")
        print(f"  [{i}/{len(args.seeds)}] Seed = {seed}")
        print(f"{'='*60}")
        res = run_one_seed(seed, X, y, ds.nuc_sequences, ds.aa_sequences,
                           groups=groups, panels=panels)
        results.append(res)
        print(f"  → CV F1 = {res['cv_mean_f1']:.4f} ± {res['cv_std_f1']:.4f}")
        print(f"    Fold F1'leri: {[f'{f:.4f}' for f in res['fold_f1s']]}")
        print()

    all_cv_means = [r["cv_mean_f1"] for r in results]
    summary = {
        "seeds":            args.seeds,
        "individual_runs":  results,
        "overall_mean_f1":  float(np.mean(all_cv_means)),
        "overall_std_f1":   float(np.std(all_cv_means)),
        "overall_min_f1":   float(np.min(all_cv_means)),
        "overall_max_f1":   float(np.max(all_cv_means)),
        "n_seeds":          len(args.seeds),
        "data_file":        args.data,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{'='*60}")
    print(f"  ÖZET — {len(args.seeds)} Seed Kararlılık Testi")
    print(f"{'='*60}")
    print(f"  Ortalama CV F1: {summary['overall_mean_f1']:.4f}")
    print(f"  Std            : ±{summary['overall_std_f1']:.4f}")
    print(f"  Min / Max      : {summary['overall_min_f1']:.4f} / {summary['overall_max_f1']:.4f}")
    print(f"  Rapor          : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

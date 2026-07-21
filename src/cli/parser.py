# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""src/cli/parser.py — CLI argument parser for VARIANT-GNN."""

from __future__ import annotations

import argparse

MODES = [
    "train",
    "tune",
    "eval",
    "predict",
    "crossval",
    "external_val",
    "adversarial_val",
    "train_panels",
    "explain",
    "panel_transfer",
    "label_quality",
    "ablation",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VARIANT-GNN: Hybrid Graph Ensemble for Variant Pathogenicity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train --config configs/pdr.yaml
  python main.py --mode predict --test_file data/jury_test.csv
  python main.py --mode external_val --test_file data/test.csv
  python main.py --mode explain --data_file data/train_variants.csv
  python main.py --mode ablation --data_file data/train_variants.csv
        """,
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="train",
        help="Çalışma modu (varsayılan: train)",
    )
    p.add_argument("--data_file", type=str, default=None, help="Etiketli veri CSV yolu")
    p.add_argument("--test_file", type=str, default=None, help="Test/jüri CSV yolu (etiketsiz)")
    p.add_argument("--config", type=str, default=None, help="YAML config dosyası (varsayılan: configs/default.yaml)")
    p.add_argument("--n_trials", type=int, default=30, help="Optuna deneme sayısı (tune modu)")
    p.add_argument("--log_file", type=str, default=None, help="Log dosyası yolu")
    p.add_argument(
        "--panel",
        type=str,
        default=None,
        choices=["General", "Hereditary_Cancer", "PAH", "CFTR"],
        help="Panel filtresi (train modu için)",
    )
    p.add_argument("--output", type=str, default=None, help="Çıktı CSV yolu (predict/external_val için)")
    p.add_argument("--tta", action="store_true", default=False, help="Test-Time Augmentation etkinleştir")
    p.add_argument("--tta_k", type=int, default=10, help="TTA augmentation sayısı (varsayılan: 10)")
    return p

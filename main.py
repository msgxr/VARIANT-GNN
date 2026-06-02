"""main.py — VARIANT-GNN CLI entry point.

All mode logic lives in src/cli/. See src/cli/runner.py for dispatch,
src/cli/modes/ for individual mode implementations.

Usage:
  python main.py --mode train  --config configs/pdr.yaml
  python main.py --mode predict --test_file data/jury_test.csv
  python main.py --mode external_val --test_file data/test.csv
  python main.py --mode explain --data_file data/train_variants.csv
  python main.py --mode ablation --data_file data/train_variants.csv
  python main.py --help
"""

from src.cli.runner import main

if __name__ == "__main__":
    main()

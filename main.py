"""main.py — VARIANT-GNN CLI entry point.

All mode logic lives in src/cli/. See src/cli/runner.py for dispatch,
src/cli/modes/ for individual mode implementations.

Usage:
  variant-gnn                                  # ARGUMANSIZ -> interaktif menu (claude benzeri)
  python main.py --mode train  --config configs/pdr.yaml
  python main.py --mode predict --test_file data/jury_test.csv
  python main.py --mode external_val --test_file data/test.csv
  python main.py --mode explain --data_file data/train_variants.csv
  python main.py --mode ablation --data_file data/train_variants.csv
  python main.py --help
"""

from __future__ import annotations


def main() -> None:
    """Giris noktasi. NOT: konsol komutu `variant-gnn` artik src.cli.entry:main'i
    kullanir (editable-dostu); bu main.py ise `python main.py` icin korunur.

    Argumansiz  -> interaktif menu (claude benzeri). Agir paket import zincirine
                   (numpy/torch ...) GIRMEDEN acilir; boylece bagimliliklar henuz
                   kurulmamis olsa bile menu gorunur ve [6] Kurulum secilebilir.
    Argumanli   -> mevcut tek-atis CLI (src.cli.runner), davranis DEGISMEDEN.
    """
    import sys

    if len(sys.argv) == 1:
        from src.cli.interactive import run_interactive

        raise SystemExit(run_interactive())

    from src.cli.runner import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()

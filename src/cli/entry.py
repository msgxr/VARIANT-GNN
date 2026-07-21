# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""src/cli/entry.py — `variant-gnn` konsol komutunun giris noktasi.

pyproject.toml [project.scripts]: ``variant-gnn = "src.cli.entry:main"``.

Bu modul KASITLI olarak hafiftir: modul duzeyinde yalnizca standart kutuphane
import edilir. Agir bagimlilik zinciri (numpy / torch / xgboost ...) yalnizca
ARGUMANLI calistirmada, ``src.cli.runner`` ice aktarildiginda devreye girer.
Boylece argumansiz ``variant-gnn`` cagrisi, bagimliliklar henuz kurulmamis olsa
bile interaktif menuyu acabilir ([6] Kurulum bu yuzden menude erisilebilir kalir).

Neden src paketinin icinde? Editable kurulumda (pip install -e . /
pipx install -e .) ``src`` paketi repoya CANLI baglidir; boylece kaynak
duzenlemeleri aninda yansir. Kok duzeyindeki main.py'yi giris noktasi yapmak,
hatchling'in onu site-packages'a kopyalamasina ve global ad alanini
kirletmesine yol acardi.
"""

from __future__ import annotations


def main() -> None:
    import sys

    # Argumansiz  -> interaktif menu (claude benzeri); agir import zinciri YOK.
    if len(sys.argv) == 1:
        from src.cli.interactive import run_interactive

        raise SystemExit(run_interactive())

    # Argumanli   -> mevcut tek-atis CLI; davranis DEGISMEDEN.
    from src.cli.runner import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()

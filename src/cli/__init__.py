"""src.cli — VARIANT-GNN command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["main"]

if TYPE_CHECKING:  # yalnizca tip denetleyiciler icin; calisma zamaninda import YOK
    from src.cli.runner import main


def __getattr__(name: str):  # PEP 562 — tembel oznitelik erisimi
    # `from src.cli import main` calismaya devam eder, ANCAK src.cli paketini
    # (veya src.cli.interactive gibi hafif bir alt modulu) import etmek artik
    # agir bagimlilik zincirini (numpy/torch ...) ONCEDEN tetiklemez. Bu, argumansiz
    # `variant-gnn` menusunun bagimliliklar kurulmadan da acilmasini saglar.
    if name == "main":
        from src.cli.runner import main as _main

        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

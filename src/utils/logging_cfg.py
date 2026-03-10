"""
src/utils/logging_cfg.py
Central logging configuration.
Call ``setup_logging()`` once at application entry point.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level:      int            = logging.INFO,
    log_file:   Optional[str]  = None,
    fmt:        str            = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    date_fmt:   str            = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure root logger with console + optional file handler.

    Parameters
    ----------
    level    : Logging level (default INFO).
    log_file : Optional path to a log file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level    = level,
        format   = fmt,
        datefmt  = date_fmt,
        handlers = handlers,
        force    = True,
    )

    # Suppress noisy third-party loggers
    for noisy in ("torch", "torch_geometric", "optuna", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

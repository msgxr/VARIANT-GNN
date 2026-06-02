"""
src/utils/logging_cfg.py
Central logging configuration.
Call ``setup_logging()`` once at application entry point.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional


def _utf8_stdout() -> io.TextIOWrapper | io.TextIOBase:
    """UTF-8 stream over stdout — safe on Windows Turkish locale (cp1254)."""
    # Streamlit overrides stdout; re-wrapping it causes "I/O operation on closed file"
    if "streamlit" in sys.modules:
        return sys.stdout

    try:
        if hasattr(sys.stdout, "buffer"):
            return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass
    return sys.stdout


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure root logger with console + optional file handler.

    Parameters
    ----------
    level    : Logging level (default INFO).
    log_file : Optional path to a log file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(_utf8_stdout())]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy in ("torch", "torch_geometric", "optuna", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

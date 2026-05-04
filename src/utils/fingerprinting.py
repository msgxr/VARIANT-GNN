"""
src/utils/fingerprinting.py
SHA-256 based fingerprinting for DataFrames, files, and config dicts.
Used by the reproducibility layer to detect data drift or config changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a SHA-256 fingerprint of a DataFrame's numeric content.

    Only shape and rounded numeric values are hashed — no PII or raw values
    are logged.

    Parameters
    ----------
    df : DataFrame to fingerprint.

    Returns
    -------
    Hex string digest (64 characters).
    """
    hasher = hashlib.sha256()
    # Include shape
    hasher.update(f"shape={df.shape}".encode())
    # Include column names (order matters)
    hasher.update("|".join(df.columns).encode())
    # Include numeric values (rounded to avoid float noise)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        arr = np.round(numeric.fillna(-9999.0).values, 6)
        hasher.update(arr.tobytes())
    digest = hasher.hexdigest()
    logger.debug("DataFrame fingerprint: shape=%s  digest=%s…", df.shape, digest[:8])
    return digest


def file_fingerprint(path: Path) -> str:
    """
    Compute a SHA-256 fingerprint of a file's byte content.

    Parameters
    ----------
    path : File path.

    Returns
    -------
    Hex digest string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def config_fingerprint(config: dict) -> str:
    """
    Compute a SHA-256 fingerprint of a config dictionary.

    Parameters
    ----------
    config : Serialisable dict.

    Returns
    -------
    Hex digest string.
    """
    serialised = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()

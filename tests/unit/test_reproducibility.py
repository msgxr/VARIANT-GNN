# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""tests/unit/test_reproducibility.py"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.artifact_manifest import build_manifest, load_manifest, save_manifest
from src.utils.fingerprinting import (
    config_fingerprint,
    dataframe_fingerprint,
    file_fingerprint,
)
from src.utils.reproducibility import setup_reproducibility, snapshot_environment

# ---------------------------------------------------------------------------
# setup_reproducibility
# ---------------------------------------------------------------------------


def test_setup_reproducibility_sets_seed():
    setup_reproducibility(seed=123)
    a = np.random.rand(5)
    setup_reproducibility(seed=123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_setup_reproducibility_python_hash_seed():
    setup_reproducibility(seed=42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_snapshot_environment_keys():
    env = snapshot_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "PYTHONHASHSEED" in env


# ---------------------------------------------------------------------------
# Fingerprinting — dataframe
# ---------------------------------------------------------------------------


def test_dataframe_fingerprint_deterministic():
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    h1 = dataframe_fingerprint(df)
    h2 = dataframe_fingerprint(df)
    assert h1 == h2
    assert len(h1) == 64


def test_dataframe_fingerprint_changes_with_data():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1.0, 2.0]})
    df2 = pd.DataFrame({"a": [1.0, 9.0]})
    assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)


def test_dataframe_fingerprint_changes_with_columns():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
    df2 = pd.DataFrame({"c": [1.0], "d": [2.0]})
    assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)


def test_dataframe_fingerprint_empty():
    import pandas as pd

    df = pd.DataFrame()
    h = dataframe_fingerprint(df)
    assert isinstance(h, str) and len(h) == 64


# ---------------------------------------------------------------------------
# Fingerprinting — file
# ---------------------------------------------------------------------------


def test_file_fingerprint_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = file_fingerprint(f)
    h2 = file_fingerprint(f)
    assert h1 == h2


def test_file_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content A")
    h1 = file_fingerprint(f)
    f.write_text("content B")
    h2 = file_fingerprint(f)
    assert h1 != h2


def test_file_fingerprint_missing_raises():
    with pytest.raises(FileNotFoundError):
        file_fingerprint(Path("/nonexistent/file.txt"))


# ---------------------------------------------------------------------------
# Fingerprinting — config
# ---------------------------------------------------------------------------


def test_config_fingerprint_deterministic():
    cfg = {"seed": 42, "lr": 0.001, "nested": {"a": 1}}
    h1 = config_fingerprint(cfg)
    h2 = config_fingerprint(cfg)
    assert h1 == h2


def test_config_fingerprint_key_order_invariant():
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"b": 2, "a": 1}
    assert config_fingerprint(cfg1) == config_fingerprint(cfg2)


def test_config_fingerprint_changes_with_value():
    assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_build_manifest_keys():
    manifest = build_manifest(
        model_dir=Path("."),
        config={"seed": 42},
        dataset_fingerprints={"train": "abc123"},
        model_version="1.0.0",
    )
    for key in ("model_version", "created_at", "python_version", "config_hash", "package_versions"):
        assert key in manifest


def test_save_and_load_manifest(tmp_path):
    path = save_manifest(
        model_dir=tmp_path,
        config={"seed": 42},
        model_version="2.0.0",
    )
    assert path.exists()
    loaded = load_manifest(tmp_path)
    assert loaded["model_version"] == "2.0.0"
    assert "python_version" in loaded


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_manifest_config_hash_changes():
    m1 = build_manifest(Path("."), config={"seed": 42})
    m2 = build_manifest(Path("."), config={"seed": 99})
    assert m1["config_hash"] != m2["config_hash"]

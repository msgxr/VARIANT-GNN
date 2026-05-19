"""
tests/integration/test_external_validation_runner.py
Integration tests for ExternalValidationRunner.

These tests use a minimal mock model pipeline to avoid requiring real
trained artifacts.  They verify the pipeline contract:
  - reads CSV without training
  - sanitizes leakage columns
  - produces canonical output schema
  - works fully offline
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference.prediction_schema import PREDICTION_COLUMNS, validate_prediction_frame


# ---------------------------------------------------------------------------
# Minimal fake artifacts
# ---------------------------------------------------------------------------


class _FakePreprocessor:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _FakeEnsemble:
    def predict_with_uncertainty(self, X, n_iter=10, threshold=0.5, **kwargs):
        n = X.shape[0]
        rng = np.random.default_rng(42)
        # Gercekci dagılım — hardcoded sabit degerlerden kacin
        raw = rng.uniform(0.1, 0.9, n)
        preds = (raw >= threshold).astype(int)
        proba = np.column_stack([1 - raw, raw])
        uncertainty = rng.uniform(0.05, 0.35, n)
        return preds, proba, uncertainty


def _write_fake_artifacts(model_dir: Path, feature_names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "preprocessor.pkl", "wb") as fh:
        pickle.dump(_FakePreprocessor(), fh)
    with open(model_dir / "ensemble.pkl", "wb") as fh:
        pickle.dump(_FakeEnsemble(), fh)
    (model_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    (model_dir / "feature_names.json").write_text(json.dumps(feature_names))
    (model_dir / "manifest.json").write_text(json.dumps({"model_version": "test-1.0"}))


def _make_blind_csv(path: Path, n: int = 20, with_leakage: bool = False) -> list[str]:
    rng = np.random.default_rng(42)
    feature_names = [f"feat_{i}" for i in range(10)]
    data = {"Variant_ID": [f"V{i}" for i in range(n)], "Panel": ["General"] * n}
    for fn in feature_names:
        data[fn] = rng.uniform(0, 1, size=n)
    if with_leakage:
        data["chromosome"] = ["chr1"] * n
        data["label"] = rng.integers(0, 2, size=n)
    pd.DataFrame(data).to_csv(path, index=False)
    return feature_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path):
    feature_names = [f"feat_{i}" for i in range(10)]
    model_dir = tmp_path / "models" / "final"
    _write_fake_artifacts(model_dir, feature_names)
    csv_path = tmp_path / "blind_test.csv"
    _make_blind_csv(csv_path, n=20)
    return {
        "tmp": tmp_path,
        "model_dir": model_dir,
        "csv_path": csv_path,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_produces_canonical_schema(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    out_path = env["tmp"] / "predictions.csv"
    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    result = runner.run(env["csv_path"], out_path)

    assert out_path.exists()
    validate_prediction_frame(result)
    assert list(result.columns) == PREDICTION_COLUMNS


def test_runner_row_count_matches_input(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    out_path = env["tmp"] / "predictions.csv"
    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    result = runner.run(env["csv_path"], out_path)
    assert len(result) == 20


def test_runner_drops_leakage_columns(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    leaky_csv = env["tmp"] / "leaky.csv"
    _make_blind_csv(leaky_csv, n=10, with_leakage=True)

    out_path = env["tmp"] / "predictions_leaky.csv"
    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    result = runner.run(leaky_csv, out_path)
    # Predictions still produced despite leaky columns
    assert len(result) == 10
    validate_prediction_frame(result)


def test_runner_missing_input_raises(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    with pytest.raises(FileNotFoundError):
        runner.run(Path("/nonexistent/blind.csv"), env["tmp"] / "out.csv")


def test_runner_prediction_values_valid(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    out_path = env["tmp"] / "predictions.csv"
    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    result = runner.run(env["csv_path"], out_path)
    assert set(result["Prediction"].unique()).issubset({"Pathogenic", "Benign"})


def test_runner_missing_artifact_raises(tmp_path):
    from src.inference.external_validation_runner import ExternalValidationRunner

    empty_model_dir = tmp_path / "empty_model"
    empty_model_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ExternalValidationRunner(model_dir=empty_model_dir)


def test_runner_probabilities_in_range(env):
    from src.inference.external_validation_runner import ExternalValidationRunner

    out_path = env["tmp"] / "predictions.csv"
    runner = ExternalValidationRunner(
        model_dir=env["model_dir"],
        reports_dir=env["tmp"] / "reports",
    )
    result = runner.run(env["csv_path"], out_path)
    assert (result["Pathogenic_Probability"] >= 0).all()
    assert (result["Pathogenic_Probability"] <= 1).all()
    assert (result["Benign_Probability"] >= 0).all()
    assert (result["Benign_Probability"] <= 1).all()

"""
tests/integration/test_high_risk_fixes.py
Regression guards for the 3 high-risk jury-inference fixes.

#1 positional-coupling : anonymous adapter must NOT distributionally reorder a
   frame that already has the expected width (reorder would desync the fixed
   integer _miss_cols of the preprocessor). src/inference/anonymous_inference.py
#2 342-col-cliff       : the preprocessor must FAIL-LOUD (ValueError) on an
   under-width input instead of silently dropping the missing-indicator block
   and emitting a wrong-width matrix. src/features/preprocessing.py
#3 submission-format   : the validator must accept the minimal jury schema
   (Variant_ID + prediction_label) AND the default 7-col validator must REJECT
   a minimal file. src/scientific/submission_validator.py

Tests that need the trained preprocessor auto-skip when models/preprocessor.pkl
is absent (fresh clone before training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"
_HAS_PREPROCESSOR = (_MODELS_DIR / "preprocessor.pkl").exists()


def _load_preprocessor():
    from src.inference.artifact_loader import ArtifactLoader

    return ArtifactLoader(_MODELS_DIR).load_preprocessor()


# ── #2 — 342-col cliff: fail-loud on under-width input ─────────────────────
@pytest.mark.skipif(not _HAS_PREPROCESSOR, reason="shipped models/preprocessor.pkl required")
def test_preprocessing_rejects_underwidth_input():
    """An input narrower than _miss_cols.max()+1 must raise, not silently skip."""
    pre = _load_preprocessor()
    need = int(np.asarray(pre._miss_cols).max()) + 1
    too_narrow = np.random.default_rng(0).normal(size=(4, need - 1)).astype(float)
    with pytest.raises(ValueError):
        pre.transform(too_narrow)


@pytest.mark.skipif(not _HAS_PREPROCESSOR, reason="shipped models/preprocessor.pkl required")
def test_preprocessing_output_width_invariant_holds_on_happy_path():
    """Full-width raw input still yields exactly n_output_features columns."""
    pre = _load_preprocessor()
    n_in = int(pre._imputer.n_features_in_)
    X = np.random.default_rng(1).normal(size=(6, n_in)).astype(float)
    out = pre.transform(X)
    assert out.shape[1] == int(pre.n_output_features)


# ── #1 — positional coupling: no reorder at expected width ─────────────────
@pytest.mark.skipif(not _HAS_PREPROCESSOR, reason="needs n_features_in_ for expected width")
def test_anonymous_skips_reorder_at_expected_width(monkeypatch):
    """_distributional_align must be SKIPPED when width == expected_n_features."""
    from src.inference.anonymous_inference import AnonymousDataAdapter

    pre = _load_preprocessor()
    n_in = int(pre._imputer.n_features_in_)

    adapter = AnonymousDataAdapter(
        expected_n_features=n_in,
        feature_signatures={"0": {"mean": 0.0, "std": 1.0}},  # non-empty → guard reachable
    )

    calls = {"n": 0}

    def _spy(df):
        calls["n"] += 1
        return df

    monkeypatch.setattr(adapter, "_distributional_align", _spy)

    rng = np.random.default_rng(2)
    # width == expected → reorder must be skipped
    df_match = pd.DataFrame(rng.normal(size=(5, n_in)), columns=[f"c{i}" for i in range(n_in)])
    df_match.insert(0, "Variant_ID", [f"V{i}" for i in range(5)])
    adapter.adapt(df_match)
    assert calls["n"] == 0, "reorder ran on an already-expected-width frame (desync risk)"

    # width != expected → reorder must run (off-width anonymized input)
    df_off = pd.DataFrame(rng.normal(size=(5, n_in - 10)), columns=[f"c{i}" for i in range(n_in - 10)])
    df_off.insert(0, "Variant_ID", [f"V{i}" for i in range(5)])
    adapter.adapt(df_off)
    assert calls["n"] == 1, "reorder should run on an off-width frame"


# ── #3 — submission format: validator accepts minimal, rejects when default ─
def _write_minimal_csv(path: Path) -> Path:
    pd.DataFrame({"Variant_ID": ["V1", "V2", "V3"], "prediction_label": [1, 0, 1]}).to_csv(path, index=False)
    return path


def test_validator_accepts_minimal_schema(tmp_path):
    from src.scientific.submission_validator import JURY_MINIMAL_COLUMNS, SubmissionValidator

    csv = _write_minimal_csv(tmp_path / "min.csv")
    report = SubmissionValidator(expected_columns=JURY_MINIMAL_COLUMNS).validate(submission_path=csv)
    assert report.passed, "minimal-schema validator must accept Variant_ID+prediction_label"


def test_default_validator_rejects_minimal_schema(tmp_path):
    from src.scientific.submission_validator import SubmissionValidator

    csv = _write_minimal_csv(tmp_path / "min.csv")
    report = SubmissionValidator().validate(submission_path=csv)  # default = 7-col JURY_COLUMNS
    assert not report.passed, "default 7-col validator must reject a 2-col file"


# ── #4 (LOW) — named branch reorders BY NAME → pozisyonel-kuplajın robust yolu ──
@pytest.mark.skipif(not _HAS_PREPROCESSOR, reason="shipped models/preprocessor.pkl required")
def test_runner_named_branch_reorders_by_name():
    """feature_names.json (named branch) varsa kolonlar İSİMLE hizalanır — girdi
    sırasından bağımsız. Bu, pozisyonel-kuplajın 'robust yol' çözümünü kanıtlar (#4):
    jüri kolonları farklı sırada gelse bile named-branch eğitim sırasına yeniden dizer,
    böylece sabit tamsayı _miss_cols doğru kolonu okur. (no-feature_names erken-dönüş
    dalı sıra-korumalıdır ama yeniden-sıralamaz; bu yüzden named-branch daha sağlam.)
    """
    from src.inference.external_validation_runner import ExternalValidationRunner

    pre = _load_preprocessor()
    n_in = int(pre._imputer.n_features_in_)
    feature_names = [f"F{i}" for i in range(n_in)]
    mc = int(np.asarray(pre._miss_cols)[0])  # bir eksiklik-gösterge kolonu

    rng = np.random.default_rng(7)
    data = {f"F{i}": rng.normal(size=5) for i in range(n_in)}
    data[f"F{mc}"][2] = np.nan  # eksikliği İSİMLE F{mc}'ye koy
    # Kolonları TERS sırada ver — named-branch eğitim sırasına dizmeli
    df_reversed = pd.DataFrame(data)[[f"F{i}" for i in reversed(range(n_in))]]
    assert list(df_reversed.columns)[0] == f"F{n_in - 1}"  # gerçekten ters

    runner = ExternalValidationRunner.__new__(ExternalValidationRunner)
    runner._feature_names = feature_names

    X = runner._align_features(df_reversed)
    # named-branch isimle hizaladıysa NaN, girdi sırasında (son sütun) değil
    # eğitim sırasındaki mc konumunda olmalı:
    assert np.isnan(X[:, mc]).any(), "named branch isimle hizalamadı (pozisyonel desync riski)"
    out = pre.transform(X)
    assert out.shape[1] == int(pre.n_output_features)

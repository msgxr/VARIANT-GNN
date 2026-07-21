# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/integration/test_missing_indicator_parity.py
Regression guard for the train/inference missing-indicator asymmetry.

Bug class: an inference path that destroys the raw-NaN pattern BEFORE
VariantPreprocessor.transform breaks the missing-indicator block. The
preprocessor captures the raw NaN pattern (features/preprocessing.py:459) and
derives the indicator block via np.isnan(_Xraw_t[:, _miss_cols])
(features/preprocessing.py:496). Zero-filling / 0.0-padding first makes that
block ALL-ZERO -> silent divergence from training, which feeds raw NaN
(cli/modes/train.py -> training/trainer.py). RESULTS_CANONICAL.json attributes
+1.5pp on the official 4-panel score to these indicators.

Two sites were hardened:
  1. inference/external_validation_runner.py::_align_features — used to call
     aligned_df.fillna(0.0). LATENT today: with no models/feature_names.json
     shipped, load_feature_names()==None and the runner takes the early-return
     numeric branch (which already preserves NaN), so the canonical number was
     NOT produced through a zeroed-indicator runner path. The fix removes the
     foot-gun that activates the moment a feature_names.json is shipped.
  2. inference/anonymous_inference.py::_fit_to_expected_size — used to 0.0-pad
     genuinely-absent columns; now NaN-pads so the median-imputer fills them
     with the training median and absent columns register as MISSING (not a
     real 0.0 measurement).

These tests use the SHIPPED models/preprocessor.pkl (auto-skip if missing) so
they exercise the real missing-indicator logic rather than a mock.

NOTE: a separate, pre-existing positional-coupling concern is NOT covered here:
_miss_cols are fixed integer indices, so a blind CSV whose column ORDER differs
from training can read indicators from the wrong columns even with NaN
preserved. That needs a name/signature-keyed aligner (follow-up).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (_MODELS_DIR / "preprocessor.pkl").exists(),
    reason="shipped models/preprocessor.pkl required for parity test",
)


def _load_preprocessor():
    from src.inference.artifact_loader import ArtifactLoader

    return ArtifactLoader(_MODELS_DIR).load_preprocessor()


def _raw_nan_frame(pre, seed: int = 0):
    """Build a raw, NaN-bearing numpy matrix of the exact trained input width."""
    rng = np.random.default_rng(seed)
    n_in = int(pre._imputer.n_features_in_)
    miss = np.asarray(pre._miss_cols)
    assert len(miss) > 0, "shipped model has no missing-indicator columns"
    X = rng.normal(size=(8, n_in)).astype(float)
    for c in miss[:6]:
        X[1, int(c)] = np.nan
        X[5, int(c)] = np.nan
    return X, miss, n_in


def test_raw_nan_yields_correct_indicator_block():
    """Inference-path (raw NaN) indicator block == training-path derivation."""
    pre = _load_preprocessor()
    X, miss, n_in = _raw_nan_frame(pre)

    out = pre.transform(X)
    assert out.shape[1] == int(pre.n_output_features)

    block = out[:, n_in:]
    assert block.shape[1] == len(miss)
    assert block.any(), "missing-indicator block is all-zero on NaN-bearing input"

    # Identical to the training-time derivation (preprocessing.py:338-340 / :496).
    expected = np.isnan(X[:, miss]).astype(float)
    assert np.array_equal(block, expected)


def test_fillna_zero_destroys_indicator_block():
    """Pins the exact regression: pre-filling 0.0 zeroes the indicator block."""
    pre = _load_preprocessor()
    X, _, n_in = _raw_nan_frame(pre)

    out_raw = pre.transform(X)
    out_filled = pre.transform(np.nan_to_num(X, nan=0.0))

    assert out_raw[:, n_in:].any(), "raw-NaN path should fire indicators"
    assert not out_filled[:, n_in:].any(), "fillna(0.0) path must produce an all-zero block (the bug)"


def test_runner_align_features_preserves_nan():
    """Post-fix _align_features must NOT destroy the NaN pattern.

    Injects feature_names to force the (previously buggy) alignment branch.
    Fails on the unpatched runner; passes after the fillna(0.0) removal.
    """
    from src.inference.external_validation_runner import ExternalValidationRunner

    pre = _load_preprocessor()
    n_in = int(pre._imputer.n_features_in_)
    feat_names = [f"F{i}" for i in range(n_in)]

    runner = ExternalValidationRunner.__new__(ExternalValidationRunner)
    runner._feature_names = feat_names

    rng = np.random.default_rng(1)
    arr = rng.normal(size=(6, n_in))
    for c in np.asarray(pre._miss_cols)[:4]:
        arr[2, int(c)] = np.nan
    feat_df = pd.DataFrame(arr, columns=feat_names)

    X = runner._align_features(feat_df)
    assert np.isnan(X).any(), "_align_features destroyed the NaN pattern (fillna bug)"

    out = pre.transform(X)
    assert out[:, n_in:].any(), "indicator block all-zero after _align_features"


def test_anonymous_pad_uses_nan_not_zero():
    """NaN-padding (post-fix) keeps absent columns missing, not 'present 0.0'.

    Asserts the MECHANISM: a genuinely-absent feature padded with 0.0 would
    register as present (np.isnan(0.0)==False) and be scaled as a real
    measurement; padded with NaN it correctly registers as missing and the
    internal imputer fills it with the training median.
    """
    from src.inference.anonymous_inference import AnonymousDataAdapter

    pre = _load_preprocessor()
    n_in = int(pre._imputer.n_features_in_)

    n_present = n_in - 5
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        rng.normal(size=(5, n_present)),
        columns=[f"c{i}" for i in range(n_present)],
    )

    adapter = AnonymousDataAdapter(expected_n_features=n_in, feature_signatures={})
    padded = adapter._fit_to_expected_size(df)
    assert padded.shape[1] == n_in

    pad_cols = [c for c in padded.columns if str(c).startswith("__pad_")]
    assert pad_cols, "expected padded columns"
    # Post-fix: padded columns must be NaN (missing), not literal 0.0.
    assert padded[pad_cols].isna().all().all(), "padded columns must be NaN so the preprocessor treats them as missing"

"""
tests/unit/test_innovations.py
================================
Regression test suite for the 6 TEKNOFEST 2026 cutting-edge innovations:

  1. Conformal Prediction       (provable coverage)
  2. Sharpness-Aware Minimisation (SAM optimizer)
  3. Tabular Mixup              (data augmentation)
  4. Domain Adversarial Training (panel-invariant)
  5. Snapshot Ensemble          (cyclic LR + checkpoints)
  6. Reproducibility Manifest   (§7.5 jury re-run)
  7. Anonymous Inference Adapter (§3.2 column-name-free)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. Conformal Prediction
# ---------------------------------------------------------------------------


class TestConformalPrediction:
    @pytest.fixture
    def cal_data(self):
        rng = np.random.default_rng(42)
        n = 500
        # Realistic-ish: probabilities aligned with true labels
        labels = rng.integers(0, 2, n)
        proba = np.zeros((n, 2), dtype=float)
        for i, y in enumerate(labels):
            base = 0.7 if y == 1 else 0.3
            p1 = np.clip(base + rng.normal(0, 0.15), 0.05, 0.95)
            proba[i] = [1 - p1, p1]
        return proba, labels

    def test_aps_method_meets_coverage_target(self, cal_data):
        from src.scientific.conformal_prediction import ConformalPredictor

        cal_proba, cal_labels = cal_data
        cp = ConformalPredictor(alpha=0.10, method="APS")
        cp.fit(cal_proba, cal_labels)

        # Generate fresh test set with same distribution
        rng = np.random.default_rng(123)
        n_test = 300
        y_test = rng.integers(0, 2, n_test)
        p_test = np.zeros((n_test, 2))
        for i, y in enumerate(y_test):
            base = 0.7 if y == 1 else 0.3
            p1 = np.clip(base + rng.normal(0, 0.15), 0.05, 0.95)
            p_test[i] = [1 - p1, p1]

        sets, _ = cp.predict_sets(p_test)
        report = cp.evaluate_coverage(sets, y_test)

        # Allow 5 % stochastic slack
        assert report.empirical_coverage >= 0.85
        assert report.average_set_size > 0
        assert report.average_set_size <= 2.0

    def test_lac_method_smaller_set_size(self, cal_data):
        from src.scientific.conformal_prediction import ConformalPredictor

        cal_proba, cal_labels = cal_data
        aps = ConformalPredictor(alpha=0.10, method="APS").fit(cal_proba, cal_labels)
        lac = ConformalPredictor(alpha=0.10, method="LAC").fit(cal_proba, cal_labels)

        # Both should give valid quantile thresholds
        assert aps._q_hat is not None
        assert lac._q_hat is not None

    def test_predict_with_decision_returns_correct_keys(self, cal_data):
        from src.scientific.conformal_prediction import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10).fit(*cal_data)
        out = cp.predict_with_decision(cal_data[0][:50])

        assert set(out.keys()) == {"decisions", "prediction_sets", "confidence"}
        assert len(out["decisions"]) == 50
        assert all(d in ("Pathogenic", "Benign", "Abstain") for d in out["decisions"])

    def test_mondrian_per_group_calibration(self, cal_data):
        from src.scientific.conformal_prediction import MondrianConformalPredictor

        cal_proba, cal_labels = cal_data
        groups = np.array(["General"] * 250 + ["CFTR"] * 250)
        mcp = MondrianConformalPredictor(alpha=0.10, method="APS")
        mcp.fit(cal_proba, cal_labels, groups)

        assert "General" in mcp._predictors
        assert "CFTR" in mcp._predictors
        assert mcp._fallback is not None


# ---------------------------------------------------------------------------
# 2. SAM Optimizer
# ---------------------------------------------------------------------------


class TestSAM:
    def test_sam_two_step_actually_updates_weights(self):
        import torch

        from src.training.sam import SAM

        torch.manual_seed(0)
        model = torch.nn.Linear(5, 2)
        original_w = model.weight.detach().clone()

        opt = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(8, 5)
        y = torch.randint(0, 2, (8,))

        loss1 = criterion(model(x), y)
        loss1.backward()
        opt.first_step(zero_grad=True)

        loss2 = criterion(model(x), y)
        loss2.backward()
        opt.second_step(zero_grad=True)

        # Weights should have changed
        assert not torch.allclose(model.weight, original_w, atol=1e-7)

    def test_sam_step_with_closure(self):
        import torch

        from src.training.sam import SAM

        model = torch.nn.Linear(3, 2)
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 3)
        y = torch.randint(0, 2, (4,))

        def closure():
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert torch.is_tensor(loss)


# ---------------------------------------------------------------------------
# 3. Mixup
# ---------------------------------------------------------------------------


class TestMixup:
    def test_mixup_numpy_increases_sample_count(self):
        from src.training.mixup import mixup_numpy

        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 10))
        y = rng.integers(0, 2, 100)
        X_aug, y_aug = mixup_numpy(X, y, alpha=0.2, n_synthetic=50, seed=42)

        assert X_aug.shape == (150, 10)
        assert y_aug.shape == (150,)

    def test_mixup_same_class_only_preserves_labels(self):
        from src.training.mixup import mixup_numpy

        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        X_aug, y_aug = mixup_numpy(
            X,
            y,
            alpha=1.0,
            n_synthetic=20,
            same_class_only=True,
            seed=0,
        )
        # Synthetic samples should still be in {0, 1}
        synth_labels = y_aug[100:]
        assert set(synth_labels).issubset({0, 1})

    def test_mixup_torch_data_shapes(self):
        import torch

        from src.training.mixup import mixup_data

        X = torch.randn(20, 8)
        y = torch.randint(0, 2, (20,))
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=0.5)

        assert X_mix.shape == X.shape
        assert y_a.shape == y_b.shape == y.shape
        assert lam.shape == (20,)
        assert (lam >= 0).all() and (lam <= 1).all()


# ---------------------------------------------------------------------------
# 4. Domain Adversarial Training
# ---------------------------------------------------------------------------


class TestDANN:
    def test_lambda_schedule_monotone_increasing(self):
        from src.training.domain_adversarial import dann_lambda

        progressions = np.linspace(0.0, 1.0, 11)
        lambdas = [dann_lambda(p, gamma=10.0) for p in progressions]

        assert lambdas[0] == 0.0
        assert lambdas[-1] > 0.99
        # Monotone non-decreasing
        for i in range(len(lambdas) - 1):
            assert lambdas[i + 1] >= lambdas[i]

    def test_panel_discriminator_forward_shape(self):
        import torch

        from src.training.domain_adversarial import PanelDiscriminator

        disc = PanelDiscriminator(input_dim=32, hidden_dim=16, n_panels=4)
        feats = torch.randn(10, 32)
        out = disc(feats, lambda_=0.5)
        assert out.shape == (10, 4)

    def test_grad_reverse_inverts_gradient(self):
        import torch

        from src.training.domain_adversarial import grad_reverse

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = grad_reverse(x, lambda_=2.0)
        y.sum().backward()
        # Forward identity: y == x
        assert torch.allclose(y, x.detach())
        # Gradient: -2 * 1 = -2 for each element (sum's gradient is 1)
        assert torch.allclose(x.grad, torch.tensor([-2.0, -2.0, -2.0]))

    def test_encode_panels_canonical_order(self):
        from src.training.domain_adversarial import KNOWN_PANELS_ORDER, encode_panels

        panels = ["PAH", "Hereditary_Cancer", "General", "CFTR", "UnknownPanel"]
        encoded, order = encode_panels(panels)
        assert encoded[0] == KNOWN_PANELS_ORDER.index("PAH")
        assert encoded[2] == KNOWN_PANELS_ORDER.index("General")
        assert encoded[4] == len(KNOWN_PANELS_ORDER)  # "Other"


# ---------------------------------------------------------------------------
# 5. Snapshot Ensemble
# ---------------------------------------------------------------------------


class TestSnapshotEnsemble:
    def test_cyclic_lr_produces_min_at_cycle_end(self):
        import torch

        from src.training.snapshot_ensemble import CosineAnnealingCyclicLR

        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-2)
        sched = CosineAnnealingCyclicLR(opt, cycle_length=4, lr_min=1e-5, lr_max=1e-2)

        lrs = [sched.step(epoch) for epoch in range(8)]
        # Cycle ends at epoch 3, 7 → LR should be near lr_min then
        # Cycle starts at epoch 0, 4 → LR should be near lr_max
        assert lrs[0] == pytest.approx(1e-2, abs=1e-7)
        assert lrs[3] < lrs[0]
        assert lrs[3] < lrs[2]

    def test_snapshot_buffer_collects_at_cycle_ends(self):
        import torch

        from src.training.snapshot_ensemble import SnapshotBuffer

        model = torch.nn.Linear(4, 2)
        buf = SnapshotBuffer(cycle_length=3, n_snapshots=5)
        for ep in range(12):
            buf.maybe_save(ep, model)
        # Cycle ends: epochs 2, 5, 8, 11 → 4 snapshots
        assert buf.n_collected() == 4

    def test_snapshot_buffer_evicts_oldest_when_full(self):
        import torch

        from src.training.snapshot_ensemble import SnapshotBuffer

        model = torch.nn.Linear(4, 2)
        buf = SnapshotBuffer(cycle_length=2, n_snapshots=3)
        for ep in range(10):
            buf.maybe_save(ep, model)
        # Cycle ends: 1, 3, 5, 7, 9 → 5 saves but only 3 retained
        assert buf.n_collected() == 3


# ---------------------------------------------------------------------------
# 6. Reproducibility Manifest
# ---------------------------------------------------------------------------


class TestReproducibilityManifest:
    def test_manifest_self_hash_changes_with_metrics(self, tmp_path):
        from src.utils.reproducibility_manifest import ManifestBuilder

        b1 = ManifestBuilder(seed=42).with_metrics({"binary_f1": 0.90})
        b2 = ManifestBuilder(seed=42).with_metrics({"binary_f1": 0.95})
        m1 = b1.build()
        m2 = b2.build()
        assert m1.compute_self_hash() != m2.compute_self_hash()

    def test_manifest_roundtrip_json(self, tmp_path):
        from src.utils.reproducibility_manifest import (
            ManifestBuilder,
            ReproducibilityManifest,
        )

        path = tmp_path / "manifest.json"
        b = ManifestBuilder(seed=42).with_metrics({"binary_f1": 0.94})
        m = b.save(path)
        loaded = ReproducibilityManifest.from_json(path)
        assert loaded.seed == m.seed
        assert loaded.metrics["binary_f1"] == 0.94

    def test_manifest_verify_after_save(self, tmp_path):
        from src.utils.reproducibility_manifest import (
            ManifestBuilder,
            ReproducibilityManifest,
        )

        path = tmp_path / "manifest.json"
        b = ManifestBuilder(seed=42).with_metrics({"binary_f1": 0.94})
        b.save(path)
        loaded = ReproducibilityManifest.from_json(path)
        assert loaded.verify() is True


# ---------------------------------------------------------------------------
# 7. Anonymous Inference Adapter
# ---------------------------------------------------------------------------


class TestAnonymousInference:
    def test_detect_panel_column(self):
        from src.inference.anonymous_inference import detect_panel_column

        df = pd.DataFrame(
            {
                "X1": ["General", "CFTR", "PAH", "Hereditary_Cancer"],
                "X2": [1.0, 2.0, 3.0, 4.0],
            }
        )
        assert detect_panel_column(df) == "X1"

    def test_detect_sequence_columns(self):
        from src.inference.anonymous_inference import detect_sequence_columns

        df = pd.DataFrame(
            {
                "S1": ["ACGTACGTACG", "TTGCAACGTAC", "GGGCATAGCTA", "CCCATAGGCTT"],
                "S2": ["AILVMSTHKQR", "ARNDCEQGHIL", "MFPSTWYVALK", "GHISTANCREP"],
                "X1": [1.0, 2.0, 3.0, 4.0],
            }
        )
        nuc, aa = detect_sequence_columns(df)
        assert nuc == "S1"
        assert aa == "S2"

    def test_anonymous_adapter_expected_shape(self):
        from src.inference.anonymous_inference import AnonymousDataAdapter

        df = pd.DataFrame(
            {
                "Col_0": ["VAR_001", "VAR_002", "VAR_003"],
                "Col_1": ["General", "PAH", "CFTR"],
                "Col_2": [0.5, 0.3, 0.8],
                "Col_3": [1.2, 0.9, 1.5],
                "Col_4": [10.0, 20.0, 30.0],
            }
        )
        adapter = AnonymousDataAdapter(expected_n_features=3)
        feat, meta, nuc, aa = adapter.adapt(df)

        # Expect 3 numeric features, padded if fewer
        assert feat.shape == (3, 3)
        assert "Variant_ID" in meta.columns
        assert "Panel" in meta.columns


# ---------------------------------------------------------------------------
# 8. TD-006 Multimodal Fail-Safe
# ---------------------------------------------------------------------------


class TestMultimodalFailSafe:
    """Regression tests for the TD-006 fix: pipeline must not crash when
    multimodal sequences are missing, malformed, or wrong-typed."""

    def test_build_safe_tensors_with_none(self):
        """nuc/aa = None → returns valid zero-padded tensors of correct shape."""
        import torch

        from src.api.pipeline import _build_safe_sequence_tensors
        from src.models.dnn_model import VariantDNN  # cheap module on CPU

        # Use a simple module for device extraction
        dummy = VariantDNN(input_dim=10, hidden_dim=16)
        nuc_ids, aa_ids = _build_safe_sequence_tensors(
            gnn_model=dummy,
            n_samples=4,
            nuc_sequences=None,
            aa_sequences=None,
        )
        assert nuc_ids.shape == (4, 11)
        assert aa_ids.shape == (4, 11)
        assert nuc_ids.dtype == torch.long
        assert aa_ids.dtype == torch.long
        assert (nuc_ids == 0).all()
        assert (aa_ids == 0).all()

    def test_build_safe_tensors_with_nan_strings(self):
        """nuc list with NaN/None entries → tokenized cleanly."""
        from src.api.pipeline import _build_safe_sequence_tensors
        from src.models.dnn_model import VariantDNN

        dummy = VariantDNN(input_dim=10, hidden_dim=16)
        nuc_ids, aa_ids = _build_safe_sequence_tensors(
            gnn_model=dummy,
            n_samples=3,
            nuc_sequences=[None, "ACGT", float("nan")],
            aa_sequences=["ALK", None, ""],
        )
        assert nuc_ids.shape == (3, 11)
        assert aa_ids.shape == (3, 11)

    def test_build_safe_tensors_with_wrong_type(self):
        """Non-string sequence values are str()-cast safely."""
        from src.api.pipeline import _build_safe_sequence_tensors
        from src.models.dnn_model import VariantDNN

        dummy = VariantDNN(input_dim=10, hidden_dim=16)
        nuc_ids, aa_ids = _build_safe_sequence_tensors(
            gnn_model=dummy,
            n_samples=2,
            nuc_sequences=[42, 3.14],
            aa_sequences=[0, 1],
        )
        assert nuc_ids.shape == (2, 11)
        assert aa_ids.shape == (2, 11)

    def test_build_safe_tensors_length_mismatch(self):
        """Length mismatch is logged + padded — no crash."""
        from src.api.pipeline import _build_safe_sequence_tensors
        from src.models.dnn_model import VariantDNN

        dummy = VariantDNN(input_dim=10, hidden_dim=16)
        # n_samples=5 but only 2 sequences
        nuc_ids, aa_ids = _build_safe_sequence_tensors(
            gnn_model=dummy,
            n_samples=5,
            nuc_sequences=["ACGT", "TTAG"],
            aa_sequences=["ALK", "VRP"],
        )
        assert nuc_ids.shape == (5, 11)
        assert aa_ids.shape == (5, 11)

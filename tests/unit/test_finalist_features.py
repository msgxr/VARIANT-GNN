"""
tests/unit/test_finalist_features.py
TEKNOFEST 2026 finalist özelliklerinin unit testleri.

- MCDropoutEstimator (belirsizlik tahmini)
- FocalLoss (zor örneklere odaklanma)
- adversarial_validate (domain shift tespiti)
- evaluate_per_panel (panel bazlı metrikler)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# MC Dropout Uncertainty
# ---------------------------------------------------------------------------

class TestMCDropoutEstimator:
    def test_estimate_dnn_shapes(self):
        """MC Dropout DNN çıktı boyutlarını doğrular."""
        from src.models.dnn import VariantDNN
        from src.inference.uncertainty import MCDropoutEstimator

        model = VariantDNN(input_dim=20, hidden_dim=32, num_classes=2, dropout=0.5)
        estimator = MCDropoutEstimator(model, n_forward=10)
        x = torch.randn(8, 20)
        mean_probs, uncertainty = estimator.estimate_dnn(x)

        assert mean_probs.shape == (8, 2)
        assert uncertainty.shape == (8,)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)

    def test_estimate_gnn_shapes(self):
        """MC Dropout GNN çıktı boyutlarını doğrular."""
        from src.models.gnn import VariantSAGEGNN
        from src.inference.uncertainty import MCDropoutEstimator

        model = VariantSAGEGNN(numeric_dim=16, hidden_dim=32, num_classes=2)
        estimator = MCDropoutEstimator(model, n_forward=10)
        x = torch.randn(6, 16)
        edges = torch.randint(0, 6, (2, 15))
        mean_probs, uncertainty = estimator.estimate_gnn(x, edges)

        assert mean_probs.shape == (6, 2)
        assert uncertainty.shape == (6,)

    def test_uncertainty_category(self):
        """Belirsizlik kategorileri doğru atanıyor mu."""
        from src.inference.uncertainty import MCDropoutEstimator

        scores = np.array([0.1, 0.3, 0.6])
        cats = MCDropoutEstimator.uncertainty_category(scores)
        assert cats[0] == "Yüksek Güven"
        assert cats[1] == "Orta Güven"
        assert cats[2] == "Düşük Güven"

    def test_predictive_entropy_deterministic(self):
        """Kesin tahmin için entropy ~0 olmalı."""
        from src.inference.uncertainty import MCDropoutEstimator

        probs = np.array([[0.99, 0.01], [0.01, 0.99]])
        entropy = MCDropoutEstimator._predictive_entropy(probs)
        assert np.all(entropy < 0.15), f"Kesin tahmin için entropy düşük olmalı: {entropy}"


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_forward_shape(self):
        """Focal Loss çıktı skaler olmalı."""
        from src.training.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_gamma_zero_equals_ce(self):
        """gamma=0 → standart CrossEntropy'ye eşit olmalı."""
        from src.training.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(20, 2)
        targets = torch.randint(0, 2, (20,))

        focal = FocalLoss(gamma=0.0)(logits, targets)
        ce = torch.nn.CrossEntropyLoss()(logits, targets)

        assert torch.allclose(focal, ce, atol=1e-5), f"gamma=0 CE'ye eşit olmalı: {focal} vs {ce}"

    def test_from_labels_creates_balanced_weights(self):
        """from_labels() class weight'ları doğru hesaplıyor mu."""
        from src.training.focal_loss import FocalLoss

        y = np.array([0, 0, 0, 0, 1, 1])  # 4:2 imbalanced
        fl = FocalLoss.from_labels(y, gamma=2.0)
        assert fl.alpha is not None
        # Minority class (1) should have higher weight
        assert fl.alpha[1] > fl.alpha[0]

    def test_gradient_flows(self):
        from src.training.focal_loss import FocalLoss

        model = torch.nn.Linear(10, 2)
        loss_fn = FocalLoss(gamma=2.0)
        x = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(model(x), targets)
        loss.backward()

        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Adversarial Validation
# ---------------------------------------------------------------------------

class TestAdversarialValidation:
    def test_similar_distributions_low_auc(self):
        """Benzer dağılımlar → AUC ~0.5"""
        from src.evaluation.adversarial_validation import adversarial_validate

        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_test = np.random.randn(100, 10)  # aynı dağılım

        result = adversarial_validate(X_train, X_test)
        assert result.auc_mean < 0.65, f"Benzer dağılım AUC düşük olmalı: {result.auc_mean}"
        assert "İyi" in result.verdict

    def test_different_distributions_high_auc(self):
        """Farklı dağılımlar → AUC yüksek"""
        from src.evaluation.adversarial_validation import adversarial_validate

        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_test = np.random.randn(100, 10) + 5  # shifted

        result = adversarial_validate(X_train, X_test)
        assert result.auc_mean > 0.85, f"Farklı dağılım AUC yüksek olmalı: {result.auc_mean}"

    def test_with_feature_names(self):
        """Feature isimleri verilince top_shift_features dolu dönmeli."""
        from src.evaluation.adversarial_validation import adversarial_validate

        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5) + 2
        names = [f"feat_{i}" for i in range(5)]

        result = adversarial_validate(X_train, X_test, feature_names=names)
        assert len(result.top_shift_features) > 0


# ---------------------------------------------------------------------------
# Panel-Based Evaluation
# ---------------------------------------------------------------------------

class TestEvaluatePerPanel:
    def test_returns_reports_per_panel(self):
        """Her panel için ayrı EvaluationReport dönmeli."""
        from src.evaluation.metrics import evaluate_per_panel

        np.random.seed(42)
        N = 100
        y_true = np.random.randint(0, 2, N)
        y_prob = np.column_stack([1 - np.random.rand(N), np.random.rand(N)])
        panels = np.array(["General"] * 50 + ["CFTR"] * 30 + ["PAH"] * 20)

        reports = evaluate_per_panel(y_true, y_prob, panels)
        assert "General" in reports
        assert "CFTR" in reports
        assert "PAH" in reports
        for name, report in reports.items():
            assert 0 <= report.macro_f1 <= 1

    def test_skips_tiny_panels(self):
        """Çok az örnekli paneller atlanmalı."""
        from src.evaluation.metrics import evaluate_per_panel

        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
        panels = np.array(["A", "A", "A", "A", "B"])  # B has only 1 sample

        reports = evaluate_per_panel(y_true, y_prob, panels)
        assert "A" in reports
        assert "B" not in reports  # too few samples

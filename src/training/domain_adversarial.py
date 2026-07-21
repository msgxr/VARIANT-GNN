# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/training/domain_adversarial.py
====================================
Domain Adversarial Neural Network (DANN) — Panel-Invariant Variant Learning

Referanslar:
  - Ganin & Lempitsky (2015). "Unsupervised Domain Adaptation by Backpropagation."
    ICML 2015.  https://arxiv.org/abs/1409.7495
  - Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks."
    JMLR 17.  https://arxiv.org/abs/1505.07818

TEKNOFEST 2026 motivasyonu:
  Şartname §3.2 dört bağımsız hastalık paneli tanımlar:
    General, Hereditary_Cancer, PAH, CFTR.

  Her panelin biyolojik özellik dağılımı belirgin ölçüde farklıdır:
    - PAH varyantları çoğunlukla küçük missense, yüksek konservasyon
    - CFTR varyantları çoğunlukla splice/regulatory, gnomAD frekansları çok düşük
    - Hereditary Cancer varyantları DNA repair gen ailesinde yoğunlaşır

  Naive bir model panel-spesifik shortcut'lar öğrenebilir (örn. CFTR'da
  belirli bir feature pattern'ine bakar), bu da §7.2 external validation'da
  yeni veri dağılımına genelleştirme problemine yol açar.

  DANN, gradyan tersleme katmanı (Gradient Reversal Layer, GRL) ile bu
  sorunu çözer:

    Feature Extractor (φ) → Variant Classifier (head_y)  : F1'i maksimize et
                          → Domain Discriminator (head_d) : panel'ı tahmin etmeye ÇALIŞIR
                            ↑ ama GRL, gradyanı tersine çevirir → φ paneli
                            AYIRT ETMEKTEN UZAKLAŞIR.

  Sonuç: φ panel-invariant biyolojik temsil öğrenir → 4 panelin tümünde
  daha iyi genelleştirme.

İmplementasyon:
  - Custom autograd.Function ile GRL
  - Dinamik λ schedule (Ganin 2016): λ(t) = (2 / (1 + e^{-γt})) - 1
  - Domain discriminator: 2-3 katmanlı MLP head
  - Multi-task loss: L = L_y - λ·L_d
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------


class _GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd op: identity forward, sign-flipped scaled gradient backward.

    Forward:    GRL(x) = x
    Backward:   ∂L/∂x = -λ · upstream_gradient
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with given strength λ."""
    return _GradientReversalFunction.apply(x, lambda_)


# ---------------------------------------------------------------------------
# λ scheduler (Ganin 2016 Section 4.5)
# ---------------------------------------------------------------------------


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """
    Compute the dynamic GRL strength based on training progress.

    progress : float in [0, 1] — current epoch / total epochs.
    gamma    : steepness coefficient (Ganin recommends 10).

    Schedule: λ = (2 / (1 + e^{-γ·progress})) - 1
       progress=0 → λ=0     (no domain adversarial signal at start)
       progress=1 → λ≈1     (full adversarial pressure at end)

    This warm-up prevents the discriminator from dominating training early
    when the feature extractor has not yet learnt useful representations.
    """
    progress = max(0.0, min(progress, 1.0))
    return float(2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)


# ---------------------------------------------------------------------------
# Panel Domain Discriminator
# ---------------------------------------------------------------------------


class PanelDiscriminator(nn.Module):
    """
    Multi-layer perceptron that classifies which panel a sample belongs to.

    Used adversarially via gradient reversal: the feature extractor learns
    to FOOL this discriminator, making representations panel-invariant.

    Parameters
    ----------
    input_dim   : feature dimensionality after feature extractor.
    hidden_dim  : MLP hidden width.
    n_panels    : number of distinct panels (4 for TEKNOFEST §3.2).
    dropout     : dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_panels: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_panels),
        )

    def forward(self, features: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Apply GRL then classify the panel."""
        reversed_features = grad_reverse(features, lambda_)
        return self.net(reversed_features)


# ---------------------------------------------------------------------------
# Multi-task DANN loss
# ---------------------------------------------------------------------------


def dann_multitask_loss(
    variant_logits: torch.Tensor,
    panel_logits: torch.Tensor,
    variant_labels: torch.Tensor,
    panel_labels: torch.Tensor,
    lambda_: float = 1.0,
    variant_criterion: Optional[nn.Module] = None,
    panel_criterion: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined classification + domain adversarial loss.

    L_total = L_variant + λ · L_panel  (where panel head receives reversed gradient)

    Returns (total, variant_loss, panel_loss) — all scalar tensors.
    """
    if variant_criterion is None:
        variant_criterion = nn.CrossEntropyLoss()
    if panel_criterion is None:
        panel_criterion = nn.CrossEntropyLoss()

    L_variant = variant_criterion(variant_logits, variant_labels)
    L_panel = panel_criterion(panel_logits, panel_labels)
    L_total = L_variant + lambda_ * L_panel
    return L_total, L_variant, L_panel


# ---------------------------------------------------------------------------
# Convenience wrapper: DANN-augmented model
# ---------------------------------------------------------------------------


class DANNWrapper(nn.Module):
    """
    Wraps a feature-extractor + variant-classifier with a domain discriminator.

    Useful for adapting any existing PyTorch classifier to a DANN regime
    without rewriting its forward pass:

        # feature_extractor + variant_head pre-existing in the model
        wrapped = DANNWrapper(
            feature_extractor = model.encoder,
            variant_head      = model.classifier,
            feat_dim          = model.feat_dim,
            n_panels          = 4,
        )

    forward(x, lambda_=λ) returns (variant_logits, panel_logits).

    During training, sum CE losses with appropriate weights. The GRL inside
    the discriminator handles gradient reversal automatically.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        variant_head: nn.Module,
        feat_dim: int,
        n_panels: int = 4,
        disc_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.variant_head = variant_head
        self.discriminator = PanelDiscriminator(
            input_dim=feat_dim,
            hidden_dim=disc_hidden,
            n_panels=n_panels,
        )

    def forward(
        self,
        x: torch.Tensor,
        lambda_: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (variant_logits, panel_logits)."""
        feats = self.feature_extractor(x)
        variant_logits = self.variant_head(feats)
        panel_logits = self.discriminator(feats, lambda_=lambda_)
        return variant_logits, panel_logits

    def predict_variant(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: return only the variant prediction."""
        feats = self.feature_extractor(x)
        return self.variant_head(feats)


# ---------------------------------------------------------------------------
# Panel encoding helper
# ---------------------------------------------------------------------------


KNOWN_PANELS_ORDER: List[str] = [
    "General",
    "Hereditary_Cancer",
    "PAH",
    "CFTR",
]


def encode_panels(panels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert string panel labels to integer indices in canonical order.

    Returns (encoded_array, panel_order).
    Unknown panels map to index ``len(KNOWN_PANELS_ORDER)`` (i.e. "Other").
    """
    panel_to_idx = {p: i for i, p in enumerate(KNOWN_PANELS_ORDER)}
    encoded = np.array(
        [panel_to_idx.get(str(p).strip(), len(KNOWN_PANELS_ORDER)) for p in panels],
        dtype=np.int64,
    )
    return encoded, KNOWN_PANELS_ORDER + ["Other"]

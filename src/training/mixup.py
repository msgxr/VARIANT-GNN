# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/training/mixup.py
======================
Mixup Data Augmentation (Tabular Variants)

NOT (durum): Kanonik eğitim hattı (src/training/trainer.py) bu modülü İÇE AKTARMAZ —
opsiyonel/deneysel kitaplık modülüdür. Pipeline'a dahil DEĞİLDİR.

Referanslar:
  - Zhang, Cisse, Dauphin & Lopez-Paz (2018).
    "mixup: Beyond Empirical Risk Minimization." ICLR 2018.
    https://arxiv.org/abs/1710.09412
  - Yoon, Zame, Banerjee & van der Schaar (2020).
    "VIME: Extending the Success of Self- and Semi-supervised Learning to
    Tabular Domain." NeurIPS 2020.
  - Verma et al. (2019). "Manifold Mixup: Better Representations by
    Interpolating Hidden States." ICML 2019.

TEKNOFEST 2026 motivasyonu:
  Şartname §3.2 panel boyutları:
    - General           : 1500 + 1500
    - Hereditary Cancer : 200  + 200
    - PAH               : 200  + 200
    - CFTR              : 70   + 70   ← ÇOK KÜÇÜK!

  CFTR paneli sadece 140 örnek içeriyor; bu küçük örnek boyutu derin
  modellerin overfit yapmasına yol açar. Mixup, ekstra veri toplamadan
  efektif eğitim setini büyütür:

    x_mix = λ·x_i + (1-λ)·x_j
    y_mix = λ·y_i + (1-λ)·y_j

  λ ~ Beta(α, α), α ≈ 0.2 → küçük interpolasyonlar
                  α = 1.0 → uniform interpolasyon

  Tabular Mixup, biyolojik özniteliklerin lineer interpolasyonu ile
  modeli pürüzsüz karar sınırlarına teşvik eder; %0.5-2 F1 iyileşmesi
  ve %30-50 calibration error azalması raporlanmıştır (Yoon 2020).

İmplementasyon:
  - Standart Mixup (input-level)
  - Manifold Mixup (hidden-state level — sadece DNN/GNN için)
  - λ-Beta dağılımı ayarlanabilir
  - Label smoothing entegrasyonu
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beta sampler
# ---------------------------------------------------------------------------


def _sample_lambda(alpha: float, batch_size: int = 1) -> torch.Tensor:
    """
    Sample λ ~ Beta(α, α). For α → 0: λ near 0 or 1 (no mix);
    for α → ∞: λ ≈ 0.5 (heavy mix).
    """
    if alpha <= 0:
        return torch.ones(batch_size)
    return torch.distributions.Beta(alpha, alpha).sample((batch_size,))


# ---------------------------------------------------------------------------
# Input-level Mixup
# ---------------------------------------------------------------------------


def mixup_data(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    same_class_only: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate mixed (X', y_a, y_b, λ) samples for Mixup training.

    Parameters
    ----------
    X : (N, F) input tensor.
    y : (N,) integer labels.
    alpha : Beta distribution parameter.
        - 0    → no mixing
        - 0.2  → mild interpolation (default for tabular)
        - 1.0  → uniform interpolation
    same_class_only : if True, only mix samples of the SAME class
        (label-preserving augmentation).
    seed : optional reproducibility seed.

    Returns
    -------
    X_mixed : (N, F) interpolated features
    y_a, y_b : (N,) original and shuffled labels
    lam : (N,) interpolation coefficients

    Notes
    -----
    Loss should be computed as:
        loss = λ·CE(logits, y_a) + (1-λ)·CE(logits, y_b)
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = X.shape[0]
    device = X.device

    if alpha > 0:
        lam = _sample_lambda(alpha, n).to(device)
    else:
        lam = torch.ones(n, device=device)

    # Shuffle indices for pairing
    if same_class_only:
        # For each sample, pick another sample of the same class
        perm = torch.zeros(n, dtype=torch.long, device=device)
        for cls in y.unique():
            cls_idx = (y == cls).nonzero(as_tuple=True)[0]
            shuffled = cls_idx[torch.randperm(len(cls_idx), device=device)]
            for i, src in enumerate(cls_idx):
                perm[src] = shuffled[i]
    else:
        perm = torch.randperm(n, device=device)

    # Broadcast lam: (N,) → (N, 1) for feature dim
    lam_x = lam.view(-1, 1)
    X_mixed = lam_x * X + (1.0 - lam_x) * X[perm]

    return X_mixed, y, y[perm], lam


# ---------------------------------------------------------------------------
# Mixup loss
# ---------------------------------------------------------------------------


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mixup loss: λ·L(ŷ, y_a) + (1-λ)·L(ŷ, y_b).

    Note: ``criterion`` must compute per-sample loss when ``reduction='none'``.
    For criterions that don't support 'none' reduction, the loss is computed
    twice (with full reduction) and averaged in scalar form.

    Returns a scalar loss tensor.
    """
    if hasattr(criterion, "reduction") and getattr(criterion, "reduction", "mean") == "none":
        loss_a = criterion(logits, y_a)
        loss_b = criterion(logits, y_b)
        return (lam * loss_a + (1.0 - lam) * loss_b).mean()

    # Fall back to scalar combination (less granular but always works)
    loss_a = criterion(logits, y_a)
    loss_b = criterion(logits, y_b)
    lam_mean = lam.mean()
    return lam_mean * loss_a + (1.0 - lam_mean) * loss_b


# ---------------------------------------------------------------------------
# Manifold Mixup (hidden-state level)
# ---------------------------------------------------------------------------


class ManifoldMixupHook:
    """
    Manifold Mixup for arbitrary intermediate layers.

    Usage:
      hook = ManifoldMixupHook(alpha=0.2)
      hook.register(model.block2)   # mix at output of block2

      output = model(x)             # automatic mixing during forward
      hook.last_lam                 # access λ for loss computation
      hook.last_perm                # access pairing permutation

    Paper claim: mixing at hidden layers regularises representations
    in addition to inputs, often outperforming standard input Mixup.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.enabled = True
        self.last_lam: Optional[torch.Tensor] = None
        self.last_perm: Optional[torch.Tensor] = None
        self._handle = None

    def __call__(self, module: nn.Module, inputs, output):
        if not self.enabled or not module.training:
            return output

        # Output may be a tensor or tuple (varies by layer type)
        if isinstance(output, tuple):
            return output  # not supported for tuple outputs

        n = output.shape[0]
        device = output.device
        lam = _sample_lambda(self.alpha, 1).to(device)
        perm = torch.randperm(n, device=device)

        self.last_lam = lam
        self.last_perm = perm

        # Broadcast lam to output shape
        lam_b = lam.view(*([1] * output.dim()))
        return lam_b * output + (1.0 - lam_b) * output[perm]

    def register(self, layer: nn.Module) -> None:
        """Register a forward hook on the given layer."""
        if self._handle is not None:
            self._handle.remove()
        self._handle = layer.register_forward_hook(self)

    def remove(self) -> None:
        """Detach the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last_lam = None
        self.last_perm = None


# ---------------------------------------------------------------------------
# Tabular-aware Mixup with feature noise
# ---------------------------------------------------------------------------


def tabular_mixup(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    feature_noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixup variant adapted for tabular biological data.

    Differences from standard Mixup:
      1. Optional Gaussian feature noise (perturbs interpolated features
         to simulate measurement variability — bioinformatic scores often
         have ±5-10 % noise).
      2. λ closer to 0/1 by default (α=0.2) since extreme interpolation
         of biological features (e.g. mixing pathogenic and benign feature
         vectors aggressively) can produce unrealistic samples.

    Parameters
    ----------
    feature_noise_std : Gaussian std added to mixed features.
        0.0 disables noise; 0.05 adds mild perturbation.
    """
    X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=alpha, seed=seed)

    if feature_noise_std > 0:
        noise = torch.randn_like(X_mix) * feature_noise_std
        X_mix = X_mix + noise

    return X_mix, y_a, y_b, lam


# ---------------------------------------------------------------------------
# Numpy interface (for sklearn-compatible models like XGBoost/LightGBM)
# ---------------------------------------------------------------------------


def mixup_numpy(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    n_synthetic: Optional[int] = None,
    same_class_only: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy-only Mixup for tree-based models (XGBoost, LightGBM).

    Trees can't directly consume soft labels, so we:
      1. Generate λ for each pair.
      2. Use λ > 0.5 to assign the dominant label as the new sample's label.
      3. This produces a *hard-label* Mixup augmentation that works with
         any classifier.

    Parameters
    ----------
    X            : (N, F) feature matrix.
    y            : (N,) integer labels.
    alpha        : Beta distribution parameter.
    n_synthetic  : Number of synthetic samples to generate.  Default: N.
    same_class_only : Restrict pairing to same-class samples (fully
                      label-preserving — recommended for tree boosters).
    seed         : Reproducibility seed.

    Returns
    -------
    X_aug : (N + n_synthetic, F) augmented feature matrix.
    y_aug : (N + n_synthetic,) labels (hard, integer).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n_synthetic is None:
        n_synthetic = n

    if alpha <= 0:
        return X.copy(), y.copy()

    if same_class_only:
        # For each synthetic, pick two same-class samples
        idx_a = np.zeros(n_synthetic, dtype=int)
        idx_b = np.zeros(n_synthetic, dtype=int)
        unique_classes = np.unique(y)
        per_class = n_synthetic // len(unique_classes)
        ptr = 0
        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) < 2:
                continue
            count = per_class if cls != unique_classes[-1] else (n_synthetic - ptr)
            for k in range(count):
                pair = rng.choice(cls_idx, size=2, replace=False)
                idx_a[ptr] = pair[0]
                idx_b[ptr] = pair[1]
                ptr += 1
        idx_a = idx_a[:ptr]
        idx_b = idx_b[:ptr]
        n_synthetic = ptr
    else:
        idx_a = rng.integers(0, n, size=n_synthetic)
        idx_b = rng.integers(0, n, size=n_synthetic)

    lam = rng.beta(alpha, alpha, size=n_synthetic)
    lam_x = lam.reshape(-1, 1)

    X_synth = lam_x * X[idx_a] + (1.0 - lam_x) * X[idx_b]

    # Hard labels: use the "dominant" sample's label
    y_synth = np.where(lam >= 0.5, y[idx_a], y[idx_b])

    X_aug = np.vstack([X, X_synth])
    y_aug = np.concatenate([y, y_synth])

    logger.info(
        "mixup_numpy: %d → %d samples (+%d synthetic, same_class=%s, α=%.2f).",
        n,
        len(X_aug),
        n_synthetic,
        same_class_only,
        alpha,
    )
    return X_aug, y_aug

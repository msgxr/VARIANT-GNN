"""
src/training/focal_loss.py
Focal Loss — TEKNOFEST 2026.

Zor sınıflandırılabilir örneklere odaklanır. Kolay örneklerin
ağırlığını (1-p)^gamma ile azaltır.

Lin et al. (2017) "Focal Loss for Dense Object Detection"

Özellikle küçük panellerde (CFTR: 70+70) ve sınır bölgesindeki
varyantlarda etkilidir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class weights.

    L_focal = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : Focusing parameter. γ=0 → standard CE. γ=2 (default) strongly
            down-weights easy examples.
    alpha : Per-class weight tensor. If None, all classes weighted equally.
    reduction : 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [N, C] raw model output (before softmax).
        targets : [N] integer class indices.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def from_labels(
        y: "np.ndarray",
        gamma: float = 2.0,
        num_classes: int = 2,
    ) -> "FocalLoss":
        """Factory: compute balanced alpha weights from label array."""
        import numpy as np
        counts = np.bincount(y, minlength=num_classes).astype(float)
        alpha = len(y) / (num_classes * counts)
        return FocalLoss(
            gamma=gamma,
            alpha=torch.tensor(alpha, dtype=torch.float),
        )

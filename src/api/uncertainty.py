"""
src/inference/uncertainty.py
Monte Carlo Dropout belirsizlik tahmini — TEKNOFEST 2026.

Epistemic uncertainty: Dropout'u inference sırasında açık bırakarak
N stokastik forward pass yapılır. Tahminlerdeki varyans = model
belirsizliği.

Klinik etki: "Bu tahmine güvenilir mi?" sorusuna yanıt veren
güvenilirlik metriği. Jüri için güçlü diferansiyatör.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MCDropoutEstimator:
    """
    Monte Carlo Dropout ile epistemic belirsizlik tahmini.

    Kullanım:
        estimator = MCDropoutEstimator(model, n_forward=30)
        mean_probs, uncertainty = estimator.estimate(x, edge_index)

    Dönen değerler:
        mean_probs:  [N, num_classes] — ortalama sınıf olasılıkları
        uncertainty: [N] — her varyant için belirsizlik skoru (0=kesin, 1=belirsiz)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_forward: int = 30,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.n_forward = n_forward
        self.device = device or torch.device("cpu")
        self.seed = seed

    def _enable_dropout(self) -> None:
        """Tüm Dropout katmanlarını train moduna al (inference'da da drop yapar)."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def estimate_gnn(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GNN modeli için MC Dropout tahmin."""
        self.model.eval()
        self._enable_dropout()

        all_probs = []
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            for i in range(self.n_forward):
                torch.manual_seed(self.seed + i)
                logits = self.model(x, edge_index)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        stacked = np.stack(all_probs, axis=0)  # [n_forward, N, C]
        mean_probs = stacked.mean(axis=0)       # [N, C]
        # Predictive entropy as uncertainty
        uncertainty = self._predictive_entropy(mean_probs)

        return mean_probs, uncertainty

    def estimate_dnn(
        self,
        x: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """DNN modeli için MC Dropout tahmin."""
        self.model.eval()
        self._enable_dropout()

        all_probs = []
        x = x.to(self.device)

        with torch.no_grad():
            for i in range(self.n_forward):
                torch.manual_seed(self.seed + i)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        stacked = np.stack(all_probs, axis=0)
        mean_probs = stacked.mean(axis=0)
        uncertainty = self._predictive_entropy(mean_probs)

        return mean_probs, uncertainty

    @staticmethod
    def _predictive_entropy(probs: np.ndarray) -> np.ndarray:
        """
        Predictive entropy (bilgi-kuramsal belirsizlik).

        H(p) = -Σ p_i * log(p_i)
        Normalized to [0, 1] by dividing by log(num_classes).

        Returns: [N] uncertainty scores in [0, 1].
        """
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        max_entropy = np.log(probs.shape[1])
        return entropy / max_entropy

    @staticmethod
    def uncertainty_category(scores: np.ndarray) -> np.ndarray:
        """
        Belirsizlik skorlarını klinik kategorilere çevirir.

        Returns: string array — 'Yüksek Güven', 'Orta Güven', 'Düşük Güven'
        """
        categories = np.where(
            scores < 0.2, "Yüksek Güven",
            np.where(scores < 0.5, "Orta Güven", "Düşük Güven")
        )
        return categories

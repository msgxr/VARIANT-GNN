"""
src/models/dnn_model.py
========================
Kanonik VariantDNN tanımı (TEKNOFEST 2026 — TD-007 çözümü).

Önceden 3 ayrı yerde tutulan duplikat tanımlar (src/core/dnn.py,
src/core/models/dnn.py, src/models/dnn.py) bu tek modüle birleştirilmiştir.
Tüm proje bu dosyadan import etmelidir:

    from src.models.dnn_model import VariantDNN

Geriye dönük import yolları (src.core.dnn, src.core.models.dnn, src.models.dnn)
ya doğrudan kaldırılmıştır ya da bu modülü yeniden ihraç eden ince shim'lere
indirilmiştir.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VariantDNN(nn.Module):
    """
    Tablo verisi üzerinde çalışan ileri beslemeli derin sinir ağı.

    TEKNOFEST 2026 §3.2 öznitelik vektörü için tasarlanmıştır; girdi
    boyutu ön-işleme sonrası dinamik olarak belirlenir.

    Parameters
    ----------
    input_dim : int
        Girdi öznitelik sayısı (preprocessor.transform sonrası).
    hidden_dim : int
        İlk gizli katmanın genişliği. İkinci katman ``hidden_dim // 2``
        boyutunu kullanır.
    num_classes : int
        Çıkış sınıf sayısı (Patojenik=1 / Benign=0 için 2).
    dropout : float
        İlk katman dropout oranı; ikinci katman ``dropout / 2`` kullanır.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standart forward gecisi — (N, input_dim) → (N, num_classes) logits."""
        # BatchNorm1d tek ornekte eval modunda NaN uretebilir — egitim modunda gec
        if x.shape[0] == 1 and not self.training:
            self.train()
            out = self.net(x)
            self.eval()
            return out
        return self.net(x)


__all__ = ["VariantDNN"]

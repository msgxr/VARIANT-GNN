"""
src/features/autoencoder.py
Tabular AutoEncoder as a sklearn-compatible transformer.
Wraps the PyTorch autoencoder in fit/transform API so it can sit cleanly
inside fold preprocessing without data leakage.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _TorchAutoEncoder(nn.Module):
    """Internal PyTorch autoencoder architecture."""

    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        super().__init__()
        hidden = max(input_dim // 2, encoding_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, encoding_dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that concatenates autoencoder latent
    representations to the original features.

    Parameters
    ----------
    encoding_dim   : Latent space dimensionality.
    epochs         : Training epochs.
    batch_size     : Mini-batch size.
    lr             : Adam learning rate.
    device         : ``'cpu'`` or ``'cuda'``; ``'auto'`` selects automatically.
    random_state   : Seed for reproducibility.
    append         : If True, append encoded features to the original matrix.
                     If False, return only encoded features.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "auto",
        random_state: int = 42,
        append: bool = True,
    ) -> None:
        self.encoding_dim  = encoding_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.device        = device
        self.random_state  = random_state
        self.append        = append

        self._net: Optional[_TorchAutoEncoder] = None
        self._device_obj: Optional[torch.device] = None

    # ------------------------------------------------------------------
    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoEncoderTransformer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._device_obj = self._resolve_device()
        input_dim        = X.shape[1]
        self._net        = _TorchAutoEncoder(input_dim, self.encoding_dim).to(self._device_obj)

        tensor_X = torch.FloatTensor(X).to(self._device_obj)
        dataset  = TensorDataset(tensor_X, tensor_X)
        loader   = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                _, decoded = self._net(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                logger.debug(
                    "AutoEncoder epoch %d/%d | loss=%.4f",
                    epoch, self.epochs, total_loss / len(loader),
                )

        self._net.eval()
        logger.info(
            "AutoEncoder trained: input_dim=%d → encoding_dim=%d", input_dim, self.encoding_dim
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("AutoEncoderTransformer must be fitted before transform.")
        self._net.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self._device_obj)
            encoded, _ = self._net(tensor_X)
            encoded_np = encoded.cpu().numpy()

        if self.append:
            return np.hstack([X, encoded_np])
        return encoded_np

    def get_encoder_state(self) -> dict:
        """Return state dict for serialisation."""
        if self._net is None:
            raise RuntimeError("Not fitted.")
        return self._net.state_dict()

    def load_encoder_state(self, state_dict: dict, input_dim: int) -> None:
        """Restore from saved state dict."""
        self._device_obj = self._resolve_device()
        self._net = _TorchAutoEncoder(input_dim, self.encoding_dim).to(self._device_obj)
        self._net.load_state_dict(state_dict)
        self._net.eval()

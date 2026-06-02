"""
src/training/swa.py
===================
Stochastic Weight Averaging (SWA) for GNN and DNN models.

Reference:
  Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization."
  UAI 2018.  https://arxiv.org/abs/1803.05407

Scientific motivation for TEKNOFEST 2026:
  External validation uses completely new data (§7.2 / §7.3). SWA navigates
  to flatter loss minima, which generalise significantly better to unseen
  variant distributions than a single last-epoch checkpoint.

  In practice on tabular-biological tasks SWA yields +1–3% F1 on held-out
  sets with zero additional inference cost (only the averaging happens during
  training, the saved model is a single standard checkpoint).

Implementation:
  - SWABuffer collects model state_dicts from the final ``swa_start_epoch``
    epochs of any training loop.
  - ``apply_swa`` averages all buffered checkpoints component-wise and
    loads the averaged weights back into the model.
  - ``update_batch_norm`` fixes BatchNorm running statistics after weight
    averaging (PyTorch SWA best practice).
  - Integration hooks: trainer._train_gatv2 and trainer._train_dnn call
    ``swa_buffer.push(epoch, model)`` when epoch >= swa_start_epoch.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Iterator

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SWABuffer:
    """
    Collects model checkpoints for Stochastic Weight Averaging.

    Parameters
    ----------
    swa_start_fraction : float
        Fraction of total epochs after which SWA collection starts.
        E.g. 0.75 → collect checkpoints from the last 25 % of training.
    max_checkpoints : int
        Maximum number of checkpoints to retain in the buffer (memory guard).
        If more checkpoints arrive, the oldest is evicted (circular buffer).

    Usage
    -----
    During training loop::

        swa = SWABuffer(swa_start_fraction=0.75)
        for epoch in range(1, total_epochs + 1):
            ... train ...
            swa.push(epoch, total_epochs, model)

        averaged_model = swa.apply(model)
    """

    def __init__(
        self,
        swa_start_fraction: float = 0.75,
        max_checkpoints: int = 10,
    ) -> None:
        if not 0.0 <= swa_start_fraction < 1.0:
            raise ValueError("swa_start_fraction must be in [0, 1).")
        self.swa_start_fraction = swa_start_fraction
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Deque[Dict[str, torch.Tensor]] = deque(maxlen=max_checkpoints)
        self._n_collected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_collect(self, epoch: int, total_epochs: int) -> bool:
        """Return True if this epoch's weights should be collected."""
        start = int(total_epochs * self.swa_start_fraction)
        return epoch >= start

    def push(
        self,
        epoch: int,
        total_epochs: int,
        model: nn.Module,
    ) -> bool:
        """
        Push a checkpoint if we're in the SWA collection window.

        Returns True if checkpoint was stored, False otherwise.
        """
        if not self.should_collect(epoch, total_epochs):
            return False

        # Deep-copy state dict to CPU to avoid GPU memory pressure.
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._checkpoints.append(sd)  # deque(maxlen) otomatik evict eder
        self._n_collected += 1
        return True

    def apply(self, model: nn.Module) -> nn.Module:
        """
        Average all buffered checkpoints and load into ``model``.

        If fewer than 2 checkpoints were collected (e.g. very short training),
        the model is returned unchanged with a warning.

        Parameters
        ----------
        model : nn.Module
            The model instance whose weights will be replaced by the SWA
            average.  The model is modified **in-place** and also returned.

        Returns
        -------
        model : nn.Module
            The same model instance with averaged weights loaded.
        """
        n = len(self._checkpoints)
        if n < 2:
            logger.warning(
                "SWA: only %d checkpoint(s) collected (need ≥ 2) — returning original model unchanged.",
                n,
            )
            return model

        # Compute parameter-wise mean across all checkpoints.
        avg_sd: Dict[str, torch.Tensor] = {}
        for key in self._checkpoints[0]:
            tensors = [ckpt[key].float() for ckpt in self._checkpoints]
            avg_sd[key] = torch.stack(tensors).mean(dim=0)

        # Cast back to original dtype before loading.
        original_sd = model.state_dict()
        for key in avg_sd:
            avg_sd[key] = avg_sd[key].to(original_sd[key].dtype)

        model.load_state_dict(avg_sd)
        logger.info(
            "SWA: averaged %d checkpoints → model weights updated.",
            n,
        )
        return model

    @property
    def n_collected(self) -> int:
        """Number of checkpoints collected so far."""
        return self._n_collected

    def clear(self) -> None:
        """Free buffer memory."""
        self._checkpoints.clear()
        self._n_collected = 0

    def __len__(self) -> int:
        return len(self._checkpoints)


# ---------------------------------------------------------------------------
# BatchNorm update after SWA
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_batch_norm(
    model: nn.Module,
    data_loader: Iterator,
    device: torch.device,
    n_batches: int = 50,
) -> None:
    """
    Recompute BatchNorm running statistics after SWA weight averaging.

    SWA invalidates running_mean / running_var because the averaged weights
    produce a different internal distribution.  This function performs a
    forward pass over up to ``n_batches`` mini-batches in train mode
    (which triggers BN accumulation) while disabling gradient computation.

    Parameters
    ----------
    model      : Model with SWA weights already loaded.
    data_loader: Iterator yielding (x, ...) batches.
    device     : Target device.
    n_batches  : Number of batches to accumulate statistics over.
    """
    # Reset running statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average
            module.num_batches_tracked.zero_()

    model.train()
    i = 0
    for i, batch in enumerate(data_loader):
        if i >= n_batches:
            break
        try:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            model(x)
        except Exception:
            break
    model.eval()
    logger.info("SWA: BatchNorm statistics updated over %d batches.", min(i + 1, n_batches))


# ---------------------------------------------------------------------------
# SWA Scheduler helper  (cosine annealing for SWA warm-up)
# ---------------------------------------------------------------------------


class CyclicSWAScheduler:
    """
    Cyclic learning rate schedule for SWA warm-up phase.

    Implements a simplified version of the cyclic LR used in the original SWA
    paper: learning rate oscillates between ``lr_min`` and ``lr_max`` with a
    period of ``cycle_length`` epochs, creating diverse local minima for the
    buffer to average over.

    Usage
    -----
    ::

        scheduler = CyclicSWAScheduler(
            optimizer, lr_min=1e-5, lr_max=5e-3, cycle_length=5
        )
        for epoch in range(1, total_epochs + 1):
            if swa_buffer.should_collect(epoch, total_epochs):
                scheduler.step(epoch)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_min: float = 1e-5,
        lr_max: float = 5e-3,
        cycle_length: int = 5,
    ) -> None:
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_length = cycle_length

    def step(self, epoch_in_swa: int) -> float:
        """
        Update learning rate for ``epoch_in_swa`` (0-indexed position inside
        SWA phase).  Returns the new LR.
        """
        cl = max(1, self.cycle_length)  # sıfıra bölme koruması
        t = (epoch_in_swa % cl) / cl
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 - np.cos(np.pi * t))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return float(lr)

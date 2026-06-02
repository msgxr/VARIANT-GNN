"""
src/training/snapshot_ensemble.py
==================================
Snapshot Ensembles — Cyclic Cosine Annealing Training

Referans:
  Huang, Li, Pleiss, Liu, Hopcroft & Weinberger (2017).
  "Snapshot Ensembles: Train 1, Get M for Free."  ICLR 2017.
  https://arxiv.org/abs/1704.00109

  Loshchilov & Hutter (2017).
  "SGDR: Stochastic Gradient Descent with Warm Restarts."  ICLR 2017.
  https://arxiv.org/abs/1608.03983

TEKNOFEST 2026 motivasyonu:
  External validation (§7.2) yeni veri dağılımına dayanır. Klasik ensemble
  M model eğitmeyi gerektirir (M× compute cost). Snapshot Ensemble ise
  TEK eğitim koşusunda M tane "snapshot" modeli üretir:

    1. Cyclic cosine annealing LR: η(t) = η_max · 0.5 · (1 + cos(π·t/T))
    2. Her cycle sonunda model checkpoint'i kaydet → M snapshot
    3. Inference'ta tüm snapshot olasılıklarını ortala

  SWA (zaten implement edildi) tek bir modelin son K epoch'unu ortalar;
  SE FARKLI lokal minima'lara yakınsayan M modeli toplar — daha çeşitli.
  İki yaklaşım birbirinin tamamlayıcısıdır:
    - SWA  : Aynı havzada flat minimum (eğitim sonu)
    - SE   : Farklı havzalarda M çeşit minimum (cycle başına)

  Bu modül SWA ile birlikte kullanılabilir; combined ensemble +1-3% F1
  improvement on external test (Huang 2017).

İmplementasyon:
  - CosineAnnealingWarmRestarts schedule (PyTorch built-in)
  - Snapshot collection at LR minima
  - Ensemble inference: predict_proba'yı tüm snapshot'lar üzerinden ortala
  - Disk persistence: her snapshot ayrı .pth dosyası
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cyclic Cosine Annealing Scheduler
# ---------------------------------------------------------------------------


class CosineAnnealingCyclicLR:
    """
    Cosine annealing learning rate with warm restarts (SGDR).

    LR formula at step ``t`` within a cycle of length ``T``:
        η(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(π·t/T))

    Parameters
    ----------
    optimizer    : torch.optim.Optimizer to schedule.
    cycle_length : Number of epochs per cycle.
    lr_min       : Minimum LR at end of each cycle.
    lr_max       : Maximum LR at start of each cycle.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cycle_length: int = 5,
        lr_min: float = 1e-6,
        lr_max: float = 1e-2,
    ) -> None:
        if cycle_length < 1:
            raise ValueError("cycle_length must be >= 1")
        self.optimizer = optimizer
        self.cycle_length = cycle_length
        self.lr_min = lr_min
        self.lr_max = lr_max
        self._epoch = 0

    def step(self, epoch: Optional[int] = None) -> float:
        """Update optimizer LR for the given epoch (or auto-increment)."""
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch

        t = self._epoch % self.cycle_length
        cos_factor = 0.5 * (1 + np.cos(np.pi * t / self.cycle_length))
        lr = self.lr_min + (self.lr_max - self.lr_min) * cos_factor

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return float(lr)

    def is_cycle_end(self, epoch: int) -> bool:
        """Return True if ``epoch`` is the last epoch of a cycle (LR at min)."""
        return ((epoch + 1) % self.cycle_length) == 0


# ---------------------------------------------------------------------------
# Snapshot Buffer
# ---------------------------------------------------------------------------


class SnapshotBuffer:
    """
    Collects model checkpoints at the end of each LR cycle.

    Differs from SWABuffer in that it captures DIFFERENT lokal minima
    (one per cycle) rather than averaging weights of the same minimum.

    Parameters
    ----------
    cycle_length : LR cycle length (epochs).
    n_snapshots  : Maximum number of snapshots to retain.
                   If training has fewer cycles than this, all are kept.
    save_dir     : Optional directory to persist snapshots to disk.
    """

    def __init__(
        self,
        cycle_length: int = 5,
        n_snapshots: int = 5,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.cycle_length = cycle_length
        self.n_snapshots = n_snapshots
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self._snapshots: List[Dict[str, torch.Tensor]] = []

    def maybe_save(self, epoch: int, model: nn.Module) -> bool:
        """
        Save a snapshot if ``epoch`` lies at the end of a cycle.

        Returns True if a snapshot was saved this call.
        """
        if (epoch + 1) % self.cycle_length != 0:
            return False

        # Deep copy to CPU for memory safety
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if len(self._snapshots) >= self.n_snapshots:
            # Evict oldest (FIFO) — keeps most recent N cycles
            evicted = self._snapshots.pop(0)
            del evicted

        self._snapshots.append(sd)
        snapshot_idx = len(self._snapshots)

        if self.save_dir is not None:
            path = self.save_dir / f"snapshot_{epoch + 1:04d}.pth"
            torch.save(sd, str(path))
            logger.info(
                "SnapshotEnsemble: snapshot %d saved at epoch %d → %s",
                snapshot_idx,
                epoch + 1,
                path,
            )
        else:
            logger.info(
                "SnapshotEnsemble: snapshot %d collected at epoch %d (in-memory only).",
                snapshot_idx,
                epoch + 1,
            )

        return True

    def n_collected(self) -> int:
        return len(self._snapshots)

    def snapshots(self) -> List[Dict[str, torch.Tensor]]:
        return self._snapshots

    def load_from_disk(self, pattern: str = "snapshot_*.pth") -> int:
        """Reload snapshots from ``save_dir``. Returns number loaded."""
        if self.save_dir is None:
            return 0
        files = sorted(self.save_dir.glob(pattern))
        self._snapshots = [torch.load(str(f), map_location="cpu") for f in files]
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# Snapshot Ensemble Predictor
# ---------------------------------------------------------------------------


class SnapshotEnsemblePredictor:
    """
    Inference helper for averaging predictions across all snapshots.

    Loads snapshots into a fresh copy of the model architecture and produces
    averaged class probabilities.

    Usage
    -----
    ::

        predictor = SnapshotEnsemblePredictor(
            model_factory = lambda: VariantDNN(input_dim=43, hidden_dim=128, num_classes=2),
            snapshots     = buffer.snapshots(),
            device        = torch.device("cpu"),
        )
        proba = predictor.predict_proba(X)        # (N, 2)
        # Or with custom forward args:
        proba = predictor.predict_proba_graph(data, nuc_ids=tok_n, aa_ids=tok_a)
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        snapshots: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> None:
        if not snapshots:
            raise ValueError("SnapshotEnsemblePredictor requires ≥ 1 snapshot.")
        self.model_factory = model_factory
        self.snapshots = snapshots
        self.device = device

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Average probabilities across all snapshots for tabular DNN.

        Parameters
        ----------
        X : (N, F) numpy feature matrix.

        Returns
        -------
        proba : (N, 2) ensemble-averaged probability matrix.
        """
        x_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        n = len(X)
        accumulator = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        for sd in self.snapshots:
            model = self.model_factory().to(self.device)
            model.load_state_dict(sd)
            model.eval()
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            accumulator += probs

        accumulator /= len(self.snapshots)
        return accumulator.cpu().numpy()

    @torch.no_grad()
    def predict_proba_graph(
        self,
        data: Any,
        nuc_ids: Optional[torch.Tensor] = None,
        aa_ids: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Snapshot-ensemble predict for a PyG Data graph.

        Returns averaged (N, 2) probability matrix.
        """
        data = data.to(self.device)
        n_nodes = data.x.shape[0]
        accumulator = torch.zeros((n_nodes, 2), dtype=torch.float32, device=self.device)

        for sd in self.snapshots:
            model = self.model_factory().to(self.device)
            model.load_state_dict(sd)
            model.eval()
            kwargs = {}
            if nuc_ids is not None:
                kwargs["nuc_ids"] = nuc_ids.to(self.device)
            if aa_ids is not None:
                kwargs["aa_ids"] = aa_ids.to(self.device)
            logits = model(data.x, data.edge_index, **kwargs)
            probs = torch.softmax(logits, dim=1)
            accumulator += probs

        accumulator /= len(self.snapshots)
        return accumulator.cpu().numpy()

    def __len__(self) -> int:
        return len(self.snapshots)

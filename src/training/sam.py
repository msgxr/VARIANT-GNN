"""
src/training/sam.py
====================
Sharpness-Aware Minimization (SAM)

Referans:
  Foret, Kleiner, Mobahi & Neyshabur (2021).
  "Sharpness-Aware Minimization for Efficiently Improving Generalization."
  ICLR 2021.  https://arxiv.org/abs/2010.01412

  Kwon, Kim, Park & Choi (2021).
  "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning."
  ICML 2021.  https://arxiv.org/abs/2102.11600

TEKNOFEST 2026 motivasyonu:
  External validation (§7.2) tamamen yeni veriler kullanır. Bu durumda
  modelin "düz" (flat) loss minima'ya yakınsamış olması — sharp minima'ya
  göre — ÇOK daha iyi genelleştirir.

  SAM, gradient descent'i iki adımda yapar:
    1. ε-perturbed weights: w + ε·∇L/||∇L||  (worst-case yön)
    2. Bu perturbed nokta üzerinde gradyan al, orijinal w'yi güncelle

  Bu, "min_w max_{||ε||≤ρ} L(w+ε)" minimax probleminin çözümüdür.
  Pratikte +1-3% F1 iyileşme sağlar (özellikle small dataset'lerde —
  CFTR paneli 70+70 örnek için ideal).

İmplementasyon:
  - Standard SAM (rho=0.05 varsayılan)
  - ASAM (Adaptive SAM) — scale-invariant variant
  - Closure-style training (PyTorch optimizer protokolüne uygun)
  - GNN ve DNN için drop-in replacement
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM optimizer wrapper
# ---------------------------------------------------------------------------


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer wrapper.

    Wraps a base optimizer (e.g. Adam, SGD) to perform the two-step SAM
    update: ascend to worst-case ε-neighbour, then descend from the original
    point using gradients computed at the ascended point.

    Parameters
    ----------
    params : iterable of torch parameters.
    base_optimizer : optimizer class (uninstantiated, e.g. ``torch.optim.Adam``).
    rho : float
        Neighborhood size for worst-case perturbation.  Typical range [0.01, 0.10].
        Higher rho → broader minima search (more regularization, slower convergence).
    adaptive : bool
        If True, use ASAM (per-parameter rho normalised by parameter magnitude).
    **kwargs : forwarded to base_optimizer constructor.

    Usage
    -----
    Standard training loop adaptation::

        optimizer = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=1e-3)
        for x, y in loader:
            # First step: compute loss & gradients at original w
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second step: compute loss & gradients at ascended w
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params: Iterable,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Synchronise param groups so base_optimizer sees the same parameters
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    # ------------------------------------------------------------------
    # SAM steps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Ascent step: w ← w + ε(w)
        where ε(w) = ρ · ∇L / ||∇L||₂ for SAM,
              ε(w) = ρ · |w| · ∇L / ||(|w|·∇L)||₂  for ASAM.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()

                if group["adaptive"]:
                    e_w = (torch.pow(p, 2) * p.grad) * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)

                p.add_(e_w)  # in-place: w ← w + ε

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Descent step: w ← w_original  ;  base_optimizer.step()
        Uses the gradient computed AT (w + ε) to update the ORIGINAL weights.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]  # restore original w

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------
    # Convenience: single-call step with closure (PyTorch protocol)
    # ------------------------------------------------------------------

    def step(self, closure: Optional[Callable] = None) -> None:
        """
        Standard PyTorch optimizer.step() with SAM's two-pass logic.

        ``closure`` MUST be provided and MUST return the loss tensor with
        gradients computed (i.e. closure() = loss; loss.backward() inside).
        It is called twice: at original w and at ascended w.
        """
        if closure is None:
            raise RuntimeError(
                "SAM.step requires a closure that recomputes loss and calls .backward(). See SAM docstring for usage."
            )

        # First pass: compute gradients at original w
        loss = closure()
        self.first_step(zero_grad=True)

        # Second pass: gradients at w + ε
        closure()
        self.second_step(zero_grad=True)
        return loss

    # ------------------------------------------------------------------
    # Internal: gradient norm
    # ------------------------------------------------------------------

    def _grad_norm(self) -> torch.Tensor:
        """
        Compute the L2 norm of gradients across all parameters
        (with adaptive scaling if ASAM is active).
        """
        # Reference device for stacking
        ref_device = self.param_groups[0]["params"][0].device

        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["adaptive"]:
                    g = (torch.abs(p) * p.grad).norm(p=2).to(ref_device)
                else:
                    g = p.grad.norm(p=2).to(ref_device)
                norms.append(g)

        if not norms:
            return torch.tensor(0.0, device=ref_device)
        return torch.norm(torch.stack(norms), p=2)

    # ------------------------------------------------------------------
    # State dict (delegate to base optimizer)
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ---------------------------------------------------------------------------
# Helper: SAM-aware training step
# ---------------------------------------------------------------------------


def sam_train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: SAM,
    extra_forward_args: Optional[dict] = None,
) -> float:
    """
    Drop-in replacement for a standard train step using SAM.

    Returns the scalar loss value (Python float) for logging.
    """
    extra_forward_args = extra_forward_args or {}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        outputs = model(inputs, **extra_forward_args)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss

    # Two-pass SAM
    loss = closure()
    optimizer.first_step(zero_grad=True)

    outputs2 = model(inputs, **extra_forward_args)
    loss2 = criterion(outputs2, targets)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    return float(loss.detach())


# ---------------------------------------------------------------------------
# SAM-aware GNN training step (full-batch graph)
# ---------------------------------------------------------------------------


def sam_train_step_graph(
    model: nn.Module,
    data,  # PyG Data object with .x, .edge_index, .y
    criterion: nn.Module,
    optimizer: SAM,
    nuc_ids: Optional[torch.Tensor] = None,
    aa_ids: Optional[torch.Tensor] = None,
) -> float:
    """
    SAM training step adapted for full-batch GNN node classification.

    Two forward passes through the graph; same edge_index / data.y both times.
    """
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, nuc_ids=nuc_ids, aa_ids=aa_ids)
    loss = criterion(logits, data.y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    logits2 = model(data.x, data.edge_index, nuc_ids=nuc_ids, aa_ids=aa_ids)
    loss2 = criterion(logits2, data.y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    return float(loss.detach())

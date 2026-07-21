# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/inference/triage.py
Uncertainty-aware triage engine for TEKNOFEST 2026.

Assigns human-readable risk flags to each prediction based on:
  - Probability margin (distance from decision threshold)
  - Shannon entropy of the probability distribution
  - Epistemic uncertainty (MC-Dropout std)

The binary Prediction column is NOT modified here.
Clinical_Flag is produced as a separate, audit-friendly column.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Flag constants ────────────────────────────────────────────────────────────
STABLE_PREDICTION = "STABLE_PREDICTION"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
BORDERLINE_PROBABILITY = "BORDERLINE_PROBABILITY"
HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
EXPERT_REVIEW_RECOMMENDED = "EXPERT_REVIEW_RECOMMENDED"

ALL_FLAGS = (
    STABLE_PREDICTION,
    LOW_CONFIDENCE,
    BORDERLINE_PROBABILITY,
    HIGH_UNCERTAINTY,
    EXPERT_REVIEW_RECOMMENDED,
)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of a Bernoulli(p) distribution, in bits."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = 1.0 - p
    return cast(np.ndarray, -(p * np.log2(p) + q * np.log2(q)))


class TriageEngine:
    """
    Assigns per-sample triage flags from predicted probabilities and uncertainty.

    Parameters
    ----------
    uncertainty_threshold : Uncertainty above which HIGH_UNCERTAINTY is flagged.
    borderline_margin     : Absolute distance from threshold below which
                           BORDERLINE_PROBABILITY is flagged.
    entropy_threshold     : Shannon entropy above which LOW_CONFIDENCE is flagged.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.30,
        borderline_margin: float = 0.10,
        entropy_threshold: float = 0.65,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.borderline_margin = borderline_margin
        self.entropy_threshold = entropy_threshold

    # ------------------------------------------------------------------
    def assign_flags(
        self,
        pathogenic_prob: np.ndarray,
        uncertainty: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Assign a single triage flag to each sample (worst-case priority).

        Priority order (highest → lowest):
          1. EXPERT_REVIEW_RECOMMENDED  — multiple risk factors
          2. HIGH_UNCERTAINTY           — uncertainty > threshold
          3. BORDERLINE_PROBABILITY     — prob near decision boundary
          4. LOW_CONFIDENCE             — high entropy
          5. STABLE_PREDICTION          — none of the above

        Parameters
        ----------
        pathogenic_prob : shape (n,) — P(Pathogenic) in [0, 1].
        uncertainty     : shape (n,) — epistemic uncertainty per sample.
        threshold       : Decision threshold (default 0.5).

        Returns
        -------
        flags : shape (n,) dtype object — one flag string per sample.
        """
        p = np.asarray(pathogenic_prob, dtype=float)
        u = np.asarray(uncertainty, dtype=float)

        if p.shape != u.shape:
            raise ValueError(f"pathogenic_prob and uncertainty must have the same shape; got {p.shape} and {u.shape}")

        margin = np.abs(p - threshold)
        entropy = _binary_entropy(p)

        is_high_uncertainty = u > self.uncertainty_threshold
        is_borderline = margin < self.borderline_margin
        is_low_confidence = entropy > self.entropy_threshold

        # Expert review: at least two risk factors co-occur
        risk_factor_count = is_high_uncertainty.astype(int) + is_borderline.astype(int) + is_low_confidence.astype(int)
        needs_expert = risk_factor_count >= 2

        flags = np.full(len(p), STABLE_PREDICTION, dtype=object)
        flags[is_low_confidence] = LOW_CONFIDENCE
        flags[is_borderline] = BORDERLINE_PROBABILITY
        flags[is_high_uncertainty] = HIGH_UNCERTAINTY
        flags[needs_expert] = EXPERT_REVIEW_RECOMMENDED

        n_expert = int(needs_expert.sum())
        if n_expert > 0:
            logger.info(
                "TriageEngine: %d/%d samples flagged as EXPERT_REVIEW_RECOMMENDED.",
                n_expert,
                len(p),
            )
        return flags

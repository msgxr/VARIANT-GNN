"""
src/scientific/conformal_prediction.py
=======================================
Conformal Prediction — Teorik Olarak Garanti Edilmiş Belirsizlik

Referanslar:
  - Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random World."
  - Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification."  arXiv:2107.07511
  - Romano, Patterson & Candès (2019). "Conformalized Quantile Regression."

TEKNOFEST 2026 bilimsel motivasyonu:
  Şartname §7.3 F1 = 2TP/(2TP+FP+FN) tek bir nokta tahmin metriği. Ancak
  klinik karar destek için (PDR §3.5) modelin "ne kadar emin?" sorusuna
  yanıt vermesi gerekir. Bayesian yaklaşımlar (MC Dropout vb.) prior
  varsayımlara dayanır ve teorik garanti vermez.

  Conformal Prediction *dağılımdan bağımsız* ve *teorik olarak garantili*
  bir framework sunar:
    P(y_test ∈ C(x_test)) ≥ 1 - α          ← marginal coverage guarantee
  burada C(x) tahmin kümesidir ve α güven seviyesi (örn. 0.10 → %90 coverage).

  Bu garanti modelin doğru olduğu varsayımına ihtiyaç DUYMAZ; sadece
  kalibrasyon ve test verisinin "değiştirilebilir" (exchangeable) olmasını
  gerektirir. External validation senaryosunda bile geçerlidir.

İmplementasyon:
  - APS (Adaptive Prediction Sets) — sınıf bazlı kümülatif olasılık eşikleme
    (Romano et al., 2020)
  - LAC (Least Ambiguous Classifier) — daha sıkı kümeler için
  - Marginal coverage + class-conditional coverage istatistikleri
  - Set-size dağılımı (singleton vs. abstain analizi)

Kullanım:
    cp = ConformalPredictor(alpha=0.10, method="APS")
    cp.fit(cal_proba, cal_labels)            # kalibrasyon setinde fit
    sets, scores = cp.predict_sets(test_proba)
    # sets[i] = list of likely classes (e.g., [1] or [0,1])
    coverage = cp.evaluate_coverage(sets, test_labels)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


ConformalMethod = Literal["APS", "LAC"]


# ---------------------------------------------------------------------------
# Sonuç kapsayıcısı
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Conformal Prediction empirical coverage analizi."""

    target_coverage: float  # 1 - alpha (örn. 0.90)
    empirical_coverage: float  # Gerçekleşen coverage
    average_set_size: float  # Ortalama küme boyutu
    singleton_rate: float  # |C(x)| = 1 olan örnekler / N
    abstention_rate: float  # |C(x)| ≥ 2 olan örnekler / N
    class_coverage: Dict[int, float]  # Sınıf bazlı coverage
    set_size_dist: Dict[int, int]  # Boyut histogramı

    def is_valid(self, tolerance: float = 0.02) -> bool:
        """Gerçek coverage hedefe ≥ tolerance içinde mi?"""
        return self.empirical_coverage >= (self.target_coverage - tolerance)

    def summary(self) -> str:
        lines = [
            "=== Conformal Prediction Coverage Raporu ===",
            f"  Hedef coverage         : {self.target_coverage:.3f}",
            f"  Empirik coverage        : {self.empirical_coverage:.3f}  "
            f"({'✓ GEÇERLİ' if self.is_valid() else '✗ DÜŞÜK'})",
            f"  Ortalama küme boyutu    : {self.average_set_size:.3f}",
            f"  Singleton oranı (kesin) : {self.singleton_rate:.1%}",
            f"  Çekimserlik (|C|≥2)     : {self.abstention_rate:.1%}",
            "",
            "  Sınıf bazlı coverage:",
        ]
        for cls, cov in self.class_coverage.items():
            name = "Pathogenic" if cls == 1 else "Benign"
            lines.append(f"    {name:12s}: {cov:.3f}")
        lines.append("")
        lines.append("  Küme boyutu dağılımı:")
        for size, count in sorted(self.set_size_dist.items()):
            lines.append(f"    |C|={size}: {count} örnek")
        return "\n".join(lines)

    def log(self) -> None:
        for line in self.summary().splitlines():
            logger.info(line)

    def as_dict(self) -> dict:
        return {
            "target_coverage": self.target_coverage,
            "empirical_coverage": round(self.empirical_coverage, 4),
            "average_set_size": round(self.average_set_size, 4),
            "singleton_rate": round(self.singleton_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "class_coverage": {str(k): round(v, 4) for k, v in self.class_coverage.items()},
            "set_size_distribution": {str(k): v for k, v in self.set_size_dist.items()},
            "is_valid": self.is_valid(),
        }


# ---------------------------------------------------------------------------
# Conformal Predictor
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """
    Split (Inductive) Conformal Prediction for binary variant classification.

    İki yöntem desteklenir:

    1. **APS** (Adaptive Prediction Sets, Romano et al. 2020):
       Conformity score = Σ p̂(y_k | x) for sorted classes until y_true.
       Daha akıllı: belirsiz örnekler için doğal olarak büyük kümeler üretir.

    2. **LAC** (Least Ambiguous Classifier, Sadinle et al. 2019):
       Conformity score = 1 - p̂(y_true | x).
       Daha sıkı: ortalama küme boyutu daha küçük ama abstain daha az.

    Parameters
    ----------
    alpha : float
        Hedef miscoverage rate (örn. 0.10 → %90 coverage garantisi).
    method : "APS" | "LAC"
        Conformity score yöntemi.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        method: ConformalMethod = "APS",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method not in ("APS", "LAC"):
            raise ValueError(f"method must be 'APS' or 'LAC', got {method!r}")
        self.alpha = alpha
        self.method = method

        self._q_hat: Optional[float] = None  # Calibrated quantile threshold
        self._n_calib: int = 0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Conformity scores
    # ------------------------------------------------------------------

    @staticmethod
    def _aps_score(proba: np.ndarray, label: int) -> float:
        """
        APS conformity score: cumulative sorted probability up to the true label.

        Sort probabilities descending; sum probabilities until the rank of the
        true class is reached.  Returns this cumulative sum.

        Higher score → less typical → larger prediction set when we threshold.
        """
        sorted_idx = np.argsort(proba)[::-1]
        cumsum = 0.0
        for idx in sorted_idx:
            cumsum += proba[idx]
            if idx == label:
                # Add a uniform random adjustment for tie-breaking (paper trick).
                # Use deterministic 0.5 for reproducibility.
                return cumsum - 0.5 * proba[idx]
        return cumsum

    @staticmethod
    def _lac_score(proba: np.ndarray, label: int) -> float:
        """LAC conformity score: 1 - p(y_true | x)."""
        return 1.0 - proba[label]

    def _compute_scores(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Vectorised conformity score computation over a calibration set."""
        scores = np.zeros(len(labels), dtype=float)
        for i in range(len(labels)):
            if self.method == "APS":
                scores[i] = self._aps_score(proba[i], int(labels[i]))
            else:  # LAC
                scores[i] = self._lac_score(proba[i], int(labels[i]))
        return scores

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        cal_proba: np.ndarray,
        cal_labels: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Calibrate the quantile threshold q̂ on a held-out validation set.

        The split-CP coverage guarantee requires:
          - cal_proba and cal_labels are NEVER seen during model training
          - test data is exchangeable with calibration data

        Parameters
        ----------
        cal_proba  : (N_cal, 2) calibration probabilities.
        cal_labels : (N_cal,) calibration labels.

        Returns self.
        """
        if cal_proba.ndim != 2 or cal_proba.shape[1] != 2:
            raise ValueError(f"cal_proba must be (N, 2); got {cal_proba.shape}")

        scores = self._compute_scores(cal_proba, cal_labels)
        self._n_calib = len(scores)

        # Minimum kalibrasyon boyutu kontrolü — istatistiksel stabilite için gerekli
        _MIN_CALIB = 30
        if self._n_calib < _MIN_CALIB:
            logger.warning(
                "ConformalPredictor: kalibrasyon seti cok kucuk (n=%d < %d). "
                "Coverage garantisi (%0.0f%%) istatistiksel olarak guvenilir olmayabilir. "
                "En az %d ornek onerilir.",
                self._n_calib,
                _MIN_CALIB,
                (1 - self.alpha) * 100,
                _MIN_CALIB,
            )

        # Conformal quantile: ⌈(n+1)(1-α)⌉ / n (finite-sample correction)
        if self._n_calib == 0:
            raise ValueError("Kalibrasyon seti bos — conformal tahmin yapilamaz.")
        q_level = np.ceil((self._n_calib + 1) * (1 - self.alpha)) / self._n_calib
        q_level = min(q_level, 1.0)
        self._q_hat = float(np.quantile(scores, q_level, method="higher"))
        self._is_fitted = True

        logger.info(
            "ConformalPredictor calibrated [method=%s, α=%.2f, n=%d]: q̂ = %.4f",
            self.method,
            self.alpha,
            self._n_calib,
            self._q_hat,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_sets(
        self,
        test_proba: np.ndarray,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Generate prediction sets with provable coverage guarantee.

        Parameters
        ----------
        test_proba : (N_test, 2) probability matrix.

        Returns
        -------
        sets   : list of prediction sets — each is a sorted list of class labels.
                 Examples:
                   [1]    — kesin Pathogenic tahmini
                   [0]    — kesin Benign tahmini
                   [0, 1] — model emin değil; iki sınıf da kümede
        scores : (N_test, 2) all-class conformity scores (low = in-set).
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_sets().")

        n = len(test_proba)
        sets: List[List[int]] = []
        all_scores = np.zeros((n, 2), dtype=float)

        for i in range(n):
            proba_i = test_proba[i]
            in_set: List[int] = []
            for cls in (0, 1):
                if self.method == "APS":
                    s = self._aps_score(proba_i, cls)
                else:
                    s = self._lac_score(proba_i, cls)
                all_scores[i, cls] = s
                if s <= self._q_hat:
                    in_set.append(cls)

            # Empty set guard: take the most probable class (theoretical edge case)
            if not in_set:
                in_set = [int(np.argmax(proba_i))]
            sets.append(sorted(in_set))

        return sets, all_scores

    # ------------------------------------------------------------------
    # Coverage evaluation
    # ------------------------------------------------------------------

    def evaluate_coverage(
        self,
        sets: List[List[int]],
        true_labels: np.ndarray,
    ) -> CoverageReport:
        """
        Compute empirical coverage and set-size statistics on labelled test set.

        Returns a ``CoverageReport`` for analysis.
        """
        n = len(true_labels)
        covered = sum(1 for i in range(n) if int(true_labels[i]) in sets[i])
        sizes = [len(s) for s in sets]
        singletons = sum(1 for s in sizes if s == 1)
        abstain = sum(1 for s in sizes if s >= 2)

        # Class-conditional coverage
        class_cov: Dict[int, float] = {}
        for cls in (0, 1):
            mask = true_labels == cls
            if mask.sum() > 0:
                cov_cls = sum(1 for i in range(n) if mask[i] and cls in sets[i])
                class_cov[cls] = cov_cls / int(mask.sum())

        size_dist: Dict[int, int] = {}
        for s in sizes:
            size_dist[s] = size_dist.get(s, 0) + 1

        return CoverageReport(
            target_coverage=1.0 - self.alpha,
            empirical_coverage=covered / n if n > 0 else 0.0,
            average_set_size=float(np.mean(sizes)) if sizes else 0.0,
            singleton_rate=singletons / n if n > 0 else 0.0,
            abstention_rate=abstain / n if n > 0 else 0.0,
            class_coverage=class_cov,
            set_size_dist=size_dist,
        )

    # ------------------------------------------------------------------
    # Convenience: end-to-end pathology decision
    # ------------------------------------------------------------------

    def predict_with_decision(
        self,
        test_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Yüksek seviyeli tahmin API'si — her örnek için klinik karar üretir:

          - "Pathogenic"      : C = {1}
          - "Benign"          : C = {0}
          - "Uzman Değerlendirmesi Gerekli" : C = {0, 1}  (model abstain)

        Parameters
        ----------
        test_proba : (N, 2) probability matrix.

        Returns
        -------
        dict with arrays:
          - decisions      : (N,) string array  ["Pathogenic" | "Benign" | "Abstain"]
          - prediction_sets: (N,) list of sorted lists
          - confidence     : (N,) float — 0 if abstain, else max-prob
        """
        sets, _ = self.predict_sets(test_proba)
        n = len(sets)
        decisions = np.empty(n, dtype=object)
        confidence = np.zeros(n, dtype=float)

        for i, s in enumerate(sets):
            if len(s) == 1:
                if s[0] == 1:
                    decisions[i] = "Pathogenic"
                else:
                    decisions[i] = "Benign"
                confidence[i] = float(test_proba[i, s[0]])
            else:
                decisions[i] = "Abstain"
                confidence[i] = 0.0

        return {
            "decisions": decisions,
            "prediction_sets": sets,
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Mondrian (panel-conditional) Conformal Predictor
# ---------------------------------------------------------------------------


class MondrianConformalPredictor:
    """
    Mondrian Conformal Prediction — group-conditional coverage guarantee.

    Standard CP guarantees marginal coverage (overall ≥ 1-α) but does NOT
    guarantee that each group (e.g. each panel) gets ≥ 1-α coverage.

    Mondrian CP fits a separate quantile q̂_g for each group g, providing:
      P(y_test ∈ C(x_test) | group=g) ≥ 1 - α   for every group g

    TEKNOFEST 2026 bağlamı:
      Şartname §3.2 dört bağımsız panel tanımlar; jüri her panelde ayrı
      F1 hesaplayabilir. Mondrian CP her panelin coverage seviyesini
      garanti altına alır → adil ve panel-bazında güvenilir tahminler.

    Kullanım:
        cp = MondrianConformalPredictor(alpha=0.10, method="APS")
        cp.fit(cal_proba, cal_labels, cal_panels)
        sets = cp.predict_sets(test_proba, test_panels)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        method: ConformalMethod = "APS",
        min_calibration_size: int = 20,
    ) -> None:
        self.alpha = alpha
        self.method = method
        self.min_calibration_size = min_calibration_size

        self._predictors: Dict[str, ConformalPredictor] = {}
        self._fallback: Optional[ConformalPredictor] = None
        self._is_fitted: bool = False

    def fit(
        self,
        cal_proba: np.ndarray,
        cal_labels: np.ndarray,
        cal_groups: np.ndarray,
    ) -> "MondrianConformalPredictor":
        """
        Fit a per-group conformal predictor + a global fallback.

        Groups with fewer than ``min_calibration_size`` samples fall back to
        the global predictor.
        """
        # Global fallback (per-marginal coverage)
        self._fallback = ConformalPredictor(alpha=self.alpha, method=self.method)
        self._fallback.fit(cal_proba, cal_labels)

        unique_groups = np.unique(cal_groups)
        for g in unique_groups:
            mask = cal_groups == g
            n_g = int(mask.sum())
            if n_g < self.min_calibration_size:
                logger.info(
                    "Mondrian CP: group '%s' has only %d samples (< %d) — using global fallback predictor.",
                    g,
                    n_g,
                    self.min_calibration_size,
                )
                continue
            cp_g = ConformalPredictor(alpha=self.alpha, method=self.method)
            cp_g.fit(cal_proba[mask], cal_labels[mask])
            self._predictors[str(g)] = cp_g
            logger.info(
                "Mondrian CP: group '%s' calibrated (n=%d, q̂=%.4f).",
                g,
                n_g,
                cp_g._q_hat,
            )

        self._is_fitted = True
        return self

    def predict_sets(
        self,
        test_proba: np.ndarray,
        test_groups: np.ndarray,
    ) -> List[List[int]]:
        """Per-group prediction set generation."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_sets().")

        n = len(test_proba)
        sets: List[List[int]] = []
        for i in range(n):
            g = str(test_groups[i])
            cp = self._predictors.get(g, self._fallback)
            cp_sets, _ = cp.predict_sets(test_proba[i : i + 1])
            sets.append(cp_sets[0])
        return sets

    def evaluate_per_group(
        self,
        test_proba: np.ndarray,
        test_labels: np.ndarray,
        test_groups: np.ndarray,
    ) -> Dict[str, CoverageReport]:
        """Per-group coverage evaluation."""
        sets = self.predict_sets(test_proba, test_groups)
        unique_groups = np.unique(test_groups)
        reports: Dict[str, CoverageReport] = {}
        for g in unique_groups:
            mask = test_groups == g
            g_sets = [sets[i] for i in range(len(sets)) if mask[i]]
            cp = self._predictors.get(str(g), self._fallback)
            reports[str(g)] = cp.evaluate_coverage(g_sets, test_labels[mask])
        return reports

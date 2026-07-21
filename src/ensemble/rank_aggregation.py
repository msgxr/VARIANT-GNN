# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/ensemble/rank_aggregation.py
==================================
Rank Aggregation Ensemble — Borda Count ve Kemeny-Young

Bilimsel motivasyon:
  Ağırlıklı olasılık ortalaması, modellerin farklı kalibrasyon seviyelerinde
  olduğu durumlarda başarısız olabilir. Rank-bazlı birleştirme yöntemleri:
    1. Bireysel modellerin mutlak olasılık ölçeğine bağımlı değildir
    2. Calibrasyon hatalarına karşı daha dayanıklıdır (robust)
    3. Ensemble üyelerinin birbirinden farklı dağılım gösterdiği heterogen
       topluluklarda (XGBoost + GNN + DNN + LGBM) daha istikrarlı sonuç verir

  Borda Count (Borda 1781):
    Her model için varyantları olasılığa göre sıralar.
    Her varyantın sıra puanları toplanarak final skor elde edilir.
    Son skor yeniden [0,1] aralığına normalize edilir.

  Referans:
    Dwork et al. (2001). Rank Aggregation Methods for the Web.
    WWW 2001.

TEKNOFEST 2026 §7.3 uyumu:
  Rank aggregation sonucu Pathogenic_Probability olarak kullanılır.
  Binary F1 (pos_label=1) üzerinde eşik optimizasyonu yapılır.

Kullanım:
    from src.ensemble.rank_aggregation import RankAggregator
    agg = RankAggregator(method="borda")
    proba_final = agg.aggregate([xgb_proba, lgbm_proba, gnn_proba, dnn_proba])
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)

AggMethod = Literal["borda", "reciprocal_rank", "geometric_mean"]


class RankAggregator:
    """
    Çok modelli rank tabanlı ensemble birleştirici.

    Parameters
    ----------
    method   : "borda" | "reciprocal_rank" | "geometric_mean"
    weights  : Model ağırlıkları (None → eşit ağırlık)
    """

    def __init__(
        self,
        method: AggMethod = "borda",
        weights: Optional[List[float]] = None,
    ) -> None:
        self.method = method
        self.weights = weights

    def aggregate(self, proba_list: List[np.ndarray]) -> np.ndarray:
        """
        Birden fazla modelin P(Pathogenic) olasılıklarını birleştirir.

        Parameters
        ----------
        proba_list : Her eleman [N, 2] veya [N] şeklinde P(Pathogenic) matrisi

        Returns
        -------
        proba_final : [N] P(Pathogenic) — normalize edilmiş rank skoru
        """
        probas_1d: List[np.ndarray] = []
        for p in proba_list:
            arr = np.asarray(p)
            if arr.ndim == 2:
                arr = arr[:, 1]
            probas_1d.append(arr)

        K = len(probas_1d)
        N = probas_1d[0].shape[0]

        if self.weights is None:
            w = np.ones(K) / K
        else:
            w = np.asarray(self.weights[:K], dtype=float)
            w = w / w.sum()

        if self.method == "borda":
            return self._borda(probas_1d, w, N)
        elif self.method == "reciprocal_rank":
            return self._reciprocal_rank(probas_1d, w, N)
        elif self.method == "geometric_mean":
            return self._geometric_mean(probas_1d, w)
        else:
            raise ValueError(f"Bilinmeyen method: {self.method}")

    # ── Borda Count ─────────────────────────────────────────────────────────

    def _borda(self, probas: List[np.ndarray], w: np.ndarray, N: int) -> np.ndarray:
        borda_sum = np.zeros(N)
        for i, p in enumerate(probas):
            ranks = self._rank(p)  # yüksek olasılık → yüksek sıra
            borda_sum += w[i] * ranks
        return self._normalize(borda_sum)

    # ── Reciprocal Rank ──────────────────────────────────────────────────────

    def _reciprocal_rank(self, probas: List[np.ndarray], w: np.ndarray, N: int) -> np.ndarray:
        rr_sum = np.zeros(N)
        for i, p in enumerate(probas):
            ranks = self._rank(p)  # 1..N
            rr_sum += w[i] * (1.0 / (N - ranks + 1))  # düşük sıra → yüksek rr
        return self._normalize(rr_sum)

    # ── Geometric Mean ───────────────────────────────────────────────────────

    def _geometric_mean(self, probas: List[np.ndarray], w: np.ndarray) -> np.ndarray:
        log_sum = np.zeros_like(probas[0], dtype=float)
        for i, p in enumerate(probas):
            log_sum += w[i] * np.log(np.clip(p, 1e-9, 1 - 1e-9))
        return cast(np.ndarray, np.exp(log_sum))

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    @staticmethod
    def _rank(p: np.ndarray) -> np.ndarray:
        """P(Pathogenic) → sıralama puanı (yüksek prob → yüksek sıra)."""
        order = np.argsort(np.argsort(p))  # 0-based rank
        return order.astype(float)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """[min,max] → [0,1] min-max normalizasyon."""
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.full_like(x, 0.5)
        return cast(np.ndarray, (x - mn) / (mx - mn))

    def compare_with_weighted_avg(
        self,
        proba_list: List[np.ndarray],
        y_true: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """
        Rank aggregation vs ağırlıklı ortalama F1 karşılaştırması.
        Her iki yöntemin Binary F1 skorunu döner.
        """
        from sklearn.metrics import f1_score

        rank_proba = self.aggregate(proba_list)
        rank_preds = (rank_proba >= threshold).astype(int)
        rank_f1 = float(f1_score(y_true, rank_preds, average="binary", pos_label=1, zero_division=0))

        probas_1d = [np.asarray(p)[:, 1] if np.asarray(p).ndim == 2 else np.asarray(p) for p in proba_list]
        if self.weights is None:
            w = np.ones(len(probas_1d)) / len(probas_1d)
        else:
            w = np.asarray(self.weights[: len(probas_1d)]) / sum(self.weights[: len(probas_1d)])

        avg_proba = sum(wi * pi for wi, pi in zip(w, probas_1d))
        avg_preds = (avg_proba >= threshold).astype(int)
        avg_f1 = float(f1_score(y_true, avg_preds, average="binary", pos_label=1, zero_division=0))

        logger.info(
            "Rank Aggregation (%s) F1=%.4f vs Weighted Avg F1=%.4f",
            self.method,
            rank_f1,
            avg_f1,
        )
        return {
            "rank_aggregation_f1": rank_f1,
            "weighted_avg_f1": avg_f1,
            "rank_method": self.method,
            "delta_f1": round(rank_f1 - avg_f1, 4),
        }

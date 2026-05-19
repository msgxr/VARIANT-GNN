"""
src/core/graph/builder.py
==========================
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Graf Oluşturma Katmanı

Strateji hiyerarşisi:
  CorrelationGraphBuilder — özellikler arası korelasyon grafı (özellikler = düğüm)
  KNNGraphBuilder         — sklearn tabanlı k-NN örnek grafı (legacy)
  SampleKNNGraphBuilder   — TEKNOFEST 2026 birincil seçim:
                            Koordinat-bağımsız, kosinüs-benzerlik tabanlı k-NN
                            örnek grafı.  Her VARİANT bir düğümdür.
                            Genomik adres (Chr/Pos) kullanılmaz — §3.2 uyumlu.

Yapılandırma uyumu:
  configs/default.yaml → gnn.knn_k = 10
                          gnn.knn_threshold = 0.30  (kenar budama eşiği)

Şartname §3.2:
  "Yarışma veri setinde varyantların genomik adres bilgileri tamamen
   gizlenmiştir."  →  Yalnızca biyolojik/hesaplamalı özellikler kullanılır.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soyut temel sınıf
# ---------------------------------------------------------------------------


class GraphBuilder(ABC):
    """Tüm graf oluşturma stratejileri için ortak protokol."""

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "GraphBuilder":
        """Eğitim verisinden graf topolojisini öğren."""
        ...

    @abstractmethod
    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        """Tek bir ön işlenmiş özellik vektörünü PyG Data nesnesine dönüştür."""
        ...

    @property
    @abstractmethod
    def edge_index(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def edge_attr(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Korelasyon bazlı özellik grafı
# ---------------------------------------------------------------------------


class CorrelationGraphBuilder(GraphBuilder):
    """
    Özellikler arasındaki Pearson korelasyonuna dayalı özellik-etkileşim grafı.

    Her düğüm bir özelliktir.  Korelasyonu ``corr_threshold``'u aşan özellik
    çiftleri kenar ile bağlanır; kenar ağırlığı |korelasyon| değeridir.

    Bu yapı orijinal VARIANT-GNN'in özellik-grafidir; VariantGATv2GNN'de
    "özellik düğümü" yaklaşımı için kullanılır.
    """

    def __init__(self, corr_threshold: float = 0.25) -> None:
        self.corr_threshold   = corr_threshold
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr:  Optional[torch.Tensor] = None
        self._n_features: int = 0

    def fit(self, X_train: np.ndarray) -> "CorrelationGraphBuilder":
        corr = np.corrcoef(X_train, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        n    = X_train.shape[1]

        edges:   List[List[int]] = []
        weights: List[float]     = []

        for i in range(n):
            for j in range(n):
                if i != j and abs(corr[i, j]) >= self.corr_threshold:
                    edges.append([i, j])
                    weights.append(float(abs(corr[i, j])))

        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self._edge_attr  = torch.tensor(weights, dtype=torch.float)
        else:
            self._edge_index = torch.empty((2, 0), dtype=torch.long)
            self._edge_attr  = torch.empty((0,), dtype=torch.float)

        self._n_features = n
        logger.info(
            "CorrelationGraph: %d özellik düğümü, %d yönlü kenar (eşik=%.2f)",
            n, len(edges), self.corr_threshold,
        )
        return self

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        x_t = torch.tensor(x_row, dtype=torch.float).unsqueeze(1)  # [N_feats, 1]
        y_t = torch.tensor([label], dtype=torch.long) if label is not None else None
        return Data(
            x          = x_t,
            edge_index = self._edge_index,
            edge_attr  = self._edge_attr,
            y          = y_t,
        )

    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError("CorrelationGraphBuilder henüz fit edilmedi.")
        return self._edge_index

    @property
    def edge_attr(self) -> torch.Tensor:
        if self._edge_attr is None:
            raise RuntimeError("CorrelationGraphBuilder henüz fit edilmedi.")
        return self._edge_attr


# ---------------------------------------------------------------------------
# Legacy sklearn k-NN örnek grafı
# ---------------------------------------------------------------------------


class KNNGraphBuilder(GraphBuilder):
    """
    Öklid mesafesi tabanlı k-NN örnek grafı (sklearn backend).

    Her ÖRNEK (varyant) bir düğümdür.  Özellik uzayında komşu olan
    varyantlar kenar ile bağlanır.  SampleKNNGraphBuilder'ın daha eski,
    sklearn-bağımlı versiyonudur.  Yeni kod için SampleKNNGraphBuilder tercih edin.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_attr:  Optional[torch.Tensor] = None

    def fit(self, X_train: np.ndarray) -> "KNNGraphBuilder":
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(X_train)), metric="euclidean")
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_train)

        edges:   List[Tuple[int, int]] = []
        weights: List[float]           = []

        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            for dist, j in zip(dists[1:], nbrs[1:]):
                edges.append((i, j))
                edges.append((j, i))
                weights.extend([float(dist), float(dist)])

        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self._edge_attr  = torch.tensor(weights, dtype=torch.float)
        else:
            self._edge_index = torch.empty((2, 0), dtype=torch.long)
            self._edge_attr  = torch.empty((0,), dtype=torch.float)

        logger.info(
            "KNNGraph: %d örnek, k=%d, %d kenar",
            len(X_train), self.k, len(edges),
        )
        return self

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        raise NotImplementedError(
            "KNNGraphBuilder örnek grafı oluşturur, satır başına değil. "
            "CorrelationGraphBuilder.row_to_graph() kullanın."
        )

    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index is None:
            raise RuntimeError("KNNGraphBuilder henüz fit edilmedi.")
        return self._edge_index

    @property
    def edge_attr(self) -> torch.Tensor:
        if self._edge_attr is None:
            raise RuntimeError("KNNGraphBuilder henüz fit edilmedi.")
        return self._edge_attr


# ---------------------------------------------------------------------------
# SampleKNNGraphBuilder — TEKNOFEST 2026 birincil seçim
# ---------------------------------------------------------------------------


class SampleKNNGraphBuilder:
    """
    Koordinat-bağımsız, kosinüs-benzerlik k-NN örnek grafı oluşturucu.

    TEKNOFEST 2026 §3.2 uyumu:
      - Genomik adres (Chr/Pos) kullanılmaz — tamamen özellik-bazlı.
      - Bağlantılar kosinüs benzerliği ile kurulur (öklid değil).
      - ``cosine_threshold`` ile zayıf kenarlar budanır (isteğe bağlı).

    Yapılandırma:
      configs/default.yaml → gnn.knn_k = 10, gnn.knn_threshold = 0.30

    Her varyant bir düğümdür.  Düğüm özellikleri: ön işlenmiş sayısal özellik
    vektörü (biyokimyasal + evrimsel + popülasyon + in-silico).

    Kullanım
    --------
    >>> builder = SampleKNNGraphBuilder(k=10, cosine_threshold=0.30)
    >>> train_data = builder.build(X_train_scaled, y_train)  # eğitim grafı
    >>> test_data  = builder.build(X_test_scaled)            # çıkarım grafı
    """

    def __init__(
        self,
        k:                int   = 10,
        cosine_threshold: float = 0.0,   # 0.0 = budama yok
        add_self_loops:   bool  = False,
    ) -> None:
        """
        Parameters
        ----------
        k                : Her düğümün bağlanacağı komşu sayısı.
                           Şartname ve default.yaml uyumu: k=10.
        cosine_threshold : Bu değerin altındaki kosinüs benzerliğine sahip
                           kenarlar kaldırılır.  0.0 = budama yok.
                           Önerilen: 0.30 (default.yaml gnn.knn_threshold).
        add_self_loops   : Her düğüme kendi kendine döngü ekle (True/False).
        """
        self.k                = k
        self.cosine_threshold = cosine_threshold
        self.add_self_loops   = add_self_loops

    # ------------------------------------------------------------------
    def build(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Data:
        """
        [N_samples, N_features] özellik matrisinden PyG Data nesnesi oluştur.

        Parameters
        ----------
        X : Ön işlenmiş ve normalize edilmiş özellik matrisi (N, F).
        y : Opsiyonel etiket dizisi (N,) — None ise etiket eklenmez.

        Returns
        -------
        PyG Data:
          x          : (N, F)  düğüm özellik matrisi
          edge_index : (2, E)  kenar bağlantı indeksleri
          edge_attr  : (E,)    kenar ağırlıkları (kosinüs benzerliği)
          y          : (N,) veya None
          num_nodes  : N (açık olarak ayarlanır)
        """
        N = X.shape[0]
        if N < 2:
            # Tek örnek: boş kenar indexi ile oluştur
            x_t  = torch.tensor(X, dtype=torch.float)
            y_t  = torch.tensor(y, dtype=torch.long) if y is not None else None
            return Data(
                x          = x_t,
                edge_index = torch.empty((2, 0), dtype=torch.long),
                edge_attr  = torch.empty((0,),   dtype=torch.float),
                y          = y_t,
                num_nodes  = N,
            )

        x_tensor = torch.tensor(X, dtype=torch.float)
        k_actual = min(self.k, N - 1)

        edge_index, cos_sim = self._build_knn_cosine(x_tensor, k_actual)

        # Kosinüs eşiği ile budama
        if self.cosine_threshold > 0.0 and edge_index.shape[1] > 0:
            keep_mask  = cos_sim >= self.cosine_threshold
            edge_index = edge_index[:, keep_mask]
            cos_sim    = cos_sim[keep_mask]

        # Self-loop ekleme (opsiyonel)
        if self.add_self_loops:
            self_idx   = torch.arange(N, dtype=torch.long).unsqueeze(0).expand(2, -1)
            self_sim   = torch.ones(N, dtype=torch.float)
            edge_index = torch.cat([edge_index, self_idx], dim=1)
            cos_sim    = torch.cat([cos_sim, self_sim], dim=0)

        y_tensor: Optional[torch.Tensor] = None
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.long)

        logger.debug(
            "SampleKNNGraph: N=%d, k=%d, eşik=%.2f, kenar=%d",
            N, k_actual, self.cosine_threshold, edge_index.shape[1],
        )

        return Data(
            x          = x_tensor,
            edge_index = edge_index,
            edge_attr  = cos_sim,
            y          = y_tensor,
            num_nodes  = N,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_knn_cosine(
        x: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kosinüs benzerliği tabanlı k-NN kenarları oluştur.

        Önce torch_geometric.nn.knn_graph (torch-cluster backend) dener.
        Kurulu değilse saf PyTorch matris çarpımı ile yedek hesaplama yapar.

        Returns
        -------
        edge_index : (2, N*k) yönlü kenar indeksleri
        cos_sim    : (N*k,)   her kenar için kosinüs benzerliği [−1, 1]
        """
        # ── Yol 1: torch_geometric.nn.knn_graph (tercih edilen) ─────────
        try:
            from torch_geometric.nn import knn_graph as _knn_graph
            edge_index = _knn_graph(x, k=k, cosine=True)   # (2, N*k) — yönlü
            src, dst   = edge_index
            x_norm     = F.normalize(x, p=2, dim=1)
            cos_sim    = (x_norm[src] * x_norm[dst]).sum(dim=1).clamp(-1.0, 1.0)
            return edge_index, cos_sim

        except (ImportError, RuntimeError):
            pass  # yedek hesaplamaya geç

        # ── Yol 2: Saf PyTorch (torch-cluster yoksa) ────────────────────
        # Tam kosinüs benzerlik matrisi: (N, N)
        _LARGE_N = 10_000
        if x.shape[0] > _LARGE_N:
            _mem_gb = x.shape[0] ** 2 * 4 / (1024 ** 3)
            logger.warning(
                "Buyuk veri seti (%d ornek): k-NN graf O(N^2) matris hesapliyor "
                "(tahmini ~%.1f GB bellek). Bellek hatasi olusabilir.",
                x.shape[0], _mem_gb,
            )
        x_norm  = F.normalize(x, p=2, dim=1)
        sim_mat = x_norm @ x_norm.t()           # (N, N)
        # Self-loop'ları dışla: köşegen −∞
        sim_mat.fill_diagonal_(float("-inf"))

        N = x.shape[0]
        topk_sim, topk_idx = sim_mat.topk(k, dim=1)   # (N, k)

        src = (
            torch.arange(N, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, k)
            .reshape(-1)
        )
        dst        = topk_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)    # (2, N*k)
        cos_sim    = topk_sim.reshape(-1).clamp(-1.0, 1.0)

        return edge_index, cos_sim

    # ------------------------------------------------------------------
    # GraphBuilder protokol uyumu (kullanılmaz ama tip uyumu için)
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "SampleKNNGraphBuilder":
        """Durum tutmaz; dönüşte self döner (API uyumluluğu)."""
        return self

    def row_to_graph(self, x_row: np.ndarray, label: Optional[int] = None) -> Data:
        raise NotImplementedError(
            "SampleKNNGraphBuilder tam örnek matrisi üzerinde çalışır. "
            "build(X, y) kullanın."
        )

    @property
    def edge_index(self) -> torch.Tensor:
        raise NotImplementedError("build(X) ile tam grafı oluşturun.")

    @property
    def edge_attr(self) -> torch.Tensor:
        raise NotImplementedError("build(X) ile tam grafı oluşturun.")


# ---------------------------------------------------------------------------
# Fabrika fonksiyonu
# ---------------------------------------------------------------------------


def get_graph_builder(strategy: str = "sample_knn", **kwargs) -> "GraphBuilder | SampleKNNGraphBuilder":
    """
    Strateji adına göre bir GraphBuilder örneği döndür.

    Desteklenen stratejiler
    -----------------------
    ``"sample_knn"``   (default) — TEKNOFEST 2026 birincil; SampleKNNGraphBuilder.
    ``"correlation"``            — Korelasyon bazlı özellik grafı.
    ``"knn"``                    — Legacy sklearn k-NN örnek grafı.

    Parameters
    ----------
    strategy : Strateji adı (büyük/küçük harf duyarsız).
    **kwargs : İlgili builder sınıfına aktarılan anahtar sözcük argümanları.
    """
    s = strategy.lower()
    if s == "sample_knn":
        return SampleKNNGraphBuilder(**kwargs)
    if s == "correlation":
        return CorrelationGraphBuilder(**kwargs)
    if s == "knn":
        return KNNGraphBuilder(**kwargs)
    raise ValueError(
        f"Bilinmeyen graf stratejisi: {strategy!r}. "
        f"Desteklenenler: 'sample_knn', 'correlation', 'knn'"
    )

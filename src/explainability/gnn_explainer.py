"""
src/explainability/gnn_explainer.py
GNNExplainer wrapper with safe fallback for environments where
torch_geometric.explain is unavailable or incompatible.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import torch

logger = logging.getLogger(__name__)

try:
    from torch_geometric.explain import Explainer, GNNExplainer

    _GNN_EXPLAINER_AVAILABLE = True
except ImportError:
    _GNN_EXPLAINER_AVAILABLE = False
    logger.warning("torch_geometric.explain not available — GNN explanations disabled.")


class GNNExplainerWrapper:
    """
    Safe wrapper around PyG GNNExplainer.

    Parameters
    ----------
    model  : Trained FeatureGNN model.
    device : torch.device for inference.
    """

    def __init__(self, model: Any, device: Optional[torch.device] = None) -> None:
        self._model = model
        self._device = device or torch.device("cpu")
        self._explainer = None

        if _GNN_EXPLAINER_AVAILABLE:
            try:
                self._explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=100),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="multiclass_classification",
                        task_level="graph",
                        return_type="raw",
                    ),
                )
                logger.info("GNNExplainer initialised.")
            except Exception as exc:
                logger.warning("GNNExplainer init failed: %s", exc)

    # ------------------------------------------------------------------
    def explain(self, data: Any) -> Optional[object]:
        """
        Produce an explanation for a single graph Data object.

        Returns the Explanation object or None if unavailable.
        """
        if self._explainer is None:
            return None
        try:
            data = data.to(self._device)
            explanation = self._explainer(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch,
            )
            return cast("object | None", explanation)
        except Exception as exc:
            logger.warning("GNNExplainer.explain failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def analyze_neighborhood_stats(
        self,
        edge_index: "torch.Tensor",
        labels: "torch.Tensor",
        edge_weight: Optional["torch.Tensor"] = None,
        mc_dropout_scores: Optional["torch.Tensor"] = None,
        uncertain_threshold: float = 0.30,
    ) -> dict:
        """Test seti üzerinde GNN komşuluk sınıf kompozisyonu analizi.

        PDR §4.4 için üretilen istatistikler:
          - Patojenik varyantların ortalama komşu sayısı ve aynı sınıf oranı
          - Benign varyantların ortalama komşu sayısı ve aynı sınıf oranı
          - Belirsiz varyantların karma komşuluk profili
          - Ortalama kenar ağırlıkları (cosine k-NN)

        Parameters
        ----------
        edge_index          : (2, E) kenar dizini (kaynak → hedef).
        labels              : (N,) binary etiket tensörü (1=Patojenik, 0=Benign).
        edge_weight         : (E,) kenar ağırlıkları; None ise tümü 1.0 kabul edilir.
        mc_dropout_scores   : (N,) MC Dropout belirsizlik skorları; None ise atlanır.
        uncertain_threshold : MC Dropout üzeri "belirsiz" sayılır.

        Returns
        -------
        dict:
          {
            "pathogenic": {"avg_neighbors": float, "same_class_ratio": float,
                           "avg_edge_weight": float},
            "benign":     {"avg_neighbors": float, "same_class_ratio": float,
                           "avg_edge_weight": float},
            "uncertain":  {"avg_neighbors": float, "mixed_ratio": float,
                           "avg_edge_weight": float, "n_variants": int},
            "overall":    {"avg_edge_weight": float, "n_edges": int, "n_nodes": int},
          }
        """
        import numpy as np

        try:
            ei = edge_index.cpu().numpy()  # (2, E)
            lbl = labels.cpu().numpy()  # (N,)
            ew = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(ei.shape[1], dtype=float)
            mcd = mc_dropout_scores.cpu().numpy() if mc_dropout_scores is not None else None
        except Exception as exc:
            logger.warning("neighborhood_stats tensor convert failed: %s", exc)
            return {}

        n_nodes = len(lbl)
        src, dst = ei[0], ei[1]

        stats: dict = {
            "pathogenic": {"neighbor_counts": [], "same_class_counts": [], "edge_weights": []},
            "benign": {"neighbor_counts": [], "same_class_counts": [], "edge_weights": []},
            "uncertain": {"neighbor_counts": [], "mixed_counts": [], "edge_weights": []},
        }

        for node in range(n_nodes):
            mask = src == node
            neighbors = dst[mask]
            weights = ew[mask]
            n_neigh = len(neighbors)

            if n_neigh == 0:
                continue

            node_lbl = int(lbl[node])
            neigh_labels = lbl[neighbors]
            same_class = int(np.sum(neigh_labels == node_lbl))
            avg_w = float(np.mean(weights))

            key = "pathogenic" if node_lbl == 1 else "benign"
            stats[key]["neighbor_counts"].append(n_neigh)
            stats[key]["same_class_counts"].append(same_class / n_neigh)
            stats[key]["edge_weights"].append(avg_w)

            if mcd is not None and float(mcd[node]) > uncertain_threshold:
                patho_neigh = int(np.sum(neigh_labels == 1))
                benign_neigh = int(np.sum(neigh_labels == 0))
                mixed = min(patho_neigh, benign_neigh) / n_neigh
                stats["uncertain"]["neighbor_counts"].append(n_neigh)
                stats["uncertain"]["mixed_counts"].append(mixed)
                stats["uncertain"]["edge_weights"].append(avg_w)

        def _agg(d: dict) -> dict:
            nc = d["neighbor_counts"]
            if not nc:
                return {"avg_neighbors": 0.0, "avg_edge_weight": 0.0}
            return {
                "avg_neighbors": round(float(np.mean(nc)), 2),
                "std_neighbors": round(float(np.std(nc)), 2),
                "avg_edge_weight": round(float(np.mean(d["edge_weights"])), 3),
            }

        result: dict = {}

        pat = _agg(stats["pathogenic"])
        sc_p = stats["pathogenic"]["same_class_counts"]
        pat["same_class_ratio"] = round(float(np.mean(sc_p)), 3) if sc_p else 0.0
        result["pathogenic"] = pat

        ben = _agg(stats["benign"])
        sc_b = stats["benign"]["same_class_counts"]
        ben["same_class_ratio"] = round(float(np.mean(sc_b)), 3) if sc_b else 0.0
        result["benign"] = ben

        unc = _agg(stats["uncertain"])
        mc_mix = stats["uncertain"]["mixed_counts"]
        unc["mixed_ratio"] = round(float(np.mean(mc_mix)), 3) if mc_mix else 0.0
        unc["n_variants"] = len(stats["uncertain"]["neighbor_counts"])
        result["uncertain"] = unc

        result["overall"] = {
            "avg_edge_weight": round(float(np.mean(ew)), 3),
            "n_edges": int(len(ew)),
            "n_nodes": n_nodes,
        }

        logger.info(
            "GNN neighborhood stats: pathogenic same_class=%.2f, "
            "benign same_class=%.2f, uncertain n=%d mixed_ratio=%.2f",
            result["pathogenic"].get("same_class_ratio", 0),
            result["benign"].get("same_class_ratio", 0),
            result["uncertain"].get("n_variants", 0),
            result["uncertain"].get("mixed_ratio", 0),
        )
        return result

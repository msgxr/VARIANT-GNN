"""
src/explainability/gnn_explainer.py
GNNExplainer wrapper with safe fallback for environments where
torch_geometric.explain is unavailable or incompatible.
"""
from __future__ import annotations

import logging
from typing import Optional

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

    def __init__(self, model, device: Optional[torch.device] = None) -> None:
        self._model  = model
        self._device = device or torch.device("cpu")
        self._explainer = None

        if _GNN_EXPLAINER_AVAILABLE:
            try:
                self._explainer = Explainer(
                    model          = model,
                    algorithm      = GNNExplainer(epochs=100),
                    explanation_type = "model",
                    node_mask_type = "attributes",
                    edge_mask_type = "object",
                    model_config   = dict(
                        mode       = "multiclass_classification",
                        task_level = "graph",
                        return_type= "log_probs",
                    ),
                )
                logger.info("GNNExplainer initialised.")
            except Exception as exc:
                logger.warning("GNNExplainer init failed: %s", exc)

    # ------------------------------------------------------------------
    def explain(self, data) -> Optional[object]:
        """
        Produce an explanation for a single graph Data object.

        Returns the Explanation object or None if unavailable.
        """
        if self._explainer is None:
            return None
        try:
            data = data.to(self._device)
            explanation = self._explainer(
                x          = data.x,
                edge_index = data.edge_index,
                batch      = data.batch,
            )
            return explanation
        except Exception as exc:
            logger.warning("GNNExplainer.explain failed: %s", exc)
            return None

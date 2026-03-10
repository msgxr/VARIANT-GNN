"""
src/explainability/feature_importance.py
Feature Importance analysis for VARIANT-GNN pathogenicity predictions.

TEKNOFEST 2026 — Final presentation requires algorithmic justification:
the module answers "WHY did the model predict Pathogenic?" by combining:

  1. SHAP (TreeExplainer on XGBoost)   — global feature importance
  2. GNN gradient-based saliency        — which node features drive GNN predictions
  3. Aggregated multi-source ranking    — merged ranking across both models

All results are exported as:
  - CSV table  : reports/feature_importance.csv
  - Bar chart  : reports/feature_importance.png
  - Single-sample explanation : reports/sample_<id>_explanation.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Optional dependency
try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    logger.warning("shap not installed — SHAP analysis will be skipped.")


# ---------------------------------------------------------------------------
# Helper: compute GNN gradient saliency
# ---------------------------------------------------------------------------

def _gnn_gradient_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    num_classes: int = 2,
) -> np.ndarray:
    """
    Compute gradient-based feature saliency for VariantSAGEGNN.

    For each sample the absolute gradient of the correct-class logit with
    respect to the input features is computed.  The mean across samples gives
    a per-feature importance score.

    Returns
    -------
    np.ndarray of shape [N_features] — mean absolute gradient importance.
    """
    model.eval()
    x_tensor     = torch.tensor(X, dtype=torch.float, requires_grad=True).to(device)
    edge_index_d = edge_index.to(device)

    logits = model(x_tensor, edge_index_d)   # [N, num_classes]
    # Sum the correct-class logits across all nodes
    targets = torch.tensor(y, dtype=torch.long, device=device)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    score   = (logits * one_hot).sum()

    score.backward()

    grads = x_tensor.grad.detach().cpu().numpy()    # [N, N_features]
    return np.abs(grads).mean(axis=0)               # [N_features]


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    """
    Combines SHAP (XGBoost) and gradient saliency (GNN) into a unified
    feature importance ranking.

    Parameters
    ----------
    feature_names   : List of feature column names.
    reports_dir     : Directory where CSV/PNG outputs are saved.
    """

    def __init__(
        self,
        feature_names: List[str],
        reports_dir:   str = "reports",
    ) -> None:
        self.feature_names = feature_names
        self.reports_dir   = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._shap_scores:    Optional[np.ndarray] = None
        self._gnn_scores:     Optional[np.ndarray] = None
        self._merged_scores:  Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap(
        self,
        xgb_model,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> None:
        """
        Compute mean |SHAP| values using XGBoost TreeExplainer.

        Parameters
        ----------
        xgb_model  : Fitted xgboost.XGBClassifier.
        X          : [N, F] processed feature matrix (training or validation).
        max_samples: Subsample for speed; set to -1 to use full X.
        """
        if not _SHAP_OK:
            logger.warning("SHAP not available; skipping SHAP computation.")
            return

        try:
            explainer = _shap.TreeExplainer(xgb_model)
            X_sub     = X[:max_samples] if max_samples > 0 else X
            sv        = explainer.shap_values(X_sub)
            # sv may be list[array] for multi-class
            if isinstance(sv, list):
                sv = np.stack([np.abs(s) for s in sv], axis=0).mean(axis=0)
            self._shap_scores = np.abs(sv).mean(axis=0)   # [N_features]
            logger.info("SHAP scores computed for %d samples.", len(X_sub))
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)

    def compute_gnn_gradients(
        self,
        gnn_model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        knn_k: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Compute mean absolute gradient saliency for VariantSAGEGNN.

        Parameters
        ----------
        gnn_model : Fitted VariantSAGEGNN model.
        X         : [N, F] processed feature matrix.
        y         : [N] integer labels.
        knn_k     : k used for cosine kNN graph construction.
        device    : Torch device; defaults to CPU.
        """
        if device is None:
            device = torch.device("cpu")

        try:
            from torch_geometric.nn import knn_graph
            x_t        = torch.tensor(X, dtype=torch.float)
            k_actual   = min(knn_k, len(X) - 1)
            edge_index = knn_graph(x_t, k=k_actual, cosine=True)

            self._gnn_scores = _gnn_gradient_importance(
                gnn_model, X, y, edge_index, device
            )
            logger.info("GNN gradient saliency computed for %d samples.", len(X))
        except Exception as exc:
            logger.warning("GNN gradient computation failed: %s", exc)

    def build_ranking(
        self,
        shap_weight: float = 0.5,
        gnn_weight:  float = 0.5,
    ) -> pd.DataFrame:
        """
        Merge SHAP and GNN scores into a combined feature ranking.

        Scores are min-max normalised before merging so they are comparable.

        Parameters
        ----------
        shap_weight : Weight applied to normalised SHAP scores.
        gnn_weight  : Weight applied to normalised GNN gradient scores.

        Returns
        -------
        DataFrame with columns:
            feature | shap_score | gnn_score | combined_score | rank
        """

        def _norm(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None:
                return np.zeros(len(self.feature_names))
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / (rng + 1e-12)

        shap_n = _norm(self._shap_scores)
        gnn_n  = _norm(self._gnn_scores)
        combined = shap_weight * shap_n + gnn_weight * gnn_n

        df = pd.DataFrame({
            "feature":        self.feature_names,
            "shap_score":     self._shap_scores if self._shap_scores is not None
                              else np.zeros(len(self.feature_names)),
            "gnn_grad_score": self._gnn_scores  if self._gnn_scores  is not None
                              else np.zeros(len(self.feature_names)),
            "combined_score": combined,
        }).sort_values("combined_score", ascending=False).reset_index(drop=True)

        df.insert(0, "rank", range(1, len(df) + 1))
        self._merged_scores = df
        return df

    def export_csv(self, filename: str = "feature_importance.csv") -> Path:
        """
        Save the feature importance ranking to CSV.

        Returns the path to the written file.
        """
        if self._merged_scores is None:
            self.build_ranking()

        out = self.reports_dir / filename
        self._merged_scores.to_csv(out, index=False)
        logger.info("Feature importance CSV saved → %s", out)
        return out

    def export_plot(
        self,
        filename:  str = "feature_importance.png",
        top_n:     int = 20,
        figsize:   Tuple[int, int] = (10, 7),
    ) -> Path:
        """
        Save a horizontal bar chart of the top-N features.

        Returns the path to the written file.
        """
        if self._merged_scores is None:
            self.build_ranking()

        df  = self._merged_scores.head(top_n)
        out = self.reports_dir / filename

        fig, ax = plt.subplots(figsize=figsize)
        y_pos   = range(len(df))
        colors  = ["#d62728" if s > df["combined_score"].median() else "#1f77b4"
                   for s in df["combined_score"]]

        ax.barh(y_pos, df["combined_score"], color=colors, edgecolor="white")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["feature"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Combined Importance Score (normalised)")
        ax.set_title(
            f"Top-{top_n} Feature Importances\n"
            "(red = above median; SHAP + GNN gradient fusion)"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature importance plot saved → %s", out)
        return out

    def explain_sample(
        self,
        xgb_model,
        x_sample: np.ndarray,
        sample_id: str = "sample",
        filename_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Per-sample explanation using SHAP.

        Returns a DataFrame with feature names and their SHAP contributions
        for the given sample.  Also exports to CSV.

        Parameters
        ----------
        xgb_model : Fitted XGBClassifier.
        x_sample  : [1, N_features] or [N_features] numpy array.
        sample_id : Identifier string used in the filename.
        """
        x_2d = x_sample.reshape(1, -1)

        shap_vals = np.zeros(x_2d.shape[1])
        if _SHAP_OK:
            try:
                explainer = _shap.TreeExplainer(xgb_model)
                sv        = explainer.shap_values(x_2d)
                if isinstance(sv, list):
                    # Multi-class: use class 1 (Pathogenic)
                    shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                else:
                    shap_vals = sv[0]
            except Exception as exc:
                logger.warning("Per-sample SHAP failed: %s", exc)

        fname = filename_prefix or f"sample_{sample_id}_explanation"
        df    = pd.DataFrame({
            "feature":    self.feature_names,
            "shap_value": shap_vals,
        }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

        out = self.reports_dir / f"{fname}.csv"
        df.to_csv(out, index=False)
        logger.info("Sample explanation saved → %s", out)
        return df

    # ------------------------------------------------------------------
    # Convenience: run all steps at once
    # ------------------------------------------------------------------

    def run(
        self,
        xgb_model,
        gnn_model: Optional[torch.nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        knn_k:   int = 5,
        top_n:   int = 20,
        device:  Optional[torch.device] = None,
    ) -> Dict[str, Path]:
        """
        Full pipeline: compute SHAP + GNN gradients → merge → export CSV + plot.

        Returns
        -------
        Dict with keys ``"csv"`` and ``"plot"`` mapping to output file Paths.
        """
        self.compute_shap(xgb_model, X)

        if gnn_model is not None:
            self.compute_gnn_gradients(gnn_model, X, y, knn_k=knn_k, device=device)

        self.build_ranking()
        csv_path  = self.export_csv()
        plot_path = self.export_plot(top_n=top_n)

        return {"csv": csv_path, "plot": plot_path}

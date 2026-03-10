"""
src/explainability/lime_explainer.py
LIME-based local explanation with safe fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lime
    import lime.lime_tabular
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False
    logger.warning("lime not installed — LIME explanations disabled.")


class LIMEExplainer:
    """
    LIME tabular explainer wrapper.

    Parameters
    ----------
    training_data  : Reference dataset for LIME background distribution.
    feature_names  : Column names (fall back to Feature_0, Feature_1 etc.).
    class_names    : Class labels for display.
    predict_fn     : Callable that takes (N, F) array and returns (N, C) probabilities.
    random_state   : Random seed for reproducibility.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]]   = None,
        predict_fn: Optional[Callable]     = None,
        random_state: int = 42,
    ) -> None:
        n = training_data.shape[1]
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(n)]
        self.class_names   = class_names   or ["Benign", "Pathogenic"]
        self.predict_fn    = predict_fn
        self._explainer    = None

        if _LIME_AVAILABLE:
            try:
                self._explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data  = training_data,
                    feature_names  = self.feature_names,
                    class_names    = self.class_names,
                    mode           = "classification",
                    random_state   = random_state,
                )
                logger.info("LIME LimeTabularExplainer initialised.")
            except Exception as exc:
                logger.warning("LIME init failed: %s", exc)

    # ------------------------------------------------------------------
    def explain_instance(
        self,
        instance:    np.ndarray,
        predict_fn:  Optional[Callable] = None,
        num_features: int               = 15,
        num_samples:  int               = 500,
        output_html:  Optional[str]     = None,
    ) -> Optional[object]:
        """Explain a single instance; return lime explanation object or None."""
        if self._explainer is None:
            return None
        fn = predict_fn or self.predict_fn
        if fn is None:
            logger.warning("No predict_fn provided to LIME explainer.")
            return None
        try:
            exp = self._explainer.explain_instance(
                data_row     = instance,
                predict_fn   = fn,
                num_features = num_features,
                num_samples  = num_samples,
            )
            if output_html:
                os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
                exp.save_to_file(output_html)
                logger.info("LIME explanation saved → %s", output_html)
            return exp
        except Exception as exc:
            logger.warning("LIME explain_instance failed: %s", exc)
            return None

"""
src/training/tune.py
Optuna-based hyperparameter optimisation for XGBoost.
Preprocessing runs inside each Optuna trial to avoid leakage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import get_settings
from src.features.preprocessing import build_preprocessor_from_config
from src.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
) -> float:
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "use_label_encoder": False,
        "n_jobs":           -1,
        "random_state":     seed,
    }

    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    f1s    = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        set_global_seed(seed + fold_idx)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        preprocessor = build_preprocessor_from_config()
        # Disable autoencoder during tuning for speed
        preprocessor.use_autoencoder = False
        X_tr_p, y_tr_r = preprocessor.fit_resample_train(X_tr, y_tr)
        X_val_p        = preprocessor.transform(X_val)

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_p, y_tr_r)
        preds = model.predict(X_val_p)
        f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

    return float(np.mean(f1s))


class ModelTuner:
    """Wraps Optuna study for XGBoost hyperparameter search."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        cv_folds: Optional[int] = None,
    ) -> None:
        self.X        = X
        self.y        = y
        self.n_trials = n_trials
        self.cfg      = get_settings()
        self.cv_folds = cv_folds or self.cfg.training.cv_folds

    def optimise_xgboost(self) -> Dict[str, Any]:
        """Run Optuna search and return the best parameter dict."""
        set_global_seed(self.cfg.seed)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _objective(
                trial, self.X, self.y, self.cv_folds, self.cfg.seed
            ),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        best = study.best_params
        logger.info("Best Optuna trial | F1=%.4f | params=%s", study.best_value, best)
        return best

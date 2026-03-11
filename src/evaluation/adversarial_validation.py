"""
src/evaluation/adversarial_validation.py
Adversarial Validation — TEKNOFEST 2026.

Train ve test veri setleri arasındaki domain shift'i tespit eder.

Yöntem:
  1. Train veriye label=0, test veriye label=1 ata
  2. Birleşik veriyi XGBoost ile sınıflandır (3-fold CV)
  3. ROC-AUC ≈ 0.5 → veri setleri benzer dağılımda (iyi)
     ROC-AUC ≈ 1.0 → ciddi domain shift var (kötü)

Jüri için: "Veri kalitesini de test ettik" argümanı.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Adversarial validation sonuçları."""
    auc_mean: float
    auc_std: float
    auc_per_fold: List[float]
    verdict: str  # 'İyi — Dağılımlar benzer', 'Uyarı — Hafif shift', 'Tehlike — Ciddi shift'
    top_shift_features: List[str]  # Domain shift'e en çok katkıda bulunan öznitelikler

    def log(self) -> None:
        logger.info("=== Adversarial Validation ===")
        logger.info("AUC: %.4f ± %.4f", self.auc_mean, self.auc_std)
        logger.info("Karar: %s", self.verdict)
        if self.top_shift_features:
            logger.info("En çok shift gösteren öznitelikler: %s",
                       ", ".join(self.top_shift_features[:5]))


def adversarial_validate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_folds: int = 3,
    random_state: int = 42,
) -> AdversarialResult:
    """
    Train/test domain shift analizi.

    Parameters
    ----------
    X_train : Training feature matrix [N_train, D].
    X_test  : Test feature matrix [N_test, D].
    feature_names : Feature isimleri (shift analizi için).
    n_folds : Cross-validation fold sayısı.

    Returns
    -------
    AdversarialResult with AUC scores, verdict, and top shift features.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    # Train=0, Test=1
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([
        np.zeros(len(X_train)),
        np.ones(len(X_test)),
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    importances = np.zeros(X_combined.shape[1])

    for train_idx, val_idx in skf.split(X_combined, y_combined):
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=random_state,
        )
        clf.fit(X_combined[train_idx], y_combined[train_idx])
        proba = clf.predict_proba(X_combined[val_idx])[:, 1]
        auc = roc_auc_score(y_combined[val_idx], proba)
        aucs.append(auc)
        importances += clf.feature_importances_

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))

    # Verdict
    if auc_mean < 0.55:
        verdict = "İyi — Dağılımlar benzer (domain shift yok)"
    elif auc_mean < 0.70:
        verdict = "Uyarı — Hafif domain shift tespit edildi"
    else:
        verdict = "Tehlike — Ciddi domain shift var"

    # Top shift features
    top_shift_features = []
    if feature_names is not None:
        top_idx = np.argsort(importances)[::-1][:10]
        top_shift_features = [feature_names[i] for i in top_idx]

    result = AdversarialResult(
        auc_mean=auc_mean,
        auc_std=auc_std,
        auc_per_fold=aucs,
        verdict=verdict,
        top_shift_features=top_shift_features,
    )
    result.log()
    return result

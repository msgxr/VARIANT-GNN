"""
src/inference/tta.py
=====================
Test-Time Augmentation (TTA) — Varyans Azaltma ve Belirsizlik Tahmini

Bilimsel motivasyon:
  Sınır bölgesi varyantları (P(Pathogenic) ≈ 0.5) için tek bir forward pass
  yüksek varyansa sahip. TTA k kez hafif pertürbasyon uygulayarak tahmin
  dağılımının momentlerini (ortalama, std) tahmin eder.

  Referans:
    Ayhan & Berens (2018). Test-time Data Augmentation for Estimation of
    Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks.
    MIDL 2018.

Şartname uyumu (§3.2, §7.3):
  - Dış veri kaynağı yok
  - Yalnızca çıkarım aşamasında — eğitim pipeline'ını etkilemez
  - Sonuç: tek bir Pathogenic_Probability skoru + uncertainty_tta

TTA gürültü stratejileri:
  1. Gaussian: N(0, σ × feature_std) — in-silico skorların ölçüm gürültüsü
  2. Dropout: model Dropout katmanlarını açık bırak (MC Dropout ile benzer)
  3. Combo: Gaussian + Dropout birleşimi (en güçlü varyans tahmini)

Kullanım:
    from src.inference.tta import TTAPredictor
    tta = TTAPredictor(n_augmentations=10, noise_scale=0.03)
    proba_mean, proba_std = tta.predict(ensemble, preprocessor, X_df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TTAResult:
    """TTA çıktısı."""
    proba_mean: np.ndarray       # [N, 2] — ortalama P(Benign), P(Pathogenic)
    proba_std:  np.ndarray       # [N] — P(Pathogenic) standart sapması
    predictions: np.ndarray     # [N] — 0/1 hard label (ortalama > thr)
    uncertainty_tta: np.ndarray # [N] — TTA belirsizlik skoru [0,1]
    n_augmentations: int


class TTAPredictor:
    """
    Test-Time Augmentation ile belirsizlik-farkında çıkarım.

    Parameters
    ----------
    n_augmentations : Kaç kez pertürbasyon uygulanacak (önerilen: 10-20)
    noise_scale     : Gaussian gürültü ölçeği — feature_std'nin katı
    threshold       : Patojenik karar eşiği (varsayılan: 0.5)
    seed            : Tekrarlanabilirlik seed'i
    """

    def __init__(
        self,
        n_augmentations: int = 10,
        noise_scale: float = 0.03,
        threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_augmentations = n_augmentations
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def predict(
        self,
        ensemble,
        preprocessor,
        X: np.ndarray,
        feature_stds: Optional[np.ndarray] = None,
    ) -> TTAResult:
        """
        TTA ile çıkarım yapar.

        Parameters
        ----------
        ensemble     : HybridEnsemble — predict(X_scaled) → (labels, proba)
        preprocessor : VariantPreprocessor — transform(X) → X_scaled
        X            : Ham özellik matrisi [N, F]
        feature_stds : Her özelliğin standart sapması (None → varyansdan hesaplanır)

        Returns
        -------
        TTAResult
        """
        X = np.asarray(X, dtype=np.float64)
        N, F = X.shape

        if feature_stds is None:
            feature_stds = np.nanstd(X, axis=0).clip(1e-8)

        all_probas: list = []

        for aug_idx in range(self.n_augmentations):
            if aug_idx == 0:
                X_aug = X.copy()
            else:
                noise = self.rng.normal(
                    loc=0.0,
                    scale=self.noise_scale * feature_stds,
                    size=X.shape,
                )
                X_aug = X + noise

            try:
                X_scaled = preprocessor.transform(X_aug)
                _, proba = ensemble.predict(X_scaled)
                proba = np.asarray(proba)
                if proba.ndim == 1:
                    proba = np.column_stack([1 - proba, proba])
                all_probas.append(proba)
            except Exception as exc:
                logger.warning("TTA aug %d başarısız (atlandı): %s", aug_idx, exc)
                continue

        if not all_probas:
            raise RuntimeError("TTA: hiçbir augmentation başarılı olmadı.")

        stack = np.stack(all_probas, axis=0)   # [K, N, 2]
        proba_mean = stack.mean(axis=0)         # [N, 2]
        proba_std  = stack[:, :, 1].std(axis=0) # [N] — P(Pathogenic) std

        predictions = (proba_mean[:, 1] >= self.threshold).astype(int)

        uncertainty_tta = np.clip(2.0 * proba_std, 0, 1)

        logger.info(
            "TTA tamamlandı: %d augmentation, N=%d, ortalama P(Path)=%.4f, "
            "ortalama tta_uncertainty=%.4f",
            len(all_probas), N,
            float(proba_mean[:, 1].mean()),
            float(uncertainty_tta.mean()),
        )

        return TTAResult(
            proba_mean      = proba_mean,
            proba_std       = proba_std,
            predictions     = predictions,
            uncertainty_tta = uncertainty_tta,
            n_augmentations = len(all_probas),
        )

    def predict_with_threshold(
        self,
        ensemble,
        preprocessor,
        X: np.ndarray,
        threshold: Optional[float] = None,
        feature_stds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Basit arayüz: (predictions, proba_pathogenic, uncertainty) döner.
        """
        thr = threshold if threshold is not None else self.threshold
        result = self.predict(ensemble, preprocessor, X, feature_stds)
        preds = (result.proba_mean[:, 1] >= thr).astype(int)
        return preds, result.proba_mean[:, 1], result.uncertainty_tta


def tta_predict_dataframe(
    ensemble,
    preprocessor,
    df: pd.DataFrame,
    feature_columns: list,
    n_augmentations: int = 10,
    noise_scale: float = 0.03,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    DataFrame → DataFrame: TTA ile zenginleştirilmiş tahmin çıktısı.

    Orijinal sütunlara şunları ekler:
      TTA_Probability     — ortalama P(Pathogenic)
      TTA_Uncertainty     — TTA std-bazlı belirsizlik [0,1]
      TTA_Prediction      — Patojenik / Benign
      TTA_Flag            — HIGH_UNCERTAINTY eğer > 0.30
    """
    X = df[feature_columns].values
    tta = TTAPredictor(n_augmentations=n_augmentations, noise_scale=noise_scale,
                       threshold=threshold, seed=seed)
    result = tta.predict(ensemble, preprocessor, X)

    out = df.copy()
    out["TTA_Probability"]  = result.proba_mean[:, 1]
    out["TTA_Uncertainty"]  = result.uncertainty_tta
    out["TTA_Prediction"]   = np.where(result.predictions == 1, "Pathogenic", "Benign")
    out["TTA_Flag"]         = np.where(result.uncertainty_tta > 0.30,
                                       "HIGH_UNCERTAINTY", "OK")
    return out

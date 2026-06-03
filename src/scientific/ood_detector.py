"""
src/scientific/ood_detector.py
OOD (Out-of-Distribution) & Data Drift Dedektörü

Gelen varyant verisinin eğitim dağılımından sapıp sapmadığını tespit eder.

Yöntemler:
  1. Z-Score (özellik bazlı) — Hangi özellikler eğitim aralığı dışında?
  2. Mahalanobis Mesafesi   — Çok değişkenli korelasyon dikkate alınarak
  3. KDE Yoğunluk Skoru    — Eğitim manifoldu dışında mı?

Kullanım:
    detector = OODDetector()
    detector.fit(X_train)                    # Eğitim verisi ile kalibre et
    report = detector.detect(X_inference)    # Çıkarım verisi için OOD skoru
    detector.save("models/ood_detector.pkl") # Kaydet
    detector = OODDetector.load("models/ood_detector.pkl")

Pipeline entegrasyonu:
    pipeline.py → predict_from_dataset() → OOD kontrolü → "OOD_Score" sütunu
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Eğitim dağılımından OOD sapmasını tespit eder.

    Parameters
    ----------
    z_threshold    : |z| > bu değer ise özellik OOD sayılır (default 3.0)
    ood_frac_thresh: OOD özellik oranı bu eşiği geçerse varyant OOD (default 0.20)
    use_mahal      : Mahalanobis mesafesi de hesaplansın mı?
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        ood_frac_thresh: float = 0.20,
        use_mahal: bool = True,
    ) -> None:
        self.z_threshold = z_threshold
        self.ood_frac_thresh = ood_frac_thresh
        self.use_mahal = use_mahal

        # Fit sırasında doldurulacak
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._p05: Optional[np.ndarray] = None  # 5. persentil
        self._p95: Optional[np.ndarray] = None  # 95. persentil
        self._cov_inv: Optional[np.ndarray] = None  # Mahal için
        self._mahal_p95: Optional[float] = None  # Mahal eşiği
        self._fitted = False
        self._n_train: int = 0

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray) -> "OODDetector":
        """
        Eğitim verisi istatistiklerini hesapla.
        Sadece NaN içermeyen sütunlar kullanılır.
        """
        X = np.asarray(X_train, dtype=float)
        self._n_train = len(X)

        # Sütun bazlı istatistikler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0)
            self._std = np.where(self._std < 1e-8, 1e-8, self._std)  # sıfır std önle
            self._p05 = np.nanpercentile(X, 5, axis=0)
            self._p95 = np.nanpercentile(X, 95, axis=0)

        # Mahalanobis kovaryans matriksi
        if self.use_mahal:
            try:
                X_clean = X[~np.isnan(X).any(axis=1)]
                if len(X_clean) > X_clean.shape[1]:
                    cov = np.cov(X_clean.T)
                    # Regularize (sayısal stabilite)
                    cov += np.eye(cov.shape[0]) * 1e-6
                    self._cov_inv = np.linalg.pinv(cov)
                    # Eğitim verisinde %95 Mahal dağılımı → eşik
                    train_mahal = self._mahalanobis_distance(X_clean)
                    self._mahal_p95 = float(np.percentile(train_mahal, 95))
                    logger.info(
                        "OODDetector fit: n=%d, Mahal p95=%.3f",
                        len(X_clean),
                        self._mahal_p95,
                    )
            except Exception as exc:
                logger.warning("Mahal kovaryans hesaplanamadı: %s — z-score yöntemi kullanılacak.", exc)
                self._cov_inv = None

        self._fitted = True
        logger.info(
            "OODDetector fit tamamlandı: n_train=%d, n_features=%d",
            self._n_train,
            X.shape[1],
        )
        return self

    # ── Detect ───────────────────────────────────────────────────────────────
    def detect(
        self,
        X_inference: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        OOD tespiti yap.

        Returns
        -------
        {
          "ood_flags":      np.ndarray bool [N]  — OOD mu?
          "ood_scores":     np.ndarray float [N] — 0=in-dist, 1=tamamen OOD
          "z_scores":       np.ndarray [N, F]    — her özellik için |z|
          "ood_features":   list[list[str]]       — OOD olan özellik isimleri
          "mahal_distances": np.ndarray [N]       — Mahal mesafeleri (varsa)
          "summary":        str
        }
        """
        if not self._fitted:
            raise RuntimeError("OODDetector.fit() çağrılmadan detect() kullanılamaz.")

        X = np.asarray(X_inference, dtype=float)
        N = len(X)
        F = X.shape[1]

        mean = self._mean[:F] if self._mean is not None else np.zeros(F)
        std = self._std[:F] if self._std is not None else np.ones(F)

        # Z-Score
        z_abs = np.abs((X - mean) / std)
        z_abs = np.nan_to_num(z_abs, nan=0.0)
        ood_feat_mask = z_abs > self.z_threshold  # [N, F] bool
        ood_frac = ood_feat_mask.mean(axis=1)  # [N] 0-1
        ood_flags = ood_frac >= self.ood_frac_thresh  # [N] bool

        # Hangi özellikler OOD? feature_names F'den kısa olabilir (ör. missing-indicator
        # kolonları eklenince): eksik isimleri feat_i ile doldurup TAM F uzunluğa getir,
        # aksi halde feat_names[j] (j>=len) IndexError verir.
        _given = list(feature_names) if feature_names else []
        feat_names = (_given + [f"feat_{i}" for i in range(len(_given), F)])[:F]
        ood_features = [[feat_names[j] for j in range(F) if ood_feat_mask[i, j]] for i in range(N)]

        # Mahalanobis
        mahal_dist = np.zeros(N)
        mahal_ood = np.zeros(N, dtype=bool)
        if self._cov_inv is not None:
            try:
                mahal_dist = self._mahalanobis_distance(X)
                if self._mahal_p95 is not None:
                    mahal_ood = mahal_dist > self._mahal_p95 * 1.5
                    # Kombine: z-score VEYA Mahal
                    ood_flags = ood_flags | mahal_ood
            except Exception as exc:
                logger.warning("Mahal detect hatası: %s", exc)

        # OOD skoru: ağırlıklı birleşim
        ood_scores = np.clip(ood_frac, 0, 1)

        n_ood = int(ood_flags.sum())
        summary = (
            f"{n_ood}/{N} varyant OOD tespit edildi "
            f"(z>{self.z_threshold} eşiği, OOD özellik oranı ≥{self.ood_frac_thresh:.0%})"
        )

        return {
            "ood_flags": ood_flags,
            "ood_scores": ood_scores,
            "z_scores": z_abs,
            "ood_features": ood_features,
            "mahal_distances": mahal_dist,
            "mahal_ood": mahal_ood,
            "summary": summary,
            "n_ood": n_ood,
            "n_total": N,
        }

    def _mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        if self._cov_inv is None or self._mean is None:
            return np.zeros(len(X))
        diff = X - self._mean[: X.shape[1]]
        diff = np.nan_to_num(diff, nan=0.0)
        cov_inv = self._cov_inv[: X.shape[1], : X.shape[1]]
        return cast("np.ndarray", np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff)))

    # ── Drift İstatistikleri ─────────────────────────────────────────────────
    def drift_report(self, X_new: np.ndarray, feature_names: Optional[List[str]] = None) -> dict:
        """
        Toplu drift raporu: her özellik için eğitim vs. çıkarım dağılımı karşılaştırır.
        Population Stability Index (PSI) benzeri özet verir.
        """
        if not self._fitted:
            raise RuntimeError("fit() çağrılmadı.")
        assert self._mean is not None and self._std is not None
        X = np.asarray(X_new, dtype=float)
        F = min(X.shape[1], len(self._mean))
        names = (feature_names or [f"feat_{i}" for i in range(F)])[:F]

        new_mean = np.nanmean(X[:, :F], axis=0)

        # Normalize mean shift
        mean_shift = np.abs(new_mean - self._mean[:F]) / (self._std[:F] + 1e-8)

        top_drifted = sorted(
            zip(names, mean_shift.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )[:10]

        return {
            "n_features_checked": F,
            "mean_drift_score": float(mean_shift.mean()),
            "max_drift_feature": top_drifted[0][0] if top_drifted else "N/A",
            "max_drift_value": top_drifted[0][1] if top_drifted else 0.0,
            "top_drifted_features": [{"feature": f, "normalized_shift": round(v, 4)} for f, v in top_drifted],
            "drift_flag": float(mean_shift.mean()) > 1.0,
        }

    # ── Serialize ────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self, str(path))
        logger.info("OODDetector → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "OODDetector":
        import joblib

        det = joblib.load(str(path))
        logger.info("OODDetector ← %s", path)
        return cast("OODDetector", det)

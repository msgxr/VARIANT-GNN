"""
src/scientific/label_quality.py
=================================
Confident Learning — Gürültülü Etiket Tespiti

Referans:
  Northcutt, Jiang & Chuang (2021). "Confident Learning: Estimating Uncertainty
  in Dataset Labels." JAIR 70, 1373–1411.  https://arxiv.org/abs/1911.00068

TEKNOFEST 2026 bağlamı:
  ClinVar'daki 3-4 yıldızlı varyantlar bile belirli oranda gürültü taşıyabilir
  (Benet-Pagès et al., 2023: ~4 % mislabelling rate in 4-star ClinVar entries).
  Etiket gürültüsü modeli sistematik olarak zayıflatır; eğitimden önce bunları
  tespit etmek §7.3 F1 skorunu +1–4 % artırabilir.

Algoritma:
  1. Çapraz-doğrulama ile her örnek için P̂(y=k | x) tahmin olasılıkları üret.
  2. Her sınıf için kalibre edilmiş olasılık eşikleri hesapla: t_k = mean P̂_k.
  3. Güven matrisi C[ŷ, y] oluştur: C[ŷ, y] = Σ_{i: P̂ŷ(xi) ≥ tŷ ∧ P̂y(xi) ≥ ty}
  4. Normalize et → sınıf başına beklenen gürültü oranları p(ŷ, y).
  5. Eğer p(ŷ ≠ y) > noise_threshold, örnek "low quality" olarak işaretlenir.

Hiçbir dış bağımlılık yok (sklearn + numpy); opsiyonel cleanlab uyumlu çıktı.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sonuç kapsayıcısı
# ---------------------------------------------------------------------------


@dataclass
class LabelQualityReport:
    """Gürültülü etiket analizi sonucu."""

    n_samples:          int
    n_flagged:          int
    estimated_noise_rate: float           # Toplam tahmini gürültü oranı [0, 1]

    flagged_indices:    List[int]         # Şüpheli örnek indisleri
    noise_matrix:       np.ndarray        # 2×2 normalize gürültü matrisi
    confidence_scores:  np.ndarray        # Her örnek için güven skoru [0, 1]
    label_quality_scores: np.ndarray      # Yüksek → temiz etiket

    # Sınıf başına gürültü oranları
    class_noise_rates:  Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Label Quality Report ===",
            f"  Toplam örnek         : {self.n_samples}",
            f"  Şüpheli etiket       : {self.n_flagged}  "
            f"({100*self.n_flagged/max(1,self.n_samples):.1f} %)",
            f"  Tahmini gürültü oranı: {100*self.estimated_noise_rate:.2f} %",
        ]
        for cls, rate in self.class_noise_rates.items():
            name = "Pathogenic" if cls == 1 else "Benign"
            lines.append(f"  Gürültü ({name:12s}): {100*rate:.2f} %")
        return "\n".join(lines)

    def log(self) -> None:
        for line in self.summary().splitlines():
            logger.info(line)

    def clean_mask(self) -> np.ndarray:
        """Return a boolean mask: True → keep this sample."""
        mask = np.ones(self.n_samples, dtype=bool)
        mask[self.flagged_indices] = False
        return mask


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def _compute_calibrated_thresholds(
    prob_matrix: np.ndarray,
    labels:      np.ndarray,
    n_classes:   int = 2,
) -> np.ndarray:
    """
    Her sınıf k için eşik t_k = mean_{i: y_i=k} P̂(y=k | x_i).

    Bu eşik, sınıf frekanslarından bağımsız bir güven kalibrasyonu sağlar
    ve dengesiz veri setlerinde threshold=0.5 sabit eşikten daha iyidir.
    """
    thresholds = np.zeros(n_classes)
    for k in range(n_classes):
        mask = labels == k
        if mask.sum() > 0:
            thresholds[k] = prob_matrix[mask, k].mean()
        else:
            thresholds[k] = 1.0 / n_classes
    return thresholds


def _build_confident_joint(
    prob_matrix: np.ndarray,
    labels:      np.ndarray,
    thresholds:  np.ndarray,
    n_classes:   int = 2,
) -> np.ndarray:
    """
    Güven matrisi C[ŷ, y]:
      C[ŷ, y] = |{i : argmax P̂(xi) = ŷ  AND  P̂(y | xi) ≥ t_y}|

    Bu, hem modelin en yüksek olasılıklı tahminini hem de gerçek sınıfın
    olasılığının kalibre eşiğini aşmasını gerektiren sıkı koşuldur.
    """
    cj = np.zeros((n_classes, n_classes), dtype=float)
    predicted_class = prob_matrix.argmax(axis=1)

    for i in range(len(labels)):
        y_hat = predicted_class[i]
        y_true = int(labels[i])
        # Geçerli sınıf koşulu: P̂(y_true | xi) ≥ t_{y_true}
        if prob_matrix[i, y_true] >= thresholds[y_true]:
            cj[y_hat, y_true] += 1.0

    return cj


def _label_quality_score(
    prob_matrix: np.ndarray,
    labels:      np.ndarray,
    noise_matrix: np.ndarray,
) -> np.ndarray:
    """
    Her örnek için etiket kalite skoru:
      LQS(i) = P̂(y=yi | xi) / (P̂(y=yi | xi) + Σ_{k≠yi} p(ŷ=k, y=yi) / p(y=yi))

    0 → düşük kalite (muhtemelen yanlış etiket)
    1 → yüksek kalite (etiket güvenilir)
    """
    n = len(labels)
    scores = np.zeros(n, dtype=float)
    # Sınıf frekansları (prior)
    class_counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    class_prior  = class_counts / class_counts.sum()

    for i in range(n):
        y = int(labels[i])
        p_correct = prob_matrix[i, y]

        # Expected noise contribution from the other class
        # Σ_{k≠y} noise_matrix[k, y] / p(y)
        noise_contamination = 0.0
        for k in range(noise_matrix.shape[0]):
            if k != y:
                noise_contamination += noise_matrix[k, y] / max(class_prior[y], 1e-9)

        scores[i] = p_correct / max(p_correct + noise_contamination, 1e-9)

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------


class ConfidentLearner:
    """
    Varyant veri setindeki gürültülü etiket örnekleri tespit eder.

    Parameters
    ----------
    noise_threshold : float
        Bir örneğin şüpheli sayılması için gereken maksimum etiket kalite
        skoru (LQS < noise_threshold → şüpheli).  Varsayılan 0.50 makul
        bir F1/temizlik dengesi sağlar; daha agresif temizlik için 0.40
        kullanılabilir.
    cv_folds : int
        Çapraz-doğrulama dilim sayısı. En az 3 önerilir.
    base_estimator : optional
        Olasılık tahmini için kullanılacak sklearn-uyumlu sınıflandırıcı.
        None verilirse dahili olarak XGBClassifier kullanılır.
    """

    def __init__(
        self,
        noise_threshold: float = 0.45,
        cv_folds: int = 5,
        base_estimator = None,
    ) -> None:
        self.noise_threshold = noise_threshold
        self.cv_folds        = cv_folds
        self._base_estimator = base_estimator

    # ------------------------------------------------------------------
    # Olasılık üretimi (çapraz-doğrulama)
    # ------------------------------------------------------------------

    def _get_oof_probabilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Out-of-fold predict_proba via StratifiedKFold.

        Returns
        -------
        prob_matrix : (N, 2) array of [P(Benign), P(Pathogenic)]
        """
        n_classes   = 2
        prob_matrix = np.zeros((len(X), n_classes), dtype=float)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            estimator = self._make_estimator()
            estimator.fit(X_tr, y_tr)
            proba = estimator.predict_proba(X_val)
            prob_matrix[val_idx] = proba

            preds_val = proba.argmax(axis=1)
            fold_f1 = f1_score(y_val, preds_val, average="binary", pos_label=1, zero_division=0)
            logger.debug("LabelQuality CV fold %d | F1=%.4f", fold_idx, fold_f1)

        return prob_matrix

    def _make_estimator(self):
        """Return a fresh copy of the base estimator."""
        if self._base_estimator is not None:
            from sklearn.base import clone
            return clone(self._base_estimator)
        # Default: lightweight XGBoost
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                n_jobs=-1, verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    # ------------------------------------------------------------------
    # Ana analiz
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        panel_labels: Optional[np.ndarray] = None,
    ) -> LabelQualityReport:
        """
        Etiket kalite analizi yap.

        Parameters
        ----------
        X : (N, F) özellik matrisi (ön işlenmiş veya ham).
        y : (N,) binary etiket dizisi (0=Benign, 1=Pathogenic).
        panel_labels : (N,) opsiyonel panel adı dizisi (her panel için
            ayrı gürültü istatistikleri).

        Returns
        -------
        LabelQualityReport
        """
        logger.info(
            "ConfidentLearner: %d örnek üzerinde %d-fold OOF çalışıyor …",
            len(X), self.cv_folds,
        )

        # 1. OOF olasılıkları
        prob_matrix = self._get_oof_probabilities(X, y)

        # 2. Kalibre eşikler
        thresholds = _compute_calibrated_thresholds(prob_matrix, y)
        logger.debug("Calibrated thresholds: Benign=%.3f  Pathogenic=%.3f",
                     thresholds[0], thresholds[1])

        # 3. Güven matrisi
        cj = _build_confident_joint(prob_matrix, y, thresholds)

        # 4. Normalize: p(ŷ, y)
        total = cj.sum()
        noise_matrix = cj / max(total, 1e-9)

        # 5. Etiket kalite skoru
        lqs = _label_quality_score(prob_matrix, y, noise_matrix)

        # 6. Şüpheli örnekler (LQS < eşik)
        flagged = np.where(lqs < self.noise_threshold)[0].tolist()

        # 7. Sınıf bazlı gürültü oranları
        class_noise = {}
        for k in range(2):
            off_diag = sum(noise_matrix[j, k] for j in range(2) if j != k)
            class_noise[k] = float(off_diag)

        # 8. Panel bazlı istatistik (opsiyonel)
        if panel_labels is not None:
            self._log_panel_noise(lqs, y, panel_labels)

        estimated_noise = float(sum(noise_matrix[j, k]
                                    for j in range(2) for k in range(2) if j != k))

        report = LabelQualityReport(
            n_samples            = len(X),
            n_flagged            = len(flagged),
            estimated_noise_rate = estimated_noise,
            flagged_indices      = flagged,
            noise_matrix         = noise_matrix,
            confidence_scores    = prob_matrix.max(axis=1),
            label_quality_scores = lqs,
            class_noise_rates    = class_noise,
        )
        report.log()
        return report

    def _log_panel_noise(
        self,
        lqs: np.ndarray,
        y: np.ndarray,
        panel_labels: np.ndarray,
    ) -> None:
        """Panel bazlı düşük kaliteli örnek oranını logla."""
        for panel in np.unique(panel_labels):
            mask  = panel_labels == panel
            n_p   = int(mask.sum())
            n_bad = int((lqs[mask] < self.noise_threshold).sum())
            if n_p > 0:
                logger.info(
                    "Panel %-22s | n=%4d | düşük kalite=%3d (%5.1f %%)",
                    panel, n_p, n_bad, 100 * n_bad / n_p,
                )

    # ------------------------------------------------------------------
    # Temiz veri subkümesi
    # ------------------------------------------------------------------

    def clean_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        report: LabelQualityReport,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Şüpheli örnekleri çıkararak temiz X, y ve orijinal indisleri döndür.

        Returns
        -------
        X_clean      : (M, F) temiz özellik matrisi.
        y_clean      : (M,) temiz etiket dizisi.
        clean_indices: (M,) orijinal veri setindeki indisler.
        """
        mask = report.clean_mask()
        clean_indices = np.where(mask)[0]
        logger.info(
            "ConfidentLearner: %d → %d örnek (%d şüpheli çıkarıldı, %.1f %% loss).",
            len(X), int(mask.sum()), report.n_flagged,
            100 * report.n_flagged / max(len(X), 1),
        )
        return X[clean_indices], y[clean_indices], clean_indices

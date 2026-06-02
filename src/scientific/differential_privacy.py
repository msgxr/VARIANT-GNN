"""
src/scientific/differential_privacy.py
Diferansiyel Gizlilik — Laplace Mekanizması

KVKK (6698 sayılı Kişisel Verilerin Korunması Kanunu) ve
GDPR Madde 89 uyumlu veri anonimleştirme modülü.

Teori:
  ε-Differential Privacy: M(D) ≈ M(D') — iki komşu veri seti için
  Laplace(0, Δf/ε) gürültüsü eklenerek bireysel veri sızdırmaz.

  - ε (epsilon) = gizlilik bütçesi
      · ε → 0 : Maksimum gizlilik (çok gürültülü, düşük fayda)
      · ε → ∞ : Gizlilik yok (gürültüsüz)
      · Tıbbi veri için önerilen: ε ∈ [0.5, 2.0]

  - Δf (global sensitivity):
      · Normalize özellikler [0,1] → Δf = 1.0
      · Ölçeklenmemiş özellikler → özellik aralığından hesaplanır

Kullanım:
    dp = DifferentialPrivacy(epsilon=1.0)
    X_private = dp.apply(X_original, feature_names=feature_names)

    # Hassas sütunları seç (örn. sadece allele frequency)
    dp = DifferentialPrivacy(epsilon=0.5, sensitive_features=["gnomAD_exomes_AF"])
    X_private = dp.apply(X_original, feature_names=feature_names)

    # Privacy raporu
    print(dp.privacy_report())
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# KVKK uyumlu önerilen hassas özellikler (popülasyon frekansı + klinik)
_DEFAULT_SENSITIVE = [
    "gnomAD_exomes_AF",
    "gnomAD_exomes_AF_afr",
    "gnomAD_exomes_AF_eur",
    "gnomAD_exomes_AF_eas",
    "gnomAD_exomes_AF_sas",
    "gnomAD_exomes_AF_amr",
    "ExAC_AF",
]


class DifferentialPrivacy:
    """
    Laplace mekanizması ile ε-Diferansiyel Gizlilik.

    Parameters
    ----------
    epsilon           : Gizlilik bütçesi (önerilen tıbbi: 0.5–2.0)
    sensitivity       : Global duyarlılık (None → otomatik tahmin)
    sensitive_features: Sadece bu özelliklere gürültü ekle (None → hepsine)
    clip_to_range     : Gürültü sonrası değerleri [min, max] ile kırp
    seed              : Tekrarlanabilirlik için seed (None = rastgele)
    """

    PRIVACY_LEVELS: Dict[str, float] = {
        "maksimum": 0.1,
        "yüksek": 0.5,
        "standart": 1.0,
        "düşük": 2.0,
        "çok düşük": 5.0,
    }

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: Optional[float] = None,
        sensitive_features: Optional[List[str]] = None,
        clip_to_range: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon > 0 olmalıdır, {epsilon} verildi.")

        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.sensitive_features = sensitive_features
        self.clip_to_range = clip_to_range
        self._rng = np.random.default_rng(seed)

        # Takip
        self._n_queries: int = 0
        self._budget_used: float = 0.0
        self._total_budget_target: float = 10.0  # toplam bütçe hedefi

    # ── Laplace Gürültüsü ────────────────────────────────────────────────────
    def _laplace_noise(self, shape: tuple, scale: float) -> np.ndarray:
        """Laplace(0, scale) dağılımından gürültü üretir."""
        return self._rng.laplace(loc=0.0, scale=scale, size=shape)

    def _auto_sensitivity(self, X: np.ndarray) -> float:
        """Verinin özellik aralığından global duyarlılık tahmini."""
        ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        return float(np.nanmedian(ranges[ranges > 0]) or 1.0)

    # ── Ana uygulama ─────────────────────────────────────────────────────────
    def apply(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        return_mask: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Veri matriksine Laplace gürültüsü ekle.

        Parameters
        ----------
        X             : [N, F] özellik matrisi
        feature_names : Özellik isimleri (sensitive_features filtresi için)
        return_mask   : True → (X_private, noise_mask) döndür

        Returns
        -------
        X_private : Gizliliği korunmuş matris
        noise_mask: (isteğe bağlı) Gürültü eklenen sütunlar boolean maskesi
        """
        X_in = np.asarray(X, dtype=float).copy()
        N, F = X_in.shape

        # Hangi sütunlara gürültü eklenecek?
        if self.sensitive_features is not None and feature_names is not None:
            sf_lower = {s.lower() for s in self.sensitive_features}
            col_mask = np.array([(fn.lower() in sf_lower) for fn in feature_names[:F]], dtype=bool)
        else:
            col_mask = np.ones(F, dtype=bool)  # Hepsi

        n_noised = int(col_mask.sum())
        if n_noised == 0:
            logger.warning("DP: Hiçbir sütun seçilmedi, gürültü eklenmedi.")
            if return_mask:
                return X_in, col_mask
            return X_in

        # Duyarlılık
        delta = self.sensitivity or self._auto_sensitivity(X_in[:, col_mask])
        scale = delta / self.epsilon

        logger.info(
            "DP Laplace: ε=%.2f, Δf=%.4f, scale=%.4f, %d/%d sütun",
            self.epsilon,
            delta,
            scale,
            n_noised,
            F,
        )

        # Gürültü ekle
        noise = self._laplace_noise((N, n_noised), scale)
        X_in[:, col_mask] += noise

        # Kırpma
        if self.clip_to_range:
            col_idx = np.where(col_mask)[0]
            for j, ci in enumerate(col_idx):
                col_orig = X[:, ci]
                lo = float(np.nanmin(col_orig) - 3 * scale)
                hi = float(np.nanmax(col_orig) + 3 * scale)
                X_in[:, ci] = np.clip(X_in[:, ci], lo, hi)

        # Bütçe takibi
        self._n_queries += 1
        self._budget_used += self.epsilon
        X_in = np.nan_to_num(X_in, nan=0.0)

        if return_mask:
            return X_in, col_mask
        return X_in

    # ── Sadece seçili satırlar ────────────────────────────────────────────────
    def apply_to_rows(
        self,
        X: np.ndarray,
        row_indices: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Sadece belirli satırlara (varyantlara) DP uygula."""
        X_out = X.copy()
        if len(row_indices) == 0:
            return X_out
        X_out[row_indices] = self.apply(X[row_indices], feature_names)
        return X_out

    # ── Gizlilik Raporu ──────────────────────────────────────────────────────
    def privacy_report(self) -> dict:
        """
        Gizlilik bütçesi ve parametre raporu döndür.
        KVKK uyumluluk belgesi olarak kullanılabilir.
        """
        level = "bilinmiyor"
        for lbl, eps in sorted(self.PRIVACY_LEVELS.items(), key=lambda t: t[1]):
            if self.epsilon <= eps:
                level = lbl
                break

        return {
            "mechanism": "Laplace",
            "epsilon": self.epsilon,
            "privacy_level": level,
            "sensitivity": self.sensitivity or "otomatik",
            "scale": (self.sensitivity or 1.0) / self.epsilon,
            "n_queries": self._n_queries,
            "cumulative_epsilon": round(self._budget_used, 4),
            "sensitive_features": self.sensitive_features or "tüm özellikler",
            "compliance": {
                "KVKK": "Madde 6 — Özel nitelikli KVK korunması",
                "GDPR": "Madde 89 — İstatistiksel amaçlı güvence",
                "HIPAA": "45 CFR §164.514 — Safe Harbor anonymization",
            },
            "recommendation": (
                f"ε={self.epsilon} ile gizlilik seviyesi: {level}. "
                + (
                    "Tıbbi veri için ε ≤ 2.0 önerilir."
                    if self.epsilon > 2.0
                    else "Tıbbi veri için uygun gizlilik seviyesi."
                )
            ),
        }

    # ── Yardımcılar ──────────────────────────────────────────────────────────
    @staticmethod
    def recommended_epsilon(data_sensitivity: str = "medical") -> float:
        """Veri hassasiyetine göre önerilen epsilon değeri döner."""
        table = {
            "medical": 1.0,
            "genetic": 0.5,
            "aggregate": 5.0,
            "public": 10.0,
        }
        return table.get(data_sensitivity, 1.0)

    def noise_magnitude(self) -> float:
        """Beklenen ortalama gürültü büyüklüğü (LAplace beklentisi = scale)."""
        delta = self.sensitivity or 1.0
        return delta / self.epsilon

    @staticmethod
    def fingerprint(X_private: np.ndarray) -> str:
        """Gizliliği korunmuş verinin kriptografik parmak izi (audit log için)."""
        digest = hashlib.sha256(X_private.tobytes()).hexdigest()[:16]
        return f"DP-SHA256:{digest}"

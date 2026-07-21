# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/evaluation/panel_transfer.py
=================================
Çapraz Panel Genelleştirme Matrisi (Cross-Panel Generalization Matrix)

TEKNOFEST 2026 bağlamı:
  Şartname §3.2 dört bağımsız hastalık paneli tanımlar:
    General, Hereditary_Cancer (Kalıtsal Kanser), PAH (Fenilketonüri), Kistik Fibrozis (CFTR)

  Jüri external validation'da her paneli bağımsız değerlendirir.  Bir modelin
  *gerçek genelleştirme kapasitesi* şu soruyla ölçülür:
    "Panel X üzerinde eğitilen model Panel Y üzerinde ne kadar performans gösterir?"

  Bu analiz hem PDR (Proje Detay Raporu) hem de finalist sunum için kritik
  kanıt sağlar: modelin hastalık alanına özgü bir "ezber" yapmadığını,
  biyolojik özellikleri genel olarak öğrendiğini kanıtlar.

Çıktı:
  - 4×4 F1 matrisi (her hücre: train_panel → test_panel arası F1)
  - Diyagonal = in-distribution performans
  - Diyagonal dışı = cross-domain transfer kabiliyeti
  - JSON raporu + PNG görselleştirme

Kullanım (CLI):
  python main.py --mode panel_transfer --data_file data/train.csv

Kullanım (Python):
  from src.evaluation.panel_transfer import CrossPanelEvaluator
  evaluator = CrossPanelEvaluator()
  result = evaluator.evaluate(X, y, panel_labels)
  result.plot(save_path="reports/figures/panel_transfer.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sonuç kapsayıcısı
# ---------------------------------------------------------------------------


@dataclass
class PanelTransferResult:
    """Çapraz panel genelleştirme analizi sonucu."""

    panels: List[str]  # Panel isimleri (satır/sütun)
    f1_matrix: np.ndarray  # (N_panels × N_panels) F1 skoru matrisi
    support: Dict[str, int]  # Panel başına örnek sayısı
    in_dist_mean: float  # Diyagonal ortalama (in-distribution)
    cross_mean: float  # Diyagonal dışı ortalama (transfer)
    transfer_gap: float  # in_dist_mean - cross_mean  (küçük = iyi)

    def summary(self) -> str:
        lines = ["=== Çapraz Panel Genelleştirme Matrisi ==="]
        col_label = "Eğitim \\ Test"
        header = f"{col_label:22s}" + "".join(f"{p:22s}" for p in self.panels)
        lines.append(header)
        lines.append("-" * len(header))
        for i, train_panel in enumerate(self.panels):
            row = f"{train_panel:22s}"
            for j, test_panel in enumerate(self.panels):
                marker = " *" if i == j else "  "
                row += f"{self.f1_matrix[i, j]:6.3f}  {marker:2s}         "
            lines.append(row)
        lines.append("-" * len(header))
        lines.append(f"  In-distribution F1 (diyagonal ort.) : {self.in_dist_mean:.4f}")
        lines.append(f"  Cross-domain F1  (diyagonal dışı)   : {self.cross_mean:.4f}")
        lines.append(f"  Transfer gap (küçük = iyi)           : {self.transfer_gap:.4f}")
        return "\n".join(lines)

    def log(self) -> None:
        for line in self.summary().splitlines():
            logger.info(line)

    def as_dict(self) -> dict:
        return {
            "panels": self.panels,
            "f1_matrix": self.f1_matrix.tolist(),
            "support": self.support,
            "in_dist_mean": round(self.in_dist_mean, 4),
            "cross_mean": round(self.cross_mean, 4),
            "transfer_gap": round(self.transfer_gap, 4),
        }

    def plot(
        self,
        save_path: Optional[str | Path] = None,
        title: str = "Cross-Panel Generalization — TEKNOFEST 2026",
    ) -> None:
        """Matris ısı haritası çiz ve opsiyonel olarak kaydet."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(self.f1_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Binary F1")

            ax.set_xticks(range(len(self.panels)))
            ax.set_yticks(range(len(self.panels)))
            ax.set_xticklabels(self.panels, rotation=30, ha="right", fontsize=9)
            ax.set_yticklabels(self.panels, fontsize=9)
            ax.set_xlabel("Test Paneli", fontsize=10)
            ax.set_ylabel("Eğitim Paneli", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

            for i in range(len(self.panels)):
                for j in range(len(self.panels)):
                    val = self.f1_matrix[i, j]
                    color = "white" if val < 0.4 else "black"
                    marker = "★" if i == j else ""
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}{marker}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold" if i == j else "normal",
                        color=color,
                    )

            ax.text(
                0.5,
                -0.22,
                f"In-dist F1={self.in_dist_mean:.3f}  |  Cross F1={self.cross_mean:.3f}  "
                f"|  Gap={self.transfer_gap:.3f}",
                ha="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#555",
            )

            plt.tight_layout()
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
                logger.info("Panel transfer matrisi → %s", save_path)
            plt.close()
        except Exception as exc:
            logger.warning("Panel transfer grafiği oluşturulamadı: %s", exc)


# ---------------------------------------------------------------------------
# Değerlendirici
# ---------------------------------------------------------------------------


class CrossPanelEvaluator:
    """
    Çapraz panel genelleştirme matrisi hesaplayıcı.

    Her eğitim paneli için bağımsız bir lightweight model eğitir
    (XGBoost varsayılan olarak kullanılır) ve tüm test panellerinde test eder.
    Bu, tam ensemble'ın çapraz-panel davranışını yaklaşık ama hızlı tahmin eder.

    Parameters
    ----------
    min_panel_size : int
        Bir panelin dahil edilmesi için minimum örnek sayısı.
    test_size : float
        Her panel split'inde test fraksiyonu.
    random_state : int
        Reprodusibilite için sabit seed.
    """

    def __init__(
        self,
        min_panel_size: int = 30,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.min_panel_size = min_panel_size
        self.test_size = test_size
        self.random_state = random_state

    # ------------------------------------------------------------------

    def _make_estimator(self) -> Any:
        """Lightweight hızlı panel başına model (tam pipeline değil)."""
        try:
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
                random_state=self.random_state,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        panel_labels: np.ndarray,
        groups: "np.ndarray | None" = None,
    ) -> PanelTransferResult:
        """
        Panel başına model eğit ve tüm panellerde test et.

        Parameters
        ----------
        X            : (N, F) özellik matrisi.
        y            : (N,) binary etiket.
        panel_labels : (N,) her örnek için panel adı.

        Returns
        -------
        PanelTransferResult
        """
        unique_panels = sorted(set(panel_labels))
        # Yeterli örnekli panelleri filtrele
        valid_panels = [p for p in unique_panels if (panel_labels == p).sum() >= self.min_panel_size]

        if len(valid_panels) < 2:
            logger.warning(
                "CrossPanelEvaluator: geçerli panel sayısı < 2 (%d mevcut). En az %d örnek gerekli.",
                len(valid_panels),
                self.min_panel_size,
            )

        n = len(valid_panels)
        f1_matrix = np.zeros((n, n), dtype=float)
        support = {}

        # Panel başına eğitim verisi ve test verisi ayır
        panel_data: Dict[str, dict] = {}
        for panel in valid_panels:
            mask = panel_labels == panel
            Xp, yp = X[mask], y[mask]
            gp = groups[mask] if groups is not None else None
            support[panel] = int(mask.sum())

            if len(np.unique(yp)) < 2 or len(yp) < self.min_panel_size:
                panel_data[panel] = {"X_tr": Xp, "y_tr": yp, "X_te": Xp, "y_te": yp}
                continue

            # GROUP-AWARE within-panel split (Variant_ID) — aynı varyant panel-içi
            # train/test'i çaprazlamasın. groups verilmezse stratified'e düşer.
            if gp is not None and len(np.unique(gp)) >= 2:
                from sklearn.model_selection import GroupShuffleSplit

                _tr, _te = next(
                    GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state).split(
                        Xp, yp, groups=gp
                    )
                )
                X_tr, X_te, y_tr, y_te = Xp[_tr], Xp[_te], yp[_tr], yp[_te]
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xp,
                    yp,
                    test_size=self.test_size,
                    stratify=yp,
                    random_state=self.random_state,
                )
            panel_data[panel] = {
                "X_tr": X_tr,
                "y_tr": y_tr,
                "X_te": X_te,
                "y_te": y_te,
            }

        # Eğitim × test kombinasyonları
        for i, train_panel in enumerate(valid_panels):
            data_tr = panel_data[train_panel]

            # Model eğitimi
            estimator = self._make_estimator()
            try:
                estimator.fit(data_tr["X_tr"], data_tr["y_tr"])
            except Exception as exc:
                logger.warning(
                    "Panel %s eğitimi başarısız (%s) — sıfır F1 atandı.",
                    train_panel,
                    exc,
                )
                continue

            # Tüm panellerde test et
            for j, test_panel in enumerate(valid_panels):
                data_te = panel_data[test_panel]
                X_te = data_te["X_te"]
                y_te = data_te["y_te"]

                try:
                    preds = estimator.predict(X_te)
                    f1 = float(
                        f1_score(
                            y_te,
                            preds,
                            average="binary",
                            pos_label=1,
                            zero_division=0,
                        )
                    )
                    f1_matrix[i, j] = f1
                    logger.debug(
                        "Transfer %s → %s : F1=%.4f (n_test=%d)",
                        train_panel,
                        test_panel,
                        f1,
                        len(y_te),
                    )
                except Exception as exc:
                    logger.warning(
                        "Transfer %s → %s başarısız (%s).",
                        train_panel,
                        test_panel,
                        exc,
                    )

        # Özet istatistikler
        if n >= 2:
            diag = [f1_matrix[i, i] for i in range(n)]
            off_diag = [f1_matrix[i, j] for i in range(n) for j in range(n) if i != j]
            in_dist = float(np.mean(diag))
            cross = float(np.mean(off_diag)) if off_diag else 0.0
            gap = in_dist - cross
        else:
            in_dist = cross = gap = float(f1_matrix[0, 0]) if n == 1 else 0.0

        result = PanelTransferResult(
            panels=valid_panels,
            f1_matrix=f1_matrix,
            support=support,
            in_dist_mean=in_dist,
            cross_mean=cross,
            transfer_gap=gap,
        )
        result.log()
        return result

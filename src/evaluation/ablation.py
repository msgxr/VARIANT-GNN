"""
src/evaluation/ablation.py
============================
TEKNOFEST 2026 — Ablation Analizi

Şartname §4 (Yarışmaya Katılım Koşulları):
  "Geliştirilen modelin mimarisi, eğitim süreçleri ve iç test (validasyon)
   sonuçları detaylı olarak sunulacaktır."

  PDR'de model bileşenlerinin katkısının kanıtlanması beklenir.

Bu modül, ensemble'ın her bir bileşenini ve preprocessing seçeneğini
sırayla devre dışı bırakarak validation Binary F1 (Pathogenic, §7.3)
üzerindeki etkisini ölçer:

  Tam ensemble (XGB + LGB + GNN + DNN)        — baseline
  - XGB    : XGBoost'u ensemble'dan çıkar
  - LGB    : LightGBM'i ensemble'dan çıkar
  - GNN    : GATv2 GNN'i ensemble'dan çıkar
  - DNN    : DNN'i ensemble'dan çıkar
  - SMOTE  : SMOTE re-sampling'i kapat
  (NOT: AE / SelectKBest canonical hatta zaten KAPALI olduğundan ablasyon spec'leri
   kaldırıldı — delta≈0 yanıltıcı satır üretmesinler diye.)

Her ablation için tam pipeline yeniden eğitilir, Binary F1 hesaplanır,
JSON raporu üretilir:

  reports/ablation_report.json
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.features.preprocessing import build_preprocessor_from_config
from src.training.trainer import VariantTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sonuç kapsayıcısı
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Tek bir ablation koşusunun sonucu."""

    name: str  # "baseline", "no_xgb", "no_smote", ...
    description: str
    binary_f1: float
    delta_f1: float = 0.0  # baseline'a göre fark (negatif = düşüş)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationReport:
    """Tüm ablation koşularının raporu."""

    baseline: AblationResult
    ablations: List[AblationResult] = field(default_factory=list)
    n_samples: int = 0
    test_size: float = 0.2
    seed: int = 42

    def summary(self) -> str:
        lines = [
            "=== Ablation Analizi Raporu ===",
            f"  N samples            : {self.n_samples}",
            f"  Test size            : {self.test_size}",
            f"  Seed                 : {self.seed}",
            f"  Baseline Binary F1   : {self.baseline.binary_f1:.4f}",
            "",
            f"  {'Ablation':25s} {'F1':>8s}  {'Δ F1':>8s}  {'Açıklama':40s}",
            "  " + "-" * 90,
        ]
        for r in self.ablations:
            sign = "↓" if r.delta_f1 < 0 else "↑" if r.delta_f1 > 0 else "·"
            lines.append(f"  {r.name:25s} {r.binary_f1:>8.4f}  {sign} {abs(r.delta_f1):>6.4f}  {r.description:40s}")
        return "\n".join(lines)

    def log(self) -> None:
        for line in self.summary().splitlines():
            logger.info(line)

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "test_size": self.test_size,
            "seed": self.seed,
            "baseline": {
                "name": self.baseline.name,
                "binary_f1": self.baseline.binary_f1,
                "description": self.baseline.description,
            },
            "ablations": [
                {
                    "name": r.name,
                    "description": r.description,
                    "binary_f1": r.binary_f1,
                    "delta_f1": r.delta_f1,
                }
                for r in self.ablations
            ],
        }

    def pdr_table(self) -> str:
        """PDR §4.5 için Markdown ablation tablosu üretir.

        Format:
            | Konfigürasyon | Binary F1 | Δ F1 | Notlar |
            |--------------|-----------|------|--------|
            | Tam Ensemble | 0.945     | —    | ...    |
            ...
        """
        header = "| Konfigürasyon | Binary F1 | Delta F1 | Notlar |\n|--------------|-----------|----------|--------|\n"
        baseline_row = (
            f"| **Tam Ensemble (baseline)** | **{self.baseline.binary_f1:.3f}** | -- | {self.baseline.description} |\n"
        )

        rows = []
        for r in sorted(self.ablations, key=lambda x: x.delta_f1):
            sign = "-" if r.delta_f1 < 0 else "+"
            delta = f"{sign}{abs(r.delta_f1):.3f}"
            notlar = "Kritik" if r.delta_f1 < -0.02 else ("Orta" if r.delta_f1 < 0 else "minimal")
            rows.append(f"| {r.description} | {r.binary_f1:.3f} | {delta} | {notlar} |")

        return header + baseline_row + "\n".join(rows)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Ablation raporu → %s", path)

    def to_markdown(self, path: Path) -> None:
        """PDR tablosu olarak Markdown dosyasına yazar."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("## Ablation Analizi — PDR §4.5\n\n")
            fh.write(self.pdr_table())
            fh.write(
                f"\n\n*N={self.n_samples}, test_size={self.test_size}, "
                f"seed={self.seed}, metric=Binary F1 (Patojenik, §7.3)*\n"
            )
        logger.info("Ablation Markdown tablosu → %s", path)


# ---------------------------------------------------------------------------
# Ablation çalıştırıcı
# ---------------------------------------------------------------------------


class AblationRunner:
    """
    TEKNOFEST 2026 ablation analizi için yardımcı sınıf.

    Her ablation çalışması için:
      1. Preprocessor + ensemble eğitilir (yapılandırma override'lı).
      2. Hold-out test setinde Binary F1 hesaplanır (§7.3).
      3. Baseline ile fark loglanır.

    Parameters
    ----------
    X            : Eğitim/test özellik matrisi (N, F).
    y            : Binary etiket (N,).
    nuc_seqs     : Opsiyonel multimodal Nuc_Context dizileri.
    aa_seqs      : Opsiyonel multimodal AA_Context dizileri.
    test_size    : Hold-out test fraksiyonu (varsayılan 0.20).
    seed         : Reproducibility seed.
    threshold    : Binary classification threshold (varsayılan: 0.50).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nuc_seqs: Optional[List[str]] = None,
        aa_seqs: Optional[List[str]] = None,
        test_size: float = 0.20,
        seed: int = 42,
        threshold: float = 0.50,
    ) -> None:
        self.X = X
        self.y = y
        self.nuc_seqs = nuc_seqs
        self.aa_seqs = aa_seqs
        self.test_size = test_size
        self.seed = seed
        self.threshold = threshold

        # Tek bir tutarlı train/test split — tüm ablation'lar aynı split'te
        # değerlendirilir (apples-to-apples karşılaştırma).
        idx = np.arange(len(X))
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
        self._idx_tr, self._idx_te = idx_tr, idx_te
        self._X_tr, self._X_te = X[idx_tr], X[idx_te]
        self._y_tr, self._y_te = y[idx_tr], y[idx_te]
        self._nuc_tr = [nuc_seqs[i] for i in idx_tr] if nuc_seqs else None
        self._aa_tr = [aa_seqs[i] for i in idx_tr] if aa_seqs else None

    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        ablation_overrides: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Verilen override ile bir pipeline eğitir, hold-out test üzerinde
        Binary F1 (Pathogenic, §7.3) döner.

        Override anahtarları:
          - drop_models    : List[str] — ['xgb','lgbm','gnn','dnn'] alt kümesi
          - smote_enabled  : bool
          - use_autoencoder: bool
          - use_feature_selection : bool
        """

        cfg = get_settings()
        # Settings'i kirletmemek için derin kopya yap
        local_cfg = copy.deepcopy(cfg)

        if "smote_enabled" in ablation_overrides:
            local_cfg.preprocessing.smote_enabled = ablation_overrides["smote_enabled"]
        if "use_autoencoder" in ablation_overrides:
            local_cfg.preprocessing.use_autoencoder = ablation_overrides["use_autoencoder"]
        if "use_feature_selection" in ablation_overrides:
            local_cfg.preprocessing.use_feature_selection = ablation_overrides["use_feature_selection"]

        # Geçici Settings override — reset_settings() ile atomik geri alma
        from src.config import settings as _settings_mod

        _orig = _settings_mod._settings
        _settings_mod._settings = local_cfg
        try:
            # Preprocessor: local_cfg override'lı build
            preprocessor = build_preprocessor_from_config()
            # Direkt preprocessor attribute'larını da override et (çift güvence)
            for key in ("smote_enabled", "use_autoencoder", "use_feature_selection"):
                if key in ablation_overrides and hasattr(preprocessor, key):
                    setattr(preprocessor, key, ablation_overrides[key])

            trainer = VariantTrainer()
            preprocessor_local, ensemble, X_inner_val, y_inner_val = trainer._fit_single(
                self._X_tr,
                self._y_tr,
                nuc_seqs=self._nuc_tr,
                aa_seqs=self._aa_tr,
            )
        finally:
            # Her durumda orijinal settings'i geri yükle
            _settings_mod._settings = _orig

        # Drop models post-fit if requested (zero out by setting to None)
        drop_models = ablation_overrides.get("drop_models", [])
        if "xgb" in drop_models:
            ensemble.xgb = None
        if "lgbm" in drop_models:
            ensemble.lgbm = None
        if "gnn" in drop_models:
            ensemble.gnn = None
        if "dnn" in drop_models:
            ensemble.dnn = None
        # Yeniden normalize ensemble weights
        active_idx = [
            i for i, m in enumerate([ensemble.xgb, ensemble.lgbm, ensemble.gnn, ensemble.dnn]) if m is not None
        ]
        if not active_idx:
            raise RuntimeError("All ensemble members dropped — invalid ablation.")

        # Hold-out test üzerinde değerlendirme
        X_te_proc = preprocessor_local.transform(self._X_te)
        preds, _ = ensemble.predict(X_te_proc, threshold=self.threshold)
        bf1 = float(
            f1_score(
                self._y_te,
                preds,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )

        meta = {
            "active_models": [name for i, name in enumerate(["xgb", "lgbm", "gnn", "dnn"]) if i in active_idx],
            "smote_enabled": preprocessor_local.smote_enabled,
            "use_autoencoder": preprocessor_local.use_autoencoder,
            "use_feature_selection": preprocessor_local.use_feature_selection,
        }
        return bf1, meta

    # ------------------------------------------------------------------

    def run(self) -> AblationReport:
        """Run all ablation configurations and return the report."""

        ablation_specs: List[Tuple[str, str, Dict[str, Any]]] = [
            ("baseline", "Tüm ensemble + tüm preprocessing", {}),
            ("no_xgb", "XGBoost devre dışı", {"drop_models": ["xgb"]}),
            ("no_lgbm", "LightGBM devre dışı", {"drop_models": ["lgbm"]}),
            ("no_gnn", "GATv2 GNN devre dışı", {"drop_models": ["gnn"]}),
            ("no_dnn", "DNN devre dışı", {"drop_models": ["dnn"]}),
            ("no_smote", "SMOTE devre dışı", {"smote_enabled": False}),
            # NOT: no_autoencoder / no_feature_selection spec'leri KALDIRILDI — AE+SelectKBest
            # canonical hatta zaten KAPALI (configs/default.yaml), bu yüzden override delta≈0
            # üretip §4.5 ablasyon tablosunu yanıltırdı.
        ]

        results: List[AblationResult] = []
        baseline_f1: Optional[float] = None

        for name, desc, overrides in ablation_specs:
            logger.info("Ablation çalışıyor: %s (%s)", name, desc)
            try:
                f1, meta = self._train_and_evaluate(overrides)
            except Exception as exc:
                logger.warning("Ablation '%s' başarısız: %s", name, exc)
                f1, meta = float("nan"), {"error": str(exc)}

            result = AblationResult(
                name=name,
                description=desc,
                binary_f1=f1,
                metadata=meta,
            )
            if name == "baseline":
                baseline_f1 = f1
                baseline_result = result
            else:
                if baseline_f1 is not None and not np.isnan(f1):
                    result.delta_f1 = f1 - baseline_f1
                results.append(result)

            logger.info("Ablation '%s' Binary F1 = %.4f", name, f1)

        report = AblationReport(
            baseline=baseline_result,  # type: ignore[arg-type]
            ablations=results,
            n_samples=len(self.X),
            test_size=self.test_size,
            seed=self.seed,
        )
        report.log()
        return report


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    nuc_seqs: Optional[List[str]] = None,
    aa_seqs: Optional[List[str]] = None,
    output: Path = Path("reports/ablation_report.json"),
) -> AblationReport:
    """
    Tek satırlık ablation entrypoint'i.

    Parameters
    ----------
    X        : (N, F) eğitim özellikleri.
    y        : (N,) binary etiketler.
    nuc_seqs : Opsiyonel multimodal sekanslar.
    aa_seqs  : Opsiyonel multimodal sekanslar.
    output   : JSON çıktı yolu.

    Returns
    -------
    AblationReport
    """
    runner = AblationRunner(X, y, nuc_seqs=nuc_seqs, aa_seqs=aa_seqs)
    report = runner.run()
    report.to_json(output)
    return report

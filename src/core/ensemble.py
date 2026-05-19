"""
src/core/ensemble.py
=====================
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Hibrit Topluluk Modeli

Dört baz modelin ağırlıklı topluluğu:
  1. XGBoost      (gradient boosting, tabular güç)
  2. LightGBM     (hızlı gradient boosting, düşük bellek)
  3. VariantGATv2GNN  (graf dikkat ağı — özellikler arası ilişkiler)
  4. VariantDNN       (derin sinir ağı — doğrusal-olmayan örüntüler)

Birleştirme stratejisi (öncelik sırası):
  a) Meta-öğrenici stacking (LogisticRegression; fit_meta_learner() çağrıldıysa)
  b) Nelder-Mead optimize edilmiş ağırlıklı ortalama (optimise_weights() çağrıldıysa)
  c) Yapılandırma dosyasındaki varsayılan ağırlıklar

TEKNOFEST §7.3: Birincil değerlendirme metriği —
  F1 = 2·TP / (2·TP + FP + FN)  (Pathogenic sınıfı, pos_label=1, binary F1)

Belirsizlik nicemleme: GATv2 MC-Dropout ile epistemik belirsizlik tahmini.
Panel bazlı eşik: Her panel (General / Hereditary_Cancer / PAH / CFTR) için
  ayrı optimal eşik kullanılabilir (predict_with_panel_threshold()).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from src.config import get_settings
from src.core.models.gnn import VariantGATv2GNN
from src.models.dnn_model import VariantDNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcı işlevler
# ---------------------------------------------------------------------------


def _dnn_predict_proba(
    model: VariantDNN,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """DNN'den (N, 2) olasılık matrisi döndür."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device))
        return F.softmax(logits, dim=1).cpu().numpy()


def _build_knn_graph(X_scaled: np.ndarray, knn_k: int):
    """X_scaled üzerinde k-NN örnek grafiği kurar (PyG Data nesnesi)."""
    from src.core.graph.builder import SampleKNNGraphBuilder
    return SampleKNNGraphBuilder(k=knn_k).build(X_scaled, y=None)


# ---------------------------------------------------------------------------
# HybridEnsemble
# ---------------------------------------------------------------------------


class HybridEnsemble:
    """
    TEKNOFEST 2026 Hibrit Topluluk Modeli.

    Parameters
    ----------
    xgb_model   : Eğitilmiş XGBClassifier (veya None).
    lgbm_model  : Eğitilmiş LGBMClassifier (veya None).
    gnn_model   : Eğitilmiş VariantGATv2GNN (veya None).
    dnn_model   : Eğitilmiş VariantDNN (veya None).
    weights     : [w_xgb, w_lgb, w_gnn, w_dnn] — otomatik normalize edilir.
    device      : Torch device (None → CUDA varsa CUDA, yoksa CPU).
    """

    LABEL_MAP: Dict[int, str] = {0: "Benign", 1: "Pathogenic"}

    # TEKNOFEST §3.2: desteklenen 4 panel
    KNOWN_PANELS: Tuple[str, ...] = (
        "General", "Hereditary_Cancer", "PAH", "CFTR",
    )

    def __init__(
        self,
        xgb_model:  Optional[xgb.XGBClassifier] = None,
        lgbm_model: Optional[Any]                = None,
        gnn_model:  Optional[VariantGATv2GNN]    = None,
        dnn_model:  Optional[VariantDNN]          = None,
        weights:    Optional[List[float]]         = None,
        device:     Optional[torch.device]        = None,
    ) -> None:
        cfg = get_settings()

        self.xgb  = xgb_model
        self.lgbm = lgbm_model
        self.gnn  = gnn_model
        self.dnn  = dnn_model

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Ağırlıkları normalize et; eksikse yapılandırmadan al
        raw_w = list(weights) if weights is not None else list(cfg.ensemble.weights)
        if len(raw_w) != 4:
            raw_w = [0.30, 0.30, 0.25, 0.15]
        w_sum = sum(raw_w)
        if w_sum <= 0:
            logger.warning(
                "Ensemble weights toplami <= 0 (%s) — varsayilan agirliklar kullaniliyor.",
                raw_w,
            )
            raw_w = [0.30, 0.30, 0.25, 0.15]
            w_sum = 1.0
        self.weights: List[float] = [w / w_sum for w in raw_w]

        # Stacking meta-öğrenicisi (fit_meta_learner() ile doldurulur)
        self.meta_learner: Optional[Any] = None

        # Panel bazlı optimal eşikler (optimise_panel_thresholds() ile)
        self.panel_thresholds: Dict[str, float] = {}

        logger.info(
            "HybridEnsemble: models=%s, weights=%s, device=%s",
            [m for m, v in [("XGB", xgb_model), ("LGB", lgbm_model),
                             ("GNN", gnn_model), ("DNN", dnn_model)] if v is not None],
            [round(w, 3) for w in self.weights],
            self.device,
        )

    # ------------------------------------------------------------------
    # Baz model olasılıkları
    # ------------------------------------------------------------------

    def predict_proba_all(
        self,
        X_scaled: np.ndarray,
        nuc_ids:  Optional[torch.Tensor] = None,
        aa_ids:   Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[np.ndarray],  # xgb  (N, 2)
        Optional[np.ndarray],  # lgbm (N, 2)
        Optional[np.ndarray],  # gnn  (N, 2)
        Optional[np.ndarray],  # dnn  (N, 2)
    ]:
        """
        Dört baz modelden ham olasılık matrislerini döndür.
        Her çıktı (N, 2) şeklinde [P(Benign), P(Pathogenic)] içerir
        veya model yoksa None olur.
        """
        # XGBoost
        xgb_probs: Optional[np.ndarray] = None
        if self.xgb is not None:
            xgb_probs = self.xgb.predict_proba(X_scaled)

        # LightGBM
        lgb_probs: Optional[np.ndarray] = None
        if self.lgbm is not None:
            lgb_probs = self.lgbm.predict_proba(X_scaled)

        # GNN (GAT v2 — örnek grafiği üzerinde)
        gnn_probs: Optional[np.ndarray] = None
        if self.gnn is not None:
            gnn_probs = self._gat_predict_proba(
                X_scaled, nuc_ids=nuc_ids, aa_ids=aa_ids
            )

        # DNN
        dnn_probs: Optional[np.ndarray] = None
        if self.dnn is not None:
            dnn_probs = _dnn_predict_proba(
                self.dnn.to(self.device), X_scaled, self.device
            )

        return xgb_probs, lgb_probs, gnn_probs, dnn_probs

    def _gat_predict_proba(
        self,
        X_scaled: np.ndarray,
        nuc_ids:  Optional[torch.Tensor] = None,
        aa_ids:   Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        VariantGATv2GNN'den örnek k-NN grafiği üzerinde olasılık döndür.
        Tek satırlık batch'ler de güvenle çalışır (BatchNorm bypass).
        """
        cfg   = get_settings()
        knn_k = getattr(cfg.gnn, "knn_k", 10)
        data  = _build_knn_graph(X_scaled, knn_k)

        model = self.gnn.to(self.device)
        model.eval()
        data  = data.to(self.device)

        with torch.no_grad():
            kwargs: Dict[str, Any] = {}
            if nuc_ids is not None:
                kwargs["nuc_ids"] = nuc_ids.to(self.device)
            if aa_ids is not None:
                kwargs["aa_ids"] = aa_ids.to(self.device)
            logits = model(data.x, data.edge_index, **kwargs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    # ------------------------------------------------------------------
    # Birleştirme
    # ------------------------------------------------------------------

    def combine(
        self,
        xgb_proba:  Optional[np.ndarray],
        lgb_proba:  Optional[np.ndarray],
        gnn_proba:  Optional[np.ndarray],
        dnn_proba:  Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Baz model olasılıklarını birleştirir.

        Öncelik:
          1. Meta-öğrenici stacking (fit_meta_learner() çağrıldıysa)
          2. Ağırlıklı ortalama (self.weights)

        Returns
        -------
        (N, 2) birleşik olasılık matrisi.
        """
        # ── Stacking yolu ──────────────────────────────────────────────
        if self.meta_learner is not None:
            cols = [
                p[:, 1]
                for p in [xgb_proba, lgb_proba, gnn_proba, dnn_proba]
                if p is not None
            ]
            if len(cols) >= 2:
                try:
                    meta_X    = np.column_stack(cols)            # (N, n_models)
                    meta_proba = self.meta_learner.predict_proba(meta_X)  # (N, 2)
                    return meta_proba
                except Exception as exc:
                    logger.warning(
                        "Meta-öğrenici tahmin başarısız (%s) — "
                        "ağırlıklı ortalamaya geçildi.", exc
                    )

        # ── Ağırlıklı ortalama ────────────────────────────────────────
        pairs = [
            (xgb_proba,  self.weights[0]),
            (lgb_proba,  self.weights[1]),
            (gnn_proba,  self.weights[2]),
            (dnn_proba,  self.weights[3]),
        ]
        available = [(p, w) for p, w in pairs if p is not None]
        if not available:
            raise ValueError(
                "HybridEnsemble.combine(): Hicbir baz modelden tahmin alinamadi. "
                "En az bir model (XGB/LGB/GNN/DNN) yuklenmis olmali."
            )

        total_w = sum(w for _, w in available)
        if total_w <= 0:
            raise ValueError("Mevcut modellerin agirlik toplami <= 0.")

        # Model eksikse agirliklar yeniden normalize edilir — bunu logla
        if len(available) < 4:
            active_names = [
                name for (p, _), name in zip(pairs, ["XGB", "LGB", "GNN", "DNN"])
                if p is not None
            ]
            logger.debug(
                "Ensemble: %d/4 model aktif (%s), agirliklar yeniden normalize edildi.",
                len(available), active_names,
            )

        return sum((w / total_w) * p for p, w in available)

    # ------------------------------------------------------------------
    # Tahmin API'si
    # ------------------------------------------------------------------

    def predict(
        self,
        X_scaled:  np.ndarray,
        threshold: float = 0.50,
        nuc_ids:   Optional[torch.Tensor] = None,
        aa_ids:    Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Klasik ensemble tahmini.

        Parameters
        ----------
        X_scaled  : Ön işlenmiş özellik matrisi (N, F).
        threshold : Sınıflandırma eşiği (§7.3: default 0.50 dengeli veri için).
        nuc_ids   : (opsiyonel) GNN multimodal nükleotid token tensörü.
        aa_ids    : (opsiyonel) GNN multimodal amino asit token tensörü.

        Returns
        -------
        preds : (N,) binary tahmin dizisi (0=Benign, 1=Pathogenic).
        proba : (N, 2) olasılık matrisi [P(Benign), P(Pathogenic)].
        """
        xp, lp, gp, dp = self.predict_proba_all(X_scaled, nuc_ids=nuc_ids, aa_ids=aa_ids)
        proba = self.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gp, dnn_proba=dp)
        preds = (proba[:, 1] >= threshold).astype(int)
        return preds, proba

    def predict_with_uncertainty(
        self,
        X_scaled:  np.ndarray,
        n_iter:    int   = 15,
        threshold: float = 0.50,
        nuc_ids:   Optional[torch.Tensor] = None,
        aa_ids:    Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC-Dropout ile epistemik belirsizlik nicemleme.

        GATv2 modeli ``n_iter`` kez dropout aktif olarak çalıştırılır;
        diğer modeller (XGB, LGB, DNN) belirsizlik tahmini için gerekli
        stokastisiteyi taşımadığından yalnızca bir kez çalıştırılır.

        Parameters
        ----------
        n_iter    : MC-Dropout tekrar sayısı (önerilen: 10-30).
        threshold : Sınıflandırma eşiği.

        Returns
        -------
        preds       : (N,)   binary tahminler.
        proba       : (N, 2) birleşik olasılık matrisi.
        uncertainty : (N,)   epistemik belirsizlik std (0=kesin, yüksek=belirsiz).
        """
        if self.gnn is None:
            # GNN yoksa normal tahmine geri dön, sıfır belirsizlik
            preds, proba = self.predict(X_scaled, threshold=threshold)
            uncertainty  = np.zeros(len(X_scaled), dtype=np.float32)
            return preds, proba, uncertainty

        cfg   = get_settings()
        knn_k = getattr(cfg.gnn, "knn_k", 10)
        data  = _build_knn_graph(X_scaled, knn_k)

        model = self.gnn.to(self.device)
        data  = data.to(self.device)

        kwargs: Dict[str, Any] = {}
        if nuc_ids is not None:
            kwargs["nuc_ids"] = nuc_ids.to(self.device)
        if aa_ids is not None:
            kwargs["aa_ids"] = aa_ids.to(self.device)

        # GNN MC-Dropout
        gnn_mean, gnn_std = model.predict_with_uncertainty(
            data.x, data.edge_index, n_iter=n_iter, **kwargs
        )
        gnn_probs     = gnn_mean.cpu().numpy()   # (N, 2)
        uncertainty   = gnn_std[:, 1].cpu().numpy()  # P(Pathogenic) std

        # Diğer modeller (tek pass)
        xp = self.xgb.predict_proba(X_scaled) if self.xgb  is not None else None
        lp = self.lgbm.predict_proba(X_scaled) if self.lgbm is not None else None
        dp = _dnn_predict_proba(
            self.dnn.to(self.device), X_scaled, self.device
        ) if self.dnn is not None else None

        proba = self.combine(
            xgb_proba=xp, lgb_proba=lp, gnn_proba=gnn_probs, dnn_proba=dp
        )
        preds = (proba[:, 1] >= threshold).astype(int)

        return preds, proba, uncertainty

    def predict_with_panel_threshold(
        self,
        X_scaled:    np.ndarray,
        panel_labels: Sequence[str],
        default_threshold: float = 0.50,
        nuc_ids:    Optional[torch.Tensor] = None,
        aa_ids:     Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Panel bazlı optimal eşiklerle tahmin.

        Her örnek kendi paneline ait optimise edilmiş eşikle sınıflandırılır.
        Eşik bulunamazsa ``default_threshold`` kullanılır.

        Parameters
        ----------
        panel_labels : Her örnek için panel adı dizisi (len=N).

        Returns
        -------
        preds : (N,) binary tahminler.
        proba : (N, 2) birleşik olasılık matrisi.
        """
        xp, lp, gp, dp = self.predict_proba_all(
            X_scaled, nuc_ids=nuc_ids, aa_ids=aa_ids
        )
        proba = self.combine(xgb_proba=xp, lgb_proba=lp, gnn_proba=gp, dnn_proba=dp)

        preds = np.empty(len(X_scaled), dtype=int)
        for i, panel in enumerate(panel_labels):
            thr      = self.panel_thresholds.get(panel, default_threshold)
            preds[i] = int(proba[i, 1] >= thr)

        return preds, proba

    # ------------------------------------------------------------------
    # Ağırlık optimizasyonu (§5.3 — Nelder-Mead)
    # ------------------------------------------------------------------

    def optimise_weights(
        self,
        X_val:  np.ndarray,
        loader: Any,          # API uyumluluğu için — None geçin
        y_val:  np.ndarray,
        nuc_ids:  Optional[torch.Tensor] = None,
        aa_ids:   Optional[torch.Tensor] = None,
        n_iter_nm: int = 600,
    ) -> None:
        """
        Nelder-Mead ile topluluk ağırlıklarını validation kümesinde optimize et.

        Hedef fonksiyon: Binary F1 (Pathogenic sınıfı, §7.3) maksimize edilir.
        Sonuç ``self.weights`` içine in-place yazılır.

        Parameters
        ----------
        X_val      : Ön işlenmiş doğrulama özellik matrisi.
        loader     : Kullanılmıyor; API uyumluluğu için None geçin.
        y_val      : Doğrulama etiketleri (0=Benign, 1=Pathogenic).
        n_iter_nm  : Nelder-Mead maksimum iterasyon sayısı.
        """
        xp, lp, gp, dp = self.predict_proba_all(
            X_val, nuc_ids=nuc_ids, aa_ids=aa_ids
        )
        matrices   = [xp, lp, gp, dp]
        avail_idx  = [i for i, m in enumerate(matrices) if m is not None]

        if len(avail_idx) < 2:
            logger.info(
                "optimise_weights: en az 2 aktif model gerekli (%d mevcut) — atlandı.",
                len(avail_idx),
            )
            return

        avail_m = [matrices[i] for i in avail_idx]

        def _neg_binary_f1(w_raw: np.ndarray) -> float:
            """§7.3: Binary F1 (Pathogenic, pos_label=1) negatifi."""
            w       = np.abs(w_raw)
            w       = w / (w.sum() + 1e-12)
            blended = sum(wi * mi for wi, mi in zip(w, avail_m))
            preds   = (blended[:, 1] >= 0.5).astype(int)
            return -f1_score(
                y_val, preds,
                average="binary", pos_label=1, zero_division=0,
            )

        x0     = np.array([self.weights[i] for i in avail_idx])
        result = minimize(
            _neg_binary_f1, x0, method="Nelder-Mead",
            options={"maxiter": n_iter_nm, "xatol": 1e-5, "fatol": 1e-5},
        )

        opt_w = np.abs(result.x)
        opt_w = opt_w / (opt_w.sum() + 1e-12)

        new_weights = [0.0] * 4
        for j, idx in enumerate(avail_idx):
            new_weights[idx] = float(opt_w[j])
        ws = sum(new_weights)
        self.weights = [w / ws for w in new_weights] if ws > 0 else new_weights

        logger.info(
            "optimise_weights: ağırlıklar=%s  (val Binary F1=%.4f)",
            [round(w, 4) for w in self.weights], -result.fun,
        )

    # ------------------------------------------------------------------
    # Panel bazlı eşik optimizasyonu
    # ------------------------------------------------------------------

    def optimise_panel_thresholds(
        self,
        X_val:        np.ndarray,
        y_val:        np.ndarray,
        panel_labels: Sequence[str],
        n_steps:      int = 200,
    ) -> Dict[str, float]:
        """
        Her panel için F1-optimal sınıflandırma eşiği bul.

        Şartname §3.2: 4 panel (General, Hereditary_Cancer, PAH, CFTR).
        Her panel için eşik bağımsız olarak optimize edilir.

        Parameters
        ----------
        X_val        : Ön işlenmiş doğrulama özellik matrisi.
        y_val        : Doğrulama etiketleri.
        panel_labels : Her örnek için panel adı dizisi.
        n_steps      : Eşik tarama adım sayısı.

        Returns
        -------
        {panel_name: optimal_threshold} sözlüğü.
        ``self.panel_thresholds`` da güncellenir.
        """
        _, proba = self.predict(X_val, threshold=0.5)
        p1       = proba[:, 1]
        panels_arr = np.array(panel_labels)
        thresholds = np.linspace(0.01, 0.99, n_steps)

        results: Dict[str, float] = {}

        # Global (tüm paneller)
        f1_global = np.array([
            f1_score(
                y_val,
                (p1 >= thr).astype(int),
                average="binary", pos_label=1, zero_division=0,
            )
            for thr in thresholds
        ])
        best_global = float(thresholds[np.argmax(f1_global)])
        results["__global__"] = best_global
        logger.info(
            "Panel __global__: optimal eşik=%.3f  F1=%.4f",
            best_global, f1_global.max(),
        )

        # Panel bazlı
        for panel in self.KNOWN_PANELS:
            mask = panels_arr == panel
            if mask.sum() < 10:
                results[panel] = best_global
                logger.warning(
                    "Panel %s: yetersiz örnek (%d) — global eşik kullanıldı.",
                    panel, mask.sum(),
                )
                continue

            y_p  = y_val[mask]
            p1_p = p1[mask]
            f1_p = np.array([
                f1_score(
                    y_p,
                    (p1_p >= thr).astype(int),
                    average="binary", pos_label=1, zero_division=0,
                )
                for thr in thresholds
            ])
            best_p = float(thresholds[np.argmax(f1_p)])
            results[panel] = best_p
            logger.info(
                "Panel %-22s: optimal eşik=%.3f  F1=%.4f  (n=%d)",
                panel, best_p, f1_p.max(), mask.sum(),
            )

        self.panel_thresholds = results
        return results

    # ------------------------------------------------------------------
    # Meta-öğrenici stacking (§5.3)
    # ------------------------------------------------------------------

    def fit_meta_learner(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        nuc_ids: Optional[torch.Tensor] = None,
        aa_ids:  Optional[torch.Tensor] = None,
    ) -> None:
        """
        Doğrulama kümesi tahminleri üzerine LogisticRegression meta-öğrenicisi eğit.

        Eğitim tamamlandıktan sonra ``combine()`` ağırlıklı ortalama yerine
        meta-öğreniciyi kullanır.

        Parameters
        ----------
        X_val : Ön işlenmiş doğrulama özellik matrisi.
        y_val : Doğrulama etiketleri (0=Benign, 1=Pathogenic).
        """
        from sklearn.linear_model import LogisticRegression

        xp, lp, gp, dp = self.predict_proba_all(
            X_val, nuc_ids=nuc_ids, aa_ids=aa_ids
        )
        cols = [
            p[:, 1]
            for p in [xp, lp, gp, dp]
            if p is not None
        ]
        if len(cols) < 2:
            logger.info(
                "fit_meta_learner: en az 2 aktif model gerekli (%d mevcut) — atlandı.",
                len(cols),
            )
            return

        meta_X = np.column_stack(cols)   # (N, n_active_models)
        lr     = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",   # dengeli veri için gereksiz ama zararsız
        )
        lr.fit(meta_X, y_val)
        self.meta_learner = lr

        # Eğitim sonrası doğrulama Binary F1 (§7.3)
        meta_preds = lr.predict(meta_X)
        meta_f1    = f1_score(
            y_val, meta_preds,
            average="binary", pos_label=1, zero_division=0,
        )
        logger.info(
            "fit_meta_learner: %d örnek, %d model kolonu → val Binary F1=%.4f",
            len(y_val), meta_X.shape[1], meta_f1,
        )

    # ------------------------------------------------------------------
    # Yardımcı statik/sınıf metodları
    # ------------------------------------------------------------------

    @staticmethod
    def pathogenic_risk_score(ensemble_proba: np.ndarray) -> np.ndarray:
        """
        P(Pathogenic) × 100 olarak kalibre edilmiş risk skoru.

        Returns
        -------
        (N,) float dizisi [0, 100] aralığında.
        """
        return (ensemble_proba[:, 1] * 100).round(2)

    def score(
        self,
        X_scaled:  np.ndarray,
        y_true:    np.ndarray,
        threshold: float = 0.50,
    ) -> float:
        """
        Topluluğun Binary F1 skorunu döndür (§7.3).

        Kullanım kolaylığı amacıyla: sklearn-benzeri arayüz.
        """
        preds, _ = self.predict(X_scaled, threshold=threshold)
        return float(
            f1_score(
                y_true, preds,
                average="binary", pos_label=1, zero_division=0,
            )
        )

    def summary(self) -> str:
        """İnsan okunur topluluk özeti."""
        active = [
            name
            for name, model in [
                ("XGBoost", self.xgb),
                ("LightGBM", self.lgbm),
                ("VariantGATv2GNN", self.gnn),
                ("VariantDNN", self.dnn),
            ]
            if model is not None
        ]
        weights_str = " | ".join(
            f"{n}={w:.3f}"
            for n, w in zip(
                ["XGB", "LGB", "GNN", "DNN"], self.weights
            )
        )
        meta = "MetaLearner=AKTIF" if self.meta_learner is not None else "MetaLearner=YOK"
        panel_thr = (
            f"PanelEşikler={list(self.panel_thresholds.keys())}"
            if self.panel_thresholds
            else "PanelEşikler=YOK"
        )
        return (
            f"HybridEnsemble[\n"
            f"  Aktif modeller : {', '.join(active)}\n"
            f"  Ağırlıklar     : {weights_str}\n"
            f"  {meta}\n"
            f"  {panel_thr}\n"
            f"  Device         : {self.device}\n"
            f"]"
        )

    def __repr__(self) -> str:
        return self.summary()

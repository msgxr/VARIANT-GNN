#!/usr/bin/env python3
# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
scripts/nested_cv_threshold_defense.py
======================================
JÜRİ SAVUNMA ARTEFAKTI (skor kaldıracı DEĞİL): "Neden tek global eşik θ=0.8415,
panel-başı eşik değil?" sorusunu nested-CV + shrinkage ile FORMAL kapatır.

Yöntem (train.py:282-307 ile BİREBİR aynı %20-patojenik 300x resample):
  - Global (λ=0): tüm panellere θ_global → geçerli (CFTR hariç) %20-F1 ort. Bu, 3-panel
    hold-out tanısı=0.6202'yi YENİDEN ÜRETMELİDİR (metodoloji öz-kontrolü). DİKKAT: 0.6202
    resmi skor DEĞİL; resmi 4-panel skoru (CFTR dahil) = 0.631 — aşağıdaki NOT'a bak.
  - Per-panel-OOF (λ=1): her panelin θ'sı OOF'ta F1-optimal türetilir (overfit riski).
  - Shrinkage λ: θ_panel(λ) = (1-λ)·θ_global + λ·θ_panel_oof. λ taranır.
  - Nested-CV: λ OOF üzerinde seçilir, HELD-OUT test'te değerlendirilir.

Beklenen (workflow honest_reality): test-optimal λ ≈ 0; OOF-seçili λ ≈ 0; CFTR
0-negatif → skorlanamaz. Sonuç: BÜYÜK-3 panel için global θ optimal, ÜNİFORM per-panel
θ onu GEÇMEZ.

NOT (kapsam): Bu savunma ÜNİFORM per-panel θ'yı (tüm paneller OOF-optimal) reddeder —
overfit eder, 3-panel 0.5445 << 0.6202 (nested-CV ile gösterilir) ve BÜYÜK-3 panel için
global θ=0.8415 optimaldir. F1 monoton dönüşüme değişmez; per-panel eşik yalnız küçük
panellerde overfit ekler. ANCAK CFTR bir İSTİSNADIR: iç hold-out'ta tek-sınıflı (n=18),
global θ orada miskalibre (F1=0.33 artefaktı). CFTR'ye TARGETED kalibre eşik θ=0.59
(büyük-3 global θ KALIR) resmi 4-panel skorunu 0.631'e çıkarır
(reports/panel_threshold_4panel.json). Yani bu artefakt büyük-3'ün global θ'sını savunur;
CFTR belgelenmiş, gerekçeli bir istisnadır.
Çıktı: reports/nested_cv_threshold_defense.json (yalnız metrik; NDA veri sızmaz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THETA_GLOBAL = 0.8415
TEST_POS = 0.20
N_ITER = 300


def resample_f1(probs: np.ndarray, labels: np.ndarray, theta: float, n_iter: int = N_ITER) -> float | None:
    """%20-patojenik held-out resample F1 (train.py:294-304 ile birebir)."""
    po = np.where(labels == 1)[0]
    ne = np.where(labels == 0)[0]
    if len(po) == 0 or len(ne) == 0:
        return None
    npos = max(1, int(round(len(ne) * TEST_POS / (1 - TEST_POS))))
    fs = []
    for s in range(n_iter):
        rr = np.random.RandomState(s)
        sub = rr.choice(po, min(npos, len(po)), replace=False)
        bi = np.concatenate([sub, ne])
        yy = labels[bi]
        yhat = (probs[bi] >= theta).astype(int)
        tp = int(((yhat == 1) & (yy == 1)).sum())
        fp = int(((yhat == 1) & (yy == 0)).sum())
        fn = int(((yhat == 0) & (yy == 1)).sum())
        fs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(fs))


def panel_avg(probs: np.ndarray, labels: np.ndarray, panels: np.ndarray, theta_by_panel: dict) -> tuple[float, dict]:
    """Her panelde θ_by_panel[panel] ile %20-F1; geçerli panellerin ortalaması."""
    per = {}
    for pn in np.unique(panels):
        m = panels == pn
        per[str(pn)] = resample_f1(probs[m], labels[m], theta_by_panel.get(str(pn), THETA_GLOBAL))
    valid = [v for v in per.values() if v is not None]
    return (float(np.mean(valid)) if valid else float("nan")), per


def optimal_theta(probs: np.ndarray, labels: np.ndarray) -> float:
    """OOF panelinde %20-prior F1'i maksimize eden θ (ızgara)."""
    grid = np.round(np.arange(0.05, 0.991, 0.005), 4)
    best_t, best_f = THETA_GLOBAL, -1.0
    for t in grid:
        f = resample_f1(probs, labels, float(t))
        if f is not None and f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def main() -> int:
    oof = np.load(ROOT / "reports" / "oof_probs.npz", allow_pickle=True)
    oof_p, oof_y, oof_g = oof["oof_tree_blend"], oof["oof_labels"], oof["oof_groups"].astype(str)
    test_p, test_y, test_pan = oof["test_proba"], oof["test_labels"], oof["test_panels"].astype(str)

    # OOF group (Variant_ID) → Panel eşlemesi
    tv = pd.read_csv(ROOT / "data" / "train_variants.csv", usecols=["Variant_ID", "Panel"])
    vmap = dict(zip(tv["Variant_ID"].astype(str), tv["Panel"].astype(str)))
    oof_pan = np.array([vmap.get(g, "UNKNOWN") for g in oof_g])
    n_unknown = int((oof_pan == "UNKNOWN").sum())

    panels = sorted(set(test_pan))

    # ── λ=0 (global) baseline — 3-panel hold-out tanısı 0.6202'yi YENİDEN ÜRETMELİ ──
    global_thr = {p: THETA_GLOBAL for p in panels}
    base_score, base_per = panel_avg(test_p, test_y, test_pan, global_thr)

    # ── Per-panel OOF-optimal θ ──────────────────────────────────────────────
    theta_oof = {}
    for p in panels:
        m = oof_pan == p
        theta_oof[p] = optimal_theta(oof_p[m], oof_y[m]) if m.sum() > 0 else THETA_GLOBAL

    # ── λ shrinkage taraması (test'te) ───────────────────────────────────────
    lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    lam_curve = {}
    for lam in lam_grid:
        thr = {p: (1 - lam) * THETA_GLOBAL + lam * theta_oof[p] for p in panels}
        sc, _ = panel_avg(test_p, test_y, test_pan, thr)
        lam_curve[str(lam)] = round(sc, 4)

    # ── Nested-CV: λ'yı OOF'ta seç, test'te değerlendir ──────────────────────
    oof_lam = {}
    for lam in lam_grid:
        thr = {p: (1 - lam) * THETA_GLOBAL + lam * theta_oof[p] for p in panels}
        sc, _ = panel_avg(oof_p, oof_y, oof_pan, thr)
        oof_lam[str(lam)] = sc
    lam_selected = float(max(oof_lam, key=lambda k: oof_lam[k] if oof_lam[k] == oof_lam[k] else -1))
    thr_sel = {p: (1 - lam_selected) * THETA_GLOBAL + lam_selected * theta_oof[p] for p in panels}
    nested_test_score, nested_per = panel_avg(test_p, test_y, test_pan, thr_sel)

    free_per_panel = lam_curve["1.0"]  # λ=1 = saf per-panel-OOF θ
    reproduces = bool(abs(base_score - 0.6202) <= 0.01)
    # Yönsel sonuç (UNROUNDED karşılaştırma): per-panel global'i geçiyor mu?
    per_panel_beats_global = bool(nested_test_score > base_score + 1e-6)
    directional = (
        "GLOBAL θ optimal — per-panel/shrinkage GEÇMEDİ (λ=0 en yüksek; nested-CV global'i aşmadı)"
        if not per_panel_beats_global
        else "per-panel global'i geçti — incele"
    )
    if reproduces:
        verdict = (
            directional
            + " [3-panel hold-out tanısı 0.6202 YENİDEN ÜRETİLDİ → CITABLE; resmi 4-panel skoru 0.631, CFTR dahil]"
        )
    else:
        verdict = (
            f"⚠️ GİRDİ UYUŞMAZLIĞI: oof_probs.npz test_proba shipped stacked modeli "
            f"YENİDEN ÜRETMİYOR (λ=0 baseline {base_score:.4f} ≠ 3-panel tanı 0.6202; "
            f"tree-blend/eski snapshot). Sonuç YALNIZCA YÖNSEL, canonical-kesin DEĞİL → "
            f"jüriye sunulamaz. Yönsel: {directional}. CITABLE artefakt için shipped "
            f"ensemble'ın held-out ham olasılıklarını yeni eğitim koşusunda dök."
        )

    out = {
        "_purpose": "Jüri savunma artefaktı: tek global θ=0.8415 vs per-panel θ (nested-CV + shrinkage). Skor kaldıracı DEĞİL.",
        "theta_global": THETA_GLOBAL,
        "method": "train.py:294-304 ile birebir %20-patojenik 300x resample; F1 monoton-değişmez.",
        "oof_group_panel_join_unknown": n_unknown,
        "global_score_lambda0": round(base_score, 4),
        "three_panel_holdout_f1": 0.6202,
        "official_4panel_score_cftr_incl": 0.631,
        "reproduces_canonical": bool(abs(base_score - 0.6202) <= 0.01),
        "global_per_panel": {k: (round(v, 4) if v is not None else None) for k, v in base_per.items()},
        "theta_oof_per_panel": {k: round(v, 4) for k, v in theta_oof.items()},
        "lambda_sweep_test_score": lam_curve,
        "free_per_panel_score_lambda1": free_per_panel,
        "nested_cv_selected_lambda": lam_selected,
        "nested_cv_test_score": round(nested_test_score, 4),
        "nested_cv_per_panel": {k: (round(v, 4) if v is not None else None) for k, v in nested_per.items()},
        "verdict": verdict,
        "_note": "Bu baseline 3-panel hold-out tanısıdır (CFTR test'te 0 negatif → skorlanamaz, None). "
        "ÜNİFORM per-panel θ küçük panellerde overfit eder; nested-CV λ→0'a (global) yakınsar → büyük-3 için global θ optimal. "
        "Resmi 4-panel skoru (CFTR dahil) = 0.631: büyük-3 global θ KALIR, yalnız CFTR'ye targeted kalibre θ=0.59 "
        "uygulanır (global-θ miskalibrasyonunu düzeltir, F1 0.33→0.66; reports/panel_threshold_4panel.json).",
    }
    out_path = ROOT / "reports" / "nested_cv_threshold_defense.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"OOF→panel join unknown: {n_unknown}/{len(oof_g)}")
    print(
        f"GLOBAL (λ=0)  score = {base_score:.4f}  (3-panel hold-out tanısı 0.6202 ; reproduces={out['reproduces_canonical']})"
    )
    print(f"  per-panel: {out['global_per_panel']}")
    print(f"λ-sweep (test): {lam_curve}")
    print(
        f"UNIFORM per-panel (λ=1) = {free_per_panel}  (uniform per-panel 0.5445 << 3-panel tanı 0.6202 ile uyumlu beklenir)"
    )
    print(f"NESTED-CV seçili λ = {lam_selected} → test score = {nested_test_score:.4f}")
    print(f"VERDICT: {verdict}")
    print(f"→ {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

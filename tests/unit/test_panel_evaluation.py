"""
tests/unit/test_panel_evaluation.py
TEKNOFEST 2026 §7.3 — Panel bazlı binary F1 değerlendirme testleri.

Şartname gereksinimleri:
  - 4 panel: General, Hereditary_Cancer, PAH, CFTR
  - Her panel için ayrı Binary F1 (TP/FP/FN, pos_label=1) hesaplanmalı
  - Temel metrik binary F1; macro F1 ikincil
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score


def _make_panel_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    panels_cfg = {
        "General": (1000, 1000),
        "Hereditary_Cancer": (100, 100),
        "PAH": (100, 100),
        "CFTR": (30, 30),
    }
    all_labels, all_probs, all_panels = [], [], []
    for panel, (n_pat, n_ben) in panels_cfg.items():
        n = n_pat + n_ben
        y = np.array([1] * n_pat + [0] * n_ben)
        p = np.clip(
            y * rng.normal(0.7, 0.15, n) + (1 - y) * rng.normal(0.3, 0.15, n),
            0.01, 0.99,
        )
        all_labels.append(y)
        all_probs.append(p)
        all_panels.extend([panel] * n)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.array(all_panels),
    )


class TestPanelEvaluationF1:
    def test_evaluate_per_panel_returns_all_panels(self):
        from src.evaluation.metrics import evaluate_per_panel
        y, p, panels = _make_panel_data()
        proba = np.column_stack([1 - p, p])
        reports = evaluate_per_panel(y, proba, panels, threshold=0.5)
        for panel in ["General", "Hereditary_Cancer", "PAH", "CFTR"]:
            assert panel in reports, f"Panel '{panel}' raporda bulunamadı"

    def test_binary_f1_is_primary_metric_per_panel(self):
        """§7.3 — Her panel için binary_f1 birincil metrik olmalı."""
        from src.evaluation.metrics import evaluate_per_panel
        y, p, panels = _make_panel_data()
        proba = np.column_stack([1 - p, p])
        reports = evaluate_per_panel(y, proba, panels, threshold=0.5)
        for panel, rep in reports.items():
            preds = (p[panels == panel] >= 0.5).astype(int)
            expected = float(f1_score(
                y[panels == panel], preds, average="binary", pos_label=1, zero_division=0
            ))
            assert abs(rep.binary_f1 - expected) < 1e-6, (
                f"Panel {panel}: binary_f1 beklenen {expected:.6f}, alınan {rep.binary_f1:.6f}"
            )

    def test_cftr_panel_has_minimum_samples(self):
        """CFTR eğitim: 70P + 70B = 140; test: 30P + 30B = 60 — şartname §3.2."""
        from src.evaluation.metrics import evaluate_per_panel
        y, p, panels = _make_panel_data()
        proba = np.column_stack([1 - p, p])
        cftr_mask = panels == "CFTR"
        assert cftr_mask.sum() == 60, f"CFTR test seti 60 örnek olmalı, {cftr_mask.sum()} bulundu"
        reports = evaluate_per_panel(y, proba, panels, threshold=0.5)
        assert "CFTR" in reports
        assert reports["CFTR"].binary_f1 >= 0.0

    def test_general_panel_test_size(self):
        """General test: 1000P + 1000B = 2000 — şartname §3.2."""
        _, _, panels = _make_panel_data()
        assert (panels == "General").sum() == 2000

    def test_all_panel_binary_f1_in_range(self):
        from src.evaluation.metrics import evaluate_per_panel
        y, p, panels = _make_panel_data()
        proba = np.column_stack([1 - p, p])
        reports = evaluate_per_panel(y, proba, panels, threshold=0.5)
        for panel, rep in reports.items():
            assert 0.0 <= rep.binary_f1 <= 1.0, (
                f"Panel {panel}: binary_f1={rep.binary_f1} geçerli aralık dışında"
            )

    def test_panel_dict_threshold(self):
        """Panel bazlı farklı eşik değerleri desteklenmeli."""
        from src.evaluation.metrics import evaluate_per_panel
        y, p, panels = _make_panel_data()
        proba = np.column_stack([1 - p, p])
        thr_dict = {
            "General": 0.5,
            "Hereditary_Cancer": 0.45,
            "PAH": 0.40,
            "CFTR": 0.35,
        }
        reports = evaluate_per_panel(y, proba, panels, threshold=thr_dict)
        assert len(reports) == 4

    def test_panel_counts_match_spec(self):
        """Şartname §3.2 test seti sayıları doğrulanmalı."""
        _, _, panels = _make_panel_data()
        counts = {p: (panels == p).sum() for p in np.unique(panels)}
        assert counts["General"] == 2000
        assert counts["Hereditary_Cancer"] == 200
        assert counts["PAH"] == 200
        assert counts["CFTR"] == 60

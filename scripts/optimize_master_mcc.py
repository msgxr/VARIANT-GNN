# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
scripts/optimize_master_mcc.py
================================
MASTER paneli için MCC-optimal eşiği RAPORLAR (analiz-only).

ÖNEMLİ: Bu script artık models/ altındaki HİÇBİR artefaktı (threshold.json,
panel_thresholds.json) DEĞİŞTİRMEZ. Kanonik karar eşiği θ=0.8415'tir ve yalnız
eğitim hattınca (src/cli/modes/train.py) yazılır. Burada yalnız bir tanı analizi
(group-aware OOF, ensemble olasılığı) üretilip reports/'a yazılır.

Çıktı: reports/master_mcc_threshold_analysis.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, matthews_corrcoef  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.config import get_settings, reset_settings  # noqa: E402
from src.data.loader import load_csv  # noqa: E402
from src.utils.seeds import set_global_seed  # noqa: E402
from src.utils.serialization import ModelStore  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
set_global_seed(42)
reset_settings()
cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))

print("Veri yükleniyor (MASTER / General panel)...")
ds = load_csv(str(REPO / "data" / "train_variants.csv"))
X_all = ds.features.values

# Variant_ID + Panel'i ham CSV'den al (group-aware split + MASTER filtresi için)
raw_df = pd.read_csv(str(REPO / "data" / "train_variants.csv"))
panel_all = raw_df.get("Panel", pd.Series(["General"] * len(ds.labels))).astype(str).str.strip().values
vid_all = raw_df.get("Variant_ID", pd.Series([f"v{i}" for i in range(len(ds.labels))])).astype(str)
base_ids_all = vid_all.str.replace(r"_aug\d*$", "", regex=True).to_numpy()

mask = np.array([p in ("General", "MASTER", "") for p in panel_all])
if not mask.any():
    mask = np.ones(len(ds.labels), dtype=bool)

X = X_all[mask]
y = np.array(ds.labels)[mask]
groups = base_ids_all[mask]
print(f"  MASTER örnekleri: {len(y)} (P={y.sum()}, B={(y == 0).sum()})")

store = ModelStore(cfg.paths.models_dir)
preproc = store.load_preprocessor()
# Tüm ensemble (XGB-only DEĞİL) — kanonik karar ensemble olasılığında verilir.
ensemble = joblib.load(str(REPO / "models" / "ensemble.pkl"))

# ── GROUP-AWARE 5-Fold OOF ile olasılık üret (satır-bazlı sızıntı yok) ─────────
print("\nGroup-aware 5-Fold OOF olasılık toplanıyor...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_probs, all_labels = [], []
for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_val_p = preproc.transform(X[val_idx])
    _out = ensemble.predict(X_val_p)
    probs = _out[1][:, 1] if isinstance(_out, tuple) else _out[:, 1]
    all_probs.extend(probs)
    all_labels.extend(y[val_idx])
    assert len(set(groups[tr_idx]) & set(groups[val_idx])) == 0, "OOF fold sızıntısı!"
    print(f"  Fold {fold + 1}: n_val={len(val_idx)}")

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Eşik taraması (analiz) ────────────────────────────────────────────────────
thresholds = np.arange(0.05, 0.96, 0.01)
f1s = np.array([f1_score(all_labels, (all_probs >= t).astype(int), pos_label=1, zero_division=0) for t in thresholds])
mccs = np.array([matthews_corrcoef(all_labels, (all_probs >= t).astype(int)) for t in thresholds])

CANON = 0.8415
baseline_idx = int(np.argmin(np.abs(thresholds - CANON)))
baseline_f1, baseline_mcc = float(f1s[baseline_idx]), float(mccs[baseline_idx])

# MCC-optimal (F1 kaybı < 0.01 şartıyla) — yalnız RAPOR amaçlı
valid_mask = f1s >= (baseline_f1 - 0.01)
best_idx = int(np.argmax(mccs * valid_mask))
best_t, best_mcc, best_f1 = float(thresholds[best_idx]), float(mccs[best_idx]), float(f1s[best_idx])

print(f"\n{'=' * 55}")
print("MASTER Panel Eşik ANALİZİ (analiz-only — models/ DEĞİŞMEZ)")
print(f"  Kanonik θ={CANON} → F1={baseline_f1:.4f}, MCC={baseline_mcc:.4f}")
print(f"  MCC-optimal (rapor) θ={best_t:.3f} → F1={best_f1:.4f}, MCC={best_mcc:.4f}")
print(f"  ΔMCC=+{best_mcc - baseline_mcc:.4f}  ΔF1={best_f1 - baseline_f1:+.4f}")

# ── reports/'a yaz (models/'a ASLA) ──────────────────────────────────────────
out = {
    "_note": (
        "ANALİZ-ONLY. Kanonik karar eşiği θ=0.8415'tir ve değiştirilmez. "
        "Bu dosya yalnız MASTER panelinde MCC-eşik ödünleşimini tanı amaçlı gösterir."
    ),
    "method": "group-aware StratifiedGroupKFold OOF (Variant_ID), ensemble probability",
    "canonical_threshold": CANON,
    "canonical_f1": round(baseline_f1, 6),
    "canonical_mcc": round(baseline_mcc, 6),
    "mcc_optimal_threshold_report_only": round(best_t, 6),
    "mcc_optimal_f1": round(best_f1, 6),
    "mcc_optimal_mcc": round(best_mcc, 6),
    "n": int(len(all_labels)),
}
rep_dir = REPO / "reports"
rep_dir.mkdir(parents=True, exist_ok=True)
(rep_dir / "master_mcc_threshold_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\n  ✅ reports/master_mcc_threshold_analysis.json yazıldı (models/ değişmedi).")
print(f"{'=' * 55}")

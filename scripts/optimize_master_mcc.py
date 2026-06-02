"""
scripts/optimize_master_mcc.py
================================
MASTER paneli için MCC-optimal eşik bulur.
Model yeniden eğitilmez — sadece karar eşiği güncellenir.
Çıktı: models/panel_thresholds.json güncellenir
"""
from __future__ import annotations
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef

from src.config import reset_settings, get_settings
from src.data.loader import load_csv
from src.utils.serialization import ModelStore
from src.utils.seeds import set_global_seed

REPO = Path(__file__).resolve().parent.parent
set_global_seed(42)
reset_settings()
cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))

print("Veri yükleniyor (MASTER / General panel)...")
ds = load_csv(str(REPO / "data" / "train_variants.csv"))
X_all = ds.features.values

# MASTER = General panel filtresi
if hasattr(ds, "panel") and ds.panel is not None:
    mask = np.array([str(p).strip() in ("General", "MASTER", "") for p in ds.panel])
else:
    # Panel kolonu yoksa tüm veriyi kullan (zaten birleşik)
    mask = np.ones(len(ds.labels), dtype=bool)

X = X_all[mask]
y = np.array(ds.labels)[mask]
print(f"  MASTER örnekleri: {len(y)} (P={y.sum()}, B={(y==0).sum()}, oran={y.sum()/(y==0).sum():.2f}:1)")

store   = ModelStore(cfg.paths.models_dir)
preproc = store.load_preprocessor()
xgb     = store.load_xgb()

# ── 5-Fold CV ile her fold'da validation prob üret ───────────────────────────
print("\n5-Fold CV ile olasılık toplanıyor...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_probs, all_labels = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    X_val_p = preproc.transform(X_val)
    probs   = xgb.predict_proba(X_val_p)[:, 1]
    all_probs.extend(probs)
    all_labels.extend(y_val)
    print(f"  Fold {fold+1}: n_val={len(y_val)}")

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Eşik taraması ─────────────────────────────────────────────────────────────
print("\nEşik taraması (0.05 – 0.95)...")
thresholds = np.arange(0.05, 0.96, 0.01)
f1s, mccs  = [], []

for t in thresholds:
    preds = (all_probs >= t).astype(int)
    f1s.append(f1_score(all_labels, preds, pos_label=1, zero_division=0))
    mccs.append(matthews_corrcoef(all_labels, preds))

f1s  = np.array(f1s)
mccs = np.array(mccs)

# Mevcut F1 baseline (θ=0.6831, canonical karar eşiği)
baseline_idx = np.argmin(np.abs(thresholds - 0.6831))
baseline_f1  = float(f1s[baseline_idx])
baseline_mcc = float(mccs[baseline_idx])

# MCC-optimal: F1 kayıbı < 0.01 şartıyla
f1_floor    = baseline_f1 - 0.01
valid_mask  = f1s >= f1_floor
best_mcc_idx = int(np.argmax(mccs * valid_mask))
best_t       = float(thresholds[best_mcc_idx])
best_mcc     = float(mccs[best_mcc_idx])
best_f1      = float(f1s[best_mcc_idx])

print(f"\n{'='*55}")
print(f"SONUÇ — MASTER Panel Eşik Optimizasyonu")
print(f"{'='*55}")
print(f"  Mevcut  θ=0.6831 → F1={baseline_f1:.4f}, MCC={baseline_mcc:.4f}")
print(f"  Optimal θ={best_t:.3f} → F1={best_f1:.4f}, MCC={best_mcc:.4f}")
print(f"  ΔMCC = +{best_mcc - baseline_mcc:.4f}")
print(f"  ΔF1  =  {best_f1 - baseline_f1:+.4f}")

# ── panel_thresholds.json güncelle ─────────────────────────────────────────
thresh_path = REPO / "models" / "panel_thresholds.json"
thresholds_dict = json.loads(thresh_path.read_text())
old_general = thresholds_dict.get("General", thresholds_dict.get("__global__", 0.241))

thresholds_dict["General"] = round(best_t, 6)
# __global__ da güncelle (tahmin pipeline'ı bu değeri kullanıyor)
thresholds_dict["__global__"] = round(best_t, 6)

thresh_path.write_text(json.dumps(thresholds_dict, indent=2))
print(f"\n  ✅ panel_thresholds.json güncellendi")
print(f"     General: {old_general:.6f} → {best_t:.6f}")

# ── threshold.json güncelle ───────────────────────────────────────────────
global_path = REPO / "models" / "threshold.json"
global_data = json.loads(global_path.read_text())
global_data["threshold"] = round(best_t, 6)
global_path.write_text(json.dumps(global_data, indent=2))
print(f"     threshold.json: {round(best_t,6)}")
print(f"{'='*55}")
print("Tüm eşikler:")
for k, v in json.loads(thresh_path.read_text()).items():
    print(f"  {k}: {v:.6f}")

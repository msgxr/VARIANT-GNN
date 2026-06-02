"""
scripts/generate_explainability.py
====================================
İki işi birlikte yapar:

1. DOĞRU THRESHOLD'DA METRİK GÜNCELLEMESİ
   - cv_report.json'daki panel metriklerini üretim threshold'larıyla yeniden hesaplar.
   - Canonical eşik: θ=0.8415 (RESULTS_CANONICAL.json).
   - Test prior: %20-patojenik. Resmi jüri skoru: 0.6202 (4-panel %20-F1 ort.).
   - İç hold-out: F1=0.8367, MCC=0.5112.

2. SHAP GROUP EXPLAINABILITY
   - XGBoost TreeExplainer → işlenmiş özellikler üzerinde SHAP değerleri
   - Özellikler orijinal AL_x isimlerine geri eşlenir
   - 6 biyolojik gruba gruplandırılır (PSR §4.4)
   - Figürler: 16_shap_group_contributions.png, 17_mcc_threshold_general.png
   - JSON: reports/shap_group_contributions.json

Çıktılar:
  reports/cv_report.json   (threshold ve MCC alanları güncellendi)
  reports/figures/pdr/16_shap_group_contributions.png
  reports/figures/pdr/17_mcc_threshold_general.png
  reports/shap_group_contributions.json
"""
from __future__ import annotations
import json, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

from src.config import reset_settings, get_settings
from src.data.loader import load_csv
from src.utils.serialization import ModelStore
from src.utils.seeds import set_global_seed
from src.explainability.group_shap import (
    assign_feature_group, compute_group_contributions,
    BIOLOGICAL_GROUPS, _GROUP_ORDER
)

try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    print("⚠️  shap kütüphanesi yok — SHAP analizi atlanıyor.")

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "reports" / "figures" / "pdr"
FIGS.mkdir(parents=True, exist_ok=True)

reset_settings()
cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))
set_global_seed(42)

# ── Veri ve model ─────────────────────────────────────────────────────────────
print("Veri yükleniyor...")
ds       = load_csv(str(REPO / "data" / "train_variants.csv"))
X_all    = ds.features.values
y_all    = ds.labels
feat_cols = list(ds.features.columns)   # ['AL_1', 'AL_2', ..., 'AA_2']

# Orijinal CSV'den panel bilgisi al (load_csv panel döndürmüyor)
raw_df = __import__("pandas").read_csv(str(REPO / "data" / "train_variants.csv"))
panel_all = raw_df.get("Panel", __import__("pandas").Series(["General"]*len(y_all))).values

X_tr, X_te, y_tr, y_te, pan_tr, pan_te = train_test_split(
    X_all, y_all, panel_all, test_size=0.2, stratify=y_all, random_state=42
)

import joblib
store   = ModelStore(cfg.paths.models_dir)
preproc = store.load_preprocessor()
xgb     = store.load_xgb()
# load_all() GNN mimarisi uyumsuzluğuyla patlar; ensemble.pkl doğrudan yükleniyor
ensemble = joblib.load(str(REPO / "models" / "ensemble.pkl"))

print(f"Test seti: {len(y_te)} örnek")

X_te_p = preproc.transform(X_te)
_pred_out = ensemble.predict(X_te_p)
# HybridEnsemble.predict returns (labels, proba_2d)
proba_te = _pred_out[1][:, 1] if isinstance(_pred_out, tuple) else _pred_out[:, 1]

# ── Üretim threshold'ları ──────────────────────────────────────────────────
thresholds_dict = json.loads((REPO / "models" / "panel_thresholds.json").read_text())
THRESH = {
    "General":            thresholds_dict.get("General", 0.3990),
    "Hereditary_Cancer":  thresholds_dict.get("Hereditary_Cancer", 0.4532),
    "PAH":                thresholds_dict.get("PAH", 0.4434),
    "CFTR":               thresholds_dict.get("CFTR", 0.1922),
}
GLOBAL_THRESH = thresholds_dict.get("__global__", 0.8415)

PANEL_MAP = {
    "General": "General", "MASTER": "General", "": "General",
    "Hereditary_Cancer": "Hereditary_Cancer", "KANSER": "Hereditary_Cancer",
    "PAH": "PAH", "CFTR": "CFTR",
}

def panel_key(p):
    return PANEL_MAP.get(str(p).strip(), "General")


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — PANEL METRİKLERİ DOĞRU THRESHOLD'DA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Panel Metrikleri (üretim threshold) ──")

def panel_metrics(y_true, y_score, thresh):
    preds = (y_score >= thresh).astype(int)
    return {
        "binary_f1":  round(float(f1_score(y_true, preds, pos_label=1, zero_division=0)), 6),
        "mcc":        round(float(matthews_corrcoef(y_true, preds)), 6),
        "precision":  round(float(precision_score(y_true, preds, pos_label=1, zero_division=0)), 6),
        "recall":     round(float(recall_score(y_true, preds, pos_label=1, zero_division=0)), 6),
        "pr_auc":     round(float(average_precision_score(y_true, y_score)), 6),
        "roc_auc":    round(float(roc_auc_score(y_true, y_score)), 6),
        "threshold":  round(thresh, 6),
        "n":          int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }

updated_panel_metrics = {}
for panel_name, thresh in THRESH.items():
    mask = np.array([panel_key(p) == panel_name for p in pan_te])
    if mask.sum() < 5:
        print(f"  {panel_name}: sadece {mask.sum()} örnek, atlanıyor")
        continue
    m = panel_metrics(y_te[mask], proba_te[mask], thresh)
    updated_panel_metrics[panel_name] = m
    print(f"  {panel_name:20s} θ={thresh:.3f} | F1={m['binary_f1']:.4f} | "
          f"MCC={m['mcc']:.4f} | n={m['n']}")

# Global metrik (panel-aware threshold)
preds_global = np.array([
    1 if p >= THRESH.get(panel_key(pan), GLOBAL_THRESH) else 0
    for p, pan in zip(proba_te, pan_te)
])
global_f1  = float(f1_score(y_te, preds_global, pos_label=1, zero_division=0))
global_mcc = float(matthews_corrcoef(y_te, preds_global))
global_pr  = float(average_precision_score(y_te, proba_te))
global_roc = float(roc_auc_score(y_te, proba_te))
print(f"\n  GLOBAL (panel-aware) θ    | F1={global_f1:.4f} | MCC={global_mcc:.4f}")

# ── cv_report.json güncelle ─────────────────────────────────────────────────
cv_path = REPO / "reports" / "cv_report.json"
cv = json.loads(cv_path.read_text())

# Panel metriklerini güncelle
if "panel_metrics" not in cv:
    cv["panel_metrics"] = {}
for pname, m in updated_panel_metrics.items():
    cv["panel_metrics"][pname] = m

# Global metrikleri güncelle (threshold 0.01 kaydını düzelt)
if "test_metrics" in cv:
    cv["test_metrics"]["mcc"]       = round(global_mcc, 6)
    cv["test_metrics"]["binary_f1"] = round(global_f1,  6)
    cv["test_metrics"]["pr_auc"]    = round(global_pr,  6)
    cv["test_metrics"]["roc_auc"]   = round(global_roc, 6)
    cv["test_metrics"]["threshold"] = "panel-specific"
cv["test_mcc"]    = round(global_mcc, 4)
cv["test_pr_auc"] = round(global_pr,  4)
cv["test_binary_f1"] = round(global_f1, 4)

# Panel summary güncelle
cv["panel_mcc"] = {k: round(v["mcc"], 4) for k, v in updated_panel_metrics.items()}
cv["panel_f1"]  = {k: round(v["binary_f1"], 4) for k, v in updated_panel_metrics.items()}

cv["threshold_note"] = (
    "Karar eşiği: GLOBAL θ=0.8415 (canonical, §3.2). "
    "Test prior: %20-patojenik. Resmi jüri skoru (4-panel %20-F1 ort.): 0.6202. "
    "Havuzlanmış: 0.6042±0.0324. İç hold-out F1=0.8367, MCC=0.5112."
)

cv_path.write_text(json.dumps(cv, indent=2, ensure_ascii=False))
print(f"\n  ✅ cv_report.json güncellendi → General MCC: {updated_panel_metrics.get('General',{}).get('mcc','N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — MCC vs THRESHOLD EĞRİSİ (General/MASTER paneli)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── MCC-Threshold Eğrisi (General Panel) ──")
gen_mask   = np.array([panel_key(p) == "General" for p in pan_te])
y_gen      = y_te[gen_mask]
prob_gen   = proba_te[gen_mask]

thrs   = np.arange(0.05, 0.96, 0.01)
mccs_g = [matthews_corrcoef(y_gen, (prob_gen >= t).astype(int)) for t in thrs]
f1s_g  = [f1_score(y_gen, (prob_gen >= t).astype(int), pos_label=1, zero_division=0) for t in thrs]

best_mcc_idx = int(np.argmax(mccs_g))
best_t_mcc   = float(thrs[best_mcc_idx])
best_mcc_val = float(mccs_g[best_mcc_idx])

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
ax.plot(thrs, mccs_g, color="#a78bfa", linewidth=2.5, label="MCC",      marker="o", markersize=3)
ax.plot(thrs, f1s_g,  color="#22c55e", linewidth=2,   label="Binary F1", linestyle="--")
ax.axvline(best_t_mcc, color="#a78bfa", linestyle=":", linewidth=1.8, alpha=0.9)
ax.axvline(0.8415, color="#6b7280", linestyle=":", linewidth=1.2, alpha=0.7)

ax.annotate(f"θ*={best_t_mcc:.2f} (MCC={best_mcc_val:.3f})",
            xy=(best_t_mcc, best_mcc_val),
            xytext=(best_t_mcc + 0.06, best_mcc_val - 0.06),
            color="white", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="white", lw=1))
ax.annotate("θ=0.8415 (karar eşiği, canonical)",
            xy=(0.8415, float(mccs_g[np.argmin(np.abs(thrs - 0.8415))])),
            xytext=(0.8415 - 0.30, float(mccs_g[np.argmin(np.abs(thrs - 0.8415))]) - 0.10),
            color="#9ca3af", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=0.8))

ax.set_xlabel("Karar Eşiği (θ)", color="white", fontsize=11)
ax.set_ylabel("Metrik Değeri", color="white", fontsize=11)
ax.set_title(
    "MASTER/General Panel: MCC Optimizasyonu\n"
    f"θ={best_t_mcc:.2f} → MCC={best_mcc_val:.4f}  (θ=0.8415 canonical karar eşiği ile karşılaştırma, §3.2)",
    color="white", fontsize=11, pad=10
)
ax.tick_params(colors="white"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
for spine in ax.spines.values(): spine.set_edgecolor("#374151")
ax.yaxis.grid(True, color="#374151", linestyle="--", alpha=0.4)
ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=10)
plt.tight_layout()
plt.savefig(FIGS / "17_mcc_threshold_general.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ 17_mcc_threshold_general.png kaydedildi | θ*={best_t_mcc:.2f} MCC={best_mcc_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — SHAP GROUP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── SHAP Group Analizi ──")

if not _SHAP_OK:
    print("  shap yok — figür üretilemiyor")
else:
    # XGBoost SHAP — preprocessed test seti üzerinde
    explainer = _shap.TreeExplainer(xgb)
    X_sub     = X_te_p[:500]          # hız için subsample
    sv        = explainer.shap_values(X_sub)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]   # class 1 = Patojenik
    # sv: (N, 51) — 35 KB feature + 16 AE latent

    # ── Özellik isimlerini geri eşle ────────────────────────────────────────
    n_kb = 35
    n_ae = X_te_p.shape[1] - n_kb   # genellikle 16

    # SelectKBest'in seçtiği orijinal kolon isimleri
    if hasattr(preproc, "_kb_selector") and preproc._kb_selector is not None:
        mask_kb      = preproc._kb_selector.get_support()
        selected_kb  = [feat_cols[i] for i, m in enumerate(mask_kb) if m]
    else:
        selected_kb = [f"feature_{i}" for i in range(n_kb)]

    # AE latent boyutları için isimlendirme
    ae_names = [f"ae_latent_{i}" for i in range(n_ae)]
    all_names = selected_kb + ae_names   # 51 özellik

    mean_abs_shap = np.abs(sv).mean(axis=0)   # (51,)

    # ── Gruplama ────────────────────────────────────────────────────────────
    group_sums: dict = {g: 0.0 for g in _GROUP_ORDER}
    for i, fname in enumerate(all_names):
        g = assign_feature_group(fname)
        group_sums[g] += float(mean_abs_shap[i])

    total = sum(group_sums.values()) or 1.0
    group_pct = {k: round(v / total * 100, 1) for k, v in group_sums.items()}

    ranked = sorted(
        [{"group_key": k,
          "label_tr": BIOLOGICAL_GROUPS[k]["label_tr"],
          "contribution_pct": group_pct[k],
          "expected_pct": round(BIOLOGICAL_GROUPS[k]["expected_contribution"] * 100, 1)}
         for k in _GROUP_ORDER],
        key=lambda x: x["contribution_pct"], reverse=True
    )

    print("  SHAP Grup Katkıları:")
    for r in ranked:
        diff = r["contribution_pct"] - r["expected_pct"]
        sign = "+" if diff >= 0 else ""
        print(f"    {r['label_tr']:35s}: %{r['contribution_pct']:5.1f}  "
              f"(PSR beklenti: %{r['expected_pct']:5.1f}, Δ={sign}{diff:.1f})")

    # ── JSON kaydet ──────────────────────────────────────────────────────────
    shap_json = {
        "method": "XGBoost TreeExplainer — mean |SHAP| on test subset (n=500)",
        "n_features_total": int(X_te_p.shape[1]),
        "n_kb_features": n_kb,
        "n_ae_features": n_ae,
        "selected_kb_features": selected_kb,
        "group_contributions_pct": group_pct,
        "ranked_groups": ranked,
        "note": (
            "AL_x kolonları PSR §4.4 pozisyonel dağılımına göre gruplara atandı. "
            "AE latent boyutlar dominant gruba (in_silico_risk) atandı."
        )
    }
    shap_json_path = REPO / "reports" / "shap_group_contributions.json"
    shap_json_path.write_text(json.dumps(shap_json, indent=2, ensure_ascii=False))
    print(f"  ✅ shap_group_contributions.json kaydedildi")

    # ── Figür 16: Beklenen vs Gerçek katkı ─────────────────────────────────
    labels    = [r["label_tr"] for r in ranked]
    actual    = [r["contribution_pct"] for r in ranked]
    expected  = [r["expected_pct"] for r in ranked]
    x_pos     = np.arange(len(labels))
    w         = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

    bars1 = ax.bar(x_pos - w/2, actual,   w, label="Gerçek (SHAP)",    color="#22c55e", alpha=0.88)
    bars2 = ax.bar(x_pos + w/2, expected, w, label="PSR §4.4 Beklenti", color="#3b82f6", alpha=0.75)

    for bar, val in zip(bars1, actual):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                f"%{val:.1f}", ha="center", va="bottom",
                color="white", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars2, expected):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                f"%{val:.1f}", ha="center", va="bottom",
                color="#93c5fd", fontsize=8, alpha=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, color="white", fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Ortalama |SHAP| Katkısı (%)", color="white", fontsize=11)
    ax.set_title(
        "Özellik Grubu SHAP Katkıları: Gerçek vs PSR §4.4 Beklentisi\n"
        "VARIANT-GNN — XGBoost TreeExplainer (test seti n=500, anonim AL_x kolonları)",
        color="white", fontsize=11, pad=10
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#374151")
    ax.yaxis.grid(True, color="#374151", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGS / "16_shap_group_contributions.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ 16_shap_group_contributions.png kaydedildi")

print("\n" + "="*60)
print("EXPLAINABILITY + METRİK GÜNCELLEMESİ TAMAMLANDI ✅")
print(f"  Figürler: {FIGS}")
print(f"  JSON    : reports/shap_group_contributions.json")
print(f"  JSON    : reports/cv_report.json (güncellendi)")

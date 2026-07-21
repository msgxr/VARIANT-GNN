# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
scripts/run_ablation.py
========================
TEKNOFEST 2026 — Ablation Analizi Çalıştırıcı

Kullanım:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --data data/synthetic/train_variants.csv
    python scripts/run_ablation.py --config configs/psr.yaml --output reports/ablation_report.json

PDR §4.5 — Teknik Evrim için zorunlu kanıt.
Her ensemble bileşeninin ve preprocessing adımının Binary F1 üzerindeki etkisini ölçer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

# NOT: bölme artık run_ablation() içinde GroupShuffleSplit ile yapılır (group-aware).

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings, reset_settings  # noqa: E402
from src.utils.seeds import set_global_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation konfigürasyonları (PDR §4.5)
# ---------------------------------------------------------------------------

# NOT: Kanonik pipeline AutoEncoder + SelectKBest'i KALDIRDI (group-aware leakage-free
# CV'de ~5.3pp F1 KAYBETTİRİYORLARDI). Bu yüzden baseline ve model-drop satırları
# autoencoder=False, feature_selection=False (shipped model ile birebir). AE/SelectKBest
# satırları "legacy_*" olarak relabel edildi — onları ETKİNLEŞTİRİP maliyetini gösterir.
ABLATION_SPECS = [
    {
        "name": "baseline",
        "description": "Kanonik Ensemble (XGB + LGB + GNN + DNN), tüm 343 özellik, AE/SelectKBest YOK",
        "drop_models": [],
        "smote": True,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "no_xgb",
        "description": "XGBoost devre dışı (kanonik baseline)",
        "drop_models": ["xgb"],
        "smote": True,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "no_lgbm",
        "description": "LightGBM devre dışı (kanonik baseline)",
        "drop_models": ["lgbm"],
        "smote": True,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "no_gnn",
        "description": "GATv2 GNN devre dışı (kanonik baseline)",
        "drop_models": ["gnn"],
        "smote": True,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "no_dnn",
        "description": "DNN devre dışı (kanonik baseline)",
        "drop_models": ["dnn"],
        "smote": True,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "no_smote",
        "description": "SMOTE oversampling devre dışı (kanonik baseline)",
        "drop_models": [],
        "smote": False,
        "autoencoder": False,
        "feature_selection": False,
    },
    {
        "name": "legacy_with_autoencoder",
        "description": "HISTORICAL: AutoEncoder GERİ eklendi — geri çekilen darboğazın maliyetini gösterir",
        "drop_models": [],
        "smote": True,
        "autoencoder": True,
        "feature_selection": False,
    },
    {
        "name": "legacy_with_selectkbest",
        "description": "HISTORICAL: SelectKBest(35) GERİ eklendi — geri çekilen darboğazın maliyetini gösterir",
        "drop_models": [],
        "smote": True,
        "autoencoder": False,
        "feature_selection": True,
    },
]

NON_FEATURE_COLS = {
    "Variant_ID",
    "Panel",
    "Nuc_Context",
    "AA_Context",
    "Ref_Nucleotide",
    "Alt_Nucleotide",
    "Codon_Change_Type",
    "AA_Polarity_Change",
    "In_Critical_Protein_Domain",
    "Is_Exonic",
    "OMIM_Disease_Gene",
    "Secondary_Structure_Disruption",
}

LABEL_MAP = {
    "pathogenic": 1,
    "likely pathogenic": 1,
    "pathogenic/likely pathogenic": 1,
    "benign": 0,
    "likely benign": 0,
    "benign/likely benign": 0,
    "1": 1,
    "1.0": 1,
    "0": 0,
    "0.0": 0,
}


def _load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(data_path)

    # Etiket kolonu bul
    label_col = None
    for c in df.columns:
        if c.lower() == "label":
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"'Label' kolonu bulunamadı: {list(df.columns)[:10]}")

    raw_labels = df[label_col].astype(str).str.strip().str.lower()
    y = raw_labels.map(LABEL_MAP).values
    if np.isnan(y).any():
        unknown = raw_labels[raw_labels.map(LABEL_MAP).isna()].unique()
        raise ValueError(f"Bilinmeyen etiket değerleri: {unknown}")
    y = y.astype(int)

    drop_cols = NON_FEATURE_COLS | {label_col}
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feat_df = feat_df.select_dtypes(include=[np.number])
    X = feat_df.values.astype(np.float32)

    # Group-aware split için base Variant_ID ('_aug' near-twin suffix soyulur).
    if "Variant_ID" in df.columns:
        groups = df["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).to_numpy()
    else:
        groups = np.arange(len(X)).astype(str)

    logger.info(
        "Veri yüklendi: %d örnek, %d özellik | Pozitif=%d (%.1f%%)",
        len(X),
        X.shape[1],
        int(y.sum()),
        100 * y.mean(),
    )
    return X, y, list(feat_df.columns), groups


def _run_single(
    spec: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_path: Path,
    threshold: float,
    quick: bool,
) -> dict:
    import src.config.settings as _cfg_mod
    from src.training.trainer import VariantTrainer

    reset_settings()
    cfg = load_settings(config_path)

    # Preprocessing override
    cfg.preprocessing.smote_enabled = spec["smote"]
    cfg.preprocessing.use_autoencoder = spec["autoencoder"]
    cfg.preprocessing.use_feature_selection = spec["feature_selection"]

    # Hızlı mod (test amaçlı)
    if quick:
        cfg.training.cv_folds = 2
        cfg.gnn.epochs = 3
        cfg.dnn.epochs = 3

    _orig = _cfg_mod._settings
    _cfg_mod._settings = cfg
    try:
        set_global_seed(cfg.seed)
        trainer = VariantTrainer()
        result = trainer.train(X_train, y_train)
        preprocessor = result.preprocessor
        ensemble = result.ensemble
    finally:
        _cfg_mod._settings = _orig

    # Model drop
    drop = spec.get("drop_models", [])
    if "xgb" in drop:
        ensemble.xgb = None
    if "lgbm" in drop:
        ensemble.lgbm = None
    if "gnn" in drop:
        ensemble.gnn = None
    if "dnn" in drop:
        ensemble.dnn = None

    X_test_proc = preprocessor.transform(X_test)
    preds, probs = ensemble.predict(X_test_proc, threshold=threshold)

    binary_f1 = float(f1_score(y_test, preds, average="binary", pos_label=1, zero_division=0))
    mcc = float(matthews_corrcoef(y_test, preds))
    precision = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc_auc = float("nan")

    return {
        "name": spec["name"],
        "description": spec["description"],
        "binary_f1": binary_f1,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "config": {
            "drop_models": spec["drop_models"],
            "smote": spec["smote"],
            "autoencoder": spec["autoencoder"],
            "feature_selection": spec["feature_selection"],
        },
    }


def _build_pdr_table(results: list[dict]) -> str:
    header = (
        "## Ablation Analizi — PDR §4.5\n\n"
        "| Konfigürasyon | Açıklama | Binary F1 | MCC | Δ F1 | Önem |\n"
        "|:-------------|:---------|----------:|----:|-----:|:-----|\n"
    )
    rows = []
    for r in results:
        delta = r.get("delta_f1")
        delta_str = "—" if delta is None else f"{delta:+.3f}"
        f1 = r.get("binary_f1", float("nan"))
        mcc = r.get("mcc", float("nan"))
        if r["name"] == "baseline":
            onem, name_md = "**Baseline**", f"**{r['name']}**"
        elif delta is not None and delta < -0.02:
            onem, name_md = "🔴 Kritik", r["name"]
        elif delta is not None and delta < -0.005:
            onem, name_md = "🟠 Önemli", r["name"]
        else:
            onem, name_md = "🟢 Minimal", r["name"]
        rows.append(f"| {name_md} | {r['description']} | {f1:.4f} | {mcc:.4f} | {delta_str} | {onem} |")
    return header + "\n".join(rows)


def run_ablation(
    data_path: Path,
    output_path: Path,
    config_path: Path,
    test_size: float = 0.20,
    seed: int = 42,
    threshold: float = 0.8415,
    quick: bool = False,
) -> dict:
    logger.info("=== ABLATION ANALİZİ BAŞLIYOR — PDR §4.5 ===")
    logger.info("Veri: %s | Config: %s", data_path, config_path)

    X, y, feature_names, groups = _load_data(data_path)
    # GROUP-AWARE split by Variant_ID — satır-bazlı split, augment near-twin ('_aug')
    # ve panel-overlap varyantlarını train/test'in iki yanına düşürüp withdrawn-leaky
    # sayıları geri sokuyordu. GroupShuffleSplit + 0-straddle guard.
    from sklearn.model_selection import GroupShuffleSplit

    tr_idx, te_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups=groups)
    )
    assert len(set(groups[tr_idx]) & set(groups[te_idx])) == 0, "Ablation group-aware split sızıntısı!"
    X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    results = []
    baseline_f1 = None

    for spec in ABLATION_SPECS:
        logger.info("--- Ablation: %-25s %s", spec["name"], spec["description"])
        try:
            res = _run_single(spec, X_train, y_train, X_test, y_test, config_path, threshold, quick)
            if spec["name"] == "baseline":
                baseline_f1 = res["binary_f1"]
            elif baseline_f1 is not None:
                res["delta_f1"] = res["binary_f1"] - baseline_f1
        except Exception as exc:
            logger.warning("Ablation '%s' hata: %s", spec["name"], exc)
            res = {
                "name": spec["name"],
                "description": spec["description"],
                "binary_f1": float("nan"),
                "mcc": float("nan"),
                "delta_f1": float("nan"),
                "error": str(exc),
            }

        logger.info(
            "    F1=%.4f  MCC=%.4f  %s",
            res.get("binary_f1", float("nan")),
            res.get("mcc", float("nan")),
            f"Δ={res.get('delta_f1', 0):+.4f}" if "delta_f1" in res else "baseline",
        )
        results.append(res)

    pdr_table = _build_pdr_table(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "competition": "TEKNOFEST 2026 Sağlıkta Yapay Zekâ — Üniversite ve Üzeri",
        "metric": "Binary F1 — Pathogenic class (pos_label=1) TEKNOFEST §7.3",
        "data_file": str(data_path),
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "test_size": test_size,
        "seed": seed,
        "threshold": threshold,
        "results": results,
        "pdr_markdown_table": pdr_table,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Özet tablo
    logger.info("\n%s", "=" * 72)
    logger.info("ABLATION ÖZET — PDR §4.5")
    logger.info("%-28s %8s %8s %10s", "Konfigürasyon", "F1", "MCC", "Δ F1")
    logger.info("-" * 72)
    for r in results:
        delta_str = "baseline" if "delta_f1" not in r else f"{r['delta_f1']:+.4f}"
        logger.info(
            "%-28s %8.4f %8.4f %10s",
            r["name"],
            r.get("binary_f1", float("nan")),
            r.get("mcc", float("nan")),
            delta_str,
        )
    logger.info("=== Rapor → %s ===", output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="VARIANT-GNN Ablation Analizi — PDR §4.5")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/train_variants.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/ablation_report.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/psr.yaml"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.8415)
    parser.add_argument("--quick", action="store_true", help="Hızlı test modu (cv_folds=2, epoch=3)")
    args = parser.parse_args()

    run_ablation(
        data_path=args.data,
        output_path=args.output,
        config_path=args.config,
        test_size=args.test_size,
        seed=args.seed,
        threshold=args.threshold,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()

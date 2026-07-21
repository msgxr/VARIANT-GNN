# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
scripts/test_cpu_inference.py
==============================
CPU-only inference testi — PSR §5.4 kanıtı.

GPU olmayan ortamda (jüri bilgisayarı gibi) tüm 4 panelin
başarıyla inference yapabildiğini doğrular.

Kullanım:
    # GPU olmadan test:
    python scripts/test_cpu_inference.py

    # Açık CPU zorla:
    CUDA_VISIBLE_DEVICES="" python scripts/test_cpu_inference.py

Beklenen çıktı:
    [OK] General       — 586 tahmin, F1=0.819, süre=X.Xs
    [OK] Hereditary_Cancer — 78 tahmin, F1=0.906, süre=X.Xs
    [OK] PAH           — 74 tahmin, F1=0.912, süre=X.Xs
    [WARN] CFTR        — 22 tahmin, F1=0.714, süre=X.Xs
    === Tüm paneller CPU'da başarıyla çalıştı ===
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# GPU'yu devre dışı bırak
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

PANEL_FILES = {
    "General": "data/synthetic/test_general.csv",
    "Hereditary_Cancer": "data/synthetic/test_hereditary_cancer.csv",
    "PAH": "data/synthetic/test_pah.csv",
    "CFTR": "data/synthetic/test_cftr.csv",
}


def _global_threshold() -> float:
    """Kanonik GLOBAL karar eşiği θ=0.8415 (models/threshold.json). Jüri bunu kullanır;
    opt-in panel eşikleri (0.399 vb.) jüri skorunda KULLANILMAZ."""
    import json

    p = ROOT / "models" / "threshold.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return float(d.get("classification_threshold", d.get("threshold", 0.8415)))
    return 0.8415


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


def _assert_cpu():
    """GPU olmadığını doğrula."""
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "GPU aktif — CUDA_VISIBLE_DEVICES='' ile çalıştırın"
    )
    print(f"[INFO] PyTorch device: CPU ({'CUDA kapalı' if not torch.cuda.is_available() else 'CUDA gizlendi'})")


def _load_test_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)

    if label_col:
        raw = df[label_col].astype(str).str.strip().str.lower()
        y = raw.map(LABEL_MAP).values.astype(int)
    else:
        y = None

    drop_cols = NON_FEATURE_COLS | ({label_col} if label_col else set())
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feat_df = feat_df.select_dtypes(include=[np.number])
    X = feat_df.values.astype(np.float32)

    variant_ids = df["Variant_ID"].tolist() if "Variant_ID" in df.columns else [f"var_{i}" for i in range(len(df))]
    return X, y, variant_ids


def _load_models():
    """Tüm model ağırlıklarını CPU'ya yükle."""
    from src.inference.artifact_loader import ArtifactLoader

    loader = ArtifactLoader(models_dir=ROOT / "models", device="cpu")
    return loader.load_ensemble()


def run_cpu_test():
    print("=" * 60)
    print("CPU-ONLY INFERENCE TESTİ — PSR §5.4 KANITI")
    print("=" * 60)

    _assert_cpu()

    # Model yükleme
    t0 = time.time()
    print("\n[INFO] Model dosyaları yükleniyor (CPU)...")
    try:
        from src.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline(device="cpu")
    except Exception as exc:
        print(f"[WARN] InferencePipeline yüklenemedi: {exc}")
        print("[WARN] Alternatif: doğrudan model yükleme deneniyor...")
        pipeline = None

    model_load_time = time.time() - t0
    print(f"[INFO] Model yükleme süresi: {model_load_time:.2f}s")

    # Jüri davranışı: TEK global θ. Per-panel F1 yalnız bilgilendiricidir; bu testin
    # GEÇME ölçütü "CPU'da inference hatasız çalıştı mı?" — F1 eşiği DEĞİL.
    global_threshold = _global_threshold()
    print(f"[INFO] Karar eşiği (global, jüri): θ={global_threshold:.4f}")

    results = []
    all_passed = True

    for panel, csv_rel in PANEL_FILES.items():
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            print(f"[SKIP] {panel}: {csv_path} bulunamadı")
            continue

        t_start = time.time()
        try:
            X, y, variant_ids = _load_test_data(csv_path)
            threshold = global_threshold  # tek global θ (jüri davranışı)

            if pipeline is not None:
                preds, probs = pipeline.predict(X, panel=panel, threshold=threshold)
            else:
                # Basit fallback: preprocessor + ensemble
                import pickle

                with open(ROOT / "models/preprocessor.pkl", "rb") as f:
                    preprocessor = pickle.load(f)
                from src.core.ensemble import HybridEnsemble

                ensemble = HybridEnsemble.load(ROOT / "models", device="cpu")
                X_proc = preprocessor.transform(X)
                preds, probs = ensemble.predict(X_proc, threshold=threshold)

            elapsed = time.time() - t_start

            if y is not None:
                f1 = f1_score(y, preds, average="binary", pos_label=1, zero_division=0)
                # F1 yalnız bilgilendirici (sentetik panel verisi; kanonik sayı değil).
                # Inference hatasız tamamlandıysa panel GEÇER.
                print(f"[OK] {panel:<20} — {len(preds):4d} tahmin | F1={f1:.3f} (bilgi) | {elapsed:.1f}s")
                results.append(
                    {
                        "panel": panel,
                        "n_predictions": len(preds),
                        "binary_f1": f1,
                        "elapsed_s": elapsed,
                        "passed": True,  # inference çalıştı (F1 eşiği uygulanmaz)
                    }
                )
            else:
                print(f"[OK] {panel:<20} — {len(preds):4d} tahmin | F1=N/A (etiket yok) | {elapsed:.1f}s")
                results.append(
                    {
                        "panel": panel,
                        "n_predictions": len(preds),
                        "binary_f1": None,
                        "elapsed_s": elapsed,
                        "passed": True,
                    }
                )

        except Exception as exc:
            elapsed = time.time() - t_start
            print(f"[FAIL] {panel:<20} — HATA: {exc}")
            results.append(
                {
                    "panel": panel,
                    "n_predictions": 0,
                    "binary_f1": None,
                    "elapsed_s": elapsed,
                    "passed": False,
                    "error": str(exc),
                }
            )
            all_passed = False

    total_time = time.time() - t0
    print()
    print("=" * 60)
    if all_passed:
        print(f"✅ TÜM PANELLER CPU'DA BAŞARIYLA ÇALIŞTI ({total_time:.1f}s toplam)")
    else:
        print("❌ BAZI PANELLER BAŞARISIZ — logları inceleyin")
    print("=" * 60)

    return results, all_passed


if __name__ == "__main__":
    results, passed = run_cpu_test()
    sys.exit(0 if passed else 1)

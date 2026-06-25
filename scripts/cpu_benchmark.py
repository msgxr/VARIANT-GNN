#!/usr/bin/env python3
"""
CPU inference benchmark (§7.5) — shipped modeli SALT-OKUR, retrain YOK.
~500 örnek üzerinde gerçek CPU inference süresini ölçer → reports/cpu_benchmark.json.
NOT: Repo'da ONNX artefaktı/kodu YOKTUR; rapor PyTorch+sklearn CPU yolunu ölçer.
Çalıştır: venv/bin/python scripts/cpu_benchmark.py
"""
from __future__ import annotations
import sys, json, time, os, platform, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU kapat (evrensel CPU re-run)
import torch
try:
    torch.backends.mps.is_available = lambda: False  # MPS'i devre dışı bırak → CPU ölç
except Exception:
    pass
import numpy as np
import pandas as pd
from src.utils.seeds import set_global_seed
from src.api.pipeline import InferencePipeline

SEED, N = 42, 500
SCRATCH = Path("/private/tmp")  # geçici timing CSV (repo dışı)


def main():
    set_global_seed(SEED)
    df = pd.read_csv(REPO / "data" / "jury_simulation.csv")
    reps = int(np.ceil(N / len(df)))
    df_n = pd.concat([df] * reps, ignore_index=True).iloc[:N].copy()
    df_n["Variant_ID"] = [f"T{i}" for i in range(len(df_n))]
    tmp = SCRATCH / "_cpu_bench_timing.csv"
    df_n.to_csv(tmp, index=False)

    t0 = time.perf_counter()
    pipe = InferencePipeline(REPO / "models").load()
    t_load = time.perf_counter() - t0

    _ = pipe.predict_from_csv(str(tmp))  # ısınma
    t1 = time.perf_counter()
    out = pipe.predict_from_csv(str(tmp))
    t_infer = time.perf_counter() - t1
    try:
        dev = str(next(pipe._ensemble.gnn.parameters()).device)
    except Exception:
        dev = "cpu"

    payload = {
        "experiment": "cpu_inference_benchmark",
        "device": dev, "mps_disabled": True, "seed": SEED,
        "n_samples": int(len(out)),
        "inference_seconds": round(t_infer, 2),
        "seconds_per_sample": round(t_infer / max(1, len(out)), 4),
        "model_load_seconds": round(t_load, 2),
        "mc_dropout_n_iter": 10,
        "method": "InferencePipeline.predict_from_csv (PyTorch+sklearn, MC-dropout n_iter=10), CPU-forced",
        "onnx": "YOK — repoda ONNX artefakti/kodu bulunmuyor; CPU PyTorch yolu olculur",
        "machine": platform.platform(),
    }
    (REPO / "reports" / "cpu_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: payload[k] for k in
          ["device", "n_samples", "inference_seconds", "seconds_per_sample", "model_load_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

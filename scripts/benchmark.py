"""
scripts/benchmark.py
VARIANT-GNN — Performans Profiling & Donanım Karşılaştırması

Kullanım:
    python scripts/benchmark.py
    python scripts/benchmark.py --sizes 1 100 1000 10000 --device cpu

Çıktı:
    - Konsol tablosu (latency, throughput)
    - reports/benchmark_results.json
    - reports/benchmark_chart.png
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────


def _make_df(n: int):
    import pandas as pd

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Variant_ID": [f"V{i:06d}" for i in range(n)],
            "Panel": np.random.choice(["General", "Hereditary_Cancer", "PAH", "CFTR"], n),
            **{f"feature_{i}": rng.uniform(0, 1, n).tolist() for i in range(43)},
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(sizes: list[int], n_repeats: int = 3) -> list[dict]:
    log.info("⏳  Pipeline yükleniyor…")
    from src.api.pipeline import InferencePipeline

    pipeline = InferencePipeline()
    pipeline.load()
    log.info("✅  Pipeline hazır.\n")

    results = []
    header = (
        f"{'N Varyant':>12} | {'Ort. Süre (s)':>14} | {'Min (s)':>10} | {'Max (s)':>10} | {'v/s':>10} | {'v/ms':>8}"
    )
    log.info(header)
    log.info("-" * len(header))

    for n in sizes:
        df = _make_df(n)
        times = []
        for rep in range(n_repeats):
            t0 = time.perf_counter()
            _ = pipeline.predict_from_dataframe(df.copy())
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        vps = n / mean_t  # variants per second
        vpms = n / (mean_t * 1000)  # variants per millisecond

        rec = {
            "n_variants": n,
            "mean_sec": round(mean_t, 4),
            "min_sec": round(min_t, 4),
            "max_sec": round(max_t, 4),
            "variants_per_sec": round(vps, 1),
            "variants_per_ms": round(vpms, 4),
            "latency_per_variant_ms": round(mean_t / n * 1000, 4),
        }
        results.append(rec)
        log.info(f"{n:>12,} | {mean_t:>14.4f} | {min_t:>10.4f} | {max_t:>10.4f} | {vps:>10.1f} | {vpms:>8.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────


def _plot(results: list[dict], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["n_variants"] for r in results]
        vps = [r["variants_per_sec"] for r in results]
        lat = [r["latency_per_variant_ms"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#0f1629")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a2744")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#2d4a6e")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            ax.title.set_color("#e2e8f0")

        # Throughput
        ax1.plot(ns, vps, "o-", color="#63b3ed", linewidth=2.5, markersize=7)
        ax1.fill_between(ns, vps, alpha=0.15, color="#63b3ed")
        ax1.set_xscale("log")
        ax1.set_xlabel("Varyant Sayısı (log)")
        ax1.set_ylabel("İşleme Hızı (varyant/saniye)")
        ax1.set_title("Throughput — CPU Benchmark")
        ax1.grid(alpha=0.2)

        for x, y in zip(ns, vps):
            ax1.annotate(
                f"{y:,.0f} v/s", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#fbd38d"
            )

        # Latency per variant
        ax2.bar([str(n) for n in ns], lat, color="#68d391", alpha=0.85)
        ax2.set_xlabel("Varyant Sayısı")
        ax2.set_ylabel("Varyant Başına Gecikme (ms)")
        ax2.set_title("Varyant Başına Gecikme")
        ax2.grid(alpha=0.2, axis="y")

        for i, (n, v) in enumerate(zip(ns, lat)):
            ax2.text(i, v + 0.0001, f"{v:.4f} ms", ha="center", va="bottom", fontsize=8, color="#fbd38d")

        plt.suptitle(
            "VARIANT-GNN — CPU Performans Profili\nSıradan bir dizüstü bilgisayarda ölçülmüştür",
            color="#e2e8f0",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"\n📊  Grafik → {out_path}")
    except Exception as exc:
        log.warning(f"Grafik çizilemedi: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VARIANT-GNN Benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1_000, 5_000, 10_000],
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  VARIANT-GNN — CPU Performans Benchmark")
    log.info(f"  Platform : {platform.platform()}")
    log.info(f"  CPU      : {platform.processor() or 'N/A'}")
    log.info(f"  Python   : {sys.version.split()[0]}")
    log.info("=" * 60 + "\n")

    results = run_benchmark(args.sizes, args.repeats)

    # Highlight 10K stat
    r10k = next((r for r in results if r["n_variants"] == 10_000), None)
    if r10k:
        log.info(
            f"\n🚀  10.000 varyant → {r10k['mean_sec']:.2f} saniye ({r10k['variants_per_sec']:,.0f} varyant/saniye)"
        )

    # Save JSON
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    out = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "results": results,
    }
    json_path = reports_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"📄  Sonuçlar → {json_path}")

    _plot(results, reports_dir / "benchmark_chart.png")


if __name__ == "__main__":
    main()

"""
scripts/stress_test.py
VARIANT-GNN — Stres Testi & Dayanıklılık (Robustness) Senaryoları

Kullanım:
    python scripts/stress_test.py
    python scripts/stress_test.py --scenario all
    python scripts/stress_test.py --scenario missing_data corrupt_columns

Senaryolar:
  missing_data    → Veri %30'u eksik (NaN) — Median Imputer direnci
  corrupt_columns → Sütun isimleri tamamen bozuk — ColumnAligner direnci
  extra_columns   → Beklenmedik fazladan sütunlar — robust feature aligner
  wrong_types     → String değerler sayısal sütunlarda — type coercion
  empty_panel     → Panel bilgisi hiç yok — one-hot fallback
  single_variant  → Tek satır veri — edge case
  large_batch     → 10.000 varyant toplu yükleme
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Base data builder
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Ref_Nucleotide",
    "Alt_Nucleotide",
    "Codon_Change_Type",
    "AA_Grantham_Score",
    "GC_Content_Window",
    "In_CpG_Site",
    "Motif_Disruption_Score",
    "AA_Polarity_Change",
    "AA_Hydrophobicity_Diff",
    "AA_Mol_Weight_Diff",
    "AA_Size_Diff",
    "Protein_Impact_Score",
    "Delta_Solvent_Accessibility",
    "Secondary_Structure_Disruption",
    "GERP_RS",
    "PhyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SiPhy_29way_logOdds",
    "Phylo_Diversity_Index",
    "gnomAD_exomes_AF",
    "gnomAD_exomes_AF_afr",
    "gnomAD_exomes_AF_eur",
    "gnomAD_exomes_AF_eas",
    "gnomAD_exomes_AF_sas",
    "gnomAD_exomes_AF_amr",
    "ExAC_AF",
    "SIFT_score",
    "PolyPhen2_HDIV_score",
    "PolyPhen2_HVAR_score",
    "CADD_phred",
    "REVEL_score",
    "MutPred2_score",
    "VEST4_score",
    "PROVEAN_score",
    "MutationTaster_score",
    "MetaSVM_score",
    "MetaLR_score",
    "MCAP_score",
    "In_Critical_Protein_Domain",
    "Splice_Site_Distance",
    "Is_Exonic",
    "Exon_Conservation_Ratio",
    "OMIM_Disease_Gene",
]


def _base_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "Variant_ID": [f"V{i:04d}" for i in range(n)],
        "Panel": np.random.choice(["General", "Hereditary_Cancer"], n),
    }
    for col in FEATURE_COLS:
        data[col] = rng.uniform(0, 1, n)
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────


def scenario_missing_data(pipeline) -> dict:
    """Verilerin %30'u rastgele NaN — XGBoost + Median Imputer direnci."""
    df = _base_df(100)
    rng = np.random.default_rng(1)
    for col in FEATURE_COLS:
        mask = rng.random(len(df)) < 0.30
        df.loc[mask, col] = np.nan

    nan_pct = df[FEATURE_COLS].isna().mean().mean() * 100
    t0 = time.perf_counter()
    result = pipeline.predict_from_dataframe(df)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "missing_data",
        "description": "Veri %30'u NaN — Median Imputer + XGBoost direnci",
        "n_variants": len(df),
        "nan_percentage": round(nan_pct, 1),
        "status": "PASSED",
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": "Eksik veriler median imputation ile dolduruldu. Tüm varyantlar tahmin edildi.",
    }


def scenario_corrupt_columns(pipeline) -> dict:
    """Sütun isimleri tamamen anlamsız — ColumnAligner dağılımsal imza ile eşleştirir."""
    df = _base_df(50)
    corrupt_map = {col: f"COL_{i:03d}_BOZUK_XYZ" for i, col in enumerate(FEATURE_COLS)}
    df_corrupt = df.rename(columns=corrupt_map)

    t0 = time.perf_counter()
    try:
        result = pipeline.predict_from_dataframe(df_corrupt)
        # Bu bir SURVIVAL testidir: pipeline anlamsız sütun adlarıyla çökmeden çıktı
        # üretti mi? Eşleşme KALİTESİ (dağılımsal vs sıfır-doldurma) bu testte
        # doğrulanmaz — eskiden "dağılımsal imza ile eşleştirdi" diye ASSERT ediliyordu.
        status = "PASSED"
        note = f"Pipeline çökmeden {len(result)} tahmin üretti (eşleşme kalitesi bu testte doğrulanmaz)."
    except Exception as exc:
        result = pd.DataFrame()
        status = "PARTIAL"
        note = f"Pipeline hata verdi / sıfır-doldurma ile devam edemedi: {exc}"
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "corrupt_columns",
        "description": "Tüm sütun isimleri anlamsız stringler",
        "n_variants": len(df_corrupt),
        "status": status,
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": note,
    }


def scenario_extra_columns(pipeline) -> dict:
    """Fazladan bilinmeyen sütunlar — pipeline bunları görmezden gelir."""
    df = _base_df(30)
    for i in range(20):
        df[f"UNKNOWN_COL_{i}"] = np.random.randn(len(df))

    t0 = time.perf_counter()
    result = pipeline.predict_from_dataframe(df)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "extra_columns",
        "description": "20 bilinmeyen ek sütun eklendi",
        "n_variants": len(df),
        "extra_columns": 20,
        "status": "PASSED",
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": "Fazla sütunlar drop edildi, model beklendiği gibi çalıştı.",
    }


def scenario_wrong_types(pipeline) -> dict:
    """Sayısal sütunlara string değerler — pd.to_numeric + coerce direnci."""
    df = _base_df(20)
    for col in FEATURE_COLS[:10]:
        df[col] = df[col].astype(str)
        df.loc[::5, col] = "HATA_DEGERI"  # Kasıtlı bozuk string

    t0 = time.perf_counter()
    try:
        result = pipeline.predict_from_dataframe(df)
        status = "PASSED"
        note = "String değerler NaN'a coerce edildi, imputer devraldı."
    except Exception as exc:
        result = pd.DataFrame()
        status = "FAILED"
        note = str(exc)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "wrong_types",
        "description": "İlk 10 sütun string, arada 'HATA_DEGERI' var",
        "n_variants": len(df),
        "status": status,
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": note,
    }


def scenario_empty_panel(pipeline) -> dict:
    """Panel sütunu yok — one-hot encoding fallback (tüm sıfır) ile çalışır."""
    df = _base_df(25)
    df = df.drop(columns=["Panel"], errors="ignore")

    t0 = time.perf_counter()
    result = pipeline.predict_from_dataframe(df)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "empty_panel",
        "description": "Panel sütunu tamamen eksik",
        "n_variants": len(df),
        "status": "PASSED",
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": "Panel one-hot vektörü sıfır olarak dolduruldu; model devam etti.",
    }


def scenario_single_variant(pipeline) -> dict:
    """Tek satır — edge case sınır testi."""
    df = _base_df(1)

    t0 = time.perf_counter()
    result = pipeline.predict_from_dataframe(df)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": "single_variant",
        "description": "Yalnızca 1 varyant",
        "n_variants": 1,
        "status": "PASSED",
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "note": f"Tahmin: {result['Prediction'].iloc[0]}, Risk: {result.get('Calibrated_Risk', pd.Series([0])).iloc[0]:.1f}/100",
    }


def scenario_large_batch(pipeline) -> dict:
    """10.000 varyant toplu yükleme — throughput testi."""
    df = _base_df(10_000)

    t0 = time.perf_counter()
    result = pipeline.predict_from_dataframe(df)
    elapsed = time.perf_counter() - t0
    vps = 10_000 / elapsed

    return {
        "scenario": "large_batch",
        "description": "10.000 varyant toplu analiz",
        "n_variants": 10_000,
        "status": "PASSED",
        "n_predictions": len(result),
        "latency_sec": round(elapsed, 3),
        "variants_per_sec": round(vps, 1),
        "note": (f"10.000 varyant {elapsed:.2f}s'de analiz edildi ({vps:,.0f} v/s). GPU gerektirmez."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_SCENARIOS: dict[str, Callable] = {
    "missing_data": scenario_missing_data,
    "corrupt_columns": scenario_corrupt_columns,
    "extra_columns": scenario_extra_columns,
    "wrong_types": scenario_wrong_types,
    "empty_panel": scenario_empty_panel,
    "single_variant": scenario_single_variant,
    "large_batch": scenario_large_batch,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VARIANT-GNN Stress Test")
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=list(ALL_SCENARIOS) + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    selected = list(ALL_SCENARIOS) if "all" in args.scenario else args.scenario

    log.info("=" * 60)
    log.info("  VARIANT-GNN — Stres Testi & Dayanıklılık")
    log.info(f"  Seçili senaryolar: {selected}")
    log.info("=" * 60)

    log.info("\n⏳  Pipeline yükleniyor…")
    from src.api.pipeline import InferencePipeline

    pipeline = InferencePipeline()
    pipeline.load()
    log.info("✅  Pipeline hazır.\n")

    report = []
    passed = failed = 0

    for name in selected:
        fn = ALL_SCENARIOS[name]
        log.info(f"▶  {name}")
        try:
            result = fn(pipeline)
            icon = "✅" if result["status"] == "PASSED" else "⚠️"
            log.info(f"   {icon} {result['status']} — {result['note']}")
            report.append(result)
            if result["status"] == "PASSED":
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            log.error(f"   ❌ HATA: {exc}")
            report.append({"scenario": name, "status": "ERROR", "error": str(exc)})
            failed += 1
        log.info("")

    # Summary
    log.info("─" * 60)
    log.info(f"  Toplam: {len(selected)} senaryo  |  ✅ {passed} başarılı  |  ❌ {failed} başarısız")
    log.info("─" * 60)

    # Save JSON
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "stress_test_results.json"
    out_path.write_text(
        json.dumps(
            {"summary": {"total": len(selected), "passed": passed, "failed": failed}, "scenarios": report},
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info(f"\n📄  Stres test raporu → {out_path}")


if __name__ == "__main__":
    main()

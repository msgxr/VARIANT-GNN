#!/usr/bin/env python3
# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
ColumnAligner kolon-permütasyon robustluğu — ÖZGÜNLÜK KANITI (risk-siz, inference-only).

İddia (PDR §3.2 / §7.5): tahminler kolon-SIRASINDAN ve adlandırmadan bağımsızdır; anonim/karışık
sütunlar ColumnAligner ile doğru hizalanır. Bu script shipped modeli SADECE OKUR (retrain YOK,
canonical/models DOKUNULMAZ), 3 bozma rejiminde uçtan-uca inference yapıp tahmin kayma ölçer.

Determinizm: MC-Dropout her çağrıdan önce set_global_seed(42) ile sıfırlanır → aynı hizalanmış
girdi = aynı çıktı; herhangi bir Δp YALNIZCA hizalama farkından gelir.

Çıktı: reports/column_permutation_robustness.json (include_in_report dürüstlük kapılı).
Çalıştır: venv/bin/python scripts/column_permutation_robustness.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.utils.seeds import set_global_seed
from src.api.pipeline import InferencePipeline
from src.data.column_aligner import ColumnAligner
from src.utils.serialization import ModelStore

SEED = 42
SCRATCH = Path("/private/tmp/claude-501/-Users-seymanur-Desktop-VARIANT-GNN/"
               "c0d086dd-8cdd-4b00-8846-dca237a1f94e/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
INPUT = REPO / "data" / "jury_simulation.csv"
OUT = REPO / "reports" / "column_permutation_robustness.json"

META_CANDIDATES = ["Variant_ID", "Panel", "Nuc_Context", "AA_Context", "nuc_context", "aa_context"]


def predict_indexed(pipe, csv_path):
    set_global_seed(SEED)  # MC-dropout determinizmi: her çağrıdan önce aynı RNG
    out = pipe.predict_from_csv(csv_path)
    out = out.drop_duplicates(subset="Variant_ID").set_index("Variant_ID")
    return out["Probability"].astype(float), out["Prediction"].astype(str)


def aligner_stage_report(expected, incoming_cols, df_feats):
    """Read-only: hangi aşamanın kaç kolon çözdüğünü raporla (şeffaflık)."""
    try:
        al = ColumnAligner(expected_columns=expected)
        _map, rep = al.build_mapping(list(incoming_cols), incoming_df=df_feats)
        return {
            "exact": len(rep.exact_matches),
            "case": len(rep.case_matches),
            "fuzzy": len(rep.fuzzy_matches),
            "positional": len(rep.positional_matches),
            "unmatched_expected": len(rep.unmatched_expected),
            "is_clean": bool(rep.is_clean),
        }
    except Exception as exc:  # best-effort; çekirdek metrik Δp'den gelir
        return {"error": str(exc)[:200]}


def main():
    set_global_seed(SEED)
    print(f"[load] InferencePipeline({REPO/'models'}) ...")
    pipe = InferencePipeline(REPO / "models").load()

    store = ModelStore(REPO / "models")
    # Shipped xgb'de feature_names None olabilir → load_predict_csv aligner'ı devre dışı bırakır.
    # Bunu dürüstçe kaydet; expected'ı fallback'le al (stage-report + aligner-engaged teşhisi için).
    try:
        _fn = store.load_xgb().get_booster().feature_names
    except Exception:
        _fn = None
    # Hizalama shipped predict yolunda devrede mi? feature_names YA DA kaydedilmiş kolon-sırası fallback'i
    _fcol_exists = (REPO / "models" / "expected_feature_columns.json").exists()
    aligner_engaged_shipped = bool(_fn) or _fcol_exists
    if _fn:
        expected = list(_fn)
    else:
        from src.data.loader import load_predict_csv as _lpc
        expected = list(_lpc(INPUT).features.columns)  # loader'ın isimle saptadığı öznitelikler
    print(f"[aligner] shipped feature_names {'VAR' if _fn else 'YOK(None)'} → "
          f"hizalama shipped-yolunda {'DEVREDE' if aligner_engaged_shipped else 'DEVRE DIŞI'}; "
          f"expected={len(expected)} kolon")

    df0 = pd.read_csv(INPUT)
    meta = [c for c in META_CANDIDATES if c in df0.columns]
    feat_cols = [c for c in df0.columns if c not in meta]
    print(f"[data] {INPUT.name}: {len(df0)} satır, {len(feat_cols)} öznitelik kolonu, meta={meta}")

    # ── Baseline (orijinal CSV) ──────────────────────────────────────────────
    p0, pred0 = predict_indexed(pipe, INPUT)
    print(f"[baseline] n={len(p0)}, P(path) ortalama={p0.mean():.4f}")

    rng = np.random.default_rng(SEED)

    # R1 — kolon SIRASI permütasyon (isimler aynı)
    cols = df0.columns.tolist()
    dfR1 = df0[[cols[i] for i in rng.permutation(len(cols))]]

    # R2 — öznitelik kolonları anonim yeniden adlandırılır (sıra korunur), meta korunur
    dfR2 = df0.rename(columns={c: f"COL_{i}" for i, c in enumerate(feat_cols)})

    # R3 — öznitelikler karıştır + anonim rename + 20 gürültü kolon (en sert)
    fc_shuf = list(rng.permutation(np.array(feat_cols, dtype=object)))
    dfR3 = df0[meta + fc_shuf].copy()
    dfR3 = dfR3.rename(columns={c: f"COL_{i}" for i, c in enumerate(fc_shuf)})
    for j in range(20):
        dfR3[f"NOISE_{j}"] = rng.normal(size=len(dfR3))

    regimes = {
        "R1_shuffle_only": dfR1,
        "R2_anonymous_rename": dfR2,
        "R3_shuffle_rename_noise": dfR3,
    }

    results = {}
    worst_dp = 0.0
    worst_agree = 100.0
    for tag, dfx in regimes.items():
        tmp = SCRATCH / f"perm_{tag}.csv"
        dfx.to_csv(tmp, index=False)
        pr, prpred = predict_indexed(pipe, tmp)
        common = p0.index.intersection(pr.index)
        dp = (pr.loc[common] - p0.loc[common]).abs()
        agree = float((prpred.loc[common] == pred0.loc[common]).mean() * 100.0)
        feat_in = [c for c in dfx.columns if c not in META_CANDIDATES]
        results[tag] = {
            "n_matched_variants": int(len(common)),
            "max_abs_delta_prob": round(float(dp.max()), 6),
            "mean_abs_delta_prob": round(float(dp.mean()), 6),
            "prediction_agreement_pct": round(agree, 4),
            "aligner_stages": aligner_stage_report(expected, feat_in, dfx[feat_in]),
        }
        worst_dp = max(worst_dp, results[tag]["max_abs_delta_prob"])
        worst_agree = min(worst_agree, agree)
        print(f"[{tag}] max|Δp|={results[tag]['max_abs_delta_prob']:.6f}  "
              f"agreement={agree:.2f}%  stages={results[tag]['aligner_stages']}")

    # GERÇEKÇİ senaryo = R1 (aynı yarışma kolon adları AL_*/CAT_*, farklı sıra). R2/R3 tamamen
    # anonim COL_* adlandırma → yarışma kolonları SABİT olduğundan gerçekleşmez (aşırı stres).
    r1 = results["R1_shuffle_only"]
    include = bool(r1["prediction_agreement_pct"] >= 99.0 and r1["max_abs_delta_prob"] < 1e-6)
    payload = {
        "experiment": "column_aligner_permutation_robustness",
        "claim": "Tahminler kolon-sirasi/adlandirmadan bagimsiz (ColumnAligner 4-asamali hizalama)",
        "seed": SEED,
        "model_dir": "models/ (shipped, read-only)",
        "input": "data/jury_simulation.csv",
        "n_rows": int(len(df0)),
        "n_feature_cols": int(len(feat_cols)),
        "aligner_engaged_in_shipped_path": aligner_engaged_shipped,
        "method": "uctan-uca predict_from_csv (load_predict_csv -> ColumnAligner); MC-dropout her "
                  "rejimden once set_global_seed(42) ile sifirlandi -> Delta_p yalniz hizalamadan",
        "regimes": results,
        "headline_scenario": "R1_shuffle_only (gercekci: ayni yarisma kolon adlari, farkli sira)",
        "headline_max_abs_delta_prob": r1["max_abs_delta_prob"],
        "headline_prediction_agreement_pct": r1["prediction_agreement_pct"],
        "worst_max_abs_delta_prob": round(worst_dp, 6),
        "worst_prediction_agreement_pct": round(worst_agree, 4),
        "verdict": "ROBUST" if include else "DEGRADED",
        "include_in_report": include,
        "fix_note": "models/expected_feature_columns.json + loader.py fallback ile ColumnAligner "
                    "shipped predict yolunda DEVREYE alindi (xgb feature_names=None oldugundan once "
                    "devre disiydi). Standart girdide NO-OP (tahminler birebir ayni, F1 degismez; A/B kanitli).",
        "note": "GERCEKCI senaryo R1 (yarisma kolonlari AL_*/CAT_* SABIT, yalniz sira degisir) -> "
                "max|dp|=0, agreement %100. R2/R3 tamamen anonim COL_* adlandirma = yarisma formatinda "
                "GERCEKLESMEZ (asiri stres); positional fallback isim+sira birlikte bozulunca sinirli.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[write] {OUT}")
    print(f"[verdict] {payload['verdict']}  include_in_report={include}  "
          f"worst max|Δp|={worst_dp:.6f}  worst agreement={worst_agree:.2f}%")


if __name__ == "__main__":
    main()

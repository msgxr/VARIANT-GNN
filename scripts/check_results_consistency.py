#!/usr/bin/env python3
"""
scripts/check_results_consistency.py
====================================
CI gate: every declared metric in the repo must match RESULTS_CANONICAL.json.

Guards against the failure mode that nearly sank this project: docs drifting away
from the shipped model, leaving multiple contradictory numbers a §7.5 jury re-run
would catch. Run in CI; non-zero exit on any mismatch.

Checks:
  1. RESULTS_CANONICAL.json headline == reports/cv_report.json test_metrics
     (binary_f1, mcc, precision, recall, roc_auc, pr_auc, brier, ece; macro_f1 if present).
  2. Internal consistency: test 2*P*R/(P+R) == test binary_f1.
  3. No WITHDRAWN/leaky/superseded numbers appear unguarded in jury-visible docs
     (incl. reports/PDR_VARIANT_GNN_2026.md, reports/mcc_analysis.md, CHANGELOG.md).
  4. No 'sentetik proxy' / 'synthetic proxy' language in jury-visible reports.
  5. Threshold single source of truth (shipped == canonical, no rival θ).
  6. Canonical jury-F1 headline (competition_jury_f1[_std], 4-panel official avg,
     per-panel %20 F1) == reports/competition_jury_f1.json.
  7. Canonical panel_test_metrics binary_f1 == reports/cv_report.json panel_metrics.
  8. models/PROVENANCE.json metrics (test_f1/precision/recall/roc_auc/pr_auc/mcc/
     brier/ece, cv_f1_mean, global_threshold, panel_f1) == canonical. PROVENANCE has
     no generator and drifted once (stale 0.9069/θ=0.59); this pins it so it can't again.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.005

# Numbers that were withdrawn as leakage-inflated / superseded — must not reappear
# as a current claim (allowed only on lines that explicitly withdraw/supersede them).
# Second block: the pre-2026-06-02 balanced-OOF θ=0.6831 retrain headline
# (test F1 0.8969 / MCC 0.5863 / balanced-jury 0.8134 / official 0.6063 / cal-θ 0.8514),
# superseded by the %20-prior θ=0.8415 canonical. Allowed only on withdraw/supersede lines.
WITHDRAWN = [
    "0.8980",
    "0.9269",
    "0.5356",
    "0.722088",
    "0.7221",
    "0.8668",
    "0.5313",
    "θ=0.241",
    "θ = 0.241",
    "0.8969",
    "0.6831",
    "0.8514",
    "0.8134",
    "0.5863",
    "0.6063",
]
# Per-line guard: a withdrawn number is tolerated only when the SAME line marks it as
# superseded/historical. Covers EN/TR withdrawal vocabulary plus changelog conventions
# ('stale', 'iddia edilemez', and the '→' / '->' change-from arrow that records history).
# A genuine *live* claim ("Test F1 = 0.8969") carries none of these and is still caught.
WITHDRAW_MARKERS = (
    r"geri çek|withdraw|supersed|şişik|leaky|leakage|ÖNCE|previous|GERİ ÇEK|"
    r"eski|önceki|geçersiz|stale|iddia edilemez|artık|→|->"
)
# Every jury/governance-facing doc — drift here is the exact failure mode that
# nearly sank the project (leakage-inflated numbers surviving in metadata files).
JURY_DOCS = [
    "README.md",
    "MODEL_CARD.md",
    "PROJECT_STATUS.md",
    "CLAUDE.md",
    "REPRODUCE.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "DATA_CARD.md",
    "RELEASE_NOTES.md",
    "ROADMAP.md",
    "models/README.md",
    "submission/SUBMISSION_CHECKLIST.md",
    "reports/PDR_VARIANT_GNN_2026.md",
    "reports/mcc_analysis.md",
    "CHANGELOG.md",
]


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def main() -> int:
    # Windows stdout varsayılanı (cp1254) θ/Türkçe karakterlerde patlar; UTF-8'e sabitle.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    errors = 0
    canon_path = ROOT / "RESULTS_CANONICAL.json"
    cv_path = ROOT / "reports" / "cv_report.json"
    if not canon_path.exists():
        print("❌ RESULTS_CANONICAL.json missing — run after a training run.")
        return 1
    canon = json.loads(canon_path.read_text(encoding="utf-8"))
    h = canon["headline"]

    # 1. canonical == cv_report
    print("1. RESULTS_CANONICAL.json vs reports/cv_report.json")
    if cv_path.exists():
        tm = json.loads(cv_path.read_text(encoding="utf-8"))["test_metrics"]
        # (canonical headline key, cv_report test_metrics key, required?)
        headline_map = [
            ("test_binary_f1", "binary_f1", True),
            ("test_mcc", "mcc", True),
            ("test_precision", "precision", True),
            ("test_recall", "recall", True),
            # discrimination + calibration: same failure mode (docs drifting off the
            # shipped model) just on the secondary metrics the jury also re-runs.
            ("test_roc_auc", "roc_auc", True),
            ("test_pr_auc", "pr_auc", True),
            ("test_brier", "brier_score", True),
            ("test_ece", "ece", True),
            # macro F1 is auxiliary — validate only if the canonical headline carries it.
            ("test_macro_f1", "macro_f1", False),
        ]
        for key, cvkey, required in headline_map:
            if key not in h:
                if required:
                    fail(f"{key}: missing from RESULTS_CANONICAL.json headline")
                    errors += 1
                continue
            if cvkey not in tm:
                fail(f"{cvkey}: missing from cv_report test_metrics")
                errors += 1
                continue
            if abs(h[key] - round(tm[cvkey], 4)) > TOL:
                fail(f"{key}: canonical {h[key]} != cv_report {tm[cvkey]:.4f}")
                errors += 1
        print("   ok" if errors == 0 else "   mismatches above")

    # 2. internal consistency
    print("2. Internal consistency: 2PR/(P+R) == binary_f1")
    P, R = h["test_precision"], h["test_recall"]
    f1_from_pr = 2 * P * R / (P + R)
    if abs(f1_from_pr - h["test_binary_f1"]) > TOL:
        fail(f"2PR/(P+R)={f1_from_pr:.4f} != binary_f1={h['test_binary_f1']}")
        errors += 1
    else:
        print("   ok")

    # 3. withdrawn numbers absent from jury docs
    print("3. Withdrawn leaky numbers absent from jury docs")
    for doc in JURY_DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for bad in WITHDRAWN:
            # allow it only on lines that explicitly withdraw/supersede it
            for ln in text.splitlines():
                if bad in ln and not re.search(WITHDRAW_MARKERS, ln, re.I):
                    fail(f"{doc}: withdrawn number {bad} still claimed → '{ln.strip()[:80]}'")
                    errors += 1

    if errors == 0:
        print("   ok")

    # 4. synthetic-proxy language
    print("4. No 'sentetik/synthetic proxy' in jury-visible reports")
    for p in (ROOT / "reports").glob("*.json"):
        rtext = p.read_text(encoding="utf-8").lower()
        if "sentetik proxy" in rtext or "synthetic proxy" in rtext:
            fail(f"{p.name}: contains synthetic-proxy language")
            errors += 1
    if errors == 0:
        print("   ok")

    # 5. threshold single-source-of-truth: shipped == canonical, no contradictory θ
    print("5. Threshold single source of truth (shipped == canonical, no rival θ)")
    thr_path = ROOT / "models" / "threshold.json"
    gthr = canon.get("global_threshold")
    if thr_path.exists() and gthr is not None:
        shipped = json.loads(thr_path.read_text(encoding="utf-8")).get("classification_threshold")
        if shipped is None or abs(shipped - gthr) > 1e-3:
            fail(f"shipped threshold {shipped} != canonical global_threshold {gthr}")
            errors += 1
    # no rival global-threshold value asserted in canonical free-text (panel block exempt)
    rivals = ["0.3367", "0.3104", "0.2562", "0.4456", "θ=0.241", "0.6831", "0.8514"]
    canon_txt = canon_path.read_text(encoding="utf-8")
    for ln in canon_txt.splitlines():
        if (
            any(r in ln for r in rivals)
            and "panel" not in ln.lower()
            and "0.764" not in ln
            and not re.search(r"önceki|geçersiz|withdraw|supersed|geri çek|eski", ln, re.I)
        ):
            fail(f"RESULTS_CANONICAL.json: rival threshold in '{ln.strip()[:70]}'")
            errors += 1
    if errors == 0:
        print("   ok")

    # 6. canonical jury-F1 block == reports/competition_jury_f1.json
    #    The official-score headline (4-panel %20-prior avg + pooled jury F1) is the
    #    number the jury actually scores on; it must trace 1:1 to its source artefact.
    print("6. Canonical jury-F1 headline vs reports/competition_jury_f1.json")
    jf1_path = ROOT / "reports" / "competition_jury_f1.json"
    if jf1_path.exists():
        jf1 = json.loads(jf1_path.read_text(encoding="utf-8"))
        # (canonical headline key, jury-json key)
        scalar_map = [
            ("competition_jury_f1", "competition_jury_f1"),
            ("competition_jury_f1_std", "competition_jury_f1_std"),
            ("official_competition_score_4panel_avg", "official_competition_score"),
        ]
        for ckey, jkey in scalar_map:
            cval, jval = h.get(ckey), jf1.get(jkey)
            if cval is None or jval is None:
                fail(f"{ckey}: missing (canonical={cval}, jury_json[{jkey}]={jval})")
                errors += 1
            elif abs(cval - jval) > TOL:
                fail(f"{ckey}: canonical {cval} != competition_jury_f1.json {jval}")
                errors += 1
        # per-panel official F1 at the 20% prior — allow None==null (e.g. CFTR small-n)
        cpanel = h.get("official_per_panel_f1_20pct", {})
        jpanel = jf1.get("official_per_panel_f1_20pct", {})
        for panel in sorted(set(cpanel) | set(jpanel)):
            cv, jv = cpanel.get(panel), jpanel.get(panel)
            if cv is None and jv is None:
                continue
            if cv is None or jv is None:
                fail(f"official_per_panel_f1_20pct[{panel}]: canonical {cv} != jury_json {jv}")
                errors += 1
            elif abs(cv - jv) > TOL:
                fail(f"official_per_panel_f1_20pct[{panel}]: canonical {cv} != jury_json {jv}")
                errors += 1
    if errors == 0:
        print("   ok")

    # 7. canonical panel_test_metrics == reports/cv_report.json panel_metrics (binary_f1)
    #    Per-panel F1 is §7.3-mandated reporting; per CLAUDE.md §VII the four panels are
    #    never one block, so each panel headline is pinned to its source.
    print("7. Canonical panel_test_metrics vs reports/cv_report.json panel_metrics")
    if cv_path.exists():
        pm = json.loads(cv_path.read_text(encoding="utf-8")).get("panel_metrics", {})
        cpm = canon.get("panel_test_metrics", {})
        for panel in sorted(set(cpm) | set(pm)):
            cf1 = cpm.get(panel, {}).get("binary_f1") if isinstance(cpm.get(panel), dict) else None
            jf1v = pm.get(panel, {}).get("binary_f1") if isinstance(pm.get(panel), dict) else None
            if cf1 is None or jf1v is None:
                fail(f"panel_test_metrics[{panel}].binary_f1: canonical {cf1} != cv_report {jf1v}")
                errors += 1
            elif abs(cf1 - jf1v) > TOL:
                fail(f"panel_test_metrics[{panel}].binary_f1: canonical {cf1} != cv_report {jf1v:.4f}")
                errors += 1
    if errors == 0:
        print("   ok")

    # 8. models/PROVENANCE.json metrics == canonical (anti-drift firewall extension).
    #    PROVENANCE has NO generator (hand-maintained) and is NOT regenerated by a jury
    #    predict run, so it historically drifted (stale test_f1=0.9069, MCC-optimal θ=0.59)
    #    until 2026-06-05. Pinning it here makes that class of drift build-failing.
    print("8. models/PROVENANCE.json vs RESULTS_CANONICAL.json")
    prov_path = ROOT / "models" / "PROVENANCE.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        prov_map = [
            ("test_f1", "test_binary_f1"),
            ("test_precision", "test_precision"),
            ("test_recall", "test_recall"),
            ("test_roc_auc", "test_roc_auc"),
            ("test_pr_auc", "test_pr_auc"),
            ("test_mcc", "test_mcc"),
            ("test_brier", "test_brier"),
            ("test_ece", "test_ece"),
            ("cv_f1_mean", "cv_binary_f1_production_oof_stacking"),
        ]
        for pkey, ckey in prov_map:
            if pkey in prov and ckey in h:
                if abs(prov[pkey] - h[ckey]) > TOL:
                    fail(f"PROVENANCE.{pkey}={prov[pkey]} != canonical {ckey}={h[ckey]}")
                    errors += 1
        gthr = canon.get("global_threshold")
        if "global_threshold" in prov and gthr is not None:
            if abs(prov["global_threshold"] - gthr) > 1e-3:
                fail(f"PROVENANCE.global_threshold={prov['global_threshold']} != canonical {gthr}")
                errors += 1
        cpm = canon.get("panel_test_metrics", {})
        for panel, pf1 in (prov.get("panel_f1") or {}).items():
            cf1 = cpm.get(panel, {}).get("binary_f1") if isinstance(cpm.get(panel), dict) else None
            if cf1 is not None and abs(pf1 - cf1) > TOL:
                fail(f"PROVENANCE.panel_f1[{panel}]={pf1} != canonical {cf1}")
                errors += 1
    if errors == 0:
        print("   ok")

    print()
    if errors:
        print(f"❌ FAIL: {errors} consistency error(s). Regenerate docs from RESULTS_CANONICAL.json.")
        return 1
    print("✅ PASS: all declared numbers are consistent with RESULTS_CANONICAL.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

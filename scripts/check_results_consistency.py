#!/usr/bin/env python3
"""
scripts/check_results_consistency.py
====================================
CI gate: every declared metric in the repo must match RESULTS_CANONICAL.json.

Guards against the failure mode that nearly sank this project: docs drifting away
from the shipped model, leaving multiple contradictory numbers a §7.5 jury re-run
would catch. Run in CI; non-zero exit on any mismatch.

Checks:
  1. RESULTS_CANONICAL.json headline == reports/cv_report.json test_metrics.
  2. Internal consistency: test 2*P*R/(P+R) == test binary_f1.
  3. No WITHDRAWN/leaky numbers (0.8980, 0.9269, 0.5356) appear in jury-visible docs.
  4. No 'sentetik proxy' / 'synthetic proxy' language in jury-visible reports.
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
WITHDRAWN = ["0.8980", "0.9269", "0.5356", "0.722088", "0.7221",
             "0.8668", "0.5313", "θ=0.241", "θ = 0.241"]
# Every jury/governance-facing doc — drift here is the exact failure mode that
# nearly sank the project (leakage-inflated numbers surviving in metadata files).
JURY_DOCS = ["README.md", "MODEL_CARD.md", "PROJECT_STATUS.md", "CLAUDE.md",
             "REPRODUCE.md", "CITATION.cff", "CODE_OF_CONDUCT.md", "DATA_CARD.md",
             "RELEASE_NOTES.md", "ROADMAP.md", "models/README.md",
             "submission/SUBMISSION_CHECKLIST.md"]


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
        for key, cvkey in [("test_binary_f1", "binary_f1"), ("test_mcc", "mcc"),
                           ("test_precision", "precision"), ("test_recall", "recall")]:
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
                if bad in ln and not re.search(
                    r"geri çek|withdraw|supersed|şişik|leaky|leakage|ÖNCE|previous|GERİ ÇEK", ln, re.I
                ):
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
        if any(r in ln for r in rivals) and "panel" not in ln.lower() and "0.764" not in ln \
                and not re.search(r"önceki|geçersiz|withdraw|supersed|geri çek|eski", ln, re.I):
            fail(f"RESULTS_CANONICAL.json: rival threshold in '{ln.strip()[:70]}'")
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

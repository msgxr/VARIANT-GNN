#!/usr/bin/env python3
"""Generate self-contained local SVG badges for README (zero external dependency).

Replaces shields.io / readme-typing-svg / GitHub Actions badge images with local
SVGs committed to docs/assets/readme/badges/. Badges are produced from the
declarative BADGES spec below, mimicking shields.io 'flat-square' and
'for-the-badge' styles. No brand logos are embedded (text-only, license-clean).

Run:  python scripts/gen_readme_badges.py
"""

from __future__ import annotations

import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "assets", "readme", "badges")

# --- text width heuristic (DejaVu Sans / Verdana-like advance widths) ----------
_NARROW = set("iljtfI.,:;'|!()[]/ ")
_WIDE = set("mwMW@%")


def _char_w(ch: str, fs: float) -> float:
    if ch in _NARROW:
        return 0.33 * fs
    if ch in _WIDE:
        return 0.90 * fs
    if ch.isupper() or ch.isdigit():
        return 0.66 * fs
    return 0.58 * fs


def _text_w(s: str, fs: float, bold: bool = False, ls: float = 0.0) -> float:
    w = sum(_char_w(c, fs) for c in s)
    if len(s) > 1:
        w += ls * (len(s) - 1)
    if bold:
        w *= 1.04
    return w


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_badge(label: str, message: str, color: str, label_color: str = "#555", style: str = "flat") -> str:
    fs = 11.0
    if style == "ftb":  # for-the-badge: uppercase, bold, letter-spaced, taller
        lab, msg = label.upper(), message.upper()
        ls, pad, h, weight, baseline = 1.4, 11, 28, "bold", 18.7
    else:  # flat-square
        lab, msg = label, message
        ls, pad, h, weight, baseline = 0.0, 6, 20, "normal", 14.2

    bold = style == "ftb"
    lw = math.ceil(_text_w(lab, fs, bold, ls) + 2 * pad)
    mw = math.ceil(_text_w(msg, fs, bold, ls) + 2 * pad)
    w = lw + mw
    # nudge centring left to compensate for trailing letter-spacing
    nudge = ls / 2.0
    lx = lw / 2.0 - nudge
    mx = lw + mw / 2.0 - nudge
    font = "'DejaVu Sans',Verdana,Geneva,sans-serif"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'role="img" aria-label="{_esc(label)}: {_esc(message)}">\n'
        f"  <title>{_esc(label)}: {_esc(message)}</title>\n"
        f'  <rect width="{lw}" height="{h}" fill="{label_color}"/>\n'
        f'  <rect x="{lw}" width="{mw}" height="{h}" fill="{color}"/>\n'
        f'  <g fill="#ffffff" font-family="{font}" font-size="{fs:g}" '
        f'font-weight="{weight}" letter-spacing="{ls:g}">\n'
        f'    <text x="{lx:.1f}" y="{baseline}" text-anchor="middle">{_esc(lab)}</text>\n'
        f'    <text x="{mx:.1f}" y="{baseline}" text-anchor="middle">{_esc(msg)}</text>\n'
        f"  </g>\n"
        f"</svg>\n"
    )


def make_typing(lines: list[str]) -> str:
    """Static, self-contained replacement for the animated typing SVG."""
    w, lh, top = 1120, 36, 50
    h = top + lh * (len(lines) - 1) + 36
    font = "'JetBrains Mono','Cascadia Code',Consolas,'Courier New',monospace"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" role="img" aria-label="VARIANT-GNN ozet">',
        "  <defs>",
        '    <linearGradient id="tp" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#0b1220"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{w}" height="{h}" rx="14" fill="url(#tp)" stroke="#1e293b" stroke-width="1.5"/>',
    ]
    for i, ln in enumerate(lines):
        y = top + lh * i
        fill = "#e2e8f0" if i == 0 else "#22d3ee"
        weight = "700" if i == 0 else "500"
        parts.append(
            f'  <text x="{w / 2:.0f}" y="{y}" text-anchor="middle" '
            f'font-family="{font}" font-size="20" font-weight="{weight}" '
            f'fill="{fill}">{_esc(ln)}</text>'
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


# --- badge spec: (filename, label, message, color, label_color, style) --------
BADGES = [
    ("badge_psr", "PSR PUANI", "93.00 / 100", "#22c55e", "#052e16", "ftb"),
    ("badge_juri_f1", "Juri F1 resmi", "0.6042±0.034", "#3b82f6", "#172554", "ftb"),
    ("badge_cv_f1", "CV F1", "0.8936±0.0004 (leakage-free)", "#0ea5e9", "#082f49", "ftb"),
    ("badge_takim", "Takim", "XYRA3 #909249", "#8b5cf6", "#2e1065", "ftb"),
    ("badge_pdr", "PDR Teslim", "29 Haziran 2026", "#f59e0b", "#431407", "ftb"),
    ("badge_ci", "CI", "GitHub Actions", "#2088ff", "#555", "flat"),
    ("badge_python", "Python", "3.12", "#3776ab", "#555", "flat"),
    ("badge_pytorch", "PyTorch", "2.8.0", "#ee4c2c", "#555", "flat"),
    ("badge_pyg", "PyG", "2.6.1", "#ff6b35", "#555", "flat"),
    ("badge_xgboost", "XGBoost", "2.1.4", "#189ab4", "#555", "flat"),
    ("badge_lightgbm", "LightGBM", "4.6.0", "#2d9a27", "#555", "flat"),
    ("badge_streamlit", "Streamlit", "1.50", "#ff4b4b", "#555", "flat"),
    ("badge_gnn", "GNN", "GATv2Conv x3 blok", "#60a5fa", "#555", "flat"),
    ("badge_dnn", "DNN", "Domain Adversarial", "#a78bfa", "#555", "flat"),
    ("badge_swa", "SWA", "Son 25pct epoch", "#8b5cf6", "#555", "flat"),
    ("badge_stacking", "Stacking", "OOF Wolpert", "#22d3ee", "#555", "flat"),
    ("badge_bio", "Bio Features", "ACMG aligned", "#16a34a", "#555", "flat"),
    ("badge_conformal", "Conformal", "LAC Mondrian", "#fb923c", "#555", "flat"),
    ("badge_nda", "TEKNOFEST NDA", "Gizli", "#ef4444", "#555", "flat"),
    ("badge_github", "GitHub", "msgxr/VARIANT-GNN", "#181717", "#555", "ftb"),
    ("badge_teknofest", "TEKNOFEST", "2026 Sanliurfa", "#ff6b35", "#555", "ftb"),
]

TYPING_LINES = [
    "PSR ASAMASI  →  93.00 / 100 PUAN",
    "JURI F1 (resmi 4-panel) = 0.6202   |   ic hold-out 0.8367",
    "Missense Varyant Patojenitesi Tahmini",
    "GATv2Conv + XGBoost + LightGBM + DANN-DNN Ensemble",
    "PDR Teslimi  →  29 Haziran 2026",
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, label, msg, color, lc, style in BADGES:
        svg = make_badge(label, msg, color, lc, style)
        with open(os.path.join(OUT_DIR, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(svg)
    typing_path = os.path.join(ROOT, "docs", "assets", "readme", "typing.svg")
    with open(typing_path, "w", encoding="utf-8") as f:
        f.write(make_typing(TYPING_LINES))
    print(f"Generated {len(BADGES)} badges in {OUT_DIR}")
    print(f"Generated typing panel at {typing_path}")


if __name__ == "__main__":
    main()

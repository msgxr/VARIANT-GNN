#!/usr/bin/env python3
"""
scripts/build_pdr_docx.py
=========================
reports/PDR_VARIANT_GNN_2026.md  →  reports/PDR_VARIANT_GNN_2026.docx

TASLAK dönüştürücü. python-docx ile başlık/paragraf/tablo/madde/figür-gömme +
TEKNOFEST format hedefi (Aptos 11pt, 1.15 satır, justified, 2.5 cm marj).

UYARI: Bu bir TASLAK üretir. Resmi şablona birebir uyum, ≤10 sayfa sınırı ve
görsel yerleşim Word'de MANUEL doğrulanmalıdır. Tablolar/figürler sayfa taşırsa
elle ayarlanır. Aptos sistemde yoksa Word ikame font kullanır.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "reports" / "PDR_VARIANT_GNN_2026.md"
OUT = ROOT / "reports" / "PDR_VARIANT_GNN_2026.docx"
FIG_DIR = ROOT

FONT = "Aptos"
SEP_RE = re.compile(r"^\s*:?-{2,}:?\s*$")
PNG_RE = re.compile(r"(reports/figures/[\w/]+\.png)")


def _set_base_style(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)
    pf = st.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Cm(2.5)
        sec.left_margin = sec.right_margin = Cm(2.5)


def _add_runs(par, text: str) -> None:
    """**bold** ve `code` işaretlerini Word run'larına çevir (basit)."""
    text = text.replace("`", "")
    for i, chunk in enumerate(re.split(r"(\*\*.+?\*\*)", text)):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            r = par.add_run(chunk[2:-2]); r.bold = True
        else:
            par.add_run(chunk)


def _heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10 if level <= 1 else 6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.strip())
    r.bold = True
    r.font.name = FONT
    r.font.size = Pt({0: 16, 1: 14, 2: 12, 3: 11}.get(level, 11))
    r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)


def _table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci in range(ncol):
            txt = row[ci] if ci < len(row) else ""
            cell = cells[ci]
            cell.paragraphs[0].text = ""
            par = cell.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(par, txt.strip())
            for run in par.runs:
                run.font.size = Pt(9)
                if ri == 0:
                    run.bold = True


def _figure(doc: Document, path_str: str, caption: str) -> bool:
    img = (FIG_DIR / path_str)
    if not img.exists():
        return False
    try:
        doc.add_picture(str(img), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(caption.strip())
            r.italic = True; r.font.size = Pt(9)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  figür gömülemedi ({path_str}): {e}")
        return False


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def main() -> int:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1254 stdout → utf-8
    lines = MD.read_text(encoding="utf-8").splitlines()
    doc = Document()
    _set_base_style(doc)

    i, n = 0, len(lines)
    n_tab = n_fig = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Tablo bloğu
        if stripped.startswith("|") and "|" in stripped[1:]:
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            rows = []
            for bl in block:
                cells = _split_row(bl)
                if all(SEP_RE.match(c) or c == "" for c in cells):
                    continue  # ayraç satırı
                rows.append(cells)
            _table(doc, rows); n_tab += 1
            continue

        # Başlık
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            _heading(doc, m.group(2), len(m.group(1)) - 1); i += 1; continue

        # Yatay çizgi
        if stripped in ("---", "***", "___"):
            i += 1; continue

        # Figür (png yolu içeren satır)
        png = PNG_RE.search(stripped)
        if png:
            caption = re.sub(r"\*|`", "", stripped)
            if _figure(doc, png.group(1), caption):
                n_fig += 1
            else:
                p = doc.add_paragraph(); _add_runs(p, stripped)
            i += 1; continue

        # Madde
        if re.match(r"^[-*]\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            _add_runs(p, re.sub(r"^[-*]\s+", "", stripped)); i += 1; continue
        if re.match(r"^\d+\.\s+", stripped):
            p = doc.add_paragraph(style="List Number")
            _add_runs(p, re.sub(r"^\d+\.\s+", "", stripped)); i += 1; continue

        # Boş satır
        if not stripped:
            i += 1; continue

        # Normal paragraf
        p = doc.add_paragraph(); _add_runs(p, stripped); i += 1

    doc.save(str(OUT))
    print(f"OK → {OUT}")
    print(f"   {n_tab} tablo, {n_fig} figür gömüldü.")
    print("   ⚠️ TASLAK: Aptos/marj/≤10-sayfa Word'de doğrulanmalı; tablo/figür taşması elle ayarlanır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

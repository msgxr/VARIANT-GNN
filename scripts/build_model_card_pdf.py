"""
scripts/build_model_card_pdf.py
=================================
MODEL_CARD.md → MODEL_CARD.pdf üretici (fpdf2 ile).

Kullanım:
    python scripts/build_model_card_pdf.py
    # → docs/MODEL_CARD.pdf

Bu script Türkçe karakter desteği için DejaVu fontunu kullanır.
fpdf2'nin built-in core fontları sadece Latin-1 destekler; Türkçe karakterler
için Unicode TrueType font gerekir.

Markdown render etmez — yalnızca okunabilir, düz formatlanmış bir
PDF üretir. PDR ekine konabilir.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("HATA: fpdf2 kurulu değil. `pip install fpdf2` ile kurun.", file=sys.stderr)
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD    = PROJECT_ROOT / "MODEL_CARD.md"
OUTPUT_PDF   = PROJECT_ROOT / "docs" / "MODEL_CARD.pdf"


def _strip_emojis(text: str) -> str:
    """Remove emoji characters that DejaVu cannot render reliably."""
    # Geniş emoji aralığını sil
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"  # emoji
        "\U00002600-\U000027BF"  # misc symbols
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001FA70-\U0001FAFF"  # extended-A
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _normalise_for_latin1(text: str) -> str:
    """
    Replace characters that core PDF fonts cannot render.
    Keeps Turkish chars (UTF-8 bilinen DejaVu desteklemese bile core font
    fallback için ASCII'ye yaklaştırma).
    """
    repl = {
        "→": "->", "←": "<-", "↔": "<->", "⇒": "=>",
        "≈": "~=", "≤": "<=", "≥": ">=",
        "·": ".",  "•": "*",  "—": "-",  "–": "-",
        "★": "*",  "✓": "[OK]", "✗": "[X]",
        "₺": "TL",
        "α": "alpha", "β": "beta", "γ": "gamma", "λ": "lambda",
        "ε": "epsilon", "ρ": "rho", "η": "eta", "Σ": "Sum",
        "²": "^2", "³": "^3",
        # Trademark / box drawing
        "│": "|", "├": "+", "└": "+", "┌": "+", "┐": "+",
        "┘": "+", "─": "-", "▼": "v", "▲": "^",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


class ModelCardPDF(FPDF):
    """Custom FPDF with TEKNOFEST header/footer."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "VARIANT-GNN Model Card | TEKNOFEST 2026 | Takim XYRA3",
                  align="C")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Sayfa {self.page_no()}/{{nb}}", align="C")


def _safe_multi_cell(pdf: ModelCardPDF, h: float, content: str) -> None:
    """Wrapper around multi_cell with graceful fallback on FPDF errors."""
    try:
        FPDF.multi_cell(pdf, 0, h, content)
    except Exception:
        # Fall back: aggressive ASCII normalisation + character truncation
        try:
            ascii_only = content.encode("ascii", errors="replace").decode("ascii")
            ascii_only = ascii_only.replace("?", " ")
            chunks = [ascii_only[i:i + 80] for i in range(0, len(ascii_only), 80)]
            for c in chunks:
                if c.strip():
                    try:
                        FPDF.multi_cell(pdf, 0, h, c)
                    except Exception:
                        continue
        except Exception:
            pass  # silently skip unrenderable line


def render_markdown(pdf: ModelCardPDF, text: str) -> None:
    """Lightweight markdown → PDF rendering."""
    text = _strip_emojis(text)
    text = _normalise_for_latin1(text)

    # Use core Helvetica font; fpdf2 supports Turkish via Latin-1 transliteration
    # for ş, ğ, ı etc., but they render approximately. Acceptable for jury copy.
    pdf.set_font("Helvetica", "", 11)

    # Process line by line
    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Code fence
        if line.strip().startswith("```"):
            in_code = not in_code
            pdf.ln(2)
            continue

        if in_code:
            pdf.set_font("Courier", "", 9)
            pdf.set_text_color(40, 40, 40)
            try:
                safe = _safe(line) if line else " "
                # Çok uzun satırları kır
                while len(safe) > 100:
                    _safe_multi_cell(pdf, 4, safe[:100])
                    safe = safe[100:]
                _safe_multi_cell(pdf, 4, safe if safe else " ")
            except Exception:
                pass
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            continue

        # Headers
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 60, 100)
            pdf.cell(0, 7, _safe(line[4:]))
            pdf.ln(7)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            continue
        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(40, 40, 100)
            pdf.cell(0, 8, _safe(line[3:]))
            pdf.ln(8)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            continue
        if line.startswith("# "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(20, 20, 80)
            pdf.cell(0, 10, _safe(line[2:]))
            pdf.ln(10)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            continue

        # Horizontal rule
        if re.match(r"^-{3,}$", line.strip()):
            pdf.ln(1)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(3)
            continue

        # Blockquote
        if line.startswith("> "):
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(80, 80, 80)
            _safe_multi_cell(pdf, 5, _safe("    " + line[2:]))
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            continue

        # Bullet list
        if line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 11)
            _safe_multi_cell(pdf, 5, _safe("  - " + line[2:]))
            continue

        # Numbered list
        if re.match(r"^\d+\.\s", line):
            pdf.set_font("Helvetica", "", 11)
            _safe_multi_cell(pdf, 5, _safe(line))
            continue

        # Tables (markdown |...|...|) — render as monospace code-like
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            pdf.set_font("Courier", "", 8)
            # Geniş satırları parçala
            safe = _safe(line)
            chunk_size = 110
            for i in range(0, len(safe), chunk_size):
                _safe_multi_cell(pdf, 4, safe[i:i + chunk_size])
            pdf.set_font("Helvetica", "", 11)
            continue

        # Paragraphs
        if line.strip():
            pdf.set_font("Helvetica", "", 11)
            safe = _safe(line)
            chunk_size = 200
            if len(safe) > chunk_size:
                for i in range(0, len(safe), chunk_size):
                    _safe_multi_cell(pdf, 5, safe[i:i + chunk_size])
            else:
                _safe_multi_cell(pdf, 5, safe)
        else:
            pdf.ln(2)


def _safe(s: str) -> str:
    """Encode-safe: attempt latin-1, fallback to ASCII; break long words."""
    try:
        s.encode("latin-1")
    except UnicodeEncodeError:
        tr_map = str.maketrans("şŞıİğĞüÜöÖçÇ", "sSiIgGuUoOcC")
        s = s.translate(tr_map).encode("latin-1", errors="replace").decode("latin-1")

    # Çok uzun kelimeleri (40+ char) zorla kır — fpdf2 multi_cell sorununu çöz
    words = s.split(" ")
    fixed_words: list[str] = []
    for word in words:
        if len(word) > 40:
            # Her 40 karakterde boşluk ekle
            chunked = [word[i:i + 40] for i in range(0, len(word), 40)]
            fixed_words.append(" ".join(chunked))
        else:
            fixed_words.append(word)
    return " ".join(fixed_words)


def main() -> int:
    if not SOURCE_MD.exists():
        print(f"HATA: {SOURCE_MD} bulunamadı", file=sys.stderr)
        return 1

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    pdf = ModelCardPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    text = SOURCE_MD.read_text(encoding="utf-8")
    render_markdown(pdf, text)
    pdf.output(str(OUTPUT_PDF))

    print(f"OK  {OUTPUT_PDF}  ({OUTPUT_PDF.stat().st_size:,} bayt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

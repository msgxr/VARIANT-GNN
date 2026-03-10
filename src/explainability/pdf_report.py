"""
VARIANT-GNN — Türkçe PDF Klinik Rapor Üretici
==============================================
SHAP, ClinVar ve risk skoru bilgilerini birleştirerek
doktora sunulmaya hazır PDF rapor üretir.

Gereksinim: fpdf2  (pip install fpdf2)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def _safe_text(text: str) -> str:
    """Latin-1'de sorun yaratabilecek karakterleri temizler."""
    replacements = {
        "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U", "ş": "s", "Ş": "S",
        "ı": "i", "İ": "I", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
        "⬆": "+", "⬇": "-", "🔴": "[KRITIK]", "🟠": "[YUKSEK]",
        "🟡": "[ORTA]", "🟢": "[DUSUK]", "⚪": "[BELIRSIZ]",
        "✅": "[OK]", "❌": "[X]", "🏥": "", "🧬": "", "📊": "",
        "→": "->", "–": "-", "—": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


class VariantReportPDF(FPDF if FPDF_AVAILABLE else object):
    """FPDF2 tabanlı klinik rapor sınıfı."""

    DARK_BG  = (15, 23, 42)
    ACCENT   = (66, 153, 225)
    TEXT_COL = (30, 30, 30)
    HEADER_BG = (30, 58, 138)

    def header(self):
        self.set_fill_color(*self.HEADER_BG)
        self.rect(0, 0, 210, 28, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.set_xy(10, 8)
        self.cell(0, 8, "VARIANT-GNN | Klinik Karar Destek Raporu", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_xy(10, 18)
        self.cell(0, 6,
                  f"Uretim Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')} | "
                  "Yalnizca Arastirma Amaclidir", ln=True)
        self.set_text_color(*self.TEXT_COL)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8,
                  f"Sayfa {self.page_no()} | VARIANT-GNN v2.0 | "
                  "Bu rapor klinik tani yerine gecirilmemelidir.",
                  align="C")

    def section_title(self, title: str):
        self.set_fill_color(235, 245, 255)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.HEADER_BG)
        self.cell(0, 8, _safe_text(title), ln=True, fill=True)
        self.set_text_color(*self.TEXT_COL)
        self.ln(1)

    def kv_row(self, key: str, value: str, bold_val: bool = False):
        self.set_font("Helvetica", "B", 9)
        self.cell(55, 6, _safe_text(key) + ":", ln=False)
        self.set_font("Helvetica", "B" if bold_val else "", 9)
        self.multi_cell(0, 6, _safe_text(str(value)))

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5.5, _safe_text(text))
        self.ln(1)

    def risk_badge(self, risk_score: float, zone_label: str):
        """Üstte büyük risk skoru rozeti."""
        if risk_score >= 75:
            r, g, b = (220, 38, 38)
        elif risk_score >= 50:
            r, g, b = (234, 88, 12)
        elif risk_score >= 25:
            r, g, b = (202, 138, 4)
        else:
            r, g, b = (22, 163, 74)

        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 22)
        self.set_xy(10, self.get_y())
        self.cell(60, 18, f"{risk_score:.1f} / 100", align="C", fill=True)
        self.set_font("Helvetica", "B", 12)
        cx = 75
        self.set_xy(cx, self.get_y() - 18)
        self.set_text_color(r, g, b)
        self.cell(120, 18, _safe_text(zone_label), align="L")
        self.set_text_color(*self.TEXT_COL)
        self.ln(6)


def generate_pdf_report(
    variant_id: Optional[str],
    risk_score: float,
    prediction: str,
    probability: float,
    clinical_insight: dict,
    top_features: list[tuple[str, float]],
    clinvar_info: Optional[dict] = None,
    shap_waterfall_path: Optional[str] = None,
) -> bytes:
    """
    Türkçe PDF klinik raporu byte dizisi olarak döner.
    Streamlit'te `st.download_button` ile kullanılabilir.
    """
    if not FPDF_AVAILABLE:
        raise ImportError("fpdf2 kurulu değil. `pip install fpdf2` komutuyla kurun.")

    pdf = VariantReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=16)

    # ── 1. Varyant Kimliği ────────────────────────────────────
    pdf.section_title("1. Varyant Bilgisi")
    pdf.kv_row("Varyant ID", variant_id or "Bilinmiyor")
    pdf.kv_row("Model Tahmini", prediction, bold_val=True)
    pdf.kv_row("Olasilik (Patojenik)", f"{probability:.2%}")
    pdf.ln(2)

    # ── 2. Risk Skoru ─────────────────────────────────────────
    pdf.section_title("2. Risk Degerlendirmesi")
    zone_label = clinical_insight.get("zone_label", "")
    pdf.risk_badge(risk_score, zone_label)
    pdf.body_text(clinical_insight.get("summary", ""))

    # ── 3. Kilit Bulgular ─────────────────────────────────────
    findings = clinical_insight.get("key_findings", [])
    if findings:
        pdf.section_title("3. Kilit Biyolojik Bulgular")
        for i, f in enumerate(findings, 1):
            direction = "+" if f.get("direction") == "artirdi" else "-"
            line = (
                f"{i}. {f.get('feature','?')} ({f.get('group','?')}) — "
                f"Riski {f.get('direction','?')} → SHAP: {f.get('shap', 0):.4f}"
            )
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, _safe_text(line), ln=True)
            pdf.body_text(f.get("insight", ""))

    # ── 4. Klinik Öneri ───────────────────────────────────────
    pdf.section_title("4. Klinik Oneri")
    pdf.body_text(clinical_insight.get("recommendation", "Öneri üretilemedi."))

    # ── 5. En Önemli Özellikler (Top Features) ────────────────
    if top_features:
        pdf.section_title("5. En Etkili Genomik Ozellikler (SHAP)")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(80, 6, "Ozellik Adi", border=1, fill=True)
        pdf.cell(40, 6, "SHAP Degeri", border=1, fill=True)
        pdf.cell(70, 6, "Etki Yonu", border=1, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 8)
        for feat_name, shap_val in top_features[:10]:
            direction = "Risk Artirici (+)" if shap_val > 0 else "Risk Azaltici (-)"
            pdf.cell(80, 5.5, _safe_text(feat_name[:40]), border=1)
            pdf.cell(40, 5.5, f"{shap_val:.4f}", border=1)
            pdf.cell(70, 5.5, direction, border=1, ln=True)
        pdf.ln(2)

    # ── 6. ClinVar Karşılaştırması ────────────────────────────
    if clinvar_info and clinvar_info.get("found"):
        pdf.section_title("6. ClinVar Karsilastirmasi (NCBI)")
        pdf.kv_row("ClinVar Siniflandirmasi",
                   clinvar_info.get("clinical_significance", "?"), bold_val=True)
        pdf.kv_row("Inceleme Durumu", clinvar_info.get("review_status", "?"))
        conds = ", ".join(clinvar_info.get("conditions", [])) or "Belirtilmemis"
        pdf.kv_row("Iliskili Durumlar", conds)
        pdf.kv_row("ClinVar URL", clinvar_info.get("url", ""))
        pdf.ln(2)

    # ── 7. SHAP Waterfall Görseli ─────────────────────────────
    if shap_waterfall_path:
        import os
        if os.path.exists(shap_waterfall_path):
            try:
                pdf.section_title("7. SHAP Waterfall Grafigi")
                pdf.image(shap_waterfall_path, w=180)
            except Exception:  # noqa: BLE001
                pass

    # ── 8. Sorumluluk Reddi ───────────────────────────────────
    pdf.ln(4)
    pdf.set_fill_color(254, 243, 199)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 60, 0)
    pdf.multi_cell(
        0, 5,
        "UYARI: Bu rapor yalnizca arastirma amaclidir. Klinik karar vermek icin "
        "bagimiz bir uzman genetikci veya tibbi genetik uzmanina basvurunuz. "
        "ACMG/AMP kriterleri cercevesinde professional degerlendirme yapilmasi onerilir.",
        fill=True,
    )

    return bytes(pdf.output())

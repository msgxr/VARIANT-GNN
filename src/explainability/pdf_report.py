"""
VARIANT-GNN — Türkçe PDF Klinik Rapor Üretici
==============================================
SHAP, ClinVar ve risk skoru bilgilerini birleştirerek
doktora sunulmaya hazır PDF rapor üretir.

Gereksinim: fpdf2  (pip install fpdf2)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def _is_nan(v: float) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


def _find_dejavu_font(style: str = "") -> Optional[str]:
    """Matplotlib paketinden DejaVuSans TTF yolunu döner (varsa)."""
    suffix = {
        "": "DejaVuSans.ttf",
        "B": "DejaVuSans-Bold.ttf",
        "I": "DejaVuSans-Oblique.ttf",
        "BI": "DejaVuSans-BoldOblique.ttf",
    }
    fname = suffix.get(style.upper(), "DejaVuSans.ttf")
    try:
        import matplotlib

        candidate = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", fname)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return None


def _safe_text(text: str) -> str:
    """Emoji ve özel karakterleri ASCII-güvenli hale getirir.

    DejaVuSans font yüklenemediğinde Helvetica (Latin-1) fallback için
    Türkçe karakterler korunur; emoji/ok karakterleri temizlenir.
    """
    replacements = {
        "⬆": "+",
        "⬇": "-",
        "🔴": "[KRITIK]",
        "🟠": "[YUKSEK]",
        "🟡": "[ORTA]",
        "🟢": "[DUSUK]",
        "⚪": "[BELIRSIZ]",
        "✅": "[OK]",
        "❌": "[X]",
        "🏥": "",
        "🧬": "",
        "📊": "",
        "→": "->",
        "–": "-",
        "—": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _safe_text_latin1(text: str) -> str:
    """DejaVu font yoksa Türkçe karakterleri de ASCII'ye çevirir (fallback)."""
    extra = {
        "ğ": "g",
        "Ğ": "G",
        "ü": "u",
        "Ü": "U",
        "ş": "s",
        "Ş": "S",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ç": "c",
        "Ç": "C",
    }
    text = _safe_text(text)
    for src, dst in extra.items():
        text = text.replace(src, dst)
    return text


class VariantReportPDF(FPDF if FPDF_AVAILABLE else object):  # type: ignore[misc]
    """FPDF2 tabanlı klinik rapor sınıfı.

    DejaVuSans TTF mevcut ise Türkçe karakterler tam olarak gösterilir.
    Yoksa _safe_text_latin1() ile ASCII fallback kullanılır.
    """

    DARK_BG = (15, 23, 42)
    ACCENT = (66, 153, 225)
    TEXT_COL = (30, 30, 30)
    HEADER_BG = (30, 58, 138)

    # DejaVu font yüklenip yüklenmediği (ilk add_page'de kontrol edilir)
    _DEJAVU_LOADED: bool = False
    _USE_DEJAVU: bool = False

    def _init_fonts(self) -> None:
        """DejaVuSans fontunu fpdf2'ye kaydet (bir kez)."""
        if VariantReportPDF._DEJAVU_LOADED:
            return
        VariantReportPDF._DEJAVU_LOADED = True
        font_path = _find_dejavu_font("")
        font_bold = _find_dejavu_font("B")
        if font_path and font_bold:
            try:
                self.add_font("DejaVu", style="", fname=font_path)
                self.add_font("DejaVu", style="B", fname=font_bold)
                VariantReportPDF._USE_DEJAVU = True
            except Exception:
                VariantReportPDF._USE_DEJAVU = False
        else:
            VariantReportPDF._USE_DEJAVU = False

    def _set_font(self, style: str = "", size: int = 10) -> None:
        """DejaVu varsa kullan, yoksa Helvetica fallback."""
        if VariantReportPDF._USE_DEJAVU:
            self.set_font("DejaVu", style=style if style in ("", "B") else "", size=size)
        else:
            self.set_font("Helvetica", style=style, size=size)

    def _txt(self, text: str) -> str:
        """Font'a göre uygun metin temizliği uygular."""
        if VariantReportPDF._USE_DEJAVU:
            return _safe_text(text)  # Türkçe karakterleri koru
        return _safe_text_latin1(text)  # Türkçe → ASCII fallback

    def header(self) -> None:
        self._init_fonts()
        self.set_fill_color(*self.HEADER_BG)
        self.rect(0, 0, 210, 28, "F")
        self.set_text_color(255, 255, 255)
        self._set_font("B", 16)
        self.set_xy(10, 8)
        self.cell(0, 8, self._txt("VARIANT-GNN | Klinik Karar Destek Raporu"), new_x="LMARGIN", new_y="NEXT")
        self._set_font("", 9)
        self.set_xy(10, 18)
        self.cell(
            0,
            6,
            self._txt(f"Uretim Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')} | Yalnizca Arastirma Amaclidir"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_text_color(*self.TEXT_COL)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-14)
        self._set_font("I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            8,
            self._txt(f"Sayfa {self.page_no()} | VARIANT-GNN v2.0 | Bu rapor klinik tani yerine gecirilmemelidir."),
            align="C",
        )

    def section_title(self, title: str) -> None:
        self.set_fill_color(235, 245, 255)
        self._set_font("B", 11)
        self.set_text_color(*self.HEADER_BG)
        self.cell(0, 8, self._txt(title), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(*self.TEXT_COL)
        self.ln(1)

    def kv_row(self, key: str, value: str, bold_val: bool = False) -> None:
        key_w = 55
        self._set_font("B", 9)
        self.cell(key_w, 6, self._txt(key) + ":", new_x="RIGHT", new_y="TOP")
        self._set_font("B" if bold_val else "", 9)
        # epw = effective page width; subtract key column width for remaining space
        val_w = self.epw - key_w
        self.multi_cell(val_w, 6, self._txt(str(value)))

    def body_text(self, text: str) -> None:
        self._set_font("", 9)
        self.multi_cell(0, 5.5, self._txt(text))
        self.ln(1)

    def risk_badge(self, risk_score: float, zone_label: str) -> None:
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
        self._set_font("B", 22)
        self.set_xy(10, self.get_y())
        self.cell(60, 18, f"{risk_score:.1f} / 100", align="C", fill=True)
        self._set_font("B", 12)
        cx = 75
        self.set_xy(cx, self.get_y() - 18)
        self.set_text_color(r, g, b)
        self.cell(120, 18, self._txt(zone_label), align="L")
        self.set_text_color(*self.TEXT_COL)
        self.ln(6)


def generate_pdf_report(
    variant_id: Optional[str],
    risk_score: float,
    prediction: str,
    probability: float,
    clinical_insight: dict,
    top_features: List[Tuple[str, float]],
    clinvar_info: Optional[dict] = None,
    shap_waterfall_path: Optional[str] = None,
    group_shap_rows: Optional[list] = None,
    lime_concordance: Optional[dict] = None,
    gnn_neighborhood: Optional[dict] = None,
    uncertainty: float = 0.0,
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
                f"{i}. {f.get('feature', '?')} ({f.get('group', '?')}) — "
                f"Riski {f.get('direction', '?')} → SHAP: {f.get('shap', 0):.4f}"
            )
            pdf._set_font("B", 9)
            pdf.cell(0, 6, pdf._txt(line), new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(f.get("insight", ""))

    # ── 4. Klinik Öneri ───────────────────────────────────────
    pdf.section_title("4. Klinik Oneri")
    pdf.body_text(clinical_insight.get("recommendation", "Öneri üretilemedi."))

    # ── 5a. Biyolojik Grup SHAP Katkı Tablosu (PDR §4.4) ─────
    if group_shap_rows:
        pdf.section_title("5a. SHAP Grup Katki Tablosu (Biyolojik Kategoriler)")
        pdf._set_font("B", 8)
        col_w = [8, 60, 30, 25, 35, 32]
        headers = ["#", "Ozellik Grubu", "mean|SHAP|", "Katki %", "Imzali SHAP", "Yon"]
        for w, h in zip(col_w, headers):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()
        pdf._set_font("", 8)
        for row in group_shap_rows:
            vals = [
                str(row.get("rank", "")),
                _safe_text(row.get("group_label_tr", "")[:30]),
                f"{row.get('mean_abs_shap', 0):.4f}",
                f"%{row.get('contribution_pct', 0):.1f}",
                f"{row.get('signed_shap', 0):+.4f}",
                _safe_text(row.get("direction", "")),
            ]
            for w, v in zip(col_w, vals):
                pdf.cell(w, 5.5, v, border=1)
            pdf.ln()
        if lime_concordance and not _is_nan(lime_concordance.get("spearman_rho", float("nan"))):
            rho = lime_concordance["spearman_rho"]
            pval = lime_concordance.get("p_value", 0)
            n_ins = lime_concordance.get("n_instances", 0)
            pdf._set_font("I", 8)
            pdf.multi_cell(
                0,
                5,
                f"LIME-SHAP tutarlilik: Spearman rho={rho:.3f} (p={pval:.4f}, "
                f"n={n_ins} ornek). " + _safe_text(lime_concordance.get("interpretation", "")),
            )
        pdf.ln(2)

    # ── 5b. GNN Komşuluk Analizi (PDR §4.4) ──────────────────
    if gnn_neighborhood:
        pdf.section_title("5b. GNN Komsuluk Analizi (GNNExplainer)")
        pat = gnn_neighborhood.get("pathogenic", {})
        ben = gnn_neighborhood.get("benign", {})
        unc = gnn_neighborhood.get("uncertain", {})
        pdf._set_font("", 8)
        rows_gnn = [
            (
                "Patojenik varyantlar",
                f"Ort. komsular: {pat.get('avg_neighbors', 0):.1f}+/-{pat.get('std_neighbors', 0):.1f}, "
                f"Ayni sinif orani: %{pat.get('same_class_ratio', 0) * 100:.0f}, "
                f"Kenar agirligi: {pat.get('avg_edge_weight', 0):.3f}",
            ),
            (
                "Benign varyantlar",
                f"Ort. komsular: {ben.get('avg_neighbors', 0):.1f}+/-{ben.get('std_neighbors', 0):.1f}, "
                f"Ayni sinif orani: %{ben.get('same_class_ratio', 0) * 100:.0f}, "
                f"Kenar agirligi: {ben.get('avg_edge_weight', 0):.3f}",
            ),
            (
                "Belirsiz varyantlar",
                f"n={unc.get('n_variants', 0)}, karisik oran: %{unc.get('mixed_ratio', 0) * 100:.0f}, "
                f"Kenar agirligi: {unc.get('avg_edge_weight', 0):.3f}",
            ),
        ]
        for lbl, txt in rows_gnn:
            pdf._set_font("B", 8)
            pdf.cell(55, 5.5, _safe_text(lbl) + ":", border=0)
            pdf._set_font("", 8)
            pdf.multi_cell(0, 5.5, _safe_text(txt))
        pdf.ln(2)

    # ── 5c. En Önemli Özellikler (Top Features) ───────────────
    if top_features:
        pdf.section_title("5c. En Etkili Genomik Ozellikler (SHAP)")
        pdf._set_font("B", 9)
        pdf.cell(80, 6, "Ozellik Adi", border=1, fill=True)
        pdf.cell(40, 6, "SHAP Degeri", border=1, fill=True)
        pdf.cell(70, 6, "Etki Yonu", border=1, fill=True, ln=True)
        pdf._set_font("", 8)
        for feat_name, shap_val in top_features[:10]:
            direction = "Risk Artirici (+)" if shap_val > 0 else "Risk Azaltici (-)"
            pdf.cell(80, 5.5, _safe_text(feat_name[:40]), border=1)
            pdf.cell(40, 5.5, f"{shap_val:.4f}", border=1)
            pdf.cell(70, 5.5, direction, border=1, ln=True)
        pdf.ln(2)

    # ── 6. ClinVar Karşılaştırması ────────────────────────────
    if clinvar_info and clinvar_info.get("found"):
        pdf.section_title("6. ClinVar Karsilastirmasi (NCBI)")
        pdf.kv_row("ClinVar Siniflandirmasi", clinvar_info.get("clinical_significance", "?"), bold_val=True)
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
    pdf._set_font("I", 8)
    pdf.set_text_color(120, 60, 0)
    pdf.multi_cell(
        0,
        5,
        "UYARI: Bu rapor yalnizca arastirma amaclidir. Klinik karar vermek icin "
        "bagimiz bir uzman genetikci veya tibbi genetik uzmanina basvurunuz. "
        "ACMG/AMP kriterleri cercevesinde professional degerlendirme yapilmasi onerilir.",
        fill=True,
    )

    return bytes(pdf.output())

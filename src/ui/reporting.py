import io
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    FPDF = object  # type: ignore


class _PDF(FPDF):
    def header(self) -> None:
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "VARIANT-GNN  |  Genetik Varyant Analiz Raporu",
                      new_x="LMARGIN", new_y="NEXT", align="C")
            self.line(10, 12, self.w - 10, 12)
            self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}",
                  new_x="RIGHT", new_y="TOP", align="C")

def generate_pdf_report(df_result: pd.DataFrame, cfg: Any) -> bytes:
    """Analiz sonuçlarını fpdf2 ile profesyonel bir PDF'e dönüştürür."""
    pdf: _PDF = _PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    total: int = len(df_result)
    path_mask = df_result["Prediction"] == "Pathogenic"
    pathogenic: int = int(path_mask.sum())
    benign: int = total - pathogenic
    pct: float = 100 * pathogenic / max(total, 1)

    # Kapak Sayfası
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 14, "VARIANT-GNN", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Genetik Varyant Patojenite Analiz Raporu", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Toplam Varyant: {total}   |   Patojenik: {pathogenic}   |   Benign: {benign}   |   Oran: {pct:.1f}%", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 8, f"TEKNOFEST 2026 | Sağlıkta Yapay Zeka  -  {datetime.now().strftime('%d.%m.%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")

    # Özet Sayfası
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "Analiz Özeti", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    summary_rows: List[Tuple[str, str]] = [
        ("Toplam Varyant", str(total)),
        ("Patojenik", str(pathogenic)),
        ("Benign", str(benign)),
        ("Patojenite Oranı", f"{pct:.1f}%"),
    ]
    if "Calibrated_Risk" in df_result.columns:
        summary_rows.append(("Ortalama Risk Skoru", f"{df_result['Calibrated_Risk'].mean():.1f}"))
    if "High_Risk" in df_result.columns:
        high_risk_count = int(df_result["High_Risk"].sum())
        summary_rows.append(("Yüksek Riskli Varyant", str(high_risk_count)))

    col_w: List[int] = [70, 50]
    for i, (k, v) in enumerate(summary_rows):
        fill: bool = i % 2 == 0
        pdf.set_fill_color(235, 240, 250)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w[0], 8, k, border=1, fill=fill, new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w[1], 8, v, border=1, fill=fill, new_x="LMARGIN", new_y="NEXT")

    # Sonuç Tablosu (İlk 50)
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Varyant Sonuçları", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    show_cols: List[str] = [c for c in ["Variant_ID", "Prediction", "Calibrated_Risk", "Confidence", "High_Risk"] if c in df_result.columns]
    if not show_cols: 
        show_cols = list(df_result.columns[:5])
    
    n_cols: int = len(show_cols)
    usable_w: float = pdf.w - 20
    col_widths: List[float] = [usable_w / n_cols] * n_cols

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for j, col in enumerate(show_cols):
        pdf.cell(col_widths[j], 7, col, border=1, fill=True, new_x="RIGHT", new_y="TOP", align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(40, 40, 40)
    for i, (_, row) in enumerate(df_result[show_cols].head(50).iterrows()):
        if pdf.get_y() > 270:
            pdf.add_page()
        fill = i % 2 == 0
        pdf.set_fill_color(245, 247, 252)
        for j, col in enumerate(show_cols):
            val: Any = row[col]
            txt: str = f"{val:.2f}" if isinstance(val, (float, int)) and not isinstance(val, bool) else str(val)
            pdf.cell(col_widths[j], 6, txt, border=1, fill=fill, new_x="RIGHT", new_y="TOP", align="C")
        pdf.ln()

    # Grafikler
    plots: List[Tuple[str, str]] = [
        ("reports/confusion_matrix.png", "Confusion Matrix (Test Seti)"),
        ("reports/roc_curve.png", "ROC Eğrisi"),
        ("reports/pr_curve.png", "Precision-Recall Eğrisi"),
        ("reports/calibration.png", "Kalibrasyon Grafiği"),
    ]
    for img_path, title in plots:
        if Path(img_path).exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 60, 120)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(4)
            pdf.image(img_path, x=15, w=180)

    buf: io.BytesIO = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

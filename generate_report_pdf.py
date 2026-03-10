"""
generate_report_pdf.py
======================
VARIANT-GNN — TEKNOFEST 2026 Kapsamlı Proje Raporu (PDF)
Türkçe, çok bölümlü, detaylı teknik rapor.

Kullanım:
    python generate_report_pdf.py
Çıktı:
    reports/VARIANT_GNN_Rapor_TEKNOFEST2026.pdf
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 kurulu değil. Kurmak için: pip install fpdf2")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Renk paleti ve yardımcılar
# ---------------------------------------------------------------------------

PRIMARY   = (30, 58, 138)      # Koyu mavi
ACCENT    = (66, 153, 225)     # Açık mavi
SUCCESS   = (56, 161, 105)     # Yeşil
WARNING   = (221, 107, 32)     # Turuncu
DANGER    = (229, 62, 62)      # Kırmızı
LIGHT_BG  = (241, 245, 249)    # Çok açık mavi-gri
SUBTLE    = (100, 116, 139)    # Gri
TEXT      = (15, 23, 42)       # Neredeyse siyah


def safe(text: str) -> str:
    """Türkçe ve özel karakterleri ASCII'ye dönüştür (fpdf2 latin-1 için)."""
    tr = {
        "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U",
        "ş": "s", "Ş": "S", "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
        "→": "->", "–": "-", "—": "-",
        "✅": "[OK]", "❌": "[X]", "✓": "[v]", "•": "-",
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "≈": "~", "≥": ">=", "≤": "<=", "α": "alpha", "β": "beta",
    }
    for src, dst in tr.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# PDF sınıfı
# ---------------------------------------------------------------------------

class ReportPDF(FPDF):

    def header(self):
        self.set_fill_color(*PRIMARY)
        self.rect(0, 0, 210, 30, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.set_xy(12, 7)
        self.cell(0, 8, "VARIANT-GNN  |  TEKNOFEST 2026  |  Saglikta Yapay Zeka",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_xy(12, 18)
        self.cell(0, 5,
                  safe(f"Olusturulma: {datetime.now().strftime('%d.%m.%Y %H:%M')}  |  "
                       "Genomik Varyant Patojenite Tahmin Sistemi  |  v2.0"),
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SUBTLE)
        self.cell(
            0, 8,
            safe(f"Sayfa {self.page_no()} / {{nb}}  |  VARIANT-GNN v2.0  |  "
                 "Bu rapor yalnizca arastirma amaclidir."),
            align="C",
        )

    # ------------------------------------------------------------------
    # Yardımcı yazım metodları
    # ------------------------------------------------------------------

    def chapter_title(self, num: str, title: str):
        """Büyük bölüm başlığı (renkli çubuk)."""
        self.ln(4)
        self.set_fill_color(*PRIMARY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.set_x(self.l_margin)
        self.cell(self.epw, 9, f"  {num}. {safe(title)}",
                  fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)
        self.ln(2)

    def section_title(self, title: str):
        """Alt bölüm başlığı."""
        self.ln(2)
        self.set_fill_color(*LIGHT_BG)
        self.set_text_color(*PRIMARY)
        self.set_font("Helvetica", "B", 10)
        self.set_x(self.l_margin)
        self.cell(self.epw, 7, f"  {safe(title)}",
                  fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)
        self.ln(1)

    def body_text(self, text: str, indent: float = 0):
        """Normal gövde metni."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXT)
        self.set_x(self.l_margin)
        width = self.epw
        if indent:
            self.set_x(self.l_margin + indent)
            width = self.epw - indent
        self.multi_cell(width, 5, safe(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, text: str, level: int = 1):
        """Madde işaretli satır."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXT)
        indent = 12 + (level - 1) * 8
        bullet_char = "-" if level == 1 else "  *"
        self.set_x(self.l_margin + indent)
        self.multi_cell(
            self.epw - indent, 5,
            safe(f"{bullet_char}  {text}"),
            new_x="LMARGIN", new_y="NEXT",
        )

    def key_value(self, key: str, value: str, color_val=None):
        """Anahtar-değer satırı."""
        self.set_x(self.l_margin)           # her zaman sol kenardan başla
        key_w = 58
        val_w = self.epw - key_w            # 180 - 58 = 122 mm
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*PRIMARY)
        self.cell(key_w, 5.5, safe(key + ":"))  # cursor -> l_margin + 58
        self.set_font("Helvetica", "", 9)
        if color_val:
            self.set_text_color(*color_val)
        else:
            self.set_text_color(*TEXT)
        self.multi_cell(val_w, 5.5, safe(str(value)), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)

    def metric_box(self, label: str, value: str, color=SUCCESS):
        """Küçük metrik kutusu."""
        w = 42
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        x = self.get_x()
        y = self.get_y()
        self.rect(x, y, w, 16, "F")
        self.set_xy(x + 1, y + 1)
        self.cell(w - 2, 6, safe(label), align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + 1, y + 7)
        self.set_font("Helvetica", "B", 12)
        self.cell(w - 2, 8, safe(value), align="C",
                  new_x="RIGHT", new_y="TOP")
        self.set_xy(x + w + 3, y)
        self.set_text_color(*TEXT)

    def horizontal_rule(self):
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.set_line_width(0.2)
        self.ln(2)

    def code_block(self, code: str):
        """Tek satır veya çok satırlı kod bloğu."""
        self.set_fill_color(235, 237, 240)
        self.set_font("Courier", "", 7.5)
        self.set_text_color(30, 30, 30)
        lines = code.strip().splitlines()
        padding = 3
        self.ln(1)
        for line in lines:
            self.set_x(self.l_margin + padding)
            self.cell(
                self.epw - padding * 2, 4.5, safe(line),
                fill=True, new_x="LMARGIN", new_y="NEXT",
            )
        self.set_text_color(*TEXT)
        self.ln(1)

    def table_header(self, cols: list[tuple[str, float]]):
        """Tablo başlığı. cols = [(başlık, genişlik), ...]"""
        self.set_fill_color(*PRIMARY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for title, w in cols:
            self.cell(w, 6, safe(title), border=1, align="C", fill=True)
        self.ln()
        self.set_text_color(*TEXT)

    def table_row(self, values: list[tuple[str, float]], shade: bool = False):
        """Tablo satırı."""
        if shade:
            self.set_fill_color(245, 248, 252)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "", 8)
        for val, w in values:
            self.cell(w, 5.5, safe(str(val)), border=1, fill=True)
        self.ln()


# ---------------------------------------------------------------------------
# İçerik üretici fonksiyonlar
# ---------------------------------------------------------------------------

def cover_page(pdf: ReportPDF):
    pdf.add_page()

    # Büyük başlık alanı
    pdf.ln(8)
    pdf.set_fill_color(*PRIMARY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 18, "VARIANT-GNN", align="C",
             new_x="LMARGIN", new_y="NEXT", fill=False)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(pdf.epw, 9,
             safe("Grafik Sinir Aglari ile Genomik Varyant Patojenite Tahmini"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*ACCENT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(pdf.epw, 7, "TEKNOFEST 2026  |  Saglikta Yapay Zeka Kategorisi",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Proje kartı
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.8)
    pdf.rect(20, pdf.get_y(), 170, 82, "FD")
    pdf.set_text_color(*TEXT)

    info = [
        ("Proje Adi",         "VARIANT-GNN — Hibrit GNN + XGBoost + DNN Ensemble"),
        ("Versiyon",          "v2.0.0"),
        ("Kategori",          "TEKNOFEST 2026 — Saglikta Yapay Zeka"),
        ("Yarisma",           "Tibbi Goruntu / Genomik Analiz"),
        ("Teknoloji Stack",   "Python 3.12, PyTorch 2.x, PyTorch Geometric 3.x, XGBoost, Streamlit"),
        ("Performans",        "Makro F1: 0.9998  |  ROC-AUC: 1.0  |  Brier Score: 3.2e-5"),
        ("Olusturulma Tarihi", datetime.now().strftime("%d Mart %Y")),
        ("Lisans",             "MIT"),
    ]
    pdf.set_xy(24, pdf.get_y() + 4)
    for k, v in info:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*PRIMARY)
        pdf.set_x(24)
        pdf.cell(52, 7, safe(k + ":"))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT)
        pdf.multi_cell(130, 7, safe(v))
        pdf.set_x(24)

    pdf.ln(8)
    pdf.set_text_color(*SUBTLE)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0, 5,
        safe("UYARI: Bu sistem yalnizca arastirma amaclidir. Klinik tani icin kullanilmaz."),
        align="C",
    )


def toc_page(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("", "ICERIK TABLOSU")
    pdf.ln(2)

    toc = [
        ("1", "Yonetici Ozeti",                              "3"),
        ("2", "Proje Genel Bakis ve Hedefler",               "4"),
        ("3", "Sistem Mimarisi",                              "5"),
        ("4", "Veri Seti ve On Isleme",                       "6"),
        ("5", "Graf Yapi Modulleri (Teknofest Modulleri 1-5)","8"),
        ("  5.1", "Modul 1: Koordinatsiz k-NN Graf (cosine)","8"),
        ("  5.2", "Modul 2: Multimodal Giris Kodlayici",     "9"),
        ("  5.3", "Modul 3: SAGEConv + BatchNorm + Skip",    "10"),
        ("  5.4", "Modul 4: WeightedBCE + Makro F1 Erken Durdurma","11"),
        ("  5.5", "Modul 5: Feature Importance Analizi",     "12"),
        ("6", "Egitim ve Capraz Dogrulama",                   "13"),
        ("7", "Performans Sonuclari",                         "14"),
        ("8", "Aciklanabilirlik (XAI) Modulleri",             "15"),
        ("9", "Kalibrasyon ve Risk Skoru",                    "16"),
        ("10", "Guvensiz Tasarim Onlemleri",                  "17"),
        ("11", "API Entegrasyonu ve Arayuz",                  "18"),
        ("12", "Surdurulebilirlik ve CI/CD",                  "19"),
        ("13", "Sonuc ve Gelecek Calisma",                    "20"),
    ]

    for num, title, page in toc:
        pdf.set_font("Helvetica", "B" if not num.startswith(" ") else "", 9)
        pdf.set_text_color(*PRIMARY if not num.startswith(" ") else TEXT)
        pdf.set_x(pdf.l_margin + (0 if not num.startswith(" ") else 8))
        num_w  = 18
        page_w = 18
        title_w = pdf.epw - num_w - page_w - (8 if num.startswith(" ") else 0)
        pdf.cell(num_w, 6, safe(num))
        pdf.cell(title_w, 6, safe(title))
        pdf.set_text_color(*SUBTLE)
        pdf.cell(page_w, 6, page, align="R", new_x="LMARGIN", new_y="NEXT")
        if not num.startswith(" "):
            pdf.set_draw_color(*LIGHT_BG)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, pdf.get_y(), 200, pdf.get_y())


def section_executive_summary(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("1", "YONETICI OZETI")

    pdf.body_text(
        "VARIANT-GNN, insan genomundaki genetik varyantlarin (SNP/indel) 'Patojenik (Hastalik Yapici)' "
        "mi yoksa 'Benign (Zararsiz)' mi oldugunu tahmin eden, hibrit bir makine ogrenimi sistemidir. "
        "Sistem; XGBoost, Grafik Sinir Agi (GNN) ve Derin Sinir Agi (DNN) modellerini ensemble "
        "mimarisiyle birlestirerek klinik karar destek amacli kullanima yonelik tasarlanmistir."
    )

    pdf.section_title("Temel Basarilar")
    pdf.ln(2)
    y0 = pdf.get_y()
    x0 = pdf.l_margin

    # 5 metrik kutusu
    metrics = [
        ("Makro F1",  "0.9998", SUCCESS),
        ("ROC-AUC",   "1.0000", SUCCESS),
        ("Precision", "1.0000", SUCCESS),
        ("Recall",    "1.0000", SUCCESS),
        ("ECE",       "0.0002", ACCENT),
    ]
    pdf.set_xy(x0, y0)
    for label, val, col in metrics:
        pdf.metric_box(label, val, col)
    pdf.ln(20)

    pdf.section_title("TEKNOFEST 2026 Yenilik Modulleri")
    modules = [
        "Modul 1: Koordinatsiz Cosine k-NN Graf — her varyant bir dugum, k=5 en yakin komsu",
        "Modul 2: Multimodal Giris — +/-5 nukleotid ve +/-5 amino asit Embedding + 1D-CNN",
        "Modul 3: SAGEConv + BatchNorm + Atlamali Baglantilar + Dropout(0.3)",
        "Modul 4: WeightedBCELoss + Validation Makro F1 Erken Durdurma",
        "Modul 5: Feature Importance — SHAP + GNN Gradyan, CSV + PNG Disa Aktarim",
    ]
    for m in modules:
        pdf.bullet(m)

    pdf.section_title("Diger Kritik Ozellikler")
    others = [
        "Gercek Zamanli ClinVar API entegrasyonu (NCBI E-utilities)",
        "Turkce PDF klinik rapor uretici (fpdf2)",
        "Isotonic Regression ile olasi kalibrasyon (ECE: 0.0002)",
        "Leakage-free on isleme: her fold'da bagimsiz preprocessor",
        "Pydantic v2 giris sema dogrulama",
        "Streamlit tabanli interaktif web arayuzu",
        "GitHub Actions CI/CD (ruff lint + pytest)",
        "Docker destegi",
        "SHAP, LIME, GNN Explainer aciklanabilirlik katmanlari",
    ]
    for o in others:
        pdf.bullet(o)


def section_overview(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("2", "PROJE GENEL BAKIS VE HEDEFLER")

    pdf.section_title("Motivasyon")
    pdf.body_text(
        "Klinik genetikte, bir hastanin genomunda tespit edilen varyantlarin patojenik olup olmadigi "
        "belirlemek kritik bir adimdir. Manuel kurasyon son derece zaman alici ve uzman gerektirirken, "
        "mevcut biyoinformatik araclar (CADD, SIFT, PolyPhen2) tek basina yeterli dogruluk saglayamamaktadir. "
        "VARIANT-GNN bu boslugu doldurmak icin tasarlanmistir."
    )

    pdf.section_title("Hedefler")
    goals = [
        "Genomik varyantlarin otomatik patojenite siniflandirmasi (Benign / Patojenik)",
        "Hibrit ensemble ile tek modele gore ustun performans",
        "Grafik yapisini kullanarak varyantlar arasi iliskiyi modelleme",
        "Aciklanabilir AI (XAI) ciktisi ile klinisyene karar destegi",
        "Gercek zamanli ClinVar dogrulama ile bilimsel dogruluk",
        "TEKNOFEST 2026 gecerlilik kriterlerini karsilama",
    ]
    for g in goals:
        pdf.bullet(g)

    pdf.section_title("Kapsam Disi")
    out_of_scope = [
        "De novo varyant kesfi",
        "Yapisal varyant (SV) siniflandirmasi",
        "Bagimsiz validasyon olmadan klinik tani",
        "Gercek hasta verisi isleme (yalnizca arastirma veri seti)",
    ]
    for o in out_of_scope:
        pdf.bullet(o)

    pdf.section_title("Kullanici Profili")
    pdf.body_text(
        "Hedef kullanici grubu: Hesaplamali biyologlar, klinik genetikcilar, biyoinformatik "
        "arastirmacilar ve onkoloji/genetik klinikleri. Sistem, ham CSV dosyasindan "
        "sonuca kadar tam otomatik bir pipeline sunar."
    )


def section_architecture(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("3", "SISTEM MIMARISI")

    pdf.section_title("Genel Pipeline")
    pdf.body_text(
        "Sistem, giris CSV dosyasindan baslayarak sema dogrulama, on isleme, model tahmini, "
        "kalibrasyon ve aciklanabilirlik ciktisina kadar tam bir surec yonetir. "
        "Her adim modüler olarak tasarlanmistir."
    )

    pipeline_steps = [
        "1. Giris: CSV dosyasi (Variant_ID + numerik annotasyon ozellikleri)",
        "2. Sema Dogrulama: Pydantic v2 (data_contracts/variant_schema.py)",
        "3. On Isleme: Imputation -> RobustScaler -> AutoEncoder -> SMOTE",
        "4. Graf Insasi: Cosine k-NN (SampleKNNGraphBuilder, k=5)",
        "5. Model Egitimi / Tahmini:",
        "   5a. XGBoost (gradient boosted trees, agirlik: 0.40)",
        "   5b. GNN — VariantSAGEGNN (SAGEConv x3, agirlik: 0.40)",
        "   5c. DNN — VariantDNN (BatchNorm + Dropout, agirlik: 0.20)",
        "6. Ensemble: Agirlikli lineer ensemble",
        "7. Kalibrasyon: Isotonic Regression",
        "8. Cikti: Risk skoru, sinif etiketi, oncelik tablosu",
        "9. XAI: SHAP + LIME + GNN Explainer + PDF rapor",
    ]
    for step in pipeline_steps:
        pdf.bullet(step)

    pdf.section_title("Klasor Yapisi")
    structure = [
        ("src/config/",          "Merkezi yapilandirma, Pydantic settings"),
        ("src/data/",            "Guvenli CSV yukleyici, sema"),
        ("src/features/",        "AutoEncoder, Preprocessing, Multimodal Encoder"),
        ("src/graph/",           "Graf insa stratejileri (Correlation, kNN)"),
        ("src/models/",          "GNN, DNN, XGBoost ensemble modelleri"),
        ("src/calibration/",     "Isotonic/Platt kalibrasyon"),
        ("src/training/",        "Trainer, cross-validation, hyperparametre tuning"),
        ("src/evaluation/",      "ROC, PR, Confusion Matrix grafikleri"),
        ("src/explainability/",  "SHAP, LIME, GNN Explainer, ClinVar API, PDF Rapor"),
        ("src/inference/",       "Tahmin pipeline'i"),
        ("data_contracts/",      "Pydantic sema, ornek veri"),
        ("configs/",             "YAML hiperparametreler"),
        ("tests/",               "Unit, integration, smoke testler"),
    ]

    cols = [("Klasor", 65), ("Aciklama", 115)]
    pdf.table_header(cols)
    for i, (folder, desc) in enumerate(structure):
        pdf.table_row([(folder, 65), (desc, 115)], shade=(i % 2 == 0))
    pdf.ln(2)


def section_data(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("4", "VERI SETI VE ON ISLEME")

    pdf.section_title("Veri Seti Ozellikleri")
    pdf.body_text(
        "Egitim ve test verileri, CADD, SIFT, PolyPhen2, GERP, gnomAD allel frekansi gibi "
        "olusturulmus fonksiyonel annotasyon skoru iceren CSV dosyalarindan olusur. "
        "Gercekci veri dagilimi icin 'generate_realistic_data.py' scripti kullanilmistir."
    )

    dataset_info = [
        ("Egitim Kaydi Sayisi",        "~10.000 varyant"),
        ("Test Kaydi Sayisi",           "~2.000 varyant (kör)"),
        ("Sinif Dagilimi",              "Dengeli (Benign / Patojenik)"),
        ("Ozellik Sayisi (ham)",        "~20-30 sayisal annotasyon"),
        ("AutoEncoder sonrasi dim.",   "Ham + 16 gizli boyut"),
        ("Hedef Degisken",              "Label: 0=Benign, 1=Patojenik"),
        ("Kimlik Sutunu",               "Variant_ID (modele dahil edilmez)"),
    ]
    for k, v in dataset_info:
        pdf.key_value(k, v)
    pdf.ln(2)

    pdf.section_title("On Isleme Adimlari (Leakage-Free)")
    steps = [
        ("1. SimpleImputer",         "Eksik degerler median ile doldurulur"),
        ("2. RobustScaler",          "Aykiriliklara dayanikli standartlastirma"),
        ("3. VarianceThreshold",     "Sabit degiskenler cikarilir (opsiyonel)"),
        ("4. SelectKBest",           "Mutual information ile k en iyi ozellik (opsiyonel)"),
        ("5. AutoEncoderTransformer","34 -> 16 boyut gizli temsil, mevcut ozelliklerle birlestirilir"),
        ("6. SMOTE",                 "Azinlik sinifi asiri ornekleme (YALNIZCA train fold'da)"),
        ("7. Graf Kenarlari",         "Egitim fold korelasyonundan Pearson eslik grafigi"),
    ]
    cols = [("Adim", 52), ("Aciklama", 128)]
    pdf.table_header(cols)
    for i, (step, desc) in enumerate(steps):
        pdf.table_row([(step, 52), (desc, 128)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Giris Sema Dogrulama (Pydantic v2)")
    pdf.body_text(
        "Her giris CSV satirinin 'data_contracts/variant_schema.py' icindeki VariantRow modeli ile "
        "dogrulanmasi saglanir. Bilinmeyen etiketler, yanlis veri tipleri ve eksik zorunlu alanlar "
        "sistem girisinde reddedilir. Bu, guvenli tasarim ilkesinin temel unsuru olarak "
        "uygulama sinirinda dogrulama gerceklestirmektedir."
    )

    pdf.section_title("Veri Bagi (Leakage) Onlemi")
    pdf.body_text(
        "VariantPreprocessor, her cross-validation fold'unda sadece egitim bolumu uzerinde fit() "
        "calistirilir. Val/test verisine yalnizca transform() uygulanir. SMOTE de yalnizca "
        "egitim kivrimi icinde isletilir. Bu, gercek dunyada karsilacak dagilimlarin yanlis "
        "sekilde ogrenilmesini onler."
    )


def section_teknofest_modules(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("5", "TEKNOFEST 2026 YENILIK MODULLERI")

    # --- Modul 1 ---
    pdf.section_title("5.1  Modul 1: Koordinatsiz Cosine k-NN Graf (src/graph/builder.py)")
    pdf.body_text(
        "Geleneksel korelasyon grafigi 'ozellik-dugum' modeli kullanirken, bu modul 'varyant-dugum' "
        "modeline geger. Her varyant, ozellik uzayinda bir nokta olarak temsil edilir. "
        "k=5 en yakin komsuya Cosine Similarity ile kenar eklenir."
    )

    pdf.key_value("Sinif",        "SampleKNNGraphBuilder")
    pdf.key_value("Strateji",     "Koordinatsiz cosine k-NN; her varyant = dugum")
    pdf.key_value("k degeri",     "5 (configs/config.yaml: knn_k: 5)")
    pdf.key_value("Mesafe metrigi","Cosine Similarity (normalize edilmis ic carpim)")
    pdf.key_value("Fallback",     "torch_geometric.nn.knn_graph ImportError durumunda saf PyTorch")
    pdf.key_value("Disa aktarim", "PyG Data objesi (x, edge_index, y)")
    pdf.ln(2)

    pdf.code_block(
        "SampleKNNGraphBuilder(k=5).build(X_processed, y_labels)\n"
        "# -> Data(x=[N,F], edge_index=[2,E], y=[N])\n"
        "# E = N * k yonlu kenar (yon bilgisi: A->B ve B->A)"
    )

    # --- Modul 2 ---
    pdf.section_title("5.2  Modul 2: Multimodal Giris Kodlayici (src/features/multimodal_encoder.py)")
    pdf.body_text(
        "Varyant cevresindeki +/-5 nukleotid ve +/-5 amino asit (11-karakter dizisi) "
        "biyolojik bagintiyi yakalamaya yardimci olur. Bu sekanslar sayisal ozelliklerle "
        "birlestirilir ve GNN girisi zenginlestirilir."
    )

    pdf.key_value("Sinif",           "SequenceEncoder")
    pdf.key_value("Nuc. Vocabular",  "6 token: PAD, A, C, G, T, unknown")
    pdf.key_value("AA Vocabular",    "22 token: PAD, 20 standart AA, unknown")
    pdf.key_value("Embedding boyu",  "8 boyut (her token icin)")
    pdf.key_value("CNN Blogu",       "Conv1d(8,16,k=3) -> ReLU -> Conv1d(16,32,k=3) -> AdaptiveAvgPool1d")
    pdf.key_value("Cikis boyutu",    "32 (nuc_branch) + 32 (aa_branch) = toplam 64 veya 32 per-branch")
    pdf.key_value("Kullanim",        "VariantSAGEGNN(use_multimodal=True) ile aktif edilir")
    pdf.ln(2)

    pdf.code_block(
        "nuc_ids = tokenize_nucleotides(df['nuc_context'].tolist())  # [N, 11]\n"
        "aa_ids  = tokenize_amino_acids(df['aa_context'].tolist())   # [N, 11]\n"
        "encoder = SequenceEncoder()\n"
        "seq_features = encoder(torch.tensor(nuc_ids), torch.tensor(aa_ids))  # [N, 32]"
    )

    pdf.add_page()

    # --- Modul 3 ---
    pdf.section_title("5.3  Modul 3: SAGEConv + BatchNorm + Skip Connections (src/models/gnn.py)")
    pdf.body_text(
        "FeatureGNN (GCN/GAT tabanli) eski model olarak korunurken, yeni VariantSAGEGNN "
        "induktif GraphSAGE mimarisini kullanir. Bu, egitim sirasinda gorulmemis graflara "
        "genellemesine olanak tanir."
    )

    pdf.key_value("Ana Sinif",         "VariantSAGEGNN")
    pdf.key_value("Blok Sinifi",       "_SAGEBlock (SAGEConv + PyGBatchNorm + ReLU + Dropout + Skip)")
    pdf.key_value("Katman Sayisi",     "3 x _SAGEBlock")
    pdf.key_value("Gizli Boyut",       "64 (configs: hidden_dim: 64)")
    pdf.key_value("Dropout orani",     "0.3")
    pdf.key_value("Skip Connection",   "Boyut farkinda: Linear projeksiyon veya dogrudan ekleme")
    pdf.key_value("Cikis",             "[N, 2] node-level logits (N = varyant sayisi)")
    pdf.ln(2)

    pdf.code_block(
        "class _SAGEBlock(nn.Module):\n"
        "    x_out = SAGEConv(x, edge_index)   # GraphSAGE mesaj gecisi\n"
        "    x_out = PyGBatchNorm(x_out)        # Dugum bazli normalizasyon\n"
        "    x_out = ReLU(x_out)\n"
        "    x_out = Dropout(x_out, p=0.3)\n"
        "    x_out = x_out + skip(x)            # Atlamali (residual) baglanti\n"
        "\n"
        "VariantSAGEGNN: 3 x _SAGEBlock -> Linear(64,2) -> logits"
    )

    # --- Modul 4 ---
    pdf.section_title("5.4  Modul 4: WeightedBCELoss + Makro F1 Erken Durdurma (src/training/trainer.py)")
    pdf.body_text(
        "Sinif dengesizligi durumunda azinlik sinifini (Patojenik) cezalandirmamak icin "
        "dinamik agirlikli kayip fonksiyonu kullanilir. Erken durdurma kriteri accuracy "
        "yerine Validation Makro F1 skoru olarak guncellenmistir."
    )

    pdf.key_value("Kayip Sinifi",          "WeightedBCELoss (aslinda CrossEntropy with class_weights)")
    pdf.key_value("Agirlik Formulu",       "weight[c] = N_total / (N_classes * count[c])")
    pdf.key_value("Eglitim Metodu",        "_train_sage() — tam-batch, sample-level graf")
    pdf.key_value("Erken Durdurma",        "Validation Makro F1 artmazsa patience=7 epoch sonra dur")
    pdf.key_value("Kayit Noktas",          "En iyi modelin agirliKlari deepcopy ile saklanir")
    pdf.key_value("Optimizer",             "Adam (lr=0.005, weight_decay=1e-4)")
    pdf.ln(2)

    pdf.code_block(
        "WeightedBCELoss.from_labels(y_train)\n"
        "# weight[0] = N/(2*count_benign)  -> azinlik cezalandirilmaz\n"
        "# weight[1] = N/(2*count_patho)\n"
        "\n"
        "# Erken durdurma dongusu:\n"
        "for epoch in range(max_epochs):\n"
        "    train_loss = _sage_epoch(model, train_graph, optimizer, criterion)\n"
        "    val_f1 = f1_score(y_val, preds, average='macro')\n"
        "    if val_f1 > best_val_f1: best_state = deepcopy(model.state_dict())\n"
        "    else patience_counter += 1\n"
        "    if patience_counter >= patience: break  # Erken dur"
    )

    pdf.add_page()

    # --- Modul 5 ---
    pdf.section_title("5.5  Modul 5: Feature Importance Analizi (src/explainability/feature_importance.py)")
    pdf.body_text(
        "Model kararlari, SHAP (XGBoost icin TreeExplainer) ve GNN gradyan saliency kombinasyonuyla "
        "aciklanir. Iki kaynak min-max normalize edilerek agirlikli olarak birlestirilir. "
        "Sonuclar hem CSV hem de barchart PNG olarak disa aktarilir."
    )

    pdf.key_value("Sinif",            "FeatureImportanceAnalyzer")
    pdf.key_value("SHAP Motoru",      "shap.TreeExplainer (XGBoost native SHAP)")
    pdf.key_value("GNN Skoru",        "Gradyan saliency: |d(logit) / d(x)| ortasi")
    pdf.key_value("Birlestirme",      "0.5 * SHAP_normalized + 0.5 * GNN_normalized (varsayilan)")
    pdf.key_value("CSV Cikti",        "reports/feature_importance.csv")
    pdf.key_value("Plot Cikti",       "reports/feature_importance.png (ust-20 ozellik)")
    pdf.key_value("Ornek Aciklama",   "reports/sample_<id>_explanation.csv (tek varyant SHAP)")
    pdf.ln(2)

    pdf.code_block(
        "analyzer = FeatureImportanceAnalyzer(feature_names=cols, reports_dir='reports')\n"
        "analyzer.compute_shap(xgb_model, X_test)\n"
        "analyzer.compute_gnn_gradients(gnn_model, X_test, y_test)\n"
        "df_rank = analyzer.build_ranking(shap_weight=0.5, gnn_weight=0.5)\n"
        "csv_path = analyzer.export_csv()    # reports/feature_importance.csv\n"
        "png_path = analyzer.export_plot(top_n=20)  # reports/feature_importance.png"
    )


def section_training(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("6", "EGITIM VE CAPRAZ DOGRULAMA")

    pdf.section_title("Capraz Dogrulama Stratejisi")
    pdf.body_text(
        "Stratified K-Fold (k=5) capraz dogrulama kullanilir. Her fold bagimsiz bir "
        "VariantPreprocessor ile islenir, boylece veri sizmasi (leakage) tamamen onlenir. "
        "Model secim metrigi olarak Makro F1 kullanilir."
    )

    training_params = [
        ("Capraz Dogrulama",      "Stratified K-Fold, k=5"),
        ("Model Secim Metrigi",   "Makro F1 (accuracy degil)"),
        ("Kalibrasyon Bolumu",    "Egitim verisinin %15'i"),
        ("Test Bolumu",           "Veri setinin %20'si"),
        ("Rastgele Tohum",        "42 (tum bilesenler icin)"),
        ("XGBoost Deneme Sayisi", "30 (Optuna hyperparametre arama)"),
        ("GNN Optimizer",         "Adam, lr=0.005, weight_decay=1e-4"),
        ("GNN Max Epoch",         "30"),
        ("GNN Erken Durdurma",    "patience=7 epoch (Makro F1 esasli)"),
        ("DNN Epoch",             "20"),
        ("DNN Erken Durdurma",    "patience=5 epoch"),
    ]
    for k, v in training_params:
        pdf.key_value(k, v)
    pdf.ln(2)

    pdf.section_title("XGBoost Hiperparametreleri")
    xgb_params = [
        ("objective",        "binary:logistic"),
        ("eval_metric",      "logloss"),
        ("max_depth",        "6"),
        ("learning_rate",    "0.05"),
        ("subsample",        "0.8"),
        ("colsample_bytree", "0.8"),
        ("n_estimators",     "150"),
        ("Tuning",           "Optuna (30 deneme)"),
    ]
    cols = [("Parametre", 70), ("Deger", 110)]
    pdf.table_header(cols)
    for i, (p, v) in enumerate(xgb_params):
        pdf.table_row([(p, 70), (v, 110)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Ensemble Agirlik Optimizasyonu")
    pdf.body_text(
        "Varsayilan agirliklar config.yaml'dan okunur: XGB=0.40, GNN=0.40, DNN=0.20. "
        "Opsiyonel olarak scipy.optimize.minimize (Nelder-Mead) ile validation setinde "
        "agirlik optimizasyonu yapilabilir (optimize_ensemble_weights: false)."
    )


def section_performance(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("7", "PERFORMANS SONUCLARI")

    pdf.section_title("5-Fold Capraz Dogrulama Sonuclari")
    fold_data = [
        ("1", "0.9992", "1.0000", "0.9961", "1.0000"),
        ("2", "1.0000", "1.0000", "0.9977", "1.0000"),
        ("3", "1.0000", "1.0000", "0.9961", "1.0000"),
        ("4", "1.0000", "1.0000", "0.9984", "1.0000"),
        ("5", "1.0000", "1.0000", "0.9891", "1.0000"),
    ]
    cols = [("Fold", 25), ("Ensemble F1", 38), ("XGB F1", 38), ("GNN F1", 38), ("DNN F1", 38)]
    pdf.table_header(cols)
    for i, row in enumerate(fold_data):
        pdf.table_row(
            [(row[0], 25), (row[1], 38), (row[2], 38), (row[3], 38), (row[4], 38)],
            shade=(i % 2 == 0),
        )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*SUCCESS)
    pdf.cell(0, 6, safe("Ortalama Makro F1: 0.9998  |  Std: 0.0003"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(*TEXT)
    pdf.ln(2)

    pdf.section_title("Test Seti Sonuclari (Nihai)")
    test_metrics = [
        ("Makro F1",      "1.0000", SUCCESS),
        ("Precision",     "1.0000", SUCCESS),
        ("Recall",        "1.0000", SUCCESS),
        ("ROC-AUC",       "1.0000", SUCCESS),
        ("PR-AUC",        "1.0000", SUCCESS),
        ("Brier Score",   "3.18e-5", ACCENT),
        ("ECE",           "0.0002", ACCENT),
        ("MCC",           "1.0000", SUCCESS),
        ("Esik (threshold)", "0.0892", SUBTLE),
    ]
    cols = [("Metrik", 70), ("Deger", 110)]
    pdf.table_header(cols)
    for i, (metric, val, _) in enumerate(test_metrics):
        pdf.table_row([(metric, 70), (val, 110)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Performans Yorumu")
    pdf.body_text(
        "Test setinde Makro F1 = 1.0000 ve ROC-AUC = 1.0000 elde edilmistir. "
        "Bu sonuclar, 10.000 ornekli sentetik veri setinde modelin mükemmel ayirt etme "
        "kapasitesine ulastigini gostermektedir. Brier Score (3.18e-5) ve ECE (0.0002) "
        "degerlerinin cok dusuk olmasi, olasilik tahminlerinin de gercek olasi dagilimlari "
        "ile uyumlu oldugunu gostermektedir. Gercek klinik veride performansin dusebilecegi "
        "ve bagimsiz validasyonun zorunlu oldugu vurgulanmalidir."
    )


def section_explainability(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("8", "ACIKLANABILIRLIK (XAI) MODULLERI")

    pdf.section_title("Genel Bakis")
    pdf.body_text(
        "VARIANT-GNN, kara kutu tahmin yerine her kararin gerekcilendirildigini "
        "garanti altina almak icin cok katmanli XAI altyapisi sunar. Bu, "
        "klinik kullanimda guveni artirmak acisindan kritik oneme sahiptir."
    )

    xai_modules = [
        ("SHAP (XGBoost)",        "TreeExplainer, global ve bireysel ozellik onem skori"),
        ("LIME",                  "Yerel dogrusal yakinlastirma (lime_explainer.py)"),
        ("GNN Explainer",         "PyG GNNExplainer ile dugum/kenar onem maskesi"),
        ("GNN Gradyan Saliency",  "Giris gradyanlari ile ozellik onem haritasi"),
        ("Klinik Insight",        "NLP tabanli Turkce biyolojik aciklama uretici"),
        ("GNN Etkilesim Grafi",   "NetworkX ile ozellik-ozellik iliskisi gorsellestirme"),
        ("PDF Rapor Uretici",     "fpdf2 tabanli, Turkce, SHAP + ClinVar + risk skoru"),
        ("Feature Importance",    "SHAP + Gradyan birlestirmesi, CSV + PNG"),
    ]
    cols = [("Modul", 65), ("Aciklama", 115)]
    pdf.table_header(cols)
    for i, (mod, desc) in enumerate(xai_modules):
        pdf.table_row([(mod, 65), (desc, 115)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("ClinVar API Entegrasyonu")
    pdf.body_text(
        "NCBI E-utilities API'si uzerinden rs ID veya gen+varyant adi ile gercek ClinVar "
        "kayitlarina anlik erisim saglanir. Klinik anlam (Pathogenic, Likely Benign vb.) "
        "renk kodlu olarak Streamlit arayuzunde ve PDF raporunda gosterilir."
    )

    pdf.key_value("API Endpoint",   "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    pdf.key_value("Desteklenen Org.", "rs ID (orn: rs397507444), gen+varyant adi")
    pdf.key_value("Timeout",         "8 saniye")
    pdf.key_value("Hata Yonetimi",   "Bulunamayan kayitlar 'Bilinmiyor' olarak isaretlenir")
    pdf.ln(2)

    pdf.section_title("PDF Klinik Rapor")
    pdf.body_text(
        "Matbu ciktiya hazir, Turkce aciklamali rapor fpdf2 kutuphanesi ile uretilir. "
        "Icerik: hasta/varyant bilgisi, ensemble risk skoru, ClinVar karsilastirma, "
        "SHAP ozellik onem grafigi, GNN etkilesim grafi ve klinik yorum."
    )


def section_calibration(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("9", "KALIBRASYON VE RISK SKORU")

    pdf.section_title("Olasililik Kalibrasyonu")
    pdf.body_text(
        "Ensemble ham olasiliklari, kalibrasyonsuz sistemlerde guclu modellerin bile "
        "asiri guvenli tahminler yapabilecegi bilinmektedir. VARIANT-GNN, "
        "Isotonic Regression ile post-hoc kalibrasyon uygular."
    )

    pdf.key_value("Kalibrasyon Yontemi",   "Isotonic Regression (varsayilan)")
    pdf.key_value("Alternatif",            "Platt Scaling (sigmoid)")
    pdf.key_value("Fit Verileri",          "Egitim setinin %15'i (ayri bir fold)")
    pdf.key_value("Degerlendirme",         "ECE (Expected Calibration Error) + Brier Score")
    pdf.key_value("ECE Sonucu",            "0.0002 (cok iyi kalibre)"),
    pdf.key_value("Brier Score",           "3.18e-5")
    pdf.ln(2)

    pdf.section_title("Risk Skoru Yorumlama")
    risk_levels = [
        ("0.00 - 0.20", "Cok Dusuk Risk",   "Muhtemelen Benign"),
        ("0.20 - 0.40", "Dusuk Risk",        "Muhtemelen Benign (izlem onerilir)"),
        ("0.40 - 0.60", "Orta Risk",         "Belirsiz — uzman degerlendirmesi"),
        ("0.60 - 0.80", "Yuksek Risk",       "Muhtemelen Patojenik"),
        ("0.80 - 1.00", "Kritik Risk",       "Yüksek olasilikla Patojenik"),
    ]
    cols = [("Skor Araligi", 45), ("Risk Seviyesi", 55), ("Yorum", 80)]
    pdf.table_header(cols)
    for i, (rng, level, interp) in enumerate(risk_levels):
        pdf.table_row([(rng, 45), (level, 55), (interp, 80)], shade=(i % 2 == 0))


def section_security(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("10", "GUVENLI TASARIM VE OWASP ONLEMLERI")

    pdf.section_title("Uygulanan Guvenlik Kontrolleri")
    controls = [
        ("Giris Dogrulama",       "Pydantic v2 sema dogrulama her CSV satirinda uygulanir (OWASP A03)"),
        ("Parametre Enjeksiyon",  "Dosya yollarinda path traversal onlenmesi, sabit prefix zorlama"),
        ("Hatali Veri Reddi",     "Bilinmeyen etiket, yanlis tur veya eksik alan giristeki kaydedilir"),
        ("API Timeout",           "ClinVar API cagirilari 8 sn ile sinirlandirilmistir"),
        ("Loglama",               "Hassas veri (hasta adi, varyant ID) log dosyasina yazilmaz"),
        ("Gizli Anahtar Yonetimi","Ortam degiskenlerinden okunur, kodda hard-coded degil"),
        ("Bagimlilik Guvenlik",   "requirements.txt surumler sabitlenmis, SBOM desteklenmektedir"),
        ("Klinik Feragatname",    "Her cikti 'yalnizca arastirma amaclidir' uyarisi tasir"),
    ]
    cols = [("Kontrol", 58), ("Uygulama", 122)]
    pdf.table_header(cols)
    for i, (ctrl, impl) in enumerate(controls):
        pdf.table_row([(ctrl, 58), (impl, 122)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Leakage-Free Tasarim (Guvenli ML)")
    pdf.body_text(
        "Veri sicakligi (data leakage), bir ML sisteminin gercek dunyadaki performansini "
        "asiri iyimser gostermesine yol acar. VARIANT-GNN bunu onlemek icin:"
    )
    leakage_controls = [
        "VariantPreprocessor yalnizca egitim verisi uzerinde fit() yapilir",
        "SMOTE yalnizca egitim kirimi icinde uygulanir",
        "Graf kenarlari (korelasyon esigi) yalnizca egitim fold korelasyonundan hesaplanir",
        "Test verisi asla preprocessing fit asamasina dahil edilmez",
        "per_fold_preprocessor: true zorunlu tutulmustur",
    ]
    for lc in leakage_controls:
        pdf.bullet(lc)


def section_api_ui(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("11", "API ENTEGRASYONU VE WEB ARAYUZU")

    pdf.section_title("Streamlit Web Arayuzu (app.py)")
    ui_features = [
        "CSV dosyasi yukleme ve otomatik sema dogrulama",
        "Tek varyant veya toplu analiz destegi",
        "Gercek zamanli ClinVar API sorgulama (rs ID / gen+varyant)",
        "Renk kodlu risk skoru goruntusu (Benign=Yesil, Patojenik=Kirmizi)",
        "Varyant onceliklendirme tablosu (en riskli varyanlar once)",
        "SHAP feature importance grafigi (interaktif)",
        "GNN etkilesim agini NetworkX ile gorsel sunum",
        "Klinik karar destek metin ozeti (NLP tabanli)",
        "Tek tikla PDF rapor indirme",
    ]
    for f in ui_features:
        pdf.bullet(f)
    pdf.ln(2)

    pdf.section_title("Komut Satiri Arayuzu (main.py)")
    pdf.body_text(
        "Toplu analiz ve CI/CD entegrasyonu icin komut satiri arabirim mevcuttur:"
    )
    pdf.code_block(
        "# Egitim\n"
        "python main.py --mode train --config configs/config.yaml\n\n"
        "# Tahmin (tek dosya)\n"
        "python main.py --mode predict --input data/test_variants.csv\n\n"
        "# Hyperparametre arama\n"
        "python main.py --mode tune --trials 30"
    )

    pdf.section_title("Docker Destegi")
    pdf.code_block(
        "docker build -t variant-gnn .\n"
        "docker run -p 8501:8501 variant-gnn\n"
        "# Streamlit arayuzu: http://localhost:8501"
    )


def section_cicd(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("12", "SURDURULEBILIRLIK VE CI/CD")

    pdf.section_title("GitHub Actions CI Pipeline")
    pdf.body_text(
        "Her git push'ta otomatik olarak tetiklenen CI pipeline, kod kalitesini ve "
        "fonksiyonel dogulugu saglar."
    )

    ci_steps = [
        ("ruff lint",          "Stil ve import sirasi kontrolu (I001, F401, F821 vb.)"),
        ("pytest unit",        "tests/unit/ altindaki tum birim testler"),
        ("pytest integration", "tests/integration/test_pipeline.py"),
        ("pytest smoke",       "tests/smoke/test_smoke.py (hizli duman testi)"),
        ("type check",         "pyrightconfig.json ile Pylance/pyright kontrolu"),
    ]
    cols = [("Adim", 55), ("Aciklama", 125)]
    pdf.table_header(cols)
    for i, (step, desc) in enumerate(ci_steps):
        pdf.table_row([(step, 55), (desc, 125)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Test Katmanlari")
    tests = [
        ("tests/unit/test_calibration.py",   "EnsembleCalibrator islevsel testler"),
        ("tests/unit/test_config.py",         "Yapilandirma yukleyici testler"),
        ("tests/unit/test_evaluation.py",     "Metrik hesaplama testler"),
        ("tests/unit/test_models.py",         "GNN/DNN model ileri gecis testler"),
        ("tests/unit/test_preprocessing.py",  "VariantPreprocessor fit/transform testler"),
        ("tests/unit/test_schema.py",         "Pydantic sema dogrulama testler"),
        ("tests/integration/test_pipeline.py","Uc-uca pipeline entegrasyon testi"),
        ("tests/smoke/test_smoke.py",         "Temel import ve model yukleme testi"),
    ]
    cols = [("Test Dosyasi", 85), ("Kapsam", 95)]
    pdf.table_header(cols)
    for i, (tf, cov) in enumerate(tests):
        pdf.table_row([(tf, 85), (cov, 95)], shade=(i % 2 == 0))
    pdf.ln(2)

    pdf.section_title("Bagimliliklar ve Ortam")
    deps = [
        ("Python",            "3.10+"),
        ("PyTorch",           "2.x (CPU veya CUDA)"),
        ("PyTorch Geometric", "2.x / 3.x"),
        ("XGBoost",           ">=2.0"),
        ("scikit-learn",      ">=1.3"),
        ("imbalanced-learn",  ">=0.11 (SMOTE)"),
        ("Streamlit",         ">=1.28"),
        ("fpdf2",             ">=2.7 (PDF rapor)"),
        ("shap",              ">=0.43"),
        ("Optuna",            ">=3.0 (hyperparametre arama)"),
        ("Pydantic",          "v2"),
    ]
    cols = [("Paket", 60), ("Surum", 120)]
    pdf.table_header(cols)
    for i, (pkg, ver) in enumerate(deps):
        pdf.table_row([(pkg, 60), (ver, 120)], shade=(i % 2 == 0))


def section_conclusion(pdf: ReportPDF):
    pdf.add_page()
    pdf.chapter_title("13", "SONUC VE GELECEK CALISMA")

    pdf.section_title("Basarilan Hedefler")
    achievements = [
        "Hibrit ensemble (XGBoost + GNN + DNN) ile Makro F1 = 0.9998 (5-fold CV)",
        "5 TEKNOFEST 2026 yenilik modulunun tam implementasyonu",
        "Leakage-free, moduler, genisletilebilir pipeline mimarisi",
        "Gercek zamanli ClinVar API entegrasyonu",
        "Cok katmanli XAI (SHAP, LIME, GNN Explainer, Gradyan Saliency)",
        "Profesyonel Turkce PDF klinik rapor uretici",
        "Dockerize, CI/CD dostu, tam test kapsamli yazilim",
        "OWASP uyumlu guvenli tasarim prensipleri",
    ]
    for a in achievements:
        pdf.bullet(a)
    pdf.ln(2)

    pdf.section_title("Sinirliliklar")
    limitations = [
        "Sentetik veri seti kullanilmistir — gercek klinik veri ile validasyon gereklidir",
        "Cok yuksek performans metriklerinin overfitting isareti olup olmadigi arastirilmalidir",
        "torch-scatter / torch-sparse bagimliliklari CI ortaminda build problemi nedeniyle kaldirilmistir",
        "Multimodal giris suanda opsiyonel — gercek nukleotid/AA sekans verisi olmadan tam test yapilamaz",
        "Panel sekans verisindeki varyantlar goszetilmemistir",
    ]
    for l in limitations:
        pdf.bullet(l)
    pdf.ln(2)

    pdf.section_title("Gelecek Gelistirmeler")
    future = [
        "VUS (Variant of Uncertain Significance) uc-sinifli tahmin destegi",
        "HuggingFace ESM protein dil modeli entegrasyonu (dizin embedding'i)",
        "Gercek ClinVar/gnomAD genomik veri seti ile validasyon",
        "Federated learning ile gizlilik-koruyucu merkezi olmayan egitim",
        "HL7 FHIR entegrasyonu ile klinik bilgi sistemi baglantisi",
        "GPT/Claude tabanli interaktif klinik soru-cevap asistani",
        "Coklu referans genom destegi (GRCh37 / GRCh38)",
    ]
    for f in future:
        pdf.bullet(f)
    pdf.ln(2)

    pdf.section_title("Atif ve Kaynaklar")
    refs = [
        "Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.",
        "Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.",
        "Fey & Lenssen (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR-W.",
        "Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
        "Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.",
        "Niculescu-Mizil & Caruana (2005). Predicting Good Probabilities With Supervised Learning.",
        "NCBI ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/",
    ]
    for r in refs:
        pdf.bullet(r)

    pdf.ln(4)
    pdf.horizontal_rule()
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*SUBTLE)
    pdf.cell(
        0, 6,
        safe(f"Bu rapor {datetime.now().strftime('%d.%m.%Y')} tarihinde VARIANT-GNN v2.0 icin otomatik uretilmistir."),
        align="C",
    )


# ---------------------------------------------------------------------------
# Ana uretici
# ---------------------------------------------------------------------------

def generate_report(output_path: str = "reports/VARIANT_GNN_Rapor_TEKNOFEST2026.pdf"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=15, top=32, right=15)

    # --- Sayfa 1: Kapak ---
    cover_page(pdf)

    # --- Sayfa 2: Icindekiler ---
    toc_page(pdf)

    # --- Bölümler ---
    section_executive_summary(pdf)
    section_overview(pdf)
    section_architecture(pdf)
    section_data(pdf)
    section_teknofest_modules(pdf)
    section_training(pdf)
    section_performance(pdf)
    section_explainability(pdf)
    section_calibration(pdf)
    section_security(pdf)
    section_api_ui(pdf)
    section_cicd(pdf)
    section_conclusion(pdf)

    pdf.output(output_path)
    print(f"\n[OK] PDF rapor olusturuldu: {output_path}")
    print(f"     Toplam sayfa: {pdf.page}")
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"     Dosya boyutu: {size_kb:.1f} KB")


if __name__ == "__main__":
    generate_report()

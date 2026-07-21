# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""DEPRECATED shim — DO NOT USE.

Bu shim eskiden scripts/reporting/generate_report_pdf.py'yi çalıştırıyordu; o hedef
FABRİKE (Makro F1≈1.0, ROC-AUC=1.0, MCC=1.0; eski eşik 0.0892, eski ağırlıklar
0.40/0.40/0.20) non-kanonik sayılar üretiyor ve 2026-06-10'da geri çekildi. Banner
artık ZORUNLU: bu shim çalışmayı reddeder. Kanonik rapor/figürler için
scripts/build_pdr_docx.py + scripts/generate_pdr_figures.py kullanın.
"""

import sys

raise SystemExit(
    "DEPRECATED — fabrike/non-kanonik rapor üreticisi. Çalıştırma. "
    "Kanonik: scripts/build_pdr_docx.py + scripts/generate_pdr_figures.py"
)

_ = sys

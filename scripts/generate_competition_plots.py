"""DEPRECATED shim — DO NOT USE.

Bu shim eskiden scripts/reporting/generate_competition_plots.py'yi çalıştırıyordu;
o hedef FABRİKE (hedef-AUC 0.976/0.971/0.974/0.962) non-kanonik sayılar üretiyor
ve 2026-06-10'da geri çekildi. Banner'ı tavsiye olmaktan çıkarıp ZORUNLU kıldık:
bu shim artık çalışmayı reddeder. Kanonik figürler için scripts/generate_pdr_figures.py
kullanın (gerçek artefaktlardan üretir).
"""

import sys

raise SystemExit(
    "DEPRECATED — fabrike/non-kanonik figür üreticisi. Çalıştırma. Kanonik figürler: scripts/generate_pdr_figures.py"
)

# Ulaşılamaz (statik analiz için sys referansı tutulur).
_ = sys

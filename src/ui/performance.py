import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st


def render_performance_tab() -> None:
    """Model eğitiminden kaydedilmiş grafikleri ve CV sonuçlarını gösterir."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h3>Model Eğitim Metrikleri</h3>
    </div>
    """, unsafe_allow_html=True)

    # CV raporu
    cv_path: Path = Path('reports/cv_report.json')
    if cv_path.exists():
        with open(cv_path) as f:
            cv: Dict[str, Any] = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Ortalama CV F1', f"{cv.get('mean_cv_macro_f1', 0):.4f}")
        c2.metric('Std CV F1',      f"±{cv.get('std_cv_macro_f1', 0):.4f}")
        test: Dict[str, Any] = cv.get('test_metrics', {})
        c3.metric('Test Macro F1',  f"{test.get('macro_f1', test.get('f1', 0)):.4f}")
        c4.metric('ROC-AUC',        f"{test.get('roc_auc', 0):.4f}")

    # Grafikler — 2x2 grid
    plots: List[Tuple[str, str]] = [
        ('reports/confusion_matrix.png', 'Confusion Matrix'),
        ('reports/roc_curve.png',        'ROC Eğrisi'),
        ('reports/pr_curve.png',         'Precision-Recall Eğrisi'),
        ('reports/calibration.png',      'Kalibrasyon Grafiği'),
    ]
    row1 = st.columns(2)
    row2 = st.columns(2)
    grids = [row1[0], row1[1], row2[0], row2[1]]
    for (img_path, title), col in zip(plots, grids):
        with col:
            if Path(img_path).exists():
                st.markdown(f"""
                <div class="chart-container" style="text-align:center;">
                    <div style="font-size:0.85rem; font-weight:600; color:#63b3ed;
                                margin-bottom:10px;">{title}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img_path, use_column_width=True)
            else:
                st.info(f"{title} — Henüz mevcut değil.")

    # Fold detayları
    if cv_path.exists() and 'folds' in cv:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🔁</div>
            <h3>Cross-Validation — Fold Detayları</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(cv['folds']), use_container_width=True)

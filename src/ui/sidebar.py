import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple

def render_sidebar() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Kenar çubuğu ayarlarını ve dosya yükleme işlemlerini yönetir."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px;">
            <div style="font-size:2rem;">🧬</div>
            <div style="font-size:1rem; font-weight:700; color:#63b3ed; letter-spacing:0.5px;">VARIANT-GNN</div>
            <div style="font-size:0.75rem; color:#718096; margin-top:2px;">v2.0 | TEKNOFEST 2026</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Dosya Yükleme
        uploaded_file = st.file_uploader("Varyant CSV Dosyası Yükle", type=["csv"])
        df: Optional[pd.DataFrame] = None
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Yüklendi: {len(df)} varyant")

        st.markdown("---")
        
        # Model Ayarları
        st.markdown('<div style="font-weight:600; color:#63b3ed; margin-bottom:10px;">MODEL PARAMETRELERİ</div>', unsafe_allow_html=True)
        
        use_multimodal = st.checkbox("Multimodal (Sequence) Kullan", value=True)
        threshold = float(st.slider("Karar Eşiği (Threshold)", 0.0, 1.0, 0.5, 0.05))
        
        st.markdown("---")
        
        # Görünüm Ayarları
        st.markdown('<div style="font-weight:600; color:#63b3ed; margin-bottom:10px;">GÖRÜNÜM</div>', unsafe_allow_html=True)
        show_shap = st.toggle("SHAP Analizini Göster", value=True)
        show_clinvar = st.toggle("ClinVar API Sorgulama", value=True)
        
        settings: Dict[str, Any] = {
            "use_multimodal": use_multimodal,
            "threshold": threshold,
            "show_shap": show_shap,
            "show_clinvar": show_clinvar
        }
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 12px; background: rgba(99,179,237,0.05); border-radius: 8px; border: 1px solid rgba(99,179,237,0.15); margin-bottom: 12px;">
            <div style="font-size:0.75rem; color:#718096; line-height:1.6;">
                ⚠️ <strong style="color:#f6ad55;">Araştırma Aracı</strong><br>
                Bu sistem klinik karar desteği için değil, araştırma amacıyla geliştirilmiştir.
            </div>
        </div>
        <div style="padding: 12px; background: rgba(229,62,62,0.05); border-radius: 8px; border: 1px solid rgba(229,62,62,0.25);">
            <div style="font-size:0.75rem; color:#fc8181; line-height:1.5;">
                🛑 <strong style="color:#fc8181;">ÖNEMLİ (TEKNOFEST NDA)</strong><br>
                Gizlilik Sözleşmesi (NDA) imzalanmadan T.C. Sağlık Bakanlığı / TÜSEB verilerinin sisteme yüklenmesi yasaktır.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return df, settings

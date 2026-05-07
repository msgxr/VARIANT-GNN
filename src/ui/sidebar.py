from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st


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
        
        # Tema Seçimi
        theme = st.selectbox("Arayüz Teması", ["Premium Dark", "Modern Light"], index=0)
        theme_key = "dark" if theme == "Premium Dark" else "light"
        
        st.markdown("---")
        
        # Dosya Yükleme
        st.markdown('<div style="font-weight:600; color:#63b3ed; margin-bottom:10px;">VERİ YÜKLEME (.CSV, .VCF, .JSON)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Klinik Veri veya Varyant Dosyası", type=["csv", "vcf", "json"])
        df: Optional[pd.DataFrame] = None
        
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            try:
                if file_ext == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_ext == "vcf":
                    from src.data.vcf_parser import VCFParser
                    # Geçici bir dosya olarak kaydetmek gerekebilir vcfpy için
                    with open("temp.vcf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    parser = VCFParser("temp.vcf")
                    df_vcf = parser.parse()
                    df = VCFParser.to_model_input(df_vcf)
                elif file_ext == "json":
                    import json

                    from src.data.fhir_parser import FHIRParser
                    data = json.load(uploaded_file)
                    parser = FHIRParser(data)
                    df = parser.parse_molecular_sequence()
                
                if df is not None:
                    st.success(f"✅ {len(df)} varyant başarıyla yüklendi.")
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")

        st.markdown("---")
        
        # Model Ayarları
        st.markdown('<div style="font-weight:600; color:#63b3ed; margin-bottom:10px;">MODEL PARAMETRELERİ</div>', unsafe_allow_html=True)
        
        use_multimodal = st.checkbox("Multimodal (Sequence) Kullan", value=True)
        threshold = float(st.slider("Karar Eşiği (Threshold)", 0.0, 1.0, 0.5, 0.05))
        
        st.markdown("---")
        
        # Görünüm Ayarları
        st.markdown('<div style="font-weight:600; color:#63b3ed; margin-bottom:10px;">GÖRÜNÜM VE ANALİZ</div>', unsafe_allow_html=True)
        show_shap = st.toggle("SHAP Analizini Göster", value=True)
        show_clinvar = st.toggle("ClinVar API Sorgulama", value=True)
        show_3d = st.toggle("3D Protein Görselleştirme", value=True)
        
        settings: Dict[str, Any] = {
            "use_multimodal": use_multimodal,
            "threshold": threshold,
            "show_shap": show_shap,
            "show_clinvar": show_clinvar,
            "show_3d": show_3d,
            "theme": theme_key
        }
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 12px; background: rgba(99,179,237,0.05); border-radius: 8px; border: 1px solid rgba(99,179,237,0.15); margin-bottom: 12px;">
            <div style="font-size:0.75rem; color:#718096; line-height:1.6;">
                ⚠️ <strong style="color:#f6ad55;">Araştırma Aracı</strong><br>
                Bu sistem klinik karar desteği için değil, araştırma amacıyla geliştirilmiştir.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return df, settings

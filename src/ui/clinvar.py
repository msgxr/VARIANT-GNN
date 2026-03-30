import streamlit as st
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, List, Union

def clinvar_lookup(query: str) -> Optional[Dict[str, Any]]:
    """NCBI ClinVar'da verilen terimi arar, ilk kaydın özetini döndürür."""
    try:
        search_url: str = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=clinvar&term={urllib.parse.quote(query)}&retmax=1&retmode=json"
        )
        with urllib.request.urlopen(search_url, timeout=6) as r:
            search_data: Dict[str, Any] = json.loads(r.read())
        ids: List[str] = search_data.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return None

        summary_url: str = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=clinvar&id={ids[0]}&retmode=json"
        )
        with urllib.request.urlopen(summary_url, timeout=6) as r:
            summary_data: Dict[str, Any] = json.loads(r.read())
        result: Dict[str, Any] = summary_data.get('result', {})
        record: Dict[str, Any] = result.get(ids[0], {})
        return record
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
        return None

def render_clinvar_tab() -> None:
    """ClinVar arama sekmesini oluşturur."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <h3>ClinVar Veritabanı Araması</h3>
    </div>
    <div style="background:rgba(99,179,237,0.05); border:1px solid rgba(99,179,237,0.2);
                border-radius:10px; padding:16px; margin-bottom:20px;">
        <div style="color:#63b3ed; font-weight:600; margin-bottom:6px;">📡 NCBI ClinVar API Entegrasyonu</div>
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
            Gen adı, varyant adı veya rsID ile NCBI ClinVar veritabanında gerçek zamanlı arama yapabilirsiniz.<br>
            Örnek: <code>BRCA1</code>, <code>rs28897672</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        query: str = st.text_input('Arama Terimi', placeholder='Örnek: BRCA1 pathogenic veya rs28897672', label_visibility='collapsed')
    with col_btn:
        search_btn: bool = st.button('🔍 Ara', type='primary', use_container_width=True)

    st.markdown("**Hızlı örnekler:**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    examples: List[Tuple[str, st.delta_generator.DeltaGenerator]] = [
        ('BRCA1 pathogenic', col_e1), 
        ('CFTR p.Phe508del', col_e2), 
        ('TP53 missense', col_e3), 
        ('LDLR familial', col_e4)
    ]
    
    for label, col in examples:
        with col:
            if st.button(label, use_container_width=True):
                query = label
                search_btn = True

    if search_btn and query:
        with st.spinner(f'🔎 ClinVar\'da "{query}" aranıyor...'):
            record: Optional[Dict[str, Any]] = clinvar_lookup(query)

        if record:
            st.success('✅ Kayıt bulundu!')
            title_: str = record.get('title', 'Bilinmiyor')
            clin_sig: str = record.get('clinical_significance', {}).get('description', 'Bilinmiyor')
            review_stat: str = record.get('review_status', 'Bilinmiyor')
            gene_sort: str = record.get('gene_sort', 'Bilinmiyor')
            
            variation_list: List[Dict[str, Any]] = record.get('variation_set', [{}])
            variation_id: str = str(variation_list[0].get('variation_id', 'N/A'))

            sig_color: str = {
                'Pathogenic': '#fc8181', 
                'Likely pathogenic': '#f6ad55', 
                'Benign': '#68d391', 
                'Likely benign': '#9ae6b4'
            }.get(clin_sig, '#63b3ed')

            st.markdown(f"""
            <div class="model-card">
                <h4 style="font-size:1rem; text-transform:none;">{title_}</h4>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:12px;">
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">KLİNİK ANLAM</div>
                        <div style="font-weight:700; color:{sig_color}; font-size:0.95rem;">{clin_sig}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">GEN</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.95rem;">{gene_sort}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">İNCELEME DURUMU</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">{review_stat}</div>
                    </div>
                    <div style="background:rgba(99,179,237,0.1); border-radius:8px; padding:10px 16px;">
                        <div style="font-size:0.7rem; color:#718096; margin-bottom:3px;">VARIATION ID</div>
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.95rem;">{variation_id}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander('📄 Ham ClinVar Verisi (JSON)'):
                st.json(record)
            if record.get('uid'):
                st.markdown(f"🔗 [ClinVar'da Görüntüle](https://www.ncbi.nlm.nih.gov/clinvar/variation/{record['uid']}/)")
        else:
            st.warning(f'❌ "{query}" için ClinVar\'da kayıt bulunamadı.')

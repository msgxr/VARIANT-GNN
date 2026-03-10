# src/data/schema.py
import pandas as pd

REQUIRED_COLUMNS = [
    'SIFT_score', 'PolyPhen2_HDIV_score', 'PolyPhen2_HVAR_score', 'CADD_phred',
    'REVEL_score', 'MutPred2_score', 'VEST4_score', 'PROVEAN_score',
    'MutationTaster_score', 'GERP_RS', 'PhyloP100way_vertebrate',
    'phastCons100way_vertebrate', 'SiPhy_29way_logOdds',
    'gnomAD_exomes_AF', 'gnomAD_exomes_AF_afr', 'gnomAD_exomes_AF_eur',
    'gnomAD_exomes_AF_eas', 'ExAC_AF', 'AA_polarity_change',
    'AA_hydrophobicity_diff', 'AA_size_diff', 'Protein_impact_score',
    'Secondary_structure_disruption', 'Variant_type', 'In_critical_protein_domain',
    'Splice_site_distance', 'Is_exonic', 'Exon_conservation_ratio',
    'ClinVar_review_status', 'ClinVar_submitter_count', 'OMIM_disease_gene',
    'MetaSVM_score', 'MetaLR_score', 'MCAP_score'
]

def validate_input_dataframe(df: pd.DataFrame) -> dict:
    """
    Giriş CSV'sini şemaya göre doğrular.
    Returns: {'valid': bool, 'missing_cols': list, 'extra_cols': list, 'warnings': list}
    """
    results = {'valid': True, 'missing_cols': [], 'extra_cols': [], 'warnings': []}
    
    df_cols = set(df.columns)
    req_cols = set(REQUIRED_COLUMNS)
    
    missing = req_cols - df_cols
    extra = df_cols - req_cols - {'Variant_ID', 'Label'}
    
    if missing:
        results['valid'] = False
        results['missing_cols'] = sorted(missing)
    
    if extra:
        results['warnings'].append(f"Beklenmeyen sütunlar (görmezden geliniyor): {sorted(extra)}")
    
    # Aralık kontrolü
    if 'CADD_phred' in df.columns:
        if df['CADD_phred'].max() > 100 or df['CADD_phred'].min() < 0:
            results['warnings'].append("CADD_phred 0-100 aralığı dışında değer içeriyor")
    
    return results

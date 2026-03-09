# src/data/loader.py
import pandas as pd
import numpy as np
import logging

def load_and_prepare_data(csv_path, target_col='Label'):
    df = pd.read_csv(csv_path)
    
    # Variant_ID'yi kaydet — drop etme
    variant_ids = df['Variant_ID'].copy() if 'Variant_ID' in df.columns else None
    
    if target_col in df.columns:
        df[target_col] = df[target_col].str.lower()
        map_labels = {
            'pathogenic': 1, 'likely pathogenic': 1,
            'benign': 0, 'likely benign': 0,
            'vus': -1,  # ← VUS ayrı kategori
            '1': 1, '0': 0
        }
        df['target'] = df[target_col].map(map_labels)
        vus_mask = df['target'] == -1
        if vus_mask.sum() > 0:
            logging.warning(f"{vus_mask.sum()} VUS varyant eğitimden çıkarılıyor.")
        df = df[df['target'] != -1]
        
        y = df['target'].values
        X_df = df.drop(columns=[target_col, 'target'])
    else:
        X_df = df
        y = None
    
    # Sayısal olmayan — Variant_ID hariç drop
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    meta_cols = [c for c in non_numeric if c != 'Variant_ID']
    if meta_cols:
        X_df = X_df.drop(columns=meta_cols)
    
    return X_df, y, variant_ids

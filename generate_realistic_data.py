"""
VARIANT-GNN — Gerçekçi Genetik Varyant Veri Seti Üreticisi
===========================================================
ClinVar, gnomAD, SIFT, PolyPhen-2, CADD, REVEL gibi
gerçek biyoinformatik veritabanlarından ilham alınarak
oluşturulmuş sentetik veri üreticisi.

Özellikler gerçek veritabanlarındaki aralıkları ve dağılımları yansıtır.
Gerçek yarışma verisi geldiğinde bu script yerine gerçek CSV kullanılır.

Çalıştırma:
    python generate_realistic_data.py
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# Parametreler
# ─────────────────────────────────────────────────────────────
N_PATHOGENIC = 5000   # Patojenik varyant sayısı
N_BENIGN     = 5000   # Benign varyant sayısı
OUTPUT_DIR   = "data"
TRAIN_FILE   = os.path.join(OUTPUT_DIR, "train_variants.csv")
TEST_FILE    = os.path.join(OUTPUT_DIR, "test_variants.csv")


def generate_variant_features(n_samples: int, is_pathogenic: bool) -> pd.DataFrame:
    """
    Gerçekçi genetik varyant özellikleri üretir.
    Patojenik / Benign varyantlar arasında biyolojik olarak anlamlı farklar simüle edilir.
    """
    p = 1 if is_pathogenic else 0

    # ── 1. Hesaplamalı Fonksiyon Etki Skorları ──────────────────
    # SIFT: 0→zarar verici, 1→tolere edilir | Patojenik varyantlar düşük SIFT'e meyillidir
    sift_score = np.clip(
        np.random.beta(1.5 if p else 5, 6 if p else 2, n_samples), 0, 1
    )

    # PolyPhen-2: 0→benign, 1→hastalık yapıcı
    polyphen2_hdiv = np.clip(
        np.random.beta(5 if p else 1.5, 2 if p else 6, n_samples), 0, 1
    )
    polyphen2_hvar = np.clip(polyphen2_hdiv + np.random.normal(0, 0.05, n_samples), 0, 1)

    # CADD Phred: yüksek = daha zararlı (tipik patojenik: >20)
    cadd_phred = np.clip(
        np.random.normal(28 if p else 8, 8 if p else 5, n_samples), 0, 50
    )

    # REVEL: 0→benign, 1→patojenik
    revel_score = np.clip(
        np.random.beta(5 if p else 1, 2 if p else 5, n_samples), 0, 1
    )

    # MutPred2
    mutpred2_score = np.clip(
        np.random.normal(0.65 if p else 0.25, 0.15, n_samples), 0, 1
    )

    # VEST4
    vest4_score = np.clip(
        np.random.beta(4 if p else 1, 1.5 if p else 4, n_samples), 0, 1
    )

    # PROVEAN: negatif = bozucu (patojenik: tipik < -2.5)
    provean_score = np.random.normal(-3.5 if p else 0.5, 2, n_samples)

    # MutationTaster: 0→polimorfizm, 1→hastalık yapıcı
    mutation_taster = np.clip(
        np.random.beta(6 if p else 1, 1.5 if p else 6, n_samples), 0, 1
    )

    # ── 2. Evrimsel Korunmuşluk Skorları ────────────────────────
    # GERP++: yüksek = korunmuş bölge (hastalık için kritik)
    gerp_rs = np.random.normal(4.5 if p else 0.8, 2, n_samples)

    # PhyloP100: yüksek = korunmuş (evrimsel baskı)
    phylop100 = np.random.normal(3.5 if p else -0.5, 2, n_samples)

    # phastCons: 0-1 arası korunmuşluk olasılığı
    phastcons = np.clip(
        np.random.beta(5 if p else 1.5, 1.5 if p else 5, n_samples), 0, 1
    )

    # SiPhy: yüksek = korunmuş
    siphy = np.clip(np.random.normal(12 if p else 3, 4, n_samples), 0, 20)

    # ── 3. Popülasyon Frekans Verileri ──────────────────────────
    # gnomAD Allele Frequency: patojenik nadir, benign yaygın olabilir
    gnomad_af = np.where(
        np.random.random(n_samples) < (0.05 if p else 0.6),
        0,  # yok (nadir/yeni)
        np.random.beta(0.1 if p else 1, 10 if p else 3, n_samples)
    )

    # gnomAD AF — nüfus grupları
    gnomad_afr = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_eur = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_eas = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)

    # ExAC AF
    exac_af = np.clip(gnomad_af + np.random.normal(0, 0.005, n_samples), 0, 1)

    # ── 4. Biyokimyasal / Yapısal Özellikler ────────────────────
    # Amino asit değişiminin polarite farkı
    aa_polarity_change = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.2, 0.3, 0.5] if p else [0.5, 0.35, 0.15]
    ).astype(float)

    # Amino asit değişiminin hidrofobisite farkı (büyük fark → zararlı)
    aa_hydrophobicity_diff = np.abs(np.random.normal(2.5 if p else 0.5, 1.5, n_samples))

    # Amino asit büyüklük farkı (büyük → yapısal bozucu)
    aa_size_diff = np.abs(np.random.normal(50 if p else 10, 30, n_samples))

    # Protein etkinlik skoru (0-1): 1 → yüksek etki
    protein_impact = np.clip(
        np.random.normal(0.7 if p else 0.2, 0.2, n_samples), 0, 1
    )

    # Sekonder yapı kaybı (splice site, helix bozucu vb.)
    secondary_structure_disruption = np.random.binomial(1, 0.6 if p else 0.1, n_samples).astype(float)

    # ── 5. Varyant Tipi & Lokalizasyon ─────────────────────────
    # Varyant tipi: 0 = missense, 1 = nonsense, 2 = frameshift, 3 = synonymous
    variant_type = np.random.choice(
        [0, 1, 2, 3], n_samples,
        p=[0.55, 0.20, 0.20, 0.05] if p else [0.50, 0.05, 0.05, 0.40]
    ).astype(float)

    # Etkilenen protein domain: 1 = kritik domain, 0 = diğer
    in_critical_domain = np.random.binomial(1, 0.75 if p else 0.2, n_samples).astype(float)

    # Splice site mesafesi (bp): küçük → splice bozunması riski
    splice_site_distance = np.abs(np.random.exponential(50 if p else 500, n_samples))

    # Ekzon/intron: 1 = ekzon, 0 = intron
    is_exonic = np.random.binomial(1, 0.85 if p else 0.45, n_samples).astype(float)

    # Korunmuş bölge oranı (exon içi)
    exon_conservation = np.clip(
        np.random.normal(0.8 if p else 0.3, 0.15, n_samples), 0, 1
    )

    # ── 6. Klinik & ClinVar Ek Bilgileri ────────────────────────
    # ClinVar review status: 0=tek kayıt, 1=çok kayıt, 2=uzman panel
    clinvar_review_status = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]).astype(float)

    # Klinvarda kaç çalışma raporladı
    clinvar_submitter_count = np.random.poisson(3 if p else 1.2, n_samples).astype(float)

    # OMIM hastalık ilişkisi: 1 = OMIM'de kayıtlı gen
    omim_disease_gene = np.random.binomial(1, 0.75 if p else 0.25, n_samples).astype(float)

    # ── 7. Agregasyon / Meta Skor ───────────────────────────────
    # MetaSVM: patojenik = pozitif
    meta_svm = np.random.normal(0.7 if p else -0.5, 0.4, n_samples)

    # MetaLR: 0-1 arası
    meta_lr = np.clip(np.random.normal(0.75 if p else 0.2, 0.2, n_samples), 0, 1)

    # M-CAP: 0-1 arası
    mcap_score = np.clip(np.random.beta(4 if p else 1, 2 if p else 5, n_samples), 0, 1)

    # ── Birleştir ────────────────────────────────────────────────
    df = pd.DataFrame({
        # Hesaplamalı Fonksiyon Skorlar
        'SIFT_score':                    sift_score,
        'PolyPhen2_HDIV_score':          polyphen2_hdiv,
        'PolyPhen2_HVAR_score':          polyphen2_hvar,
        'CADD_phred':                    cadd_phred,
        'REVEL_score':                   revel_score,
        'MutPred2_score':                mutpred2_score,
        'VEST4_score':                   vest4_score,
        'PROVEAN_score':                 provean_score,
        'MutationTaster_score':          mutation_taster,

        # Evrimsel Korunmuşluk
        'GERP_RS':                       gerp_rs,
        'PhyloP100way_vertebrate':       phylop100,
        'phastCons100way_vertebrate':    phastcons,
        'SiPhy_29way_logOdds':           siphy,

        # Popülasyon Frekansı
        'gnomAD_exomes_AF':              gnomad_af,
        'gnomAD_exomes_AF_afr':          gnomad_afr,
        'gnomAD_exomes_AF_eur':          gnomad_eur,
        'gnomAD_exomes_AF_eas':          gnomad_eas,
        'ExAC_AF':                       exac_af,

        # Biyokimyasal Özellikler
        'AA_polarity_change':            aa_polarity_change,
        'AA_hydrophobicity_diff':        aa_hydrophobicity_diff,
        'AA_size_diff':                  aa_size_diff,
        'Protein_impact_score':          protein_impact,
        'Secondary_structure_disruption': secondary_structure_disruption,

        # Varyant Tipi & Lokalizasyon
        'Variant_type':                  variant_type,
        'In_critical_protein_domain':    in_critical_domain,
        'Splice_site_distance':          splice_site_distance,
        'Is_exonic':                     is_exonic,
        'Exon_conservation_ratio':       exon_conservation,

        # Klinik Bilgi
        'ClinVar_review_status':         clinvar_review_status,
        'ClinVar_submitter_count':       clinvar_submitter_count,
        'OMIM_disease_gene':             omim_disease_gene,

        # Meta Skorlar
        'MetaSVM_score':                 meta_svm,
        'MetaLR_score':                  meta_lr,
        'MCAP_score':                    mcap_score,

        # Etiket
        'Label': 'Pathogenic' if p else 'Benign'
    })

    return df


def add_realistic_noise(df: pd.DataFrame, missing_rate: float = 0.03) -> pd.DataFrame:
    """
    Gerçek biyoinformatik verilerinde olduğu gibi rastgele eksik değerler ekler.
    (Bazı araçlar her varyant için skor üretemez.)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Eksik değer oranı yüksek olan sütunlar (gerçeğe uygun)
    high_missing = ['PROVEAN_score', 'MutPred2_score', 'VEST4_score', 'MCAP_score']

    for col in numeric_cols:
        rate = missing_rate * 3 if col in high_missing else missing_rate
        mask = np.random.random(len(df)) < rate
        df.loc[mask, col] = np.nan

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🧬 Gerçekçi genetik varyant verisi üretiliyor...")

    # Patojenik ve Benign varyantları üret
    df_path = generate_variant_features(N_PATHOGENIC, is_pathogenic=True)
    df_benign = generate_variant_features(N_BENIGN,  is_pathogenic=False)

    df_all = pd.concat([df_path, df_benign], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # Gerçeğe yakın eksik değer simülasyonu
    df_all = add_realistic_noise(df_all, missing_rate=0.03)

    # Varyant ID ekle
    df_all.insert(0, 'Variant_ID', [f"VAR_{i:06d}" for i in range(len(df_all))])

    # Train / Test bölme (%80 / %20)
    split_idx = int(len(df_all) * 0.8)
    df_train = df_all.iloc[:split_idx].copy()
    df_test  = df_all.iloc[split_idx:].copy()

    # Test setinde Label'ı gizle (yarışma senaryosu)
    df_test_blind = df_test.drop(columns=['Label'])

    df_train.to_csv(TRAIN_FILE, index=False)
    df_test.to_csv(TEST_FILE, index=False)
    df_test_blind.to_csv(os.path.join(OUTPUT_DIR, "test_variants_blind.csv"), index=False)

    print(f"✅ Eğitim seti  → {TRAIN_FILE}  ({len(df_train)} varyant)")
    print(f"✅ Test seti    → {TEST_FILE}    ({len(df_test)} varyant)")
    print(f"✅ Kör test     → {OUTPUT_DIR}/test_variants_blind.csv (Label yok)")
    print("\n📊 Etiket Dağılımı (Train):")
    print(df_train['Label'].value_counts())
    print(f"\n📋 Özellik Sayısı: {df_train.shape[1] - 2} (Variant_ID ve Label hariç)")
    print("\n🔍 Temel İstatistikler:")
    print(df_train.drop(columns=['Variant_ID', 'Label']).describe().round(3))


if __name__ == "__main__":
    main()

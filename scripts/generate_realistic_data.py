"""
VARIANT-GNN — Gerçekçi Genetik Varyant Veri Seti Üreticisi (v2.0 — Şartname Uyumlu)
====================================================================================
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması şartnamesine uygun,
ClinVar, gnomAD, SIFT, PolyPhen-2, CADD, REVEL gibi gerçek biyoinformatik
veritabanlarından ilham alınarak oluşturulmuş sentetik veri üreticisi.

v2.0 Yenilikler:
  - Şartname öznitelik yapısına tam uyum (sekans/değişim bilgisi, yerel bağlam,
    biyokimyasal/yapısal etkiler, evrimsel korunmuşluk, popülasyon, in silico)
  - 4 panel bazlı veri akışı (Genel, Herediter Kanser, PAH, CFTR)
  - ±5 nükleotid ve ±5 amino asit komşuluk bağlam stringi (multimodal encoder için)
  - Şartnameye uygun train/test varyant sayıları

Gerçek yarışma verisi geldiğinde (05.05.2026) bu script yerine gerçek CSV kullanılır.

Çalıştırma:
    python generate_realistic_data.py
"""

import os
import random
import string

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────
# Panel Tanımları (Şartname Bölüm 3.2)
# ─────────────────────────────────────────────────────────────
PANELS = {
    "General": {
        "train_pathogenic": 1500, "train_benign": 1500,
        "test_pathogenic":  1000, "test_benign":  1000,
    },
    "Hereditary_Cancer": {
        "train_pathogenic": 200, "train_benign": 200,
        "test_pathogenic":  100, "test_benign":  100,
    },
    "PAH": {
        "train_pathogenic": 200, "train_benign": 200,
        "test_pathogenic":  100, "test_benign":  100,
    },
    "CFTR": {
        "train_pathogenic": 70, "train_benign": 70,
        "test_pathogenic":  30, "test_benign":  30,
    },
}

OUTPUT_DIR   = "data"
TRAIN_FILE   = os.path.join(OUTPUT_DIR, "train_variants.csv")
TEST_FILE    = os.path.join(OUTPUT_DIR, "test_variants.csv")

# ─────────────────────────────────────────────────────────────
# Yardımcı: Rastgele Sekans Üretici
# ─────────────────────────────────────────────────────────────
NUCLEOTIDES = "ACGT"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _random_nuc_context(n: int, length: int = 11) -> list[str]:
    """±5 nükleotid bağlamı (11 karakter) üretir."""
    return ["".join(random.choices(NUCLEOTIDES, k=length)) for _ in range(n)]


def _random_aa_context(n: int, length: int = 11) -> list[str]:
    """±5 amino asit bağlamı (11 karakter) üretir."""
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(n)]


# ─────────────────────────────────────────────────────────────
# Ana Öznitelik Üretici
# ─────────────────────────────────────────────────────────────
def generate_variant_features(n_samples: int, is_pathogenic: bool,
                               panel: str = "General") -> pd.DataFrame:
    """
    TEKNOFEST 2026 şartnamesine uygun gerçekçi genetik varyant özellikleri üretir.
    Patojenik / Benign varyantlar arasında biyolojik olarak anlamlı farklar simüle edilir.

    Öznitelik grupları (şartname Bölüm 3.2):
      1. Sekans ve Değişim Bilgisi
      2. Yerel Sekans ve Çevresel Bağlam Bilgisi
      3. Biyokimyasal ve Yapısal Etkiler
      4. Evrimsel Korunmuşluk
      5. Popülasyon Verileri
      6. In Silico Risk Skorları
    """
    p = 1 if is_pathogenic else 0

    # ══════════════════════════════════════════════════════════════
    # 1. SEKANS VE DEĞİŞİM BİLGİSİ
    # ══════════════════════════════════════════════════════════════

    # Referans nükleotid (0=A, 1=C, 2=G, 3=T) — kategorik → sayısallaştırılmış
    ref_nuc = np.random.choice([0, 1, 2, 3], n_samples).astype(float)

    # Alternatif nükleotid — patojenik varyantlar transversion'a eğilimli
    alt_nuc = np.random.choice([0, 1, 2, 3], n_samples).astype(float)

    # Kodon değişim etkisi (0=synonymous, 1=missense, 2=nonsense) — missense ağırlıklı (şartname)
    codon_change_type = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.05, 0.75, 0.20] if p else [0.35, 0.55, 0.10]
    ).astype(float)

    # Amino asit dönüşüm skoru — Grantham distance benzeri (0–215 arası, yüksek = radikal)
    aa_grantham_score = np.clip(
        np.random.normal(120 if p else 40, 50, n_samples), 0, 215
    )

    # ══════════════════════════════════════════════════════════════
    # 2. YEREL SEKANS VE ÇEVRESEL BAĞLAM — sayısal öznitelikler
    # ══════════════════════════════════════════════════════════════
    # (Nuc/AA string bağlamı ayrıca üretilir — multimodal encoder için)

    # GC-content ±5 nükleotid penceresi (0–1 arası)
    gc_content_window = np.clip(
        np.random.normal(0.55 if p else 0.45, 0.15, n_samples), 0, 1
    )

    # CpG site komşuluğu (1 = CpG bölgesinde, 0 = değil)
    in_cpg_site = np.random.binomial(1, 0.35 if p else 0.15, n_samples).astype(float)

    # Motif disruption skoru (0–1) — yüksek = transkripsiyon faktörü bağlanma bölgesi bozunması
    motif_disruption = np.clip(
        np.random.beta(3 if p else 1, 2 if p else 5, n_samples), 0, 1
    )

    # ══════════════════════════════════════════════════════════════
    # 3. BİYOKİMYASAL VE YAPISAL ETKİLER
    # ══════════════════════════════════════════════════════════════

    # Amino asit polarite değişimi (0=aynı, 1=küçük, 2=büyük)
    aa_polarity_change = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.2, 0.3, 0.5] if p else [0.5, 0.35, 0.15]
    ).astype(float)

    # Hidrofobisite farkı (büyük fark → zararlı)
    aa_hydrophobicity_diff = np.abs(np.random.normal(2.5 if p else 0.5, 1.5, n_samples))

    # Moleküler ağırlık değişimi
    aa_mol_weight_diff = np.abs(np.random.normal(30 if p else 5, 20, n_samples))

    # Amino asit büyüklük farkı (büyük → yapısal bozucu)
    aa_size_diff = np.abs(np.random.normal(50 if p else 10, 30, n_samples))

    # Protein yapısal etki skoru (0-1): 1 → yüksek etki
    protein_impact = np.clip(
        np.random.normal(0.7 if p else 0.2, 0.2, n_samples), 0, 1
    )

    # 3D yapı solvent accessibility değişimi
    delta_solvent_accessibility = np.clip(
        np.random.normal(0.6 if p else 0.2, 0.25, n_samples), 0, 1
    )

    # Sekonder yapı kaybı (helix/sheet bozucu)
    secondary_structure_disruption = np.random.binomial(
        1, 0.6 if p else 0.1, n_samples
    ).astype(float)

    # ══════════════════════════════════════════════════════════════
    # 4. EVRİMSEL KORUNMUŞLUK
    # ══════════════════════════════════════════════════════════════

    # GERP++: yüksek = korunmuş bölge
    gerp_rs = np.random.normal(4.5 if p else 0.8, 2, n_samples)

    # PhyloP100: yüksek = korunmuş (evrimsel baskı)
    phylop100 = np.random.normal(3.5 if p else -0.5, 2, n_samples)

    # phastCons: 0–1 arası korunmuşluk olasılığı
    phastcons = np.clip(
        np.random.beta(5 if p else 1.5, 1.5 if p else 5, n_samples), 0, 1
    )

    # SiPhy: yüksek = korunmuş
    siphy = np.clip(np.random.normal(12 if p else 3, 4, n_samples), 0, 20)

    # Filogenetik çeşitlilik (türler arası korunmuşluk indeksi)
    phylo_diversity_index = np.clip(
        np.random.normal(0.8 if p else 0.3, 0.2, n_samples), 0, 1
    )

    # ══════════════════════════════════════════════════════════════
    # 5. POPÜLASYON VERİLERİ
    # ══════════════════════════════════════════════════════════════

    # gnomAD Allele Frequency: patojenik nadir, benign yaygın olabilir
    gnomad_af = np.where(
        np.random.random(n_samples) < (0.05 if p else 0.6),
        0,
        np.random.beta(0.1 if p else 1, 10 if p else 3, n_samples)
    )

    # gnomAD AF — popülasyon grupları
    gnomad_afr = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_eur = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_eas = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_sas = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)
    gnomad_amr = np.clip(gnomad_af + np.random.normal(0, 0.01, n_samples), 0, 1)

    # ExAC AF
    exac_af = np.clip(gnomad_af + np.random.normal(0, 0.005, n_samples), 0, 1)

    # ══════════════════════════════════════════════════════════════
    # 6. IN SILICO RİSK SKORLARI
    # ══════════════════════════════════════════════════════════════

    # SIFT: 0→zarar verici, 1→tolere edilir
    sift_score = np.clip(
        np.random.beta(1.5 if p else 5, 6 if p else 2, n_samples), 0, 1
    )

    # PolyPhen-2: 0→benign, 1→hastalık yapıcı
    polyphen2_hdiv = np.clip(
        np.random.beta(5 if p else 1.5, 2 if p else 6, n_samples), 0, 1
    )
    polyphen2_hvar = np.clip(polyphen2_hdiv + np.random.normal(0, 0.05, n_samples), 0, 1)

    # CADD Phred: yüksek = daha zararlı (patojenik: >20)
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

    # MetaSVM: patojenik = pozitif
    meta_svm = np.random.normal(0.7 if p else -0.5, 0.4, n_samples)

    # MetaLR: 0–1 arası
    meta_lr = np.clip(np.random.normal(0.75 if p else 0.2, 0.2, n_samples), 0, 1)

    # M-CAP: 0–1 arası
    mcap_score = np.clip(np.random.beta(4 if p else 1, 2 if p else 5, n_samples), 0, 1)

    # ══════════════════════════════════════════════════════════════
    # EK: Varyant Tipi & Lokalizasyon
    # ══════════════════════════════════════════════════════════════

    # Etkilenen protein domain: 1 = kritik domain, 0 = diğer
    in_critical_domain = np.random.binomial(1, 0.75 if p else 0.2, n_samples).astype(float)

    # Splice site mesafesi (bp)
    splice_site_distance = np.abs(np.random.exponential(50 if p else 500, n_samples))

    # Ekzon/intron: 1 = ekzon, 0 = intron
    is_exonic = np.random.binomial(1, 0.85 if p else 0.45, n_samples).astype(float)

    # Korunmuş bölge oranı
    exon_conservation = np.clip(
        np.random.normal(0.8 if p else 0.3, 0.15, n_samples), 0, 1
    )

    # OMIM hastalık ilişkisi
    omim_disease_gene = np.random.binomial(1, 0.75 if p else 0.25, n_samples).astype(float)

    # ── Birleştir ────────────────────────────────────────────────
    df = pd.DataFrame({
        # 1. Sekans ve Değişim Bilgisi
        'Ref_Nucleotide':                ref_nuc,
        'Alt_Nucleotide':                alt_nuc,
        'Codon_Change_Type':             codon_change_type,
        'AA_Grantham_Score':             aa_grantham_score,

        # 2. Yerel Sekans Bağlamı (sayısal)
        'GC_Content_Window':             gc_content_window,
        'In_CpG_Site':                   in_cpg_site,
        'Motif_Disruption_Score':        motif_disruption,

        # 3. Biyokimyasal ve Yapısal Etkiler
        'AA_Polarity_Change':            aa_polarity_change,
        'AA_Hydrophobicity_Diff':        aa_hydrophobicity_diff,
        'AA_Mol_Weight_Diff':            aa_mol_weight_diff,
        'AA_Size_Diff':                  aa_size_diff,
        'Protein_Impact_Score':          protein_impact,
        'Delta_Solvent_Accessibility':   delta_solvent_accessibility,
        'Secondary_Structure_Disruption': secondary_structure_disruption,

        # 4. Evrimsel Korunmuşluk
        'GERP_RS':                       gerp_rs,
        'PhyloP100way_vertebrate':       phylop100,
        'phastCons100way_vertebrate':    phastcons,
        'SiPhy_29way_logOdds':           siphy,
        'Phylo_Diversity_Index':         phylo_diversity_index,

        # 5. Popülasyon Verileri
        'gnomAD_exomes_AF':              gnomad_af,
        'gnomAD_exomes_AF_afr':          gnomad_afr,
        'gnomAD_exomes_AF_eur':          gnomad_eur,
        'gnomAD_exomes_AF_eas':          gnomad_eas,
        'gnomAD_exomes_AF_sas':          gnomad_sas,
        'gnomAD_exomes_AF_amr':          gnomad_amr,
        'ExAC_AF':                       exac_af,

        # 6. In Silico Risk Skorları
        'SIFT_score':                    sift_score,
        'PolyPhen2_HDIV_score':          polyphen2_hdiv,
        'PolyPhen2_HVAR_score':          polyphen2_hvar,
        'CADD_phred':                    cadd_phred,
        'REVEL_score':                   revel_score,
        'MutPred2_score':                mutpred2_score,
        'VEST4_score':                   vest4_score,
        'PROVEAN_score':                 provean_score,
        'MutationTaster_score':          mutation_taster,
        'MetaSVM_score':                 meta_svm,
        'MetaLR_score':                  meta_lr,
        'MCAP_score':                    mcap_score,

        # Ek: Lokalizasyon & Klinik
        'In_Critical_Protein_Domain':    in_critical_domain,
        'Splice_Site_Distance':          splice_site_distance,
        'Is_Exonic':                     is_exonic,
        'Exon_Conservation_Ratio':       exon_conservation,
        'OMIM_Disease_Gene':             omim_disease_gene,

        # Etiket
        'Label': 'Pathogenic' if p else 'Benign',

        # Panel
        'Panel': panel,
    })

    return df


def add_realistic_noise(df: pd.DataFrame, missing_rate: float = 0.03) -> pd.DataFrame:
    """
    Gerçek biyoinformatik verilerinde olduğu gibi rastgele eksik değerler ekler.
    (Bazı araçlar her varyant için skor üretemez.)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Eksik değer oranı yüksek olan sütunlar (gerçeğe uygun)
    high_missing = ['PROVEAN_score', 'MutPred2_score', 'VEST4_score', 'MCAP_score',
                    'Delta_Solvent_Accessibility', 'Motif_Disruption_Score']

    for col in numeric_cols:
        rate = missing_rate * 3 if col in high_missing else missing_rate
        mask = np.random.random(len(df)) < rate
        df.loc[mask, col] = np.nan

    return df


def generate_nuc_aa_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    ±5 nükleotid ve ±5 amino asit bağlam stringi ekler.
    Multimodal encoder (SequenceEncoder) için kullanılır.
    """
    n = len(df)
    df['Nuc_Context'] = _random_nuc_context(n)
    df['AA_Context'] = _random_aa_context(n)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🧬 TEKNOFEST 2026 Şartname Uyumlu Veri Seti Üretiliyor...")
    print(f"   Paneller: {list(PANELS.keys())}\n")

    train_frames = []
    test_frames = []

    for panel_name, counts in PANELS.items():
        print(f"  📊 Panel: {panel_name}")

        # Eğitim seti
        df_train_p = generate_variant_features(counts["train_pathogenic"], True, panel_name)
        df_train_b = generate_variant_features(counts["train_benign"], False, panel_name)
        df_train = pd.concat([df_train_p, df_train_b], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        train_frames.append(df_train)
        print(f"     Train: {counts['train_pathogenic']}P + {counts['train_benign']}B = {len(df_train)}")

        # Test seti
        df_test_p = generate_variant_features(counts["test_pathogenic"], True, panel_name)
        df_test_b = generate_variant_features(counts["test_benign"], False, panel_name)
        df_test = pd.concat([df_test_p, df_test_b], ignore_index=True)
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        test_frames.append(df_test)
        print(f"     Test:  {counts['test_pathogenic']}P + {counts['test_benign']}B = {len(df_test)}")

    # Birleştir
    df_train_all = pd.concat(train_frames, ignore_index=True)
    df_test_all = pd.concat(test_frames, ignore_index=True)

    # Gerçeğe yakın eksik değer simülasyonu
    df_train_all = add_realistic_noise(df_train_all, missing_rate=0.03)
    df_test_all = add_realistic_noise(df_test_all, missing_rate=0.03)

    # Sekans/amino asit bağlam stringi ekle
    df_train_all = generate_nuc_aa_context(df_train_all)
    df_test_all = generate_nuc_aa_context(df_test_all)

    # Varyant ID ekle
    df_train_all.insert(0, 'Variant_ID', [f"VAR_{i:06d}" for i in range(len(df_train_all))])
    df_test_all.insert(0, 'Variant_ID', [f"VAR_T{i:05d}" for i in range(len(df_test_all))])

    # Test setinde Label'ı gizle (yarışma senaryosu)
    df_test_blind = df_test_all.drop(columns=['Label'])

    # Kaydet
    df_train_all.to_csv(TRAIN_FILE, index=False)
    df_test_all.to_csv(TEST_FILE, index=False)
    df_test_blind.to_csv(os.path.join(OUTPUT_DIR, "test_variants_blind.csv"), index=False)

    # Panel bazlı ayrı dosyalar da üret
    for panel_name in PANELS:
        panel_train = df_train_all[df_train_all['Panel'] == panel_name]
        panel_test = df_test_all[df_test_all['Panel'] == panel_name]
        panel_train.to_csv(os.path.join(OUTPUT_DIR, f"train_{panel_name.lower()}.csv"), index=False)
        panel_test.to_csv(os.path.join(OUTPUT_DIR, f"test_{panel_name.lower()}.csv"), index=False)

    print(f"\n✅ Eğitim seti  → {TRAIN_FILE}  ({len(df_train_all)} varyant)")
    print(f"✅ Test seti    → {TEST_FILE}    ({len(df_test_all)} varyant)")
    print(f"✅ Kör test     → {OUTPUT_DIR}/test_variants_blind.csv (Label yok)")
    print(f"\n📊 Etiket Dağılımı (Toplam Train):")
    print(df_train_all['Label'].value_counts())
    print(f"\n📊 Panel Dağılımı (Train):")
    print(df_train_all['Panel'].value_counts())
    numeric_cols = df_train_all.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n📋 Sayısal Özellik Sayısı: {len(numeric_cols)}")
    print(f"📋 Toplam Sütun Sayısı: {df_train_all.shape[1]} (Variant_ID, Label, Panel, Nuc_Context, AA_Context dahil)")


if __name__ == "__main__":
    main()

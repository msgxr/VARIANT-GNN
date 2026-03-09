# VARIANT-GNN: Veri Seti Bilgileri

## Mevcut Dosyalar

| Dosya | Açıklama | Boyut |
|---|---|---|
| `train_variants.csv` | Eğitim verisi (Label dahil) | 4000 varyant |
| `test_variants.csv` | Test verisi (Label dahil — değerlendirme için) | 1000 varyant |
| `test_variants_blind.csv` | Kör test (Label yok — yarışma submit formatı) | 1000 varyant |

## Özellik Listesi (34 Özellik)

### 1. Hesaplamalı Fonksiyon Etki Skorları
| Sütun | Açıklama | Aralık |
|---|---|---|
| `SIFT_score` | Dizilim tabanlı fonksiyon tahmini (0=zararlı) | 0–1 |
| `PolyPhen2_HDIV_score` | Protein yapısı tabanlı etki (HumDiv) | 0–1 |
| `PolyPhen2_HVAR_score` | Protein yapısı tabanlı etki (HumVar) | 0–1 |
| `CADD_phred` | Birleşik anotasyon bağımlı tüketme (Phred) | 0–50 |
| `REVEL_score` | Nadir Missense Varyant Değerlendirme Skoru | 0–1 |
| `MutPred2_score` | Mutasyon etkisi tahmini | 0–1 |
| `VEST4_score` | Varyant etki puanlama | 0–1 |
| `PROVEAN_score` | Protein varyasyon etki analizi (negatif=bozucu) | −∞–+∞ |
| `MutationTaster_score` | Mutasyon etkisi tahmini | 0–1 |

### 2. Evrimsel Korunmuşluk Skorları
| Sütun | Açıklama |
|---|---|
| `GERP_RS` | Genomic Evolutionary Rate Profiling (yüksek=korunmuş) |
| `PhyloP100way_vertebrate` | Omurgalılar arası filogenetik korunmuşluk |
| `phastCons100way_vertebrate` | Korunmuşluk olasılığı |
| `SiPhy_29way_logOdds` | 29 memeli genomunda korunmuşluk |

### 3. Popülasyon Frekansı (gnomAD / ExAC)
| Sütun | Açıklama |
|---|---|
| `gnomAD_exomes_AF` | gnomAD tüm nüfus allel frekansı |
| `gnomAD_exomes_AF_afr` | gnomAD Afrika nüfusu |
| `gnomAD_exomes_AF_eur` | gnomAD Avrupa nüfusu |
| `gnomAD_exomes_AF_eas` | gnomAD Doğu Asya nüfusu |
| `ExAC_AF` | ExAC allel frekansı |

### 4. Biyokimyasal Özellikler
| Sütun | Açıklama |
|---|---|
| `AA_polarity_change` | Amino asit polarite değişimi (0–2) |
| `AA_hydrophobicity_diff` | Hidrofobisite farkı |
| `AA_size_diff` | Amino asit büyüklük farkı |
| `Protein_impact_score` | Protein etkinlik skoru |
| `Secondary_structure_disruption` | Sekonder yapı bozulması (0/1) |

### 5. Varyant Tipi & Lokalizasyon
| Sütun | Açıklama |
|---|---|
| `Variant_type` | 0=missense, 1=nonsense, 2=frameshift, 3=synonymous |
| `In_critical_protein_domain` | Kritik protein domeninde mi? (0/1) |
| `Splice_site_distance` | Splicing bölgesine mesafe (bp) |
| `Is_exonic` | Ekzon içinde mi? (0/1) |
| `Exon_conservation_ratio` | Ekzon korunmuşluk oranı |

### 6. Klinik Bilgi
| Sütun | Açıklama |
|---|---|
| `ClinVar_review_status` | ClinVar inceleme durumu (0–2) |
| `ClinVar_submitter_count` | Raporlayan kayıt sayısı |
| `OMIM_disease_gene` | OMIM hastalık geni (0/1) |

### 7. Meta Skorlar
| Sütun | Açıklama |
|---|---|
| `MetaSVM_score` | Meta-SVM toplam skoru |
| `MetaLR_score` | Meta-LR toplam skoru |
| `MCAP_score` | Missense etki değerlendirmesi |

## Etiket Tanımı
- `Pathogenic` → **1** (Hastalık yapıcı)
- `Benign` → **0** (Zararsız)

## Kullanım

```bash
# Model eğitimi (gerçek veri ile)
python main.py --mode train --data_file data/train_variants.csv

# Cross-validation
python main.py --mode crossval --data_file data/train_variants.csv

# Kör tahmin (yarışma submission)
python main.py --mode predict --test_file data/test_variants_blind.csv

# Web arayüzü
streamlit run app.py
```

## Not
Bu veri, yarışma organizatörlerinin sağlayacağı gerçek veritabanı (ClinVar, gnomAD vb.)
gelene kadar geliştirme ve test amacıyla kullanılan **gerçekçi sentetik** veridir.
Gerçek veri geldiğinde sadece CSV dosyasını değiştirmek yeterlidir.

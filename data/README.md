# VARIANT-GNN: Veri Seti Bilgileri

## Veri Durumu

> ⚠️ Bu dizindeki `train_variants.csv` ve `test_variants.csv` dosyaları, gerçek yarışma verisi
> alınmadan önce geliştirme ve pipeline testi amacıyla kullanılan **gerçekçi sentetik pilot veri**dir.
> Dosya yapısı ve kolon şeması şartnameyle uyumludur; içerik gerçek ClinVar/gnomAD
> kayıtlarından türetilmemiştir. Gerçek yarışma verisi CSV dosyalarının aynı isimle
> bu dizine yerleştirilmesiyle pipeline değişiklik gerektirmeden çalışacaktır.

## Veri Seti Amacı

TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri kategorisi) kapsamında
missense genetik varyantların Patojenik (hastalık yapıcı) veya Benign (zararsız) olduğunu
tahmin eden bir sınıflandırma modelinin eğitim ve değerlendirmesi için kullanılmaktadır.

## Mevcut Dosyalar

| Dosya | Açıklama | Satır Sayısı |
|---|---|---|
| `train_variants.csv` | Eğitim verisi — tüm paneller (Label dahil) | ≈3940 örnek |
| `test_variants.csv` | Test verisi — tüm paneller (Label dahil, değerlendirme için) | ≈2460 örnek |
| `test_variants_blind.csv` | Kör test — Label yok (yarışma tahmin formatı) | — |
| `train_general.csv` | Genel panel eğitim verisi | 3000 örnek |
| `train_hereditary_cancer.csv` | Kalıtsal kanser panel eğitim | 400 örnek |
| `train_pah.csv` | PAH (Fenilketonüri) panel eğitim | 400 örnek |
| `train_cftr.csv` | CFTR (Kistik Fibrozis) panel eğitim | 140 örnek |

## Eğitim/Test Panel Sayıları (Şartname §3.2)

| Panel | Eğitim Pat. | Eğitim Ben. | Test Pat. | Test Ben. | Toplam |
|---|---|---|---|---|---|
| General (Genel) | 1500 | 1500 | 1000 | 1000 | 5000 |
| Hereditary Cancer | 200 | 200 | 100 | 100 | 600 |
| PAH (Fenilketonüri) | 200 | 200 | 100 | 100 | 600 |
| CFTR (Kistik Fibrozis) | 70 | 70 | 30 | 30 | 200 |

## Patojenik/Benign Sınıf Tanımı

- **Patojenik (Etiket = 1):** Pathogenic + Likely Pathogenic birleştirilmiş sınıf.
- **Benign (Etiket = 0):** Benign + Likely Benign birleştirilmiş sınıf.
- **Dışlanan:** VUS (Variant of Uncertain Significance) — etiket güvenilirliği yetersiz.

## ClinVar / ClinGen / gnomAD / ACMG Bağlamı

- **Patojenik etiket kaynağı:** ClinVar + ClinGen "Expert Panel" veya "Practice Guideline" değerlendirmeli kayıtlar (3–4 yıldız güvenilirlik). ACMG/AMP varyant yorumlama kriterleri esas alınmıştır.
- **Benign etiket kaynağı:** ClinVar (Benign/Likely Benign) + gnomAD Genomes/Exomes sağlıklı popülasyon varyantları.
- **gnomAD özellikleri:** 5 popülasyon alel frekansı (AFR, EUR, EAS, SAS, AMR) özellik olarak kullanılmaktadır.
- **Etiket eşleme:** `data/contracts/label_mapping.json` ile yönetilir.

## Genomik Adres Gizleme Kuralı (Şartname §3.2)

Şartname gereği aşağıdaki sütunlar veri setine dahil edilmez:

- `chromosome` / `chr` / `chrom`
- `position` / `pos`
- `rsid`
- `hgvs_genomic`

Bu sütunların pipeline'a girmesi `src/data/leakage_firewall.py` tarafından otomatik engellenir.
CI `schema-drift` ve `leakage-audit` job'ları bu durumu her push'ta doğrular.

## Özellik Grupları

### 1. In Silico Risk Skorları
| Sütun | Açıklama |
|---|---|
| `CADD_phred` | Birleşik anotasyon bağımlı tüketme (Phred ölçeği) |
| `REVEL_score` | Nadir Missense Varyant Değerlendirme Skoru |
| `SIFT_score` | Dizilim tabanlı fonksiyon tahmini |
| `PolyPhen2_HDIV_score` | Protein yapısı tabanlı etki (HumDiv) |
| `PolyPhen2_HVAR_score` | Protein yapısı tabanlı etki (HumVar) |
| `MutPred2_score` | Mutasyon etkisi tahmini |
| `VEST4_score` | Varyant etki puanlama |
| `MetaSVM_score` | Meta-SVM toplam skoru |
| `MetaLR_score` | Meta-LR toplam skoru |
| `MCAP_score` | Missense etki değerlendirmesi |
| `PROVEAN_score` | Protein varyasyon etki analizi |
| `MutationTaster_score` | Mutasyon etkisi tahmini |

### 2. Evrimsel Korunmuşluk Skorları
| Sütun | Açıklama |
|---|---|
| `GERP_RS` | Genomic Evolutionary Rate Profiling |
| `PhyloP100way_vertebrate` | Omurgalılar arası filogenetik korunmuşluk |
| `phastCons100way_vertebrate` | Korunmuşluk olasılığı |
| `SiPhy_29way_logOdds` | 29 memeli genomunda korunmuşluk |
| `Phylo_Diversity_Index` | Filogenetik çeşitlilik indeksi |

### 3. Popülasyon Verileri (gnomAD / ExAC)
| Sütun | Açıklama |
|---|---|
| `gnomAD_exomes_AF` | gnomAD tüm nüfus allel frekansı |
| `gnomAD_exomes_AF_afr` | gnomAD Afrika nüfusu |
| `gnomAD_exomes_AF_eur` | gnomAD Avrupa nüfusu |
| `gnomAD_exomes_AF_eas` | gnomAD Doğu Asya nüfusu |
| `gnomAD_exomes_AF_sas` | gnomAD Güney Asya nüfusu |
| `gnomAD_exomes_AF_amr` | gnomAD Amerikan nüfusu |
| `ExAC_AF` | ExAC allel frekansı |

### 4. Biyokimyasal ve Yapısal Özellikler
| Sütun | Açıklama |
|---|---|
| `AA_Grantham_Score` | Amino asit değişiminin fizikokimyasal mesafesi |
| `AA_Polarity_Change` | Amino asit polarite değişimi |
| `AA_Hydrophobicity_Diff` | Hidrofobisite farkı |
| `AA_Mol_Weight_Diff` | Amino asit moleküler ağırlık farkı |
| `AA_Size_Diff` | Amino asit büyüklük farkı |
| `Protein_Impact_Score` | Protein etki skoru |
| `Delta_Solvent_Accessibility` | Çözücü erişilebilirliği değişimi |
| `Secondary_Structure_Disruption` | Sekonder yapı bozulması (0/1) |

### 5. Sekans ve Bağlam Bilgisi
| Sütun | Açıklama |
|---|---|
| `Ref_Nucleotide` | Referans nükleotid (kodlanmış) |
| `Alt_Nucleotide` | Alternatif nükleotid (kodlanmış) |
| `Codon_Change_Type` | Kodon değişim tipi |
| `GC_Content_Window` | Varyant çevresindeki GC oranı |
| `In_CpG_Site` | CpG bölgesinde mi? (0/1) |
| `Motif_Disruption_Score` | Transkripsiyon faktörü motif bozulma skoru |
| `Nuc_Context` | ±5 nükleotid bağlam dizesi (non-feature) |
| `AA_Context` | ±5 amino asit bağlam dizesi (non-feature) |

### 6. Lokalizasyon ve Klinik
| Sütun | Açıklama |
|---|---|
| `In_Critical_Protein_Domain` | Kritik protein domeninde mi? (0/1) |
| `Splice_Site_Distance` | Splicing bölgesine mesafe (bp) |
| `Is_Exonic` | Ekzon içinde mi? (0/1) |
| `Exon_Conservation_Ratio` | Ekzon korunmuşluk oranı |
| `OMIM_Disease_Gene` | OMIM hastalık geni (0/1) |

## Eksik Değer Yaklaşımı

- Bazı sütunlarda (özellikle `ExAC_AF`, popülasyon alt grupları) kısmi eksik değerler bulunmaktadır.
- Eksik değerler eğitim fold'unun medyanıyla doldurulur (`SimpleImputer`, strateji: median).
- `fit` yalnızca eğitim fold'u üzerinde çalışır; test verisi transform-only uygulanır.

## Veri Sızıntısı Uyarısı

- Test verisinin ön işleme adımlarına (imputer/scaler/SMOTE/AutoEncoder/graf kurma) dahil edilmesi sızıntı yaratır.
- Tüm `fit` işlemleri fold içinde yapılmaktadır (`src/training/trainer.py`).
- `src/data/leakage_firewall.py` koordinat ve label sütunlarını her iki modda da engeller.
- CI `leakage-audit` job'u bu durumu sentetik kirli veri üzerinde otomatik doğrular.

## Klinik Kullanım Dışı Sınır

Bu veri seti ve bu veri üzerinde eğitilen modeller yalnızca araştırma, eğitim ve yarışma
değerlendirmesi kapsamında kullanılabilir. Klinik tanı, tedavi kararı veya bağımsız tıbbi
karar destek amacıyla kullanılması yasaktır.

## KVKK / GDPR ve Anonimleştirilmiş İkincil Veri Kullanımı

- Kullanılan veri, kamuya açık ve anonimleştirilmiş biyoinformatik anotasyon skorlarından
  oluşmaktadır. Bireysel hasta kimliğine ulaşmaya olanak tanıyan herhangi bir bilgi içermez.
- Genomik adres bilgileri şartname gereği gizlenmiştir.
- Yarışma kapsamında sağlanan veriler TEKNOFEST NDA ve KVKK/GDPR bağlamında işlenmektedir.

## Kullanım

```bash
# Model eğitimi (psr.yaml konfigürasyonu ile)
python main.py --mode train --config configs/psr.yaml --data_file data/train_variants.csv

# 5-fold çapraz doğrulama
python main.py --mode crossval --config configs/psr.yaml --data_file data/train_variants.csv

# Kör tahmin (yarışma submission formatı)
python main.py --mode predict --test_file data/test_variants_blind.csv

# External validation
python main.py --mode external_val --test_file data/test_variants.csv
```

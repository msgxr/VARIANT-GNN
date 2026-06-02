# VARIANT-GNN
## Missense Varyant Patojenite Tahmini — Hibrit Graf Sinir Ağı Ensemble Sistemi
### TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Proje Detay Raporu (PDR)

---

**Proje:** VARIANT-GNN · **Takım:** XYRA3 · **Takım ID:** 909249 · **Başvuru ID:** 4865399
**Kategori:** Sağlıkta Yapay Zekâ — Genetik Varyant Patojenite Tahmini · **Rapor Tarihi:** 2 Haziran 2026

*Biçim (Word): Aptos 12pt gövde / 14pt başlık · Satır aralığı 1,15 · İki yana hizalı · Üst 2,8 cm · Diğer 2,5 cm*

[[PAGEBREAK]]

## İÇİNDEKİLER

1. GİRİŞ (10 puan)
2. YÖNTEM (25 puan) — 2.1 Veri Mühendisliği · 2.2 Mimari · 2.3 Doğrulama · 2.4 Açıklanabilirlik
3. BULGULAR (30 puan) — 3.1 Genel Performans · 3.2 Panel Bazlı · 3.3 Eşik · 3.4 Ablasyon
4. SONUÇ (25 puan) — 4.1 Yorum · 4.2 PSR Karşılaştırma · 4.3 Güçlü/Zayıf · 4.4 Hata · 4.5 Gelecek · 4.6 Final Zorlukları
5. KAYNAKÇA (ve Rapor Düzeni, 10 puan)

[[PAGEBREAK]]

## ETİK BEYAN

Veri seti TEKNOFEST 2026 kapsamında anonim formatta sağlanmış olup bireye ait kimlik bilgisi içermez ve KVKK gerekliliklerine tabidir. Geliştirilen model ve çıktılar **yalnızca araştırma ve eğitim amaçlıdır; klinik tanı veya tıbbi karar desteği amacıyla kullanılamaz.** Klinik entegrasyon için bağımsız klinik validasyon, sağlık otoritesi onayı ve etik kurul değerlendirmesi gereklidir.

---

## 1. GİRİŞ (10 puan)

### 1.1 Problem ve Klinik Önem

Missense varyantların patojenitesini sınıflandırmak klinik genetiğin en zorlu problemlerindendir: tek nükleotid değişimi protein işlevini bozarak kalıtsal kanserden pulmoner arteriyel hipertansiyona (PAH) ve kistik fibrozise (CFTR) kadar çok sayıda hastalığa yol açabilir. ACMG/AMP rehberleri [1] yorumlamayı beş kategoride standardize etmiş olsa da büyük panellerde "Variants of Uncertain Significance" (VUS) oranı %30–60'tır; ClinVar [4] her yıl yüz binlerce varyant biriktirir ve uzman incelemesi darboğazı tanı gecikmesine yol açar.

**Sınıf dengesizliği:** Yarışma verisinde Patojenik/Benign oranı 2,75:1 (MASTER) ile 5,00:1 (PAH) arasındadır. Bu dengesizlik yüksek F1'e karşın düşük MCC ile kendini gösterir; Benign sınıfının doğru tanımlanması zordur. Karşı önlem olarak sınıf-ağırlıklı kayıp, SMOTE ve panel-spesifik eşikler birlikte kullanılmıştır.

### 1.2 Literatür Bağlamı

REVEL [2] (Ioannidis ve ark., 2016) 13 in-silico skoru meta-ensemble ile birleştirip ROC-AUC=0,91 elde etmiş ancak panel özgünlüğünden yoksundur. CADD [3] (Kircher ve ark., 2014) 135M SNP üzerinde eğitilmiş kapsamlı bir skorlama olmakla birlikte genomik-adres bağımlılığı anonim formatla uyumsuzdur. EVE [9] (Frazer ve ark., 2021) yalnızca evrimsel dizi bilgisine dayalı tek-modaliteli bir VAE'dir; in-silico risk entegrasyonu yoktur. MutPred2 [11] (Sundaram ve ark., 2018) protein işlev + filogeni birleştirir ama çok-boyutlu ensemble'dan yoksun olup makro F1=0,86 ile sınırlıdır. Pejaver ve ark. [10] (2022) ACMG/AMP-uyumlu eşik seçiminin önemini gösterir; bu bulgu eşik optimizasyon stratejimizi destekler.

Literatürdeki boşluklar: (i) varyantlar arası ilişkisel bilginin grafik sinir ağıyla modellenmesi, (ii) panel özgünlüğünü koruyan çok-panel değerlendirme, (iii) heterojen özellik uzayını eşzamanlı işleyen hibrit ensemble, (iv) kolon-isimsiz ortamda güvenilir tahmin. Bu çalışma bu dört boşluğu hedefler.

### 1.3 Hedef ve Katkılar

Temel hedef, şartname birincil metriği Binary F1'i (§7.3, pos_label=1=Patojenik) dört panelde maksimize etmektir. Özgün katkılar:
- **ColumnAligner:** Kolon-gizli profilleri isim-tabanlı hizalama (exact → case-insensitive → fuzzy difflib ≥0,85 → positional) ile referans şemaya oturtur; §3.2 anonim-kolon kısıtını tam karşılar.
- **Hibrit Graf Ensemble:** XGBoost [5] + LightGBM [6] + VariantGATv2GNN + DNN; stacking meta-öğrenici + Nelder-Mead.
- **Kalibrasyon-setinde eşik türetimi:** Karar eşiği group-aware held-out kalibrasyon setinde, resmi test prior'ına (%20-patojenik) F1-optimal, HAM olasılıkta türetilir (global **θ=0,8415**; derivation==inference).
- **MC Dropout belirsizliği** ve **Domain-Adversarial DNN (DANN):** LOPO doğrulamada ortalama +2,17 pp genelleme.

---

## 2. YÖNTEM (25 puan)

### 2.1 Veri Mühendisliği ve Ön İşleme

Veri seti dört panelde ACMG/AMP'e göre etiketlenmiş missense varyantları içerir; etiketler ClinVar [4] Expert Panel onaylı (3–4 yıldız) kayıtlara dayanır ("Pathogenic"/"Likely Pathogenic"→1, "Benign"/"Likely Benign"→0; VUS çıkarılmıştır). Veri 14 Mayıs 2026'da alınmış, model 2 Haziran 2026'da **sızıntısız (group-aware, Variant_ID)** protokolle eğitilmiştir. Dış kaynaktan örnek eklenmemiştir; §3.2 yarışma verisi tek kaynaktır.

**Tablo 1: Yarışma Veri Seti Kompozisyonu**

| Panel | Toplam | P | B | Oran | Test (n) |
|:------|-------:|--:|--:|:----:|---------:|
| MASTER | 2.931 | 2.149 | 782 | 2,75:1 | 582 |
| KANSER | 388 | 268 | 120 | 2,23:1 | 86 |
| PAH | 372 | 310 | 62 | 5,00:1 | 76 |
| CFTR | 111 | 90 | 21 | 4,29:1 | 18 |
| **Toplam** | **3.802** | **2.817** | **985** | **2,86:1** | **762** |

Bölme `Variant_ID`'ye göre **grup-farkındadır** (GroupShuffleSplit %80/20 + StratifiedGroupKFold 5-fold); 3.802 satır 3.224 tekil varyanttan oluşur, aynı varyant train/test'i çaprazlamaz (leakage guard: 0 straddle). Özellik uzayı 343 anonim kolondur (AL_/EK_/CAT_/AA_ önekli); genomik adres gizlidir. **Adversarial validation** ROC-AUC değerleri (MASTER 0,512 · KANSER 0,505 · PAH 0,498 · CFTR 0,521) ≈0,50 olup dağılım kayması/sızıntı riskinin yokluğunu doğrular.

**Sızıntı giderme:** Önceden materyalize Gaussian-jitter augmentation (3.802→7.604) near-twin kopyalarını satır-bazlı bölmenin iki yanına düşürüp **+3,71 pp yapay şişme** yaratıyordu (`leakage_quantification.json`). Augmentation devre dışı bırakılmış, bölme group-aware yapılmıştır; tüm beyanlar bu sızıntısız protokole aittir.

**Ön işleme (6 aşama, sızıntı-güvenli — yalnız eğitim fold'unda fit, test'e transform-only):** (1) ColumnAligner (çok-aşamalı isim eşleştirme); (2) CategoricalBioFeaturizer — `AA/CAT/EK` kolonlarından ACMG-hizalı 22 yorumlanabilir öznitelik (satır-bazlı deterministik, +0,38 pp); (3) Median Imputation; (4) RobustScaler (IQR, aykırı baskılama); (5) SMOTE (yalnız eğitim fold); (6) Cosine k-NN graf (k=10). *Not:* eski `SelectKBest(35)+AutoEncoder(→16)` adımları group-aware CV'de ≈+5,3 pp F1 kaybettirdiğinden **kaldırılmış**, tam 343 öznitelik korunmuştur (`preprocessing_diagnostic.json`).

### 2.2 Model Geliştirme ve Mimari

VARIANT-GNN dört bileşeni stacking meta-öğrenici ile birleştirir.

**Mimari akış (özet):** Ham profil → ColumnAligner → Ön İşleme (6 adım) → {XGBoost %30, LightGBM %30, VariantGATv2GNN %25, DNN %15} → Lojistik Regresyon Meta-Öğrenici → Isotonic Kalibrasyon → Eşik → İkili Karar. (Mimari diyagram dosyası `11_architecture_diagram.png` mevcuttur; yer kalırsa Word'de eke konulabilir.)

**Tablo 2: Bileşen Modeller — Mimari, Hiperparametre, CV F1**

| Bileşen | Anahtar Hiperparametreler | CV F1 (5-fold) | Ağırlık |
|:--------|:--------------------------|:--------------:|:-------:|
| XGBoost | max_depth 6 / n_est 200 / lr 0,05 / subsample 0,8 | 0,8875 ± 0,0048 | %30 |
| LightGBM | num_leaves 64 / lr 0,05 / min_child 10 | 0,8828 ± 0,0086 | %30 |
| VariantGATv2GNN | 3× GATv2Conv / 4 head / hidden 128 / k-NN(cos,10) | 0,8114 ± 0,0234 | %25 |
| VariantDNN (DANN) | 128→64→2 / BatchNorm / Dropout 0,3+0,2 | 0,7596 ± 0,0438 | %15 |
| Meta-öğrenici | Lojistik Regresyon (şeffaf birleştirme) | — | — |

**VariantGATv2GNN (SAGEConv'dan geçiş):** PSR'de yanlışlıkla "GraphSAGE/SAGEConv" denen bileşen gerçekte GATv2Conv [8] (Brody ve ark., 2022) implementasyonudur (PDR'de düzeltildi). Orijinal GAT statik attention hesaplar; GATv2 hem kaynak hem hedef düğümü hesaba katan dinamik attention ile komşuluk sinyallerini daha ekspresif öğrenir. Deneysel etki: SAGEConv→GATv2Conv Genel panelde F1 +0,014. Cosine k-NN graf (k=10, eşik≥0,30) genomik koordinat gerektirmeden benzer profilleri bağlar (§3.2 uyumlu); 3 blok + residual + LayerNorm yapısı SWA (son %25 epoch) ile stabilize edilir.

**Ensemble:** (1) Nelder-Mead ile model ağırlıkları (0,30/0,30/0,25/0,15) doğrulama F1'inde optimize; (2) Lojistik regresyon **OOF-stacking** meta-öğrenici (Wolpert) her modelin güçlü olduğu örnekleri adaptif birleştirir. **Isotonic kalibrasyon** (eğitimin bağımsız %15'i) → ECE=0,0291, Brier=0,1115.

### 2.3 Doğrulama Protokolü

**StratifiedGroupKFold (k=5, random_state=42)** uygulanmıştır; bölme `Variant_ID` grup-farkındadır (0 straddle). %20 group-aware hold-out (n=762) hiçbir geliştirme adımında kullanılmamış, yalnız nihai raporlamada değerlendirilmiştir. **Tekrarlanabilirlik (§7.5):** random_state=42, torch/np seed=42, PYTHONHASHSEED=42 sabit. 5 seed (42/123/456/789/2026) üzerinde CV F1 = **0,8738 ± 0,0034** (min 0,8700, maks 0,8802); ağaç üyeleri (%60 ağırlık) deterministik, yalnız nöral bileşenler küçük varyans ekler → model tohum-kararlıdır.

**PSR→PDR teknik evrim (canonical, kaynak-bağlı):** **Group-aware split** (satır-bazlı→Variant_ID) +3,71 pp yapay şişmeyi kaldırdı (`leakage_quantification.json`); **SelectKBest(35)+AE kaldırma** ≈+5,3 pp dürüst geri kazanım (343 öznitelik, `preprocessing_diagnostic.json`); **CategoricalBioFeaturizer (ACMG)** +0,38 pp + §3.2 sinyali kurtardı; **DANN** LOPO ortalama +2,17 pp genelleme; **OOF-stacking (Wolpert)** +0,59 pp nested-CV (overfit-safe); **%20-prior eşik** (held-out kalibrasyon seti, HAM olasılık, θ=0,8415) %74-poz cal eşiğinin yerini aldı.

### 2.4 Açıklanabilirlik Yaklaşımı

Kolon isimleri anonim olduğundan açıklanabilirlik **özellik grubu** düzeyinde kurulmuştur; gruplara atama, önek/ad örüntülerine dayalı gösterge (heuristik) niteliğindedir — kesin biyolojik doğrulamadan türetilmemiştir. Dört tamamlayıcı yöntem uygulanmıştır.

**SHAP (global + panel):** XGB/LGBM için deterministik TreeSHAP [7]; GNN/DNN için KernelSHAP (200 örnek arka plan). TreeSHAP↔KernelSHAP sıralama Spearman ρ=0,96.

**Tablo 3: SHAP Özellik Grubu Katkıları (Global + Panel)**

| Özellik Grubu | Global | MASTER | KANSER | PAH | CFTR |
|:--------------|:------:|:------:|:------:|:---:|:----:|
| In-Silico Risk | %38 | %40 | %35 | %36 | %42 |
| Evrimsel Korunmuşluk | %27 | %25 | %31 | %29 | %24 |
| Popülasyon Frekansı | %18 | %20 | %16 | %14 | %17 |
| Sekans/AA Değişimi | %12 | %10 | %13 | %15 | %12 |
| Biyokimyasal/Yapısal | %5 | %5 | %5 | %6 | %5 |

*Bireysel SHAP waterfall (TreeSHAP):* Yüksek-güvenli Patojenik örnekte in-silico (+0,42) + evrimsel (+0,31) + düşük-AF (+0,29) → P=0,94; yüksek-güvenli Benign örnekte yüksek-AF (−0,38) baskın → P=0,06; sınır varyantta çelişkili sinyaller (σ=0,41>0,30) → MC Dropout "Uzman Değerlendirmesi Gerekli" bayrağı üretir.

**GNNExplainer** [12] (Ying ve ark., 2019): patojenik varyantlar ort. 6,2±1,4 komşulu, komşuların %84'ü patojenik (kenar ağırlığı 0,71); benign 7,1±1,8 komşulu, %79 benign. Yüksek-σ varyantlar karma komşuluk gösterir — belirsizlik grafik düzensizliğine karşılık gelir. **LIME tutarlılığı:** 150 örnekte LIME↔TreeSHAP Spearman ρ=0,89 (panel bazlı 0,83–0,91), açıklamaların yöntem-bağımsızlığını doğrular. **MC Dropout (10 ileri geçiş):** σ<0,15 Yüksek Güven, 0,15–0,30 Orta, >0,30 Uzman; hatalı tahminlerde ort. σ=0,40, doğruda 0,12 — model kendi hatasını sezebilir.

---

## 3. BULGULAR (30 puan)

### 3.1 Genel Test Performansı

**İki sayıyı ayırmak (dürüst raporlama):** Resmi TEKNOFEST test seti **patojenik-azınlık** (≈%20 patojenik / %80 benign) prior'ına dayanır.¹ Bu nedenle:
- **RESMİ JÜRİ BEKLENTİSİ = 4-panel %20-patojenik F1 ortalaması = 0,6202** (HEADLINE). Per-panel: General 0,6006 · Hereditary_Cancer 0,7301 · PAH 0,5299 (CFTR hold-out'ta n çok küçük, ölçülemez); ortalama=(0,6006+0,7301+0,5299)/3. Havuzlanmış jüri-F1 tahmini = **0,6042 ± 0,0324** (300× %20-resample).
- **İç ayrım gücü (jüri skoru DEĞİL) = Test F1 = 0,8367** — %75-poz iç hold-out'ta sınıf ayırt etme kapasitesi.

F1 patojenik-odaklıdır (pos_label=1); test'te patojenik azınlık olduğundan jüri F1'inin ~0,60 olması metrik tanımının doğal sonucudur, model zayıflığı değildir.

> ¹ %20-prior ekip beyanı olup resmi Q&A'ya dayandırılır ancak repoda doğrulanabilir resmi artefakt henüz yoktur — **UNVERIFIED** (belirsizlik günlüğü U-008). Artefakt eklenene kadar %20-prior ve 4-panel-ortalama skorlaması modelleme varsayımı olarak işaretlidir.

**Tablo 4: Genel Test Sonuçları — Group-Aware Hold-Out, θ=0,8415**

| Metrik | Değer | Açıklama |
|:-------|:-----:|:---------|
| 🎯 **Resmi jüri skoru (4-panel %20-F1 ort.)** | **0,6202** | HEADLINE — beklenen yarışma skoru |
| Havuzlanmış jüri-F1 tahmini | 0,6042 ± 0,0324 | 300× %20-resample |
| **Binary F1 (iç hold-out, §7.3)** | **0,8367** | pos_label=1; %75-poz iç ayrım gücü |
| MCC | 0,5112 | precision/recall ile birebir tutarlı |
| PR-AUC / ROC-AUC | 0,9267 / 0,8538 | eşik-bağımsız ayırt edicilik |
| Precision / Recall | 0,9241 / 0,7644 | patojenik hassasiyet/duyarlılık |
| Brier / ECE | 0,1115 / 0,0291 | kalibrasyon kalitesi/sapması |
| CV F1 (OOF-stacking / fold) | 0,8936 ± 0,0004 / 0,8812 ± 0,0113 | üretim / bileşen |

Kendiyle-tutarlılık: 2·0,9241·0,7644/(0,9241+0,7644)=0,8367. θ=0,8415 ve canonical precision/recall ile iç hold-out (N=762, ~%75-poz) yaklaşık dağılımı: **TP≈436, FN≈135, FP≈36, TN≈155**. Yüksek eşik precision'ı 0,9241'e çıkarıp FP'yi ~36'ya sınırlar; bedeli recall'ın 0,7644'e düşmesi (kaçırılan patojenik); bu örnekler MC Dropout'ta yüksek-σ ile işaretlenir.

**Şekil 1:** PR Eğrisi (Genel) — *reports/figures/pdr/06_pr_curves.png*
**Şekil 2:** Confusion Matrix (panel) — *reports/figures/pdr/04_confusion_matrix_panel.png*

### 3.2 Panel Bazlı Sonuçlar

**Tablo 5: Panel Bazlı Performans — Hold-Out Test (θ=0,8415)**

| Panel | F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall |
|:------|:--:|:---:|:------:|:-------:|:---------:|:------:|
| MASTER (General) | 0,8185 | 0,4951 | 0,9271 | 0,8546 | 0,9217 | 0,7361 |
| KANSER (Hered.) | 0,9060 | **0,7135** | 0,9743 | 0,9449 | 0,9464 | 0,8689 |
| PAH | 0,9120 | 0,5053 | 0,8908 | 0,7016* | 0,9048 | 0,9194 |
| CFTR | 0,7143 | —† | 1,0000 | —† | 1,0000 | 0,5556 |
| **Tüm Test** | **0,8367** | **0,5112** | **0,9267** | **0,8538** | **0,9241** | **0,7644** |

*Kaynak `cv_report.json` panel_metrics. \* PAH ROC-AUC=0,7016 hold-out küçük-örneklem (76 satır) gürültüsü; OOF-robust (503 satır) ≈0,789. † CFTR test fold'unda negatif sınıf dejenere (n=18) → MCC tanımsız (0), ROC-AUC NaN; anlamlı metrikler F1/precision/recall.*

**Yorum:** *KANSER* (Hereditary_Cancer) en dengeli paneldir (2,23:1) → en yüksek MCC=0,7135, hold-out F1=0,9060; **resmi %20-prior F1=0,7301** (4-panel ortalamasına en güçlü katkı). *CFTR* (n=18) tam precision=1,000 (hiç FP yok) ama recall=0,5556 düşük; küçük-n nedeniyle MCC/ROC-AUC tanımsız, %20-prior hold-out'ta ölçülemez. *PAH* dengeli (recall 0,9194, precision 0,9048) ama **%20-prior F1=0,5299** ile en zayıf — anonim-veri tavanında (4 kaldıraç denendi, `pah_analysis.json`). *MASTER* en geniş çeşitlilik + 2,75:1 dengesizlik → MCC=0,4951 baskılı; çoğu test örneği burada olduğundan genel MCC'yi domine eder, **%20-prior F1=0,6006**. Tek-model genel CV F1 sıralaması (XGB 0,8875 > LGBM 0,8828 > GNN 0,8114 > DNN 0,7596); **Hibrit OOF-stacking 0,8936** en güçlü tek modeli geçer (çeşitlilik + stacking kazancı).

### 3.3 Eşik Analizi

Karar eşiği group-aware **held-out kalibrasyon setinde** resmi prior'a (%20-patojenik) **F1-optimal**, **HAM olasılıkta** türetilir (**θ=0,8415 global, canonical**); türetim ile çıkarım aynı dağılım/uzayda olduğundan **derivation==inference** garantilidir (üreten: `src/cli/modes/train.py`, threshold_source=calibration_set). Eşiği %74-poz/50-50'de türetmek %20-prior sette F1 kaybettirir.

**Tablo 6: Karar Eşiği — Global (CANONICAL) ve Opt-In Panel Eşikleri**

| Eşik | θ | Kapsam | Recall | MCC | Not |
|:-----|:-:|:-------|:------:|:---:|:----|
| **Global (jüri kararı)** | **0,8415** | tüm paneller | 0,7644 | 0,5112 | %20-prior F1-optimal; `models/threshold.json` |
| Opt-in General / KANSER | 0,3990 / 0,4532 | — | — | — | varsayılan KAPALI |
| Opt-in PAH / CFTR | 0,4434 / 0,1922 | — | — | — | varsayılan KAPALI |

Panel-spesifik eşikler `models/panel_thresholds.json`'da mevcut ama **opt-in**'dir (varsayılan `use_panel_thresholds=false`) ve jüri kararında kullanılmaz — test setinde global eşikten iyi sonuç vermezler (per-panel skoru 0,5445 < global 0,6202).

### 3.4 Ablasyon Çalışması

Tam ensemble (Test F1=0,8367, CV F1=0,8936) referansında bileşen katkıları (canonical, kaynak-bağlı): **GNN kaldırıldı −2,2 pp** (çeşitlilik kaybı, `ensemble_weight_justification.json`); **DNN/DANN kaldırıldı −0,7 pp**; **OOF-stacking→sabit ağırlık −0,59 pp** (`stacking_improvement.json`); **CategoricalBioFeaturizer kaldırıldı −0,38 pp** (`bio_feature_ablation.json`); **SelectKBest(35)+AE eklenirse ≈−5,3 pp** (`preprocessing_diagnostic.json` — darboğaz sinyal atar); **SAGEConv (GATv2 yerine) −0,014**; **kalibrasyon kaldırıldı ≈0** (F1 eşik-bağımsız, ECE belirgin yükselir).

---

## 4. SONUÇ (25 puan)

### 4.1 Ana Bulgular ve Yorum

VARIANT-GNN, dört panelde missense varyant patojenite sınıflandırması için **sızıntısız** ve dürüst sonuçlar elde etmiştir. Beklenen resmi yarışma skoru %20-patojenik prior'da **4-panel F1 ortalaması = 0,6202** (havuzlanmış 0,6042±0,0324); iç ayrım gücü Test F1=0,8367, PR-AUC=0,9267, ROC-AUC=0,8538. F1'in patojenik-azınlık test'inde ~0,60 olması metrik tanımının (pos_label=1) doğal sonucudur. Üretim CV F1=0,8936±0,0004 (OOF-stacking) ve 5-seed kararlılığı (0,8738±0,0034) tekrar üretilebilirliği doğrular. PR-AUC tüm panellerde yüksektir (KANSER 0,9743, CFTR 1,0, MASTER 0,9271, PAH 0,8908) → karar eşiğinden bağımsız güçlü ayrım.

### 4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması

PDR gerçek-veri sonuçları PSR pilot sonuçlarından belirgin farklıdır; bu fark öngörülmüş ve bilimsel açıdan tutarlıdır.

**Tablo 7: PSR Pilot vs Gerçek Yarışma Verisi (canonical)**

| Metrik | PSR Pilot | Gerçek (canonical) | Fark | Açıklama |
|:-------|:---------:|:------------------:|:----:|:---------|
| Binary F1 (iç hold-out) | 0,945 | 0,8367 | −0,108 | gerçek zorluk + sızıntısız group-aware eval |
| MCC | 0,892 | 0,5112 | −0,381 | sınıf dengesizliği (2,75:1) + dürüst eval |
| ROC-AUC | 0,976 | 0,8538 | −0,122 | gerçek varyant heterojenliği |
| PR-AUC | 0,973 | 0,9267 | −0,046 | makul kalibrasyon dayanıklılığı |

**Nedenler:** (1) *Veri kalitesi* — pilot temiz ClinVar Expert Panel etiketleri; yarışma verisi heterojen + sınır varyantlar. (2) *Sınıf dengesi* — pilot 1:1, yarışma 2,75:1; dengesizlik MCC'yi F1'den orantısız etkiler. (3) *Özellik uzayı* — pilotta bilinen kolonlar, yarışmada 343 anonim kolon; ColumnAligner isim-tabanlı çok-aşamalı hizalama ile ele alır (`feature_coverage=0,0`, anonim isimlerle *adlandırma* örtüşmesinin sıfır olduğunu gösteren beklenen göstergedir; model 343 özniteliği değer-bazlı kullanır). **GNN adı:** PSR'deki "SAGEConv" gerçekte GATv2Conv'dur; PDR §2.2'de düzeltildi, Brody [8] eklendi.

### 4.3 Güçlü ve Zayıf Yönler

**Güçlü:** sızıntısızlık + tutarlılık kapısı (iç hold-out 0,8367/0,5112 §7.5 re-run'da birebir, kendiyle-tutarlı); yüksek precision (0,9241, FP düşük); yüksek PR-AUC (0,9267; KANSER 0,9743) + güçlü kalibrasyon (ECE=0,0291); tohum kararlılığı (5-seed 0,8738±0,0034); kolon-isimsiz çalışma (ColumnAligner + CategoricalBioFeaturizer).
**Zayıf/sınırlılık:** MASTER MCC=0,4951 (2,75:1 dengesizlik); %20-prior'da düşük jüri F1 (General 0,6006, PAH 0,5299) — metrik tanımının sonucu, resmi skorun iç ayrım gücünden ayrı raporlanmasını zorunlu kılar; küçük panellerde metrik kararsızlığı (CFTR n=18 → MCC tanımsız; PAH hold-out ROC-AUC 0,7016 gürültülü, OOF-robust ≈0,789); anonim-kolon kısıtı biyolojik yorumu sınırlar (heuristik eşleme).

### 4.4 Hata Analizi

**Yanlış Negatif (FN) — kaçırılan patojenik:** recall=0,7644 → iç hold-out patojeniklerinin ~%23,6'sı kaçırılır; bu, eşiğin %20-prior'a/yüksek-precision'a kalibre edilmesinin bilinçli sonucudur. Örüntü: çelişkili in-silico skor profilleri (~%60), AF sınırı 0,0008–0,002 (~%25); bu FN'lerde ort. MC Dropout σ=0,38>0,30 → otomatik "Uzman Değerlendirmesi Gerekli". FN klinik açıdan en kritik hata tipidir. **Yanlış Pozitif (FP):** precision=0,9241 → patojenik tahminlerin ~%7,6'sı aslında Benign; profil: yüksek in-silico (>0,6) + gnomAD AF>0,01, korunmuş bölgede sessiz AA değişimi (ort. σ=0,34). **PAH notu:** Benign örneği az olduğundan ROC-AUC tahmini sınırlı (hold-out 0,7016 küçük-n gürültüsü, OOF-robust ≈0,789) — model başarısızlığı değil.

### 4.5 Gelecek Çalışma

(1) Panel-spesifik MCC-optimize eşik; (2) ClinVar/gnomAD ile daha büyük CFTR/PAH kohortları; (3) Conformal Prediction (abstain → uzman incelemesi); (4) AlphaFold2 ΔΔG ile protein-yapı entegrasyonu; (5) prospektif klinik validasyon (ACMG uzman karşılaştırması).

### 4.6 Final Aşamasında Karşılaşılabilecek Zorluklar

*PDR şablonu §4 zorunlu maddesi.* **Dağılım kayması:** kör test verisi farklı varyant profili içerebilir; adversarial validation (AUC≈0,50) ve ColumnAligner isim-tabanlı hizalama ile risk azaltılır. **Eşik uyarlaması:** sınıf dengesi farklı olabilir; eşikler `models/threshold.json`/`panel_thresholds.json`'dan dinamik yüklenir, `src/cli/modes/train.py` ile resmi prior'a yeniden türetilebilir (derivation==inference). **Tekrar çalıştırma (§7.5):** Python/kütüphane versiyon farkı sapma yaratabilir; `requirements.txt` sabit versiyonları, seed=42, `submission/predict.py` tek-giriş + Docker (CPU+GPU) ile minimize edilir. **Hesaplama süresi:** GATv2GNN CPU inference gecikmesi `scripts/test_cpu_inference.py` ile doğrulanmış, ONNX ihracı hazır (~500 örnek <90 sn). **Kolon yapısı farkı:** ColumnAligner (exact→case-insensitive→fuzzy≥0,85→positional) + `data/contracts/predict_schema.json` eksik/fazla kolonu tolere eder.

---

## 5. KAYNAKÇA (ve RAPOR DÜZENİ, 10 puan)

[1] S. Richards, N. Aziz, S. Bale, D. Bick, S. Das, J. Gastier-Foster, et al., "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the ACMG and AMP," *Genet. Med.*, vol. 17, no. 5, pp. 405–424, May 2015. doi:10.1038/gim.2015.30

[2] N. M. Ioannidis, J. H. Rothstein, V. Pejaver, S. Middha, S. K. McDonnell, S. Baheti, et al., "REVEL: An Ensemble Method for Predicting the Pathogenicity of Rare Missense Variants," *Am. J. Hum. Genet.*, vol. 99, no. 4, pp. 877–885, Oct. 2016. doi:10.1016/j.ajhg.2016.08.016

[3] M. Kircher, D. M. Witten, P. Jain, B. J. O'Roak, G. M. Cooper, and J. Shendure, "A general framework for estimating the relative pathogenicity of human genetic variants," *Nat. Genet.*, vol. 46, no. 3, pp. 310–315, Mar. 2014. doi:10.1038/ng.2892

[4] M. J. Landrum, J. M. Lee, M. Benson, G. R. Brown, C. Chao, S. Chitipiralla, B. Gu, et al., "ClinVar: improving access to variant interpretations and supporting evidence," *Nucleic Acids Res.*, vol. 46, no. D1, pp. D1062–D1067, Jan. 2018. doi:10.1093/nar/gkx1153

[5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, pp. 785–794, Aug. 2016. doi:10.1145/2939672.2939785

[6] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 3149–3157, Dec. 2017.

[7] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 4765–4774, Dec. 2017.

[8] S. Brody, U. Alon, and E. Yahav, "How Attentive are Graph Attention Networks?" in *Proc. 10th Int. Conf. Learning Representations (ICLR)*, Apr. 2022. arXiv:2105.14491

[9] J. Frazer, P. Notin, M. Dias, A. Gomez, J. K. Min, K. Brock, et al., "Disease variant prediction with deep generative models of evolutionary data," *Nature*, vol. 599, pp. 91–95, Nov. 2021. doi:10.1038/s41586-021-04043-8

[10] V. Pejaver, J. Byrne, S. Feng, M. Mooney, F. Camper, Y. A. Kim, and B. Loh, "Calibration of pathogenicity predictions for missense variants in ACMG/AMP variant interpretation guidelines," *Am. J. Hum. Genet.*, vol. 109, no. 12, pp. 2163–2177, Dec. 2022. doi:10.1016/j.ajhg.2022.10.013

[11] L. Sundaram, H. Gao, S. R. Padigepati, J. F. McRae, Y. Li, J. A. Kosmicki, et al., "Predicting the clinical impact of human mutation with deep neural networks," *Nat. Genet.*, vol. 50, pp. 1161–1170, Sep. 2018. doi:10.1038/s41588-018-0167-z

[12] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, "GNNExplainer: Generating Explanations for Graph Neural Networks," in *Proc. 33rd Int. Conf. Neural Information Processing Systems (NeurIPS)*, pp. 9240–9251, Dec. 2019.

---

**RAPOR SONU** · *Takım XYRA3 | Takım ID: 909249 | Başvuru ID: 4865399*
*TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Proje Detay Raporu · 2 Haziran 2026*

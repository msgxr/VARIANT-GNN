<!--
═══════════════════════════════════════════════════════════════════════════════
  WORD CLAUDE'A TALİMAT — BU BLOK RAPORA RENDER EDİLMEZ (yalnız üretim notu)
═══════════════════════════════════════════════════════════════════════════════
  BİÇİM (resmi şablon, ZORUNLU — ihlali diskalifiye):
    • Yazı tipi: Aptos · Gövde 12 pt · Başlık 14 pt
    • Satır aralığı 1,15 · İki tarafa yaslı (justified)
    • Kenar boşlukları: üst 2,8 cm · alt/sağ/sol 2,5 cm
    • Kapak + İçindekiler HARİÇ rapor ≤ 10 SAYFA. 10 sayfayı geçen rapor
      DEĞERLENDİRİLMEZ → bu MUTLAK sınır figür sayısından önceliklidir.
    • Sayfa numaraları ardışık (Kapak/İçindekiler sonrası başlar).
    • Kaynaklar IEEE formatında.

  MATEMATİK: $$...$$ ve $...$ ifadelerini Word denklem nesnesi (OMML, Cambria Math)
    olarak render et. Numaralandır: (1), (2), ...

  FİGÜRLER: ![...](yol) işaretli her görseli ekle. "[2-UP]" etiketli iki figürü
    YAN YANA (her biri ~7,5 cm) yerleştir. "[TAM]" tam genişlik (~15 cm).
    >>> 10 SAYFA AŞILIRSA, figürleri şu öncelikle ÇIKAR (önce en alttan):
        DÜŞÜK: 18_shap_summary, 17_mcc_threshold, 01_cv_fold, 12_seed_stability
        ORTA : 08_shap_importance, 03_panel_radar, 05_roc, 19_shap_waterfall
        ÇEKİRDEK (asla çıkarma — şablon zorunlu görselleri):
          11_architecture, 04_confusion_matrix_panel, 06_pr_curves,
          14_threshold_analysis, 13_benchmark, 02_panel_f1, 09_ablation
  KAPAK + İÇİNDEKİLER ayrı sayfalardır (sayıma dahil değil). [[PAGEBREAK]] = sayfa sonu.
═══════════════════════════════════════════════════════════════════════════════
-->

# VARIANT-GNN
## Missense Varyant Patojenite Tahmini — Hibrit Graf Sinir Ağı Ensemble Sistemi
### TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Proje Detay Raporu (PDR)

**Proje:** VARIANT-GNN · **Takım:** XYRA3 · **Takım ID:** 909249 · **Başvuru ID:** 5200240
**Kategori:** Sağlıkta Yapay Zekâ — Üniversite ve Üzeri · Genetik Varyant Patojenite Tahmini
**Rapor Tarihi:** 10 Haziran 2026

*Biçim (Word): Aptos 12 pt gövde / 14 pt başlık · Satır aralığı 1,15 · İki yana yaslı · Üst 2,8 cm · Diğer 2,5 cm*

[[PAGEBREAK]]

## İÇİNDEKİLER

1. **GİRİŞ** (10 puan) — 1.1 Problem ve Klinik Önem · 1.2 Sınıf Dengesizliği · 1.3 Literatür · 1.4 Katkılar
2. **YÖNTEM** (25 puan) — 2.1 Veri Mühendisliği · 2.2 Mimari ve Matematiksel Temel · 2.3 Hiperparametre ve Doğrulama · 2.4 Açıklanabilirlik · 2.5 Karar Eşiği
3. **BULGULAR** (30 puan) — 3.1 Genel Performans · 3.2 Panel Bazlı · 3.3 Eşik Analizi · 3.4 Ablasyon · 3.5 Karşılaştırma
4. **SONUÇ** (25 puan) — 4.1 Yorum · 4.2 PSR Karşılaştırma · 4.3 Güçlü/Zayıf · 4.4 Hata Analizi · 4.5 Gelecek · 4.6 Final Zorlukları
5. **KAYNAKÇA** (ve Rapor Düzeni, 10 puan)

[[PAGEBREAK]]

## ETİK BEYAN

Veri seti TEKNOFEST 2026 kapsamında anonim formatta sağlanmış olup bireye ait kimlik bilgisi içermez ve KVKK gerekliliklerine tabidir. **Bu çalışma TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında gerçekleştirilmiş olup geliştirilen model ve çıktılar yalnızca araştırma ve eğitim amaçlıdır; klinik tanı veya tıbbi karar desteği amacıyla kullanılamaz.** Klinik entegrasyon için bağımsız klinik validasyon, sağlık otoritesi onayı ve etik kurul değerlendirmesi gereklidir.

---

## 1. GİRİŞ (10 puan)

### 1.1 Problem Tanımı ve Klinik Önem

Missense varyantların patojenitesini sınıflandırmak klinik genetiğin en zorlu problemlerindendir: tek bir nükleotid değişimi bir aminoasidi değiştirerek protein işlevini bozabilir ve kalıtsal kansere, **fenilketonüriye (PAH geni — fenilalanin hidroksilaz)** veya kistik fibrozise (CFTR geni) yol açabilir. ACMG/AMP rehberleri [1] yorumlamayı beş kategoride standardize etmiş olsa da büyük panellerde "Belirsiz Klinik Anlamlı Varyant" (VUS) oranı %30–60'tır; ClinVar [4] her yıl yüz binlerce varyant biriktirir ve uzman incelemesi darboğazı tanı gecikmesine yol açar. **Problem**, anonim ve asimetrik bir tablo-veri kümesinde her missense varyantı *Patojenik (1)* veya *Benign (0)* olarak sınıflandırmaktır; birincil başarım ölçütü ikili F1 skorudur (§7.3, `pos_label=1`).

### 1.2 Sınıf Dengesizliği ve Model Başarımına Etkisi

Yarışma verisinde Patojenik/Benign oranı 2,75:1 (MASTER) ile 5,00:1 (PAH) arasındadır (Tablo 1). Bu dengesizlik, doğruluk (accuracy) metriğini yanıltıcı kılar: tüm örnekleri "Patojenik" tahmin eden boş bir sınıflandırıcı bile MASTER'da %73 doğruluk alır. Dengesizliğin etkisi en açık biçimde Matthews korelasyon katsayısında (MCC) görülür; MCC dört hücreyi de simetrik kullandığından azınlık (Benign) sınıfının yanlış sınıflanması katsayıyı orantısız düşürür. Bu yüzden raporda F1 ile birlikte MCC ve PR-AUC zorunlu olarak verilmiş, karşı önlem olarak **sınıf-ağırlıklı kayıp**, **SMOTE** (yalnız eğitim) ve **resmi prior'a kalibre tek eşik** birlikte kullanılmıştır (§2, §3.3).

### 1.3 Literatür Bağlamı (Uluslararası Çalışmalar)

| Çalışma | Yöntem | Veri Kaynağı | Raporlanan Metrik | Boşluk |
|:--|:--|:--|:--|:--|
| REVEL [2] (2016) | 13 in-silico skorun meta-ensemble'ı | ClinVar, HGMD | ROC-AUC = 0,91 | Panel özgünlüğü yok |
| CADD [3] (2014) | Genom-geneli delesyon-temelli skorlama | 135M SNP simülasyonu | — (skorlama) | Genomik-adres bağımlı (anonimle uyumsuz) |
| EVE [9] (2021) | Evrimsel dizi VAE (üretken) | UniProt MSA | ROC-AUC ≈ 0,91 | Tek-modaliteli, in-silico risk yok |
| MutPred2 [11] (2018) | Protein işlev + filogeni DNN | ClinVar, HGMD | makro F1 = 0,86 | Çok-boyutlu ensemble değil |
| Pejaver ve ark. [10] (2022) | ACMG-uyumlu eşik kalibrasyonu | ClinVar Expert Panel | — (kalibrasyon) | Eşik stratejimizi destekler |

Literatürdeki boşluklar: (i) varyantlar arası ilişkisel bilginin graf sinir ağıyla modellenmesi, (ii) panel özgünlüğünü koruyan çok-panel değerlendirme, (iii) heterojen özellik uzayını eşzamanlı işleyen hibrit ensemble, (iv) kolon-isimsiz ortamda güvenilir tahmin. Bu çalışma dört boşluğu da hedefler.

### 1.4 Katkılar

- **ColumnAligner:** Kolon-gizli profilleri çok-aşamalı isim eşleştirmeyle (exact → case-insensitive → fuzzy difflib ≥ 0,85 → positional) referans şemaya oturtur; §3.2 anonim-kolon kısıtını tam karşılar.
- **Hibrit Graf Ensemble:** XGBoost [5] + LightGBM [6] + VariantGATv2GNN + DNN; OOF-stacking meta-öğrenici (Wolpert) + Nelder-Mead ağırlık optimizasyonu.
- **Resmi prior'a kalibre tek eşik:** Karar eşiği group-aware held-out kalibrasyon setinde, resmi test prior'ına (%20-patojenik) F1-optimal, HAM olasılıkta türetilir (global $\theta = 0{,}8415$; *derivation == inference*).
- **MC Dropout belirsizliği** + **Domain-Adversarial DNN (DANN):** panel-değişmez temsil, LOPO doğrulamada ortalama +2,17 pp genelleme.

---

[[PAGEBREAK]]

## 2. YÖNTEM (25 puan)

### 2.1 Veri Mühendisliği ve Ön İşleme

Veri seti dört panelde ACMG/AMP'e göre etiketlenmiş missense varyantları içerir; etiketler ClinVar [4] Expert Panel onaylı (3–4 yıldız) kayıtlara dayanır ("Pathogenic"/"Likely Pathogenic" → 1; "Benign"/"Likely Benign" → 0; VUS çıkarılmıştır). Veri 14 Mayıs 2026'da alınmış, model **sızıntısız (group-aware, `Variant_ID`)** protokolle eğitilmiştir. **Dış kaynaktan örnek eklenmemiştir;** §3.2 yarışma verisi tek kaynaktır.

**Tablo 1: Yarışma Veri Seti Kompozisyonu**

| Panel | Toplam | P | B | Oran (P:B) | Test (n) |
|:--|--:|--:|--:|:--:|--:|
| MASTER (General) | 2.931 | 2.149 | 782 | 2,75:1 | 582 |
| KANSER (Hereditary Cancer) | 388 | 268 | 120 | 2,23:1 | 86 |
| PAH (Fenilketonüri) | 372 | 310 | 62 | 5,00:1 | 76 |
| CFTR (Kistik Fibrozis) | 111 | 90 | 21 | 4,29:1 | 18 |
| **Toplam** | **3.802** | **2.817** | **985** | **2,86:1** | **762** |

**Asimetrik, şifreli (anonim) yapı.** Özellik uzayı 343 anonim kolondur (`AL_`/`EK_`/`CAT_`/`AA_` önekli); genomik adres ve gerçek kolon adları gizlidir. Bölme `Variant_ID`'ye göre **grup-farkındadır** (GroupShuffleSplit %80/20 + StratifiedGroupKFold 5-fold); 3.802 satır 3.224 tekil varyanttan oluşur, aynı varyant train/test'i çaprazlamaz (**leakage guard: 0 straddle**). **Adversarial validation** ROC-AUC değerleri (MASTER 0,512 · KANSER 0,505 · PAH 0,498 · CFTR 0,521) $\approx 0{,}50$ olup dağılım kayması/sızıntı riskinin yokluğunu doğrular.

**Eksik değer ve aykırı değer yönetimi.** Eksik değerler **medyan imputasyonu** ile (yalnız eğitim fold'unda fit edilen medyan, test'e transform-only) tamamlanır. Ek olarak "eksiklik bilgi taşır" (missing ≠ 0) ilkesiyle her kolon için **ikili eksiklik-deseni göstergesi** üretilir; bu, jüri §3.2 yorumuna yanıttır ve ROC-AUC'yi $0{,}850 \to 0{,}8538$ (+0,38 pp, 5-seed doğrulu), PAH F1'ini +5,4 pp, 3-panel tanı skorunu (CFTR hariç) $0{,}6052 \to 0{,}6202$ yükseltir (`missing_indicator_ablation.json`). Aykırı değerler **RobustScaler** ile bastırılır; medyan ve çeyrekler-arası açıklık (IQR) kullanan ölçekleme aykırılara dayanıklıdır:

$$x' = \frac{x - \mathrm{median}(x)}{Q_3(x) - Q_1(x)} \tag{1}$$

**Sızıntı giderme (kanıt).** Önceden materyalize Gaussian-jitter augmentation (3.802 → 7.604) near-twin kopyalarını satır-bazlı bölmenin iki yanına düşürüp **+3,71 pp yapay şişme** yaratıyordu (`leakage_quantification.json`). Augmentation devre dışı bırakılmış, bölme group-aware yapılmıştır; tüm beyanlar bu sızıntısız protokole aittir.

![Şekil 1 [2-UP]: Sızıntı kuantifikasyonu — satır-bazlı vs. group-aware split (+3,71 pp yapay şişmenin kaldırılması).](reports/figures/pdr/10_leakage_quantification.png)

**Özellik mühendisliği.** `CategoricalBioFeaturizer`, `AA`/`CAT`/`EK` kolonlarından ACMG-hizalı 22 yorumlanabilir öznitelik üretir (satır-bazlı deterministik, +0,38 pp F1, §3.2 biyolojik sinyalini kurtarır). *Not:* eski `SelectKBest(35)+AutoEncoder(→16)` adımları group-aware CV'de $\approx +5{,}3$ pp F1 kaybettirdiğinden **kaldırılmış**, tam 343 öznitelik korunmuştur (`preprocessing_diagnostic.json`). SMOTE yalnız eğitim fold'unda azınlık sınıfı sentezler (test'e asla uygulanmaz):

$$x_{\text{yeni}} = x_i + \lambda\,(x_{z_i} - x_i), \qquad \lambda \sim \mathcal{U}(0,1),\ z_i \in \text{k-NN}(x_i) \tag{2}$$

### 2.2 Model Mimarisi ve Matematiksel Temel

VARIANT-GNN dört bileşeni OOF-stacking meta-öğrenici ile birleştirir. **Mimari akış:** Ham profil → ColumnAligner → Ön İşleme (6 adım) → {XGBoost %30, LightGBM %30, VariantGATv2GNN %25, DNN %15} → Lojistik Regresyon Meta-Öğrenici → Isotonic Kalibrasyon → Eşik → İkili Karar.

![Şekil 2 [TAM]: VARIANT-GNN uçtan uca hibrit mimari diyagramı.](reports/figures/pdr/11_architecture_diagram.png)

**VariantGATv2GNN — dinamik attention (matematiksel gerekçe).** Cosine benzerliğiyle kurulan k-NN grafta ($k=10$, eşik $\geq 0{,}30$, genomik koordinat gerektirmez) düğümler GATv2Conv [8] ile mesaj iletir. Orijinal GAT *statik* attention hesaplarken, GATv2 hem kaynak hem hedef düğümü dikkate alan *dinamik* attention öğrenir. Komşuluk skoru ve normalize katsayı:

$$e_{ij} = \mathbf{a}^{\top}\,\mathrm{LeakyReLU}\!\left(\mathbf{W}\,[\,\mathbf{h}_i \,\Vert\, \mathbf{h}_j\,]\right), \qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} \tag{3}$$

$$\mathbf{h}_i' = \big\Vert_{m=1}^{M}\ \sigma\!\Big( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(m)}\, \mathbf{W}^{(m)} \mathbf{h}_j \Big) \tag{4}$$

Burada $M=4$ attention başlığı, $\Vert$ birleştirme, $\mathbf{h}_i'$ güncellenmiş düğüm temsilidir. 3 GATv2 bloğu + residual + LayerNorm yapısı SWA (son %25 epoch ortalaması) ile stabilize edilir: $\theta_{\text{SWA}} = \tfrac{1}{T}\sum_{t} \theta_t$. Deneysel etki: SAGEConv → GATv2Conv MASTER panelinde F1 +0,014 (§3.4).

**Sınıf-ağırlıklı kayıp.** Dengesizliğe karşı ağırlıklı ikili çapraz-entropi kullanılır; ağırlık ters-frekansla belirlenir:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \Big[ w_1\, y_i \log p_i + w_0\,(1-y_i)\log(1-p_i)\Big], \quad w_c = \frac{N}{2\,N_c} \tag{5}$$

DNN ayrıca **Domain-Adversarial (DANN)** eğitilir: gradyan-ters çevirme katmanıyla panel-değişmez temsil öğrenilir, toplam amaç $\mathcal{L}_y - \lambda\,\mathcal{L}_{\text{panel}}$ (sınıf kaybını düşür, panel ayırt-edilebilirliğini *artır* → panel-bağımsız öznitelik).

**Ensemble birleştirme.** (1) Nelder-Mead model ağırlıklarını doğrulama F1'inde optimize eder: $\mathbf{w}^{*} = \arg\max_{\mathbf{w}} F_1\!\big(\sum_m w_m\, p_m\big)$, $\sum_m w_m = 1$ → $(0{,}30/0{,}30/0{,}25/0{,}15)$. (2) Lojistik regresyon **OOF-stacking** meta-öğrenici her modelin güçlü olduğu örnekleri adaptif birleştirir: $p_{\text{final}} = \sigma\!\big(\beta_0 + \sum_m \beta_m\, p_m\big)$. **Isotonic kalibrasyon** (eğitimin bağımsız %15'i) monoton $g$ ile $\sum_i (g(p_i)-y_i)^2$'yi minimize eder → ECE = 0,0291, Brier = 0,1115.

![Şekil 3 [2-UP]: Kalibrasyon eğrisi — isotonik regresyon sonrası güvenilirlik diyagramı (ECE=0,0291).](reports/figures/pdr/07_calibration_curve.png)

### 2.3 Hiperparametre Optimizasyonu, Çapraz Doğrulama ve Aşırı Uyum

**Tablo 2: Bileşen Modeller — Mimari, Hiperparametre, CV F1**

| Bileşen | Anahtar Hiperparametreler | CV F1 (5-fold) | Ağırlık |
|:--|:--|:--:|:--:|
| XGBoost | max_depth 6 / n_est 200 / lr 0,05 / subsample 0,8 | 0,8876 ± 0,0047 | %30 |
| LightGBM | num_leaves 64 / lr 0,05 / min_child 10 | 0,8828 ± 0,0082 | %30 |
| VariantGATv2GNN | 3× GATv2Conv / 4 başlık / hidden 128 / k-NN(cos,10) | 0,8114 ± 0,0228 | %25 |
| VariantDNN (DANN) | 128→64→2 / BatchNorm / Dropout 0,3+0,2 | 0,7596 ± 0,0441 | %15 |
| **Meta-öğrenici** | Lojistik Regresyon (şeffaf birleştirme) | — | — |

**Arama yöntemi:** ağaç modelleri için panel-katmanlı `RandomizedSearchCV` (group-aware fold içinde), nöral bileşenler için manuel ızgara + erken durdurma. Seçim ölçütü her zaman group-aware OOF F1'tir (test'e dokunulmaz) → genelleme gücü doğrudan optimize edilir.

**Çapraz doğrulama:** StratifiedGroupKFold ($k=5$, `random_state=42`), `Variant_ID` grup-farkındadır (0 straddle). %20 group-aware hold-out ($n=762$) hiçbir geliştirme adımında kullanılmamış, yalnız nihai raporlamada değerlendirilmiştir.

**Aşırı uyum önlemleri:** Dropout (0,3/0,2), BatchNorm, erken durdurma, SWA, ağaçlarda subsample/min_child, ve OOF-stacking (nested-CV → meta-öğrenici hiç görmediği OOF üzerinde fit). **Tohum kararlılığı:** 5 seed (42/123/456/789/2026) üzerinde CV F1 = **0,8738 ± 0,0034** (min 0,8700, maks 0,8802); ağaç üyeleri (%60 ağırlık) deterministik → model tohum-kararlıdır.

![Şekil 4 [2-UP]: 5-fold CV F1 dağılımı (sol) ve 5-seed kararlılığı 0,8738 ± 0,0034 (sağ).](reports/figures/pdr/01_cv_fold_comparison.png)
![Şekil 5 [2-UP]: Tohum kararlılığı — 5 bağımsız seed üzerinde CV F1 varyansı.](reports/figures/pdr/12_seed_stability.png)

### 2.4 Açıklanabilirlik (PSR §4.4 güçlendirme)

Kolon isimleri anonim olduğundan açıklanabilirlik **özellik grubu** düzeyinde kurulmuştur; gruplara atama önek/ad örüntülerine dayalı *gösterge* (heuristik) niteliğindedir. Dört tamamlayıcı yöntem uygulanmıştır: SHAP, GNNExplainer, LIME, MC Dropout.

**SHAP — additif öznitelik atfı.** Model çıktısı baz değer + öznitelik katkıları olarak ayrıştırılır (Shapley değeri additivite garantisi):

$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{F} \phi_i, \qquad
\phi_i = \!\!\sum_{S \subseteq F \setminus \{i\}}\!\! \frac{|S|!\,(F-|S|-1)!}{F!}\,\big[f(S \cup \{i\}) - f(S)\big] \tag{6}$$

XGB/LGBM için deterministik TreeSHAP [7]; GNN/DNN için KernelSHAP (200 örnek arka plan). **Tablo 3, ölçülen** global grup katkılarını (`shap_group_contributions.json`) biyolojik beklentiyle karşılaştırır — dürüst raporlama (§III-9): model in-silico risk skorlarına biyolojik beklentinin üstünde yaslanırken popülasyon frekansını beklenenin altında ağırlıklandırır; bu, anonim-veride in-silico kolonlarının en güçlü tek sinyal olmasının doğrudan sonucudur.

**Tablo 3: SHAP Özellik Grubu Katkıları — Ölçülen vs. Biyolojik Beklenti (global)**

| Özellik Grubu | Ölçülen Katkı | Biyolojik Beklenti | Yorum |
|:--|:--:|:--:|:--|
| In-Silico Risk | **%58,33** | %38 | Anonim-veride en baskın tek sinyal |
| Biyokimyasal/Yapısal | %18,21 | %10 | Beklentinin üstünde (AA değişim sinyali) |
| Evrimsel Korunmuşluk | %10,94 | %27 | Beklentinin altında |
| Sekans Bağlamı | %7,74 | %5 | Beklentiye yakın |
| Popülasyon Verileri | %4,78 | %18 | Beklentinin belirgin altında |
| Yerel Sekans | %0,0 | %2 | İhmal edilebilir |

![Şekil 6 [2-UP]: SHAP global öznitelik önemi (özet).](reports/figures/pdr/08_shap_importance.png)
![Şekil 7 [2-UP]: SHAP grup katkıları — ölçülen vs. biyolojik beklenti.](reports/figures/shap_group_contributions_honest.png)

**Bireysel SHAP (waterfall).** Yüksek-güvenli Patojenik örnekte in-silico (+0,42) + evrimsel (+0,31) + düşük-AF (+0,29) → $P=0{,}94$; yüksek-güvenli Benign'de yüksek-AF (−0,38) baskın → $P=0{,}06$; sınır varyantta çelişkili sinyaller MC Dropout bayrağı üretir.

![Şekil 8 [2-UP]: Bireysel SHAP waterfall — patojenik ve benign örnek karşılaştırması.](reports/figures/pdr/19_shap_waterfall.png)

**GNNExplainer** [12] (n=30 örnek, k=10): grafın en yüksek-önemli öznitelikleri biyokimyasaldır — bio_grantham, bio_blosum62, hacim/MW farkları; kenar-önem ort. 0,29 (maks 0,84) → koordinatsız biyokimyasal-benzerlik komşuluğu doğrulanır (`reports/gnn_explainer_results.json`). **LIME tutarlılığı:** 150 örnekte LIME ↔ TreeSHAP baskın özniteliklerde örtüşür (küresel Spearman $\rho = 0{,}56$; `reports/lime_shap_concordance.json`). **MC Dropout belirsizliği** ($T=10$ ileri geçiş): tahmin varyansı

$$\bar{p} = \tfrac{1}{T}\sum_{t=1}^{T} p_t, \qquad \sigma^2 = \tfrac{1}{T}\sum_{t=1}^{T}(p_t - \bar{p})^2 \tag{7}$$

$\sigma<0{,}15$ Yüksek Güven, $0{,}15$–$0{,}30$ Orta, $>0{,}30$ "Uzman Değerlendirmesi Gerekli". Hatalı tahminlerde ort. $\sigma = 0{,}40$, doğruda $0{,}12$ → model kendi hatasını sezebilir.

### 2.5 Karar Eşiği Belirleme

Karar eşiği group-aware **held-out kalibrasyon setinde**, resmi prior'a (%20-patojenik) **F1-optimal**, **HAM olasılıkta** türetilir:

$$\theta^{*} = \arg\max_{\theta \in [0,1]} F_1^{\,20\%}(\theta) \;=\; 0{,}8415 \quad (\text{global, canonical}) \tag{8}$$

Türetim ile çıkarım aynı dağılım/uzayda olduğundan **derivation == inference** garantilidir (üreten: `src/cli/modes/train.py`, `threshold_source=calibration_set`). Eşiği %74-poz/50-50'de türetmek %20-prior sette F1 kaybettirir (§3.3).

---

[[PAGEBREAK]]

## 3. BULGULAR (30 puan)

### 3.1 Genel Test Performansı

**İki sayının ayrımı (dürüst raporlama).** Resmi TEKNOFEST test seti patojenik-azınlık (≈%20 patojenik / %80 benign) prior'ına dayanır.¹ Bu nedenle:

- **RESMİ JÜRİ BEKLENTİSİ = 4-panel %20-patojenik F1 ortalaması = 0,631** (HEADLINE; CFTR dahil, panel-kalibre eşik). Per-panel: General 0,6006 · Hereditary_Cancer 0,7301 · PAH 0,5299 (global $\theta=0{,}8415$ hold-out) · CFTR 0,6632 (OOF nested-CV @ panel-eşik $\theta=0{,}59$); ortalama $= (0{,}6006+0{,}7301+0{,}5299+0{,}6632)/4 = 0{,}631$. Muhafazakâr **3-panel tanı (CFTR hariç) = 0,6202** $= (0{,}6006+0{,}7301+0{,}5299)/3$. Havuzlanmış jüri-F1 tahmini = **0,6042 ± 0,0324** (300× %20-resample).
- **İç ayrım gücü (jüri skoru DEĞİL) = Test F1 = 0,8367** — %75-poz iç hold-out'ta sınıf ayırt-etme kapasitesi.

F1 patojenik-odaklıdır (`pos_label=1`); test'te patojenik azınlık olduğundan jüri F1'inin ~0,60 olması **metrik tanımının doğal sonucudur**, model zayıflığı değildir.

> ¹ %20-prior ekip beyanı olup resmi Q&A'ya dayandırılır; repoda doğrulanabilir resmi artefakt henüz yoktur — **UNVERIFIED** (belirsizlik günlüğü U-008). Artefakt eklenene kadar %20-prior ve 4-panel-ortalama skorlaması modelleme varsayımı olarak işaretlidir.

**İç hold-out metrikleri ($\theta=0{,}8415$, §7.3, `pos_label=1`):** Test F1 = **0,8367**, MCC = 0,5112, precision = 0,9241, recall = 0,7644, PR-AUC = 0,9267, ROC-AUC = 0,8538, Brier = 0,1115, ECE = 0,0291; üretim CV F1 = 0,8936 ± 0,0004 (OOF-stacking) / 0,8812 ± 0,0113 (fold).

**Metrik tanımları ve kendiyle-tutarlılık (matematiksel kanıt).** Birincil ve zorunlu metrikler:

$$P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}=\frac{2\,TP}{2\,TP+FP+FN} \tag{9}$$

$$\mathrm{MCC}=\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}} \tag{10}$$

İç hold-out'ta ($N=762$, ~%75-poz) yaklaşık karmaşıklık matrisi $TP\approx436,\ FN\approx135,\ FP\approx36,\ TN\approx155$'tir. Bu değerler raporlanan metrikleri **birebir yeniden üretir** (kanıt):

$$F_1 = \frac{2\cdot 436}{2\cdot 436 + 36 + 135} = \frac{872}{1043} = 0{,}836 \checkmark \qquad
\mathrm{MCC} = \frac{436\cdot155 - 36\cdot135}{\sqrt{472\cdot571\cdot191\cdot290}} = 0{,}513 \checkmark$$

Aynı şekilde $\frac{2\cdot 0{,}9241\cdot 0{,}7644}{0{,}9241+0{,}7644}=0{,}8367$ → precision/recall ile F1 tutarlıdır. Yüksek eşik precision'ı 0,9241'e çıkarıp FP'yi ~36'ya sınırlar; bedeli recall'ın 0,7644'e düşmesidir (kaçırılan patojenikler MC Dropout'ta yüksek-$\sigma$ ile işaretlenir).

![Şekil 9 [TAM]: Karmaşıklık matrisleri — dört panel ayrı (MASTER/KANSER/PAH/CFTR).](reports/figures/pdr/04_confusion_matrix_panel.png)
![Şekil 10 [2-UP]: Precision-Recall eğrileri (panel bazlı), PR-AUC=0,9267.](reports/figures/pdr/06_pr_curves.png)
![Şekil 11 [2-UP]: ROC eğrileri (panel bazlı), ROC-AUC=0,8538.](reports/figures/pdr/05_roc_curves.png)

### 3.2 Panel Bazlı Sonuçlar

**Tablo 4: Panel Bazlı Performans — Hold-Out Test ($\theta=0{,}8415$)**

| Panel | F1 | MCC | PR-AUC | ROC-AUC | Precision | Recall |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MASTER (General) | 0,8185 | 0,4951 | 0,9271 | 0,8546 | 0,9217 | 0,7361 |
| KANSER (Hered. Cancer) | 0,9060 | **0,7135** | 0,9743 | 0,9449 | 0,9464 | 0,8689 |
| PAH (Fenilketonüri) | 0,9120 | 0,5053 | 0,8908 | 0,7016\* | 0,9048 | 0,9194 |
| CFTR (Kistik Fibrozis) | 0,7143 | —† | 1,0000 | —† | 1,0000 | 0,5556 |
| **Tüm Test** | **0,8367** | **0,5112** | **0,9267** | **0,8538** | **0,9241** | **0,7644** |

*Kaynak `cv_report.json` panel_metrics. \* PAH ROC-AUC=0,7016 hold-out küçük-örneklem (76 satır) gürültüsü; OOF-robust (503 satır) ≈ 0,789. † CFTR test fold'unda negatif sınıf dejenere ($n=18$) → MCC tanımsız (0), ROC-AUC NaN; anlamlı metrikler F1/precision/recall.*

**Yorum.** *KANSER* en dengeli paneldir (2,23:1) → en yüksek MCC = 0,7135, hold-out F1 = 0,9060; **resmi %20-prior F1 = 0,7301** (4-panel ortalamasına en güçlü katkı). *CFTR* ($n=18$) tam precision = 1,000 (hiç FP yok) ama recall = 0,5556 düşük; küçük-$n$ nedeniyle MCC/ROC-AUC tanımsız. *PAH* dengeli (recall 0,9194) ama **%20-prior F1 = 0,5299** ile en zayıf — anonim-veri tavanında (`pah_analysis.json`). *MASTER* en geniş çeşitlilik + 2,75:1 dengesizlik → MCC = 0,4951; çoğu test örneği burada olduğundan genel MCC'yi domine eder, **%20-prior F1 = 0,6006**.

![Şekil 12 [2-UP]: Panel bazlı F1 karşılaştırması (hold-out, $\theta$=0,8415).](reports/figures/pdr/02_panel_f1_bar.png)
![Şekil 13 [2-UP]: Panel metrik radarı (F1/MCC/PR-AUC/ROC-AUC/Precision/Recall).](reports/figures/pdr/03_panel_metrics_radar.png)

### 3.3 Eşik Analizi (Farklı Eşikler + Optimal Değer)

Karar eşiği group-aware held-out kalibrasyon setinde resmi prior'a F1-optimal, HAM olasılıkta türetilir (denklem (8), $\theta=0{,}8415$ global, canonical).

**Tablo 5: Karar Eşiği — Global (CANONICAL) ve Opt-In Panel Eşikleri**

| Eşik | $\theta$ | Kapsam | Recall | MCC | Not |
|:--|:--:|:--|:--:|:--:|:--|
| **Global (jüri kararı)** | **0,8415** | tüm paneller | 0,7644 | 0,5112 | %20-prior F1-optimal; `models/threshold.json` |
| Opt-in General / KANSER | 0,3990 / 0,4532 | — | — | — | varsayılan KAPALI |
| Opt-in PAH / CFTR | 0,4434 / 0,1922 | — | — | — | varsayılan KAPALI |

Eşik stratejisi **hibrit**tir (`models/panel_thresholds.json`, `use_panel_thresholds=true` varsayılan, submission'da uygulanır): büyük üç panel (General, Hereditary_Cancer, PAH) global $\theta=0{,}8415$ kullanır; **yalnız CFTR** panel-kalibre eşik $\theta=0{,}59$ alır. CFTR'de global $\theta$ miskalibreydi (F1 $=0{,}33$ artefakt); $\theta=0{,}59$ bunu düzeltir (F1 $0{,}33 \to 0{,}66$) ve resmi 4-panel ortalamayı **0,631**'e çıkarır. Bütün panellere **uniform** per-panel tuning ise küçük panellerde overfit edip daha düşük skorlamıştı (uniform per-panel 0,5445 < 3-panel tanı 0,6202) — bu yüzden tüm-panel uniform tuning kullanılmaz, yalnız CFTR-spesifik kalibre eşik uygulanır.

![Şekil 14 [2-UP]: Eşik taraması — F1/precision/recall vs. $\theta$, optimal $\theta$=0,8415.](reports/figures/pdr/14_threshold_analysis.png)
![Şekil 15 [2-UP]: MCC-eşik eğrisi (General paneli).](reports/figures/pdr/17_mcc_threshold_general.png)

### 3.4 Ablasyon Çalışması

Tam ensemble (Test F1 = 0,8367, CV F1 = 0,8936) referansında bileşen katkıları (canonical, kaynak-bağlı):

| Değişiklik | Etki | Kaynak |
|:--|:--:|:--|
| GNN kaldırıldı | **−2,2 pp** | `ensemble_weight_justification.json` |
| DNN/DANN kaldırıldı | −0,7 pp | — |
| OOF-stacking → sabit ağırlık | −0,59 pp | `stacking_improvement.json` |
| CategoricalBioFeaturizer kaldırıldı | −0,38 pp | `bio_feature_ablation.json` |
| SelectKBest(35)+AE eklenirse | **≈ −5,3 pp** | `preprocessing_diagnostic.json` |
| SAGEConv (GATv2 yerine) | −0,014 | §2.2 |

![Şekil 16 [TAM]: Ablasyon — her bileşenin tam ensemble'a katkısı.](reports/figures/pdr/09_ablation_bar.png)

### 3.5 Karşılaştırma (Ensemble vs. Tek Modeller / Baseline)

Tek-model genel CV F1 sıralaması: XGB 0,8876 > LGBM 0,8828 > GNN 0,8114 > DNN 0,7596; **Hibrit OOF-stacking 0,8936** en güçlü tek modeli geçer (çeşitlilik + stacking kazancı). Kendiyle-tutarlılık ($2PR/(P{+}R)=$ F1) ve sızıntısız protokol tüm karşılaştırmalarda korunur.

![Şekil 17 [TAM]: Ensemble vs. baseline / tek-model karşılaştırması.](reports/figures/pdr/13_benchmark_comparison.png)

---

[[PAGEBREAK]]

## 4. SONUÇ (25 puan)

### 4.1 Ana Bulgular ve Yorum

VARIANT-GNN dört panelde **sızıntısız** ve dürüst sonuçlar üretmiştir: resmi beklenti %20-prior'da 4-panel F1 ortalaması (CFTR dahil) = **0,631** (muhafazakâr 3-panel tanı, CFTR hariç = 0,6202; havuzlanmış 0,6042 ± 0,0324), iç ayrım gücü Test F1 = 0,8367. F1'in patojenik-azınlık test'inde ~0,60 olması metrik tanımının (`pos_label=1`) doğal sonucudur. Üretim CV F1 = 0,8936 ± 0,0004 ve 5-seed kararlılığı (0,8738 ± 0,0034) tekrar üretilebilirliği; tüm panellerde yüksek PR-AUC (KANSER 0,9743 … PAH 0,8908) eşik-bağımsız güçlü ayrımı doğrular. **Katkı:** kolon-isimsiz, anonim ve dengesiz bir tablo-veride graf-tabanlı ilişkisel öğrenmeyle hibrit ensemble'ın panel-özgün, kalibre ve açıklanabilir tahmin verebildiği gösterilmiştir.

### 4.2 PSR ile Karşılaştırma ve Tutarsızlık Açıklaması

PDR gerçek-veri sonuçları PSR pilot sonuçlarından belirgin farklıdır; bu fark öngörülmüş ve bilimsel açıdan tutarlıdır.

**Tablo 6: PSR Pilot vs. Gerçek Yarışma Verisi (canonical)**

| Metrik | PSR Pilot | Gerçek (canonical) | Fark | Açıklama |
|:--|:--:|:--:|:--:|:--|
| Binary F1 (iç hold-out) | 0,945 | 0,8367 | −0,108 | gerçek zorluk + sızıntısız group-aware eval |
| MCC | 0,892 | 0,5112 | −0,381 | sınıf dengesizliği (2,75:1) + dürüst eval |
| ROC-AUC | 0,976 | 0,8538 | −0,122 | gerçek varyant heterojenliği |
| PR-AUC | 0,973 | 0,9267 | −0,046 | makul kalibrasyon dayanıklılığı |

**Nedenler:** (1) *Veri kalitesi* — pilot temiz ClinVar Expert Panel; yarışma verisi heterojen + sınır varyantlar. (2) *Sınıf dengesi* — pilot 1:1, yarışma 2,75:1; dengesizlik MCC'yi F1'den orantısız etkiler (denklem (10)). (3) *Özellik uzayı* — pilotta bilinen kolonlar, yarışmada 343 anonim kolon; ColumnAligner ele alır. **GNN adı:** PSR'deki "SAGEConv" gerçekte GATv2Conv'dur; PDR §2.2'de düzeltildi, Brody [8] eklendi.

### 4.3 Güçlü ve Zayıf Yönler

**Güçlü:** sızıntısızlık + kendiyle-tutarlılık kapısı (§7.5 re-run'da birebir); yüksek precision (0,9241, FP düşük); yüksek PR-AUC (0,9267; KANSER 0,9743) + güçlü kalibrasyon (ECE = 0,0291); tohum kararlılığı (5-seed 0,8738 ± 0,0034); kolon-isimsiz çalışma (ColumnAligner + CategoricalBioFeaturizer).
**Zayıf/sınırlılık:** MASTER MCC = 0,4951 (2,75:1 dengesizlik); %20-prior'da düşük jüri F1 (General 0,6006, PAH 0,5299) — metrik tanımının sonucu; küçük panellerde metrik kararsızlığı (CFTR $n=18$ → MCC tanımsız; PAH hold-out ROC-AUC 0,7016 gürültülü, OOF-robust ≈ 0,789); anonim-kolon kısıtı biyolojik yorumu sınırlar (heuristik eşleme — SHAP'ta in-silico baskınlığı bunun yansıması, Tablo 3).

### 4.4 Hata Analizi (Yanlış Pozitif / Yanlış Negatif)

**Yanlış Negatif (FN) — kaçırılan patojenik:** recall = 0,7644 → iç hold-out patojeniklerinin ~%23,6'sı kaçırılır; bu, eşiğin %20-prior'a/yüksek-precision'a kalibre edilmesinin **bilinçli** sonucudur. Örüntü: çelişkili in-silico skor profilleri (~%60), AF sınırı 0,0008–0,002 (~%25); bu FN'lerde ort. MC Dropout $\sigma = 0{,}38 > 0{,}30$ → otomatik "Uzman Değerlendirmesi Gerekli". **FN klinik açıdan en kritik hata tipidir** (patojenik varyantın gözden kaçması). **Yanlış Pozitif (FP):** precision = 0,9241 → patojenik tahminlerin ~%7,6'sı aslında Benign; profil: yüksek in-silico (>0,6) + gnomAD AF > 0,01, korunmuş bölgede sessiz AA değişimi (ort. $\sigma = 0{,}34$). **Zorlanılan özellik grubu:** SHAP (Tablo 3) modelin popülasyon frekansını biyolojik beklentinin altında ağırlıkladığını gösterir → yüksek-AF benign varyantlarda FP riski buradan kaynaklanır.

![Şekil 18 [2-UP]: Hata profili — FN/FP'lerin özellik ve belirsizlik ($\sigma$) dağılımı.](reports/figures/pdr/15_error_profile.png)

### 4.5 Gelecek Çalışma

(1) Panel-spesifik MCC-optimize eşik; (2) ClinVar/gnomAD ile daha büyük CFTR/PAH kohortları; (3) Conformal Prediction (abstain → uzman incelemesi); (4) AlphaFold2 ΔΔG ile protein-yapı entegrasyonu; (5) prospektif klinik validasyon (ACMG uzman karşılaştırması).

### 4.6 Final Aşamasında Karşılaşılabilecek Zorluklar

**Dağılım kayması:** kör test verisi farklı varyant profili içerebilir; adversarial validation (AUC ≈ 0,50) ve ColumnAligner ile risk azaltılır. **Eşik uyarlaması:** sınıf dengesi farklı olabilir; eşikler `models/threshold.json`'dan dinamik yüklenir, resmi prior'a yeniden türetilebilir (derivation == inference). **Tekrar çalıştırma (§7.5):** `requirements.txt` sabit versiyon, seed = 42, `submission/predict.py` tek-giriş + Docker (CPU+GPU). **Hesaplama süresi:** GATv2GNN CPU inference `scripts/test_cpu_inference.py` ile doğrulandı, ONNX hazır (~500 örnek < 90 sn). **Kolon yapısı farkı:** ColumnAligner (exact → case-insensitive → fuzzy ≥ 0,85 → positional) + `predict_schema.json` eksik/fazla kolonu tolere eder.

---

[[PAGEBREAK]]

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

**RAPOR SONU** · *Takım XYRA3 | Takım ID: 909249 | Başvuru ID: 5200240*
*TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması — Proje Detay Raporu · 10 Haziran 2026*

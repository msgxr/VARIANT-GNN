# Model Kartı — VARIANT-GNN

> **TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması** — Üniversite ve Üzeri kategorisi
> referans alınmıştır. Şartname §3.2 (varyant sınıflandırma), §7.3 (F1 metriği),
> §10 (etik kurallar), §12 (sorumluluk beyanı) ile uyumludur.

---

## ⚠️ KLİNİK KULLANIM YASAĞI (Şartname §10)

> "Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar, herhangi
> bir klinik tanı, tedavi veya tıbbi karar destek amacıyla kullanılamaz.
> Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."

**VARIANT-GNN bir araştırma prototipidir; KLİNİK BİR ARAÇ DEĞİLDİR.**

| ÖZETLE | DURUM |
|--------|-------|
| ❌ Klinik tanı koyamaz | Tıbbi cihaz değildir (CE/FDA yok) |
| ❌ Tedavi kararı üretemez | Sadece araştırma çıktısıdır |
| ❌ Doktor görüşü yerine geçemez | Human-in-the-loop zorunludur |
| ❌ Adli/yasal delil değildir | Mahkeme/sigorta süreçlerinde kullanılamaz |
| ❌ Klinik validasyon yapılmadı | Bağımsız onay gerektirir |
| ✅ Yarışma değerlendirmesi için sunulmuştur | TEKNOFEST §7.3 (Üniversite) |
| ✅ Eğitim/araştırma amaçlı kullanılabilir | UNESCO ilkesiyle uyumlu (§10) |

---

## 📋 Özet

| Alan | Değer |
|------|-------|
| **Proje Adı** | VARIANT-GNN |
| **Takım** | XYRA3 (909249) — Başvuru ID: 4865399 |
| **Yarışma** | TEKNOFEST 2026 Sağlıkta Yapay Zeka — Üniversite ve Üzeri |
| **Görev** | Missense genetik varyantların Patojenik / Benign ikili sınıflandırması (§3.2) |
| **Birincil Metrik** | Binary F1 = 2·TP / (2·TP + FP + FN) — Patojenik sınıfı, pos_label=1 (§7.3) |
| **Mimari** | XGBoost + LightGBM + VariantGATv2GNN + VariantDNN ağırlıklı topluluk |
| **Aşama** | PDR (Proje Detay Raporu) — Teslim 29 Haziran 2026 |
| **Kapsam** | 4 panel: General / Hereditary_Cancer / PAH / CFTR |
| **Veri Durumu** | ✅ **Gerçek yarışma verisi alındı (14 Mayıs 2026). Sızıntısız (group-aware) protokolle yeniden eğitildi (1 Haziran 2026). Test F1 = 0.8367, CV F1 = 0.8936 (production OOF-stacking).** |

---

## 🧬 Veri Kullanımı

### Eğitim Veri Setleri (§3.2)

| Panel | Patojenik | Benign | Toplam |
|-------|-----------|--------|--------|
| General | 2149 | 782 | 2931 |
| Hereditary Cancer (Kanser) | 268 | 120 | 388 |
| Fenilketonüri (PAH) | 310 | 62 | 372 |
| Kistik Fibrozis (CFTR) | 90 | 21 | 111 |
| **Toplam** | **2817** | **985** | **3802** |

> **Not:** Gaussian augmentation **KAPALI** — önceden materyalize edilmiş `train_variants_aug.csv`
> (3802→7604), jitter'lı near-twin kopyaları satır-bazlı split'in iki yanına düşürerek sızıntı
> yaratıyordu (`reports/leakage_quantification.json`). Eğitim 3802 orijinal örnek üzerinde,
> `Variant_ID`'ye göre **group-aware** bölme + sadece eğitim fold'unda SMOTE ile yapılır.

### Test Veri Setleri (§3.2 — asimetrik/benign-baskın "Klinik Stres Testi")

| Panel | Patojenik | Benign | Toplam |
|-------|-----------|--------|--------|
| General | 500 | 3000 | 3500 |
| Hereditary Cancer | 100 | 500 | 600 |
| PAH (Fenilketonüri) | 100 | 250 | 350 |
| CFTR (Kistik Fibrozis) | 20 | 100 | 120 |

> **Not (çelişki değil):** Yukarıdaki tablo şartname §3.2'nin **asimetrik test** tasarımıdır
> (benign-baskın "Klinik Stres Testi"; jüri günü gelecek). Raporlanan tüm metrikler ise fiili 3802-satırlık
> `data/train_variants.csv` üzerinden **group-aware %20 hold-out** ile hesaplanmıştır
> (gerçek test n=762: General 582 · KANSER 86 · PAH 76 · CFTR 18). Jüri kör test setini
> sağladığında `predict.py` ile tahmin üretilir.

### Veri Kaynakları (Şartname §10)

- **ClinVar / ClinGen** (Patojenik) — 3-4 yıldız Expert Panel etiketleri
- **gnomAD** (Benign) — popülasyon allel frekansları
- Tümü kamuya açık, anonimleştirilmiş, KVKK + GDPR uyumlu
- Genomik adres bilgileri (Chr/Pos) **gizlidir** (§3.2)

### Öznitelik Kategorileri (§3.2)

1. **Sekans ve Değişim Bilgisi** — Ref/Alt nükleotid, kodon, AA değişimi
2. **Yerel Bağlam** — 5 nükleotid + 5 amino asit komşuluğu
3. **Biyokimyasal/Yapısal Etkiler** — hidrofobiklik, polarite, MW farkı
4. **Evrimsel Korunmuşluk** — GERP, PhyloP, phastCons, SiPhy
5. **Popülasyon Verileri** — gnomAD allel frekansları (5 alt-popülasyon)
6. **In Silico Risk Skorları** — SIFT, PolyPhen2, CADD, REVEL, MutPred2 vb.

---

## 🏗️ Mimari

### Topluluk Bileşenleri

| Model | Ağırlık | Mimari | Özellik |
|-------|---------|--------|---------|
| **XGBoost** | %30 | Gradient Boosting Trees | Tabular güç, hızlı |
| **LightGBM** | %30 | Leaf-wise GBT | Düşük bellek, hızlı |
| **VariantGATv2GNN** | %25 | GATv2 + LayerNorm + skip | Komşuluk graph |
| **VariantDNN** | %15 | 3-katmanlı MLP + Dropout | Doğrusal-olmayan |

> Ağırlıklar `configs/default.yaml` üzerinden yapılandırılabilir; eğitim
> sırasında **Nelder-Mead** ile validation set üzerinde otomatik optimize
> edilir. Ek olarak **LogisticRegression meta-learner** stacking
> (`fit_meta_learner`) etkindir.

### Genelleme Teknikleri (§2/§7.3)

**Shipped pipeline'da aktif (canonical model):**
- **Stochastic Weight Averaging (SWA)** — Izmailov 2018, UAI (`src/training/swa.py`)
- **Domain Adversarial Training (DANN)** — Ganin 2016, JMLR (`configs/pdr.yaml: use_dann=true`; LOPO +2.17pp)
- **OOF-Stacking** — Wolpert 1992 (`models/meta_learner.pkl`)
- **Conformal Prediction (LAC/Mondrian)** — Angelopoulos & Bates 2021 (`reports/conformal_coverage_report.json`)
- **Confident Learning** — Northcutt 2021, JAIR (`--mode label_quality`)

**Repoda mevcut / deneysel (varsayılan kapalı, ablasyon için):**
- Sharpness-Aware Minimization (SAM) · Snapshot Ensemble · Tabular Mixup

### Destek Katmanları

| Bileşen | Modül | Amaç |
|---------|-------|------|
| Kalibrasyon | `EnsembleCalibrator` (Isotonic) | Brier + ECE optimize |
| Belirsizlik (Bayesian) | MC Dropout (10 pass) | Epistemic uncertainty |
| Belirsizlik (frequentist) | Conformal Prediction | Provable coverage |
| Açıklanabilirlik | SHAP + LIME + GNNExplainer | Per-tahmin katkılar |
| Klinik Yorumlama | ACMG/AMP 2015 Mapper | Standart genetik kriter |
| OOD Detection | Mahalanobis-based | Dağılım dışı tespit |
| Label Quality | Confident Learning | Gürültülü etiket tespit |
| Reproducibility | Manifest + SHA256 | §7.5 jüri re-run |

---

## 📊 Değerlendirme Katmanları

| Katman | Komut | Çıktı |
|--------|-------|-------|
| 5-fold CV | `--mode crossval` | `reports/cv_report.json` |
| Hold-out Test | `--mode train` (içinde) | `reports/cv_report.json` |
| Panel Kırılımı | `evaluate_per_panel()` | `cv_report.json` `panel_metrics` |
| Eksternal Validasyon (§2/§7.3) | finalde jüri YENİ verisiyle | (yerel smoke-test artefaktı geçersiz — `reports/_quarantine/`) |
| Cross-Panel Generalization | `--mode panel_transfer` | `reports/panel_transfer_matrix.json` |
| Adversarial Validation | `--mode adversarial_val` | `reports/adversarial_validation_report.json` |
| Ablation Analysis | `--mode ablation` | `reports/ablation_report.json` |
| Label Quality | `--mode label_quality` | `reports/label_quality_report.json` |
| Açıklanabilirlik | `--mode explain` | `reports/shap_*.png` + `explain_instances.json` |

---

## 🎯 Performans Sonuçları (Gerçek TEKNOFEST 2026 Verisi)

> ✅ Model 1 Haziran 2026'da gerçek yarışma verisi (3802 örnek, 3224 benzersiz varyant,
> 4 panel) üzerinde **sızıntısız (group-aware, Variant_ID)** protokolle eğitilmiştir.
> Birincil metrik 5-fold CV; test = tek group-aware hold-out (n=762, daha yüksek varyans).
> Önceki 0.8980/0.9269 sayıları satır-bazlı split sızıntısı nedeniyle geri çekildi
> (`RESULTS_CANONICAL.json`, `reports/leakage_quantification.json`).

**Genel (Hold-Out + 5-fold CV):**

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **CV Binary F1 (OOF-stacking nested)** | **0.8936 ± 0.0004** | Production OOF-stacking nested-CV (5-seed). Fold-CV=0.8812±0.0113 (bileşen). |
| Test Binary F1 (hold-out @ θ=0.8415) | 0.8367 | pos_label=1; tek group-aware bölme (n=762) |
| MCC (test) | 0.5112 | precision/recall, binary_f1'i birebir üretir |
| PR-AUC (test) | 0.9267 | Eşik bağımsız ayırt edicilik |
| ROC-AUC (test) | 0.8538 | Genel sınıf ayrımı |
| Precision / Recall (test) | 0.9241 / 0.7644 | Patojenik sınıf |
| Brier / ECE (test) | 0.1115 / 0.0291 | Kalibrasyon kalitesi/sapması |
| **Jüri beklentisi (resmi 4-panel %20-F1 ort.)** | **0.6202** | %20-patojenik jüri seti; havuzlanmış 0.6042±0.0324 |

**Panel Bazlı Sonuçlar (test, sızıntısız):**

| Panel | Binary F1 | MCC | Açıklama |
|-------|-----------|-----|----------|
| MASTER (General) | 0.8185 | 0.4951 | θ=0.8415 |
| KANSER (Hereditary_Cancer) | 0.9060 | 0.7135 | θ=0.8415 |
| PAH | 0.9120 | 0.5053 | θ=0.8415 |
| CFTR | 0.7143 | tanımsız (0) | θ=0.8415; küçük n, MCC anlamsız |

> **Not:** CFTR test n çok küçük (büyük çoğunluk patojenik) → MCC tanımsız/0; F1 daha anlamlı.
> Resmi jüri skoru: 4-panel %20-F1 ortalaması = **0.6202** (%20-patojenik jüri seti);
> havuzlanmış = 0.6042±0.0324. Tüm sayılar `RESULTS_CANONICAL.json`'dan üretilir.

---

## 🔁 Reproducibility (§7.5)

> "Yarışma jürisi, finale kalan takımların kodlarını tekrar çalıştırmasını
> ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."

VARIANT-GNN bu yetkiyi tam olarak destekler:

1. **Sabit RNG seed** — `seed=42` (`src/utils/reproducibility.py`)
2. **Deterministik PyTorch** — `cudnn.deterministic=True`
3. **Sabit paket versiyonları** — `requirements*.txt` `==` formatında
4. **İmzalı manifest** — `models/reproducibility_manifest.json`
   - Veri SHA256 + label dağılımı
   - Tüm paket versiyonları
   - Git commit + branch + dirty flag
   - Tüm artefakt SHA256 hash'leri
   - Self-hash (tamper detection)
5. **Doğrulama API'si** — `verify_manifest_chain()`

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Sanal ortam + bağımlılıklar
python3.12 -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# 2. Test (418 statik test fonksiyonu, 39 dosya — yeşil olmalı)
pytest tests/ -q

# 3. Eğitim (mevcut artifact'lar models/ altında)
python main.py --mode train --config configs/pdr.yaml

# 4. Jüri inference
python submission/predict.py \
    --input <jury_test.csv> \
    --model_dir models \
    --output submission/predictions.csv \
    --config configs/pdr.yaml

# 5. Streamlit demo
streamlit run app.py
```

Ayrıntı: [`docs/architecture.md`](docs/architecture.md),
[`docs/submission_guide.md`](docs/submission_guide.md),
[`docs/evaluation/evaluation_protocol.md`](docs/evaluation/evaluation_protocol.md)

---

## 📜 Şartname Uyumluluğu Tablosu

| § | Madde | VARIANT-GNN Uygulaması |
|---|-------|------------------------|
| §3.2 | 4 panel sınıflandırma | `src/data/loader.py` Panel one-hot |
| §3.2 | 6 öznitelik kategorisi | `src/features/feature_validator.py` |
| §3.2 | Anonim kolon adları | `src/inference/anonymous_inference.py` |
| §3.2 | Genomik adres gizliliği | Hiçbir yerde Chr/Pos kullanılmaz |
| §7.3 | F1 = 2TP/(2TP+FP+FN) | `evaluate.binary_f1` |
| §2/§7.3 | Eksternal validasyon (finalde jüri yeni verisiyle) | `external_validation_runner.py` |
| §7.5 | Jüri kod re-run | `reproducibility_manifest.py` |
| §10  | Etik beyan | `docs/ethical_statement.md` |
| §12  | Sorumluluk beyanı | Bu dosya + `docs/ethical_statement.md` |

---

## 📚 Referanslar

- Richards et al. (2015). *ACMG/AMP standards for variant interpretation.*
- Izmailov et al. (2018). *Stochastic Weight Averaging.* UAI.
- Foret et al. (2021). *Sharpness-Aware Minimization.* ICLR.
- Huang et al. (2017). *Snapshot Ensembles.* ICLR.
- Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization.* ICLR.
- Ganin & Lempitsky (2015). *Domain-Adversarial Training.* ICML.
- Vovk et al. (2005). *Algorithmic Learning in a Random World.*
- Angelopoulos & Bates (2021). *Conformal Prediction.* arXiv:2107.07511
- Northcutt et al. (2021). *Confident Learning.* JAIR.

---

## 📝 Lisans ve İletişim

- **Lisans:** TEKNOFEST 2026 Yarışma Lisansı (`LICENSE` dosyası)
- **Repository:** https://github.com/msgxr/VARIANT-GNN
- **Takım:** XYRA3 — Başvuru ID 4865399
- **İletişim:** TEKNOFEST KYS (`www.t3kys.com`) → Sağlıkta Yapay Zeka Yarışması
- **Organizasyonel sorular:** `iletisim@teknofest.org` (§11)

---

---

## ⚠️ Bilinen Sınırlamalar

| Sınırlılık | Açıklama |
|---|---|
| **CFTR örneklem küçüklüğü** | Toplam 111 örnek (test hold-out n=18); istatistiksel güç sınırlı, MCC tanımsız → F1/precision daha anlamlı; bağımsız kohortlarda doğrulama gereklidir |
| **VUS desteği yok** | "Variant of Uncertain Significance" sınıflandırması kapsam dışıdır |
| **Bağımsız klinik validasyon yok** | Harici klinik kohort üzerinde doğrulanmamıştır |
| **Genomik koordinat bağımsız** | Chr/Pos gizlendiğinden uzak konumdaki varyantlar model tarafından ayrıştırılamaz |

---

## 🏷️ Sürüm

| Alan | Değer |
|---|---|
| **Sürüm** | `v1.0.0` |
| **Durum** | Üretim — gerçek TEKNOFEST 2026 yarışma verisi ile eğitilmiş |
| **Son güncelleme** | 10 Haziran 2026 — Test F1=0.8367 @ θ=0.8415 (sızıntısız, group-aware); %20-patojenik prior Q&A-II ile doğrulandı |
| **Bir sonraki kilometre taşı** | PDR teslimi → 29 Haziran 2026 |

> **Bu Model Card belgesi canlıdır.** Yarışma süresince ve sonrasında kod
> değişiklikleriyle birlikte güncellenir. Son güncelleme: 10 Haziran 2026
> (θ=0.8415 canonical retrain; %20-patojenik test prior'ı Q&A-II transkriptiyle
> doğrulandı; tüm sayılar `RESULTS_CANONICAL.json` ile tutarlı).

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
| ✅ Yarışma değerlendirmesi için sunulmuştur | TEKNOFEST §7.2-7.3 |
| ✅ Eğitim/araştırma amaçlı kullanılabilir | UNESCO ilkesiyle uyumlu (§10) |

---

## 📋 Özet

| Alan | Değer |
|------|-------|
| **Proje Adı** | VARIANT-GNN |
| **Takım** | XYRA3 (909249) — Başvuru ID: 4865399 |
| **Yarışma** | TEKNOFEST 2026 Sağlıkta Yapay Zeka — Üniversite ve Üzeri |
| **Görev** | Missense genetik varyantların Patojenik / Benign ikili sınıflandırması (§3.2) |
| **Birincil Metrik** | Binary F1 = TP / (TP + 0.5·FP + 0.5·FN) — Patojenik sınıfı (§7.3) |
| **Mimari** | XGBoost + LightGBM + VariantGATv2GNN + VariantDNN ağırlıklı topluluk |
| **Aşama** | PDR (Proje Detay Raporu) — Teslim 29 Haziran 2026 |
| **Kapsam** | 4 panel: General / Hereditary_Cancer / PAH / CFTR |
| **Veri Durumu** | ✅ **Gerçek yarışma verisi alındı (14 Mayıs 2026). Sızıntısız (group-aware) protokolle yeniden eğitildi (1 Haziran 2026). Test F1 = 0.9069, CV F1 = 0.8936.** |

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

### Test Veri Setleri (§3.2)

| Panel | Patojenik | Benign | Toplam |
|-------|-----------|--------|--------|
| General | 1000 | 1000 | 2000 |
| Hereditary Cancer | 100 | 100 | 200 |
| PAH | 100 | 100 | 200 |
| CFTR | 30 | 30 | 60 |

> **Not:** Test seti metrikleri yukarıdaki tabloda gösterilmektedir. Tüm değerler
> gerçek TEKNOFEST 2026 yarışma verisi üzerinde hesaplanmıştır (20 Mayıs 2026).

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

### Cutting-Edge Teknikler (§7.2 generalization için)

- **Stochastic Weight Averaging (SWA)** — Izmailov 2018, ICLR
- **Sharpness-Aware Minimization (SAM)** — Foret 2021, ICLR
- **Snapshot Ensemble** — Huang 2017, ICLR
- **Tabular Mixup** — Zhang 2018, ICLR
- **Domain Adversarial Training (DANN)** — Ganin 2015, ICML
- **Conformal Prediction** — Vovk 2005, Angelopoulos 2021
- **Confident Learning** — Northcutt 2021, JAIR

### Destek Katmanları

| Bileşen | Modül | Amaç |
|---------|-------|------|
| Kalibrasyon | `EnsembleCalibrator` (Isotonic) | Brier + ECE optimize |
| Belirsizlik (Bayesian) | MC Dropout (30 pass) | Epistemic uncertainty |
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
| External Validation (§7.2) | `--mode external_val` | `reports/external_validation_report.json` |
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
| **CV Binary F1 (OOF-stacking nested)** | **0.8936 ± 0.0004** | Production OOF-stacking nested-CV (5-seed). Weighted-avg fold CV=0.8779 (bileşen). |
| Test Binary F1 (hold-out) | 0.9069 | pos_label=1; tek group-aware bölme (n=762) |
| MCC (test) | 0.5639 | precision/recall, binary_f1'i birebir üretir |
| PR-AUC (test) | 0.9114 | Eşik bağımsız ayırt edicilik |
| ROC-AUC (test) | 0.8398 | Genel sınıf ayrımı |
| Precision / Recall (test) | 0.8525 / 0.9686 | Patojenik sınıf |
| Brier / ECE (test) | 0.1197 / 0.0755 | Kalibrasyon kalitesi/sapması |

**Panel Bazlı Sonuçlar (test, sızıntısız):**

| Panel | Binary F1 | MCC | PR-AUC |
|-------|-----------|-----|--------|
| MASTER (General) | 0.8985 | 0.5410 | 0.9102 |
| KANSER (Hereditary_Cancer) | 0.9385 | 0.7753 | 0.9393 |
| PAH | 0.9173 | 0.4215 | 0.8843 |
| CFTR | 0.9714 | — (küçük n) | 1.0000 |

> **Not:** CFTR test n çok küçük (büyük çoğunluk patojenik) → MCC tanımsız/0; F1 ve PR-AUC
> daha anlamlı. Tüm sayılar `RESULTS_CANONICAL.json`'dan üretilir.

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

# 2. Test (277+ yeşil olmalı)
pytest tests/ -q

# 3. Eğitim (mevcut artifact'lar models/ altında)
python main.py --mode train --config configs/pdr.yaml

# 4. Jüri inference
python submission/predict.py \
    --input data/test_variants_blind.csv \
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
| §7.2 | External validation | `external_validation_runner.py` |
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
| **CFTR örneklem küçüklüğü (eğitim)** | 111 eğitim örneği ile istatistiksel güç sınırlıdır; bağımsız kohortlarda doğrulama gereklidir |
| **CFTR panel az örnekli** | 140 eğitim örneği; panel-genelleme riski yüksek, dikkatli yorumlanmalı |
| **VUS desteği yok** | "Variant of Uncertain Significance" sınıflandırması kapsam dışıdır |
| **Bağımsız klinik validasyon yok** | Harici klinik kohort üzerinde doğrulanmamıştır |
| **Genomik koordinat bağımsız** | Chr/Pos gizlendiğinden uzak konumdaki varyantlar model tarafından ayrıştırılamaz |

---

## 🏷️ Sürüm

| Alan | Değer |
|---|---|
| **Sürüm** | `v1.0.0` |
| **Durum** | Üretim — gerçek TEKNOFEST 2026 yarışma verisi ile eğitilmiş |
| **Son güncelleme** | 1 Haziran 2026 — Test F1=0.9069 (sızıntısız, group-aware) |
| **Bir sonraki kilometre taşı** | PDR teslimi → 29 Haziran 2026 |

> **Bu Model Card belgesi canlıdır.** Yarışma süresince ve sonrasında kod
> değişiklikleriyle birlikte güncellenir. Son güncelleme: 22 Mayıs 2026.

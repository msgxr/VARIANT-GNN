# VARIANT-GNN Sistem Mimarisi

> TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri kategorisi
> referans alınarak hazırlanmıştır (§3.2, §7.3).

## 1. Genel Bakış

VARIANT-GNN, **klinik durumu bilinmeyen genomik varyantların** Patojenik veya
Benign sınıflandırmasını yapan **çok bileşenli hibrit bir makine öğrenmesi
sistemidir**. Sistem, dört bağımsız ML modelini ağırlıklı bir topluluk halinde
birleştirir; biyolojik öznitelikleri çoklu modaliteye (sayısal +
sekans-bağlam) genişletir; jüri zamanı kolon adlarının gizli olabileceği
senaryoları ele alır (§3.2).

## 2. Pipeline Akışı

```
                    ┌──────────────────────────────────────┐
                    │  CSV (varyant öznitelikleri)         │
                    │  • 343 anonim kolon (sayısal)        │
                    │  • Sekans bağlam (Nuc_Context, AA_*)│
                    │  • Panel (4 sınıf)                   │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   src/data/loader     │
                       │  - Şema doğrulama     │
                       │  - Etiket eşleme      │
                       │  - Feature kategori   │
                       │    doğrulayıcı (§3.2) │
                       └───────────┬───────────┘
                                   ▼
                  ┌────────────────────────────────────┐
                  │ src/features/preprocessing         │
                  │  • GROUP-AWARE split (Variant_ID)  │
                  │  • ColumnAligner (anonim kolonlar) │
                  │  • CategoricalBioFeaturizer (ACMG) │
                  │  • Median Imputer                  │
                  │  • RobustScaler                    │
                  │  • SMOTE (yalnız eğitim fold)      │
                  │  • Distributional signatures       │
                  │     (anonim eşleştirme için)       │
                  │  • Cosine k-NN graph (TAM öznitelik)│
                  │  (SelectKBest/AutoEncoder KALDIRILDI)│
                  └─────────────────┬──────────────────┘
                                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                 HybridEnsemble (4 model)                       │
   │                                                                │
   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
   │   │ XGBoost  │  │ LightGBM │  │ GATv2GNN │  │ VariantDNN│    │
   │   │ (tabular)│  │ (tabular)│  │  (graph) │  │  (deep)  │     │
   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
   │        │              │              │              │           │
   │        └──────┬───────┴──────────────┴──────┬───────┘           │
   │               ▼                             ▼                    │
   │       Weighted Mean (Nelder-Mead opt)  Stacking Meta-Learner    │
   │                              │                                   │
   │                              ▼                                   │
   │                   Ensemble probability (P(Pathogenic))            │
   └─────────────────────────────┬────────────────────────────────────┘
                                 ▼
                ┌────────────────────────────────────┐
                │ EnsembleCalibrator (Isotonic)      │
                │  • Validation-set fit              │
                │  • Brier + ECE optimize            │
                └─────────────────┬──────────────────┘
                                  ▼
                ┌────────────────────────────────────┐
                │ ConformalPredictor (opsiyonel)     │
                │  • Provable coverage guarantee     │
                │  • Mondrian (panel-conditional)    │
                └─────────────────┬──────────────────┘
                                  ▼
              ┌──────────────────────────────────────────┐
              │  Output (CSV — submission/predict.py)    │
              │  • Variant_ID                            │
              │  • Prediction (Pathogenic|Benign)        │
              │  • Pathogenic_Probability                │
              │  • Calibrated_Risk                       │
              │  • Uncertainty (MC-Dropout / CP)         │
              │  • Clinical_Flag (LOW_CONFIDENCE / OK)   │
              └──────────────────────────────────────────┘
```

## 3. Bileşen Detayları

### 3.1 Veri Katmanı (`src/data/`)

| Modül | Sorumluluk |
|------|-----------|
| `loader.py` | CSV okuma + şema doğrulama + label mapping + Panel one-hot |
| `column_aligner.py` | 4 aşamalı (exact / case / fuzzy / distributional / positional) hizalama |
| `competition_sanitizer.py` | Yarışma modu sterilizasyonu (label kolonu bypass, leakage firewall) |
| `leakage_firewall.py` | Eğitim → çıkarım veri akışında label sızıntısı dedektörü |
| `schema_guard.py` | Pydantic v2 şema doğrulayıcı |

### 3.2 Özellik Mühendisliği (`src/features/`)

- **`preprocessing.py`** — Sklearn-uyumlu pipeline (ColumnAligner → CategoricalBioFeaturizer →
  median impute → RobustScaler → SMOTE [yalnız eğitim fold] → cosine k-NN graph; **TAM
  öznitelik seti**). **Tek kuralı:** her transformer YALNIZCA eğitim split'inde fit edilir.
  > **Not (sızıntısız retrain):** `VarianceThreshold + SelectKBest(35) + AutoEncoder(→16)`
  > adımları **kaldırıldı** — group-aware CV'de sinyal atıp ≈+5.3pp F1 kaybettiriyorlardı
  > (`reports/preprocessing_diagnostic.json`).
- **`autoencoder.py`** — **LEGACY (shipped pipeline'dan kaldırıldı).** 16-boyutlu latent
  temsili; sızıntısız değerlendirmede net negatif etki yaptığı için devre dışı, yalnız
  ablasyon/tarihsel referans için tutulur.
- **`multimodal_encoder.py`** — Nuc_Context (11 nükleotid) ve AA_Context
  (11 amino asit) için CNN tabanlı SequenceEncoder.
- **`feature_validator.py`** — §3.2 6 özellik kategorisini doğrular ve
  coverage skoru üretir.
- **`bio_scoring.py`** — BLOSUM62 + Grantham bilimsel skor sözlüğü.

### 3.3 Çekirdek Modeller (`src/core/`)

| Model | Mimari | Parametre |
|-------|---------|-----------|
| **XGBoost** | Gradient Boosting Trees | `n_est=200, max_depth=6` |
| **LightGBM** | Gradient Boosting (faster) | `num_leaves=63, n_est=300` |
| **VariantGATv2GNN** | 3 katmanlı GATv2 + LayerNorm + skip conn | `hidden=128, heads=4` |
| **VariantDNN** | 3 gizli katmanlı MLP + Dropout | `hidden=128, drop=0.3` |

GNN, `src/core/graph/builder.py` tarafından kurulan **kosinüs k-NN örnek
grafiği** üzerinde tam-batch node classification yapar. Genomik adres
(Chr/Pos) **kullanılmaz** (§3.2 uyumlu).

### 3.4 Eğitim (`src/training/`)

- **`trainer.py`** — `VariantTrainer.train()` → **group-aware** 5-fold StratifiedGroupKFold
  (Variant_ID) + group-aware hold-out test split (leakage guard: 0 straddle) + final fit + kalibrasyon.
- **`focal_loss.py`** — Sınıf dengesizliği için Focal Loss.
- **`swa.py`** — Stochastic Weight Averaging (Izmailov 2018).
- **`sam.py`** — Sharpness-Aware Minimization (Foret 2021).
- **`snapshot_ensemble.py`** — Cyclic LR + cycle başına checkpoint (Huang 2017).
- **`mixup.py`** — Tabular Mixup (Zhang 2018) + Manifold Mixup.
- **`domain_adversarial.py`** — DANN ile panel-invariant öğrenme (Ganin 2015).
- **`tune.py`** — Optuna hyperparameter search.
- **`cross_val.py`** — Stratified CV yardımcıları.

### 3.5 Inference (`src/inference/`, `src/api/`)

- **`api/pipeline.py` `InferencePipeline`** — Eğitilmiş modelleri yükler ve
  yeni veri üzerinde uçtan uca tahmin döner.
- **`inference/external_validation_runner.py`** — TEKNOFEST §7.2 jüri
  external validation senaryosu için hazır pipeline.
- **`inference/anonymous_inference.py`** — §3.2 anonim-kolon senaryosu için
  Variant_ID + Panel + sekans + sayısal otomatik tespit.
- **`inference/artifact_loader.py`** — ModelStore artefaktlarını sıkı
  doğrulama ile yükler.
- **`inference/triage.py`** — Belirsiz tahminleri klinik uzman
  değerlendirmesi için işaretler.

### 3.6 Değerlendirme (`src/evaluation/`)

| Modül | İçerik |
|-------|--------|
| `src/scientific/metrics/metrics.py` | Binary F1 (TEKNOFEST §7.3 birincil), Macro F1, ROC-AUC, Brier, ECE, MCC — **canlı CLI bu kopyayı import eder** (`src/cli/modes/train.py`, `evaluate.py`) |
| `panel_transfer.py` | 4×4 cross-panel generalization matrisi |
| `ablation.py` | Komponent katkı analizi |
| `abstention_analysis.py` | Belirsiz tahminlerin kalite analizi |
| `src/scientific/metrics/adversarial_validation.py` | Train/test domain shift testi — **canlı CLI bu kopyayı import eder** (`src/cli/modes/evaluate.py`) |
| `benchmarks.py` | Latency + memory benchmarks |
| `plots.py` | Matplotlib görselleştirmeler |

### 3.7 Bilimsel Modüller (`src/scientific/`)

- **`conformal_prediction.py`** — Vovk 2005 kapsama garantili belirsizlik.
- **`label_quality.py`** — Northcutt 2021 confident learning ile gürültülü
  etiket tespiti.
- **`acmg_mapper.py`** — ACMG/AMP 2015 kriter haritalayıcı.
- **`ood_detector.py`** — Distribution drift dedektörü.
- **`differential_privacy.py`** — DP-SGD wrapper'ı (opsiyonel).
- **`pubmed_rag.py`** — Klinik literatür retrieval (opsiyonel).
- **`submission_validator.py`** — §7.3 submission CSV format doğrulayıcı.

### 3.8 Yardımcılar (`src/utils/`)

- **`reproducibility.py`** — Tüm RNG seed setleyici.
- **`reproducibility_manifest.py`** — §7.5 jüri-uyumlu tam manifest
  (veri SHA256, paket versiyonları, git, artifact hash, self-hash).
- **`fingerprinting.py`** — CSV fingerprint helpers.
- **`artifact_manifest.py`** — Hızlı artefakt manifest yöneticisi.
- **`serialization.py`** — Tüm modelleri kaydet/yükle (ModelStore).
- **`seeds.py`** — Set-global-seed.
- **`logging_cfg.py`** — Logging setup.

### 3.9 Topluluk (`src/ensemble/`)

- **`panel_aware_ensemble.py`** — Panel'a göre farklı ağırlıklar uygulayan
  ensemble varyantı.
- **`weight_optimizer.py`** — Nelder-Mead ile validation-set ağırlık
  optimizasyonu.

## 4. Reproducibility Garantileri (§7.5)

Şartname §7.5 uyarınca jüri tarafından kod yeniden çalıştırılabilmelidir.
Sistem aşağıdaki garantileri sunar:

1. Tüm RNG seed'leri tek bir noktadan ayarlanır
   (`src.utils.reproducibility.setup_reproducibility`).
2. Tüm pakelerin versiyonları `requirements*.txt`'te `==` ile sabit
   (TEKNOFEST §7.5).
3. Eğitim artefaktları SHA256 ile imzalanır
   (`reproducibility_manifest.py`).
4. CV split'leri sabit seed ile deterministik.
5. CUDA deterministik algoritmalar etkin (cudnn.deterministic = True).

## 5. Dağıtım Stratejisi

| Hedef | Build dosyası | Açıklama |
|-------|--------------|----------|
| Yerel CPU | `requirements.txt` | Geliştirme + test |
| GPU (CUDA 11.8) | `requirements-gpu.txt` | Yüksek hız eğitim |
| Streamlit Cloud | `requirements-streamlit.txt` | Demo arayüzü |
| Docker | `Dockerfile` | Reproducible eğitim/inference |
| CI | `requirements-ci.txt` | GitHub Actions test pipeline |
| Colab | `requirements-colab.txt` | Notebook akışı |

## 6. Şartname Kapsam Eşlemesi

| Şartname Maddesi | Karşılayan Bileşen |
|------------------|---------------------|
| §3.2 4 panel sınıflandırma | `src/data/loader.py` Panel one-hot |
| §3.2 6 öznitelik kategorisi | `src/features/feature_validator.py` |
| §3.2 anonim kolon adları | `src/inference/anonymous_inference.py` |
| §3.2 genomik adres gizli | Hiçbir yerde Chr/Pos kullanılmaz; sample k-NN graph |
| §7.3 F1 = TP/(TP+0.5·FP+0.5·FN) | `src/scientific/metrics/metrics.py` `binary_f1` (canlı CLI import yolu; 2TP/(2TP+FP+FN) ile özdeş) |
| §7.2 external validation | `src/inference/external_validation_runner.py` |
| §7.5 jüri kod re-run | `src/utils/reproducibility_manifest.py` |
| §10 etik beyan | `docs/ethical_statement.md` |

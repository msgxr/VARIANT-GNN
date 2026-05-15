# VARIANT-GNN — Proje Bağlamı

## Takım Yapısı — XYRA3

| Kişi | KYS Rolü | Proje İçi İşlevsel Rol |
|---|---|---|
| Şeyma Nur Çebi | Takım Kaptanı, İletişim Sorumlusu | ML/İstatistik Koordinasyonu, Model Geliştirme Ana Sorumluluğu |
| Muhammed Sina Gün | Takım Üyesi | Sistem Kurgusu, Teknik Tasarım, Deney Doğrulama, Raporlama, Yarışma Uyumu |
| Şahin Kara | Takım Üyesi | Biyoinformatik Uzmanı, Veri ve Etiket Kalitesi |
| Burak Küçükcengiz | Takım Üyesi | Yazılım Geliştirici, MLOps ve Uygulama Altyapısı |
| Pınar Karadayı Ataş | Danışman | Akademik ve Yöntemsel Rehberlik |

## İşlevsel Sorumluluk Alanları (PSR'de Tanımlanan)

- **Biyoinformatik Uzmanlığı (Şahin Kara):** Veri/etiket kalitesi, ACMG uyumluluğu, ClinVar doğrulama, tutarsız profil tespiti
- **ML / İstatistik Uzmanlığı (Şeyma Nur Çebi):** Model geliştirme, ensemble, hiperparametre, açıklanabilirlik, kalibrasyon
- **Yazılım / MLOps (Burak Küçükcengiz):** CI/CD, Docker, arayüz, API, ColumnAligner, yeniden çalıştırılabilirlik
- **Sistem Kurgusu, Raporlama ve Yarışma Uyumu (Muhammed Sina Gün):** Doğrulama protokolü, panel bazlı değerlendirme, PDR, jüri anlatısı

## Problem

Missense varyantlar, tek amino asit değişikliğine yol açan nokta mutasyonlardır. Klinik önemi belirsiz varyantlar (VUS) için hesaplamalı tahmin gereklidir. Yarışma bağlamında model; anonim öznitelik vektörleri üzerinden Patojenik / Benign ikili sınıflandırması yapmalıdır.

## Hedef

Her veri paneli için (Genel, Herediter Kanser, PAH, CFTR) F1 Skoru'nu maksimize eden, veri sızıntısından arınmış, yeniden üretilebilir ve açıklanabilir bir sınıflandırma sistemi geliştirmek.

## Mevcut Teknik Yaklaşım (PSR'den doğrulanmış)

**Hibrit Ensemble:**
- XGBoost (%30) + LightGBM (%30) + VariantSAGEGNN (%25) + DNN (%15)
- Stacking meta-öğrenici: Lojistik regresyon

**VariantSAGEGNN (GNN ismi):**
- SAGEConv 3 katman, hidden_dim=128, Dropout=0.3
- Cosine k-NN grafı (k=10, eşik=0.3)
- İndüktif yapı — yeni varyantlara genelleme sağlar
- NOT: GATv2 değil, VariantSAGEGNN'dir

**Veri bölme:** %65 eğitim (CV), %15 kalibrasyon, %20 test
**CV:** Stratified 5-Fold, random_state=42
**Hiperparametre:** Optuna Bayesian TPE, 30 deneme, hedef: CV macro F1
**Kalibrasyon:** İsotonik regresyon (%15 kalibrasyon seti)
**Karar eşiği:** 0.40 (duyarlılık öncelikli)
**Açıklanabilirlik:** SHAP (grup bazlı), GNNExplainer, LIME

**GNN gerekçesi (PSR §5.1'de deneysel olarak desteklenmiş):**
Sadece XGBoost CFTR F1: 0.84±0.09; ensemble ile: 0.92. Grafik komşuluk sinyali eklenince stabil performans sağlandı.

**PSR pilot sonuçları (yarışma verisi değil, referans amaçlı):**

| Panel | Macro F1 | ROC-AUC | MCC | Brier |
|---|---|---|---|---|
| Genel | 0.945±0.003 | 0.976 | 0.892 | 0.048 |
| Herediter | 0.938±0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941±0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925±0.012 | 0.962 | 0.852 | 0.065 |

## PDR'de Kurulması Gereken Anlatı

1. Problem neden önemlidir? → Araştırma bağlamında varyant patojenitesi tahmini.
2. Mevcut yaklaşımların sınırlılığı? → Literatür referansları.
3. VARIANT-GNN buna nasıl cevap veriyor? → GNN + ensemble mimarisi ve gerekçesi.
4. Panel bazlı performans? → Her panel için ayrı F1 tablosu.
5. Hatalar nerede? → FP/FN analizi, biyolojik yorum.
6. Sınırlılıklar? → Anonim kolonlar, veri hacmi, etik sınır.

## Güçlendirilmesi Gereken Teknik Sorular

1. GNN graph oluşturma stratejisi nedir? Gerekçesi?
2. Anonim kolonlar resmî öznitelik kategorileriyle nasıl eşleşiyor?
3. CFTR panelinde 70/70 örnekle overfitting nasıl yönetiliyor?
4. Karar eşiği panel bazlı mı, global mı?
5. Açıklanabilirlik PDR'de görselleştirilebildi mi?

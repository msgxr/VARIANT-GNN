# Model Kartı — VARIANT-GNN

## Model Kimliği

| Alan | Değer |
|------|-------|
| **Model Adı** | VARIANT-GNN v0.1.0 |
| **Model Versiyonu** | 0.1.0 |
| **Geliştiren** | VARIANT-GNN Ekibi — TEKNOFEST 2026 |
| **Geliştirme Tarihi** | Mart 2026 |
| **Kullanım Amacı** | Genomik varyant patojenite sınıflandırması (Araştırma) |
| **Model Lisansı** | MIT |

---

## Model Açıklaması

VARIANT-GNN, 34 genomik özellik üzerinden en iyi olasılık tahmini yapan bir ensemble sınıflandırıcıdır:

- **Model 1 — XGBoost:** Geleneksel gradient-boosted decision tree
- **Model 2 — GNN (Graph Convolutional Network):** Özellik korelasyon grafı üzerinde mesaj geçişi
- **Model 3 — DNN (Deep Neural Network):** Tabular özellikler üzerinde derin öğrenme

Ensemble ağırlıkları: [XGB: 0.4, GNN: 0.4, DNN: 0.2]

---

## Amaçlanan Kullanım

### Birincil Kullanım
- Araştırma ortamında varyant sınıflandırma deneyleri
- Biyoinformatik pipeline prototipleme
- Makine öğrenmesi metodoloji karşılaştırmaları

### Amaçlanmayan Kullanım (Kırmızı Çizgiler)
- ❌ Klinik karar destek sistemi olarak kullanım
- ❌ Hasta tanı/tedavi kararlarında kullanım
- ❌ Regüle edilmiş tıbbi cihaz olarak sınıflandırma
- ❌ ACMG sınıflandırma standardının yerini alma

---

## Eğitim Verisi

> ⚠️ **UYARI:** Bu model sürümü gerçekçi istatistiksel sentetik veri üzerinde eğitilmiştir. Gerçek klinik veri kullanılmamıştır.

| Özellik | Değer |
|---------|-------|
| Veri Kaynağı | Sentetik (generate_realistic_data.py) |
| Eğitim Seti | 4000 varyant (2000 Pathogenic, 2000 Benign) |
| Test Seti | 1000 varyant |
| Gerçek Klinik Veri | 0 varyant |
| Eksik Değer Oranı | ~3% (simüle edilmiş) |

---

## Performans Metrikleri (Sentetik Test Seti)

> **NOT:** Aşağıdaki metrikler sentetik veri üzerindeki performansı gösterir. Gerçek ClinVar verisi üzerindeki performans bilinmemektedir.

| Metrik | Değer | Not |
|--------|-------|-----|
| F1-Macro | ~0.90+ | Sentetik veri — gerçek performans yok |
| ROC-AUC | ~0.95+ | Sentetik veri |
| Brier Score | Hesaplanmadı | Kalibrasyon belirsiz |
| ECE | Hesaplanmadı | Kalibrasyon belirsiz |

---

## Sınırlamalar ve Riskler

### Teknik Sınırlamalar
1. **Veri Sızıntısı Riski:** Cross-validation akışında preprocessing leakage riski mevcut (v0.2'de düzeltilecek)
2. **Kalibrasyon Eksikliği:** Risk skoru = prob × 100 kalibre edilmemiş; olasılık değerleri güvenilir değil
3. **Metrik Tutarsızlığı:** Model seçimi accuracy ile yapılıyor; ana metrik F1 (v0.2'de düzeltilecek)
4. **VUS Desteği Yok:** Belirsizlik kategorisi sınıflandırılamamaktadır

### Önyargı ve Adalet
- Popülasyon çeşitliliği yetersiz (gnomAD popülasyon skorları basitleştirilmiş simülasyon)
- Nadir hastalık varyantları temsil edilmemiş

### Genel Uyarı
Bu model, klinik kullanım için doğrulanmamıştır. Tıbbi karar almada kullanılmamalıdır.

---

## Açıklanabilirlik

| Yöntem | Uygulandığı Model | Durum |
|--------|------------------|-------|
| SHAP TreeExplainer | XGBoost | ✅ Aktif |
| LIME Tabular | XGBoost | ✅ Aktif |
| GNNExplainer | GNN | ⚠️ Opsiyonel (torch-geometric versiyonuna bağlı) |
| Ensemble SHAP | Ensemble | ❌ Mevcut değil |

---

## Etik ve Regülasyon

- Bu model CE İşaretleme veya FDA 510(k) onayına sahip değildir
- GDPR kapsamında kişisel genetik veri işlenmesi için izin alınmalıdır
- Klinik kullanım öncesi prospektif klinik validasyon çalışması gerekmektedir

---

## İletişim ve Atıf

GitHub: https://github.com/msgxr/VARIANT-GNN  
Yarışma: TEKNOFEST 2026 Sağlıkta Yapay Zeka

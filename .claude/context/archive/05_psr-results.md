---
Dosya: 05_psr-results.md
Klasör: context/
---

# PSR Teknik Özet + Hakem Analizi + Gerçek Yarışma Sonuçları

---

## 1. GERÇEK YARIIŞMA SONUÇLARI (reports/cv_report.json — 2026-05-15)

Yarışma verisinde çalıştırılan model sonuçları. Şeyma Nur Çebi'nin makinasında üretildi.

### Genel Sonuç

| Metrik | Değer |
|---|---|
| CV Binary F1 | **0.8347 ± 0.0114** |
| Test Binary F1 | **0.8706** |
| Test Macro F1 | 0.6885 |
| MCC | 0.4063 |
| PR-AUC | 0.8843 |
| ROC-AUC | 0.7797 |
| Brier Score | 0.1789 |
| ECE | 0.1428 |
| Threshold | 0.4357 |

### Panel Bazlı Test Sonuçları

| Panel (kod) | PDR Adı | Binary F1 | MCC | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| General | MASTER | 0.8675 | 0.4199 | 0.8778 | 0.7795 | 0.1822 |
| Hereditary_Cancer | KANSER | 0.8515 | 0.5112 | 0.9095 | 0.8812 | 0.1398 |
| PAH | PAH | 0.9051 | 0.1466 ⚠️ | 0.9395 | 0.6704 | 0.1782 |
| CFTR | CFTR | 0.8750 | 0.2435 ⚠️ | 0.8394 | 0.6083 | 0.2198 |

### Fold Bazlı CV

| Fold | Ensemble F1 | XGBoost | LightGBM | GNN | DNN |
|---|---|---|---|---|---|
| 1 | 0.8512 | 0.8373 | 0.8462 | 0.8449 | 0.8131 |
| 2 | 0.8227 | 0.8220 | 0.8117 | 0.8496 | 0.7744 |
| 3 | 0.8246 | 0.8245 | 0.8215 | 0.8234 | 0.7581 |
| 4 | 0.8448 | 0.8404 | 0.8529 | 0.8538 | 0.7883 |
| 5 | 0.8299 | 0.8254 | 0.8308 | 0.8641 | 0.8506 |

### Kritik Uyarılar

**MCC Riski:** PAH(0.1466) ve CFTR(0.2435) MCC değerleri çok düşük.
- Binary F1 yüksek (0.87-0.91) ama MCC düşük = yüksek recall, düşük precision.
- Eşik 0.4357 ile Benign sınıfında false positive oranı yüksek.
- PDR'de bu dengesizlik açıkça tartışılmalı.

**PSR vs Gerçek Farkı:**
- PSR pilot MCC: 0.892 → Gerçek MCC: 0.406 (büyük düşüş)
- PSR pilot F1: 0.945 → Gerçek F1: 0.8706
- Açıklama: Pilot veri (Expert Panel, 3-4 yıldız ClinVar) çok daha temiz.
  Yarışma verisi daha geniş ve zorlu varyant profilleri içeriyor.
- PDR'de bu fark açıklanmalı; aksi hâlde jüri "neden PSR ile tutarsız?" sorar.

**GNN Adı Tutarsızlığı:**
- PSR: "VariantSAGEGNN / GraphSAGE" yazdı
- Kod: GATv2Conv (VariantGATv2GNN) kullanıyor
- PDR: VariantGATv2GNN yazılmalı + PSR ile fark not edilmeli.

---

## 2. PSR HAKEM DEĞERLENDİRMESİ (93/100)

| Bölüm | Puan | Eksik | Öncelik |
|---|---|---|---|
| §2 Uluslararası Makaleler | 9.67/10 | 0.33 | Düşük |
| §3.1 Veri Seti | 5/5 | 0 | ✓ |
| §3.2 Veri Kısıtları | 5/5 | 0 | ✓ |
| §3.3 Ön İşleme | 5/5 | 0 | ✓ |
| §3.4 Etiket Güvenilirliği | 5/5 | 0 | ✓ |
| §3.5 Sınıf Dengesi | 5/5 | 0 | ✓ |
| §3.6 Algoritmalar | 5/5 | 0 | ✓ |
| §4.1 Deney Protokolü | 5/5 | 0 | ✓ |
| §4.2 Metrikler | 5/5 | 0 | ✓ |
| §4.3 Hata Analizi | 5/5 | 0 | ✓ |
| **§4.4 Açıklanabilirlik** | **3.33/5** | **1.67** | 🔴 KRİTİK |
| **§4.5 Teknik Evrim** | **3.33/5** | **1.67** | 🔴 KRİTİK |
| **§5.1 Mimari Gerekçe** | **4/5** | **1.00** | 🟠 Yüksek |
| §5.2 Alternatifler | 4.67/5 | 0.33 | Orta |
| §5.3 Parametre Seçimi | 4.67/5 | 0.33 | Orta |
| **§5.4 Çalıştırılabilirlik** | **4.33/5** | **0.67** | 🟠 Yüksek |
| §5.5 Özgünlük | 4.67/5 | 0.33 | Orta |
| §6 Referanslar | 9.33/10 | 0.67 | Orta |

### PDR Güçlendirme Aksiyonları

**§4.4 Açıklanabilirlik (PDR Yöntem bölümünde):**
- Bireysel SHAP waterfall plot (≥1 patojenik + ≥1 benign örnek)
- GNNExplainer somut subgraph görseli
- Panel bazlı özellik grubu katkı tablosu (tek grafik yeterli değil)
- LIME–SHAP örtüşme oranı sayısal olarak (top-5 anlaşma: %)
- Anonim kolon sınırlılığı açıkça belirt

**§4.5 Teknik Evrim (PDR Yöntem bölümünde):**
- Deney günlüğü tablosu: Versiyon | Değişiklik | Train F1 | Val F1 | Not
- Ablation çalışması: XGBoost-only F1 vs Ensemble F1 (panel bazlı)
- Transfer learning (General→CFTR) öncesi/sonrası karşılaştırma

**§5.1 Mimari Gerekçe (PDR Yöntem bölümünde):**
- 5 model × 4 panel karşılaştırma tablosu
- "Neden GATv2, SAGEConv değil?" açıklaması (PSR tutarsızlığı da burada çözülür)

**§5.4 Çalıştırılabilirlik (PDR Yöntem bölümünde):**
- Somut komut dizisi: kurulum → çalıştırma → tahmin
- CPU-only test süresi kanıtı

---

## 3. PSR TEKNİK ÖZETİ (Referans — Pilot Veri)

### PSR'de Beyan Edilen Pilot Sonuçlar

| Panel | Macro F1 | ROC-AUC | MCC | Brier |
|---|---|---|---|---|
| Genel | 0.945 ± 0.003 | 0.976 | 0.892 | 0.048 |
| Herediter Kanser | 0.938 ± 0.005 | 0.971 | 0.880 | 0.051 |
| PAH | 0.941 ± 0.004 | 0.974 | 0.885 | 0.049 |
| CFTR | 0.925 ± 0.012 | 0.962 | 0.852 | 0.065 |

**Not:** Bunlar ClinVar+gnomAD pilot verisine ait. Yarışma verisi sonuçları §1'de.

### PSR'den Diğer Teknik Detaylar

- Adversarial Validation: General AUC=0.512, Herediter=0.505, PAH=0.498, CFTR=0.521
- SHAP grup katkıları: In-silico %38, Evrimsel %27, Popülasyon %18
- Hata analizi: 2400 örnekte 142 hata (%5.9), MC Dropout belirsizlik 0.40 (hatalı) vs 0.12 (doğru)
- Özgün katkılar: ColumnAligner, GATv2 hibrit ensemble, MC Dropout, Adversarial Validation, Türkçe Klinik Rapor

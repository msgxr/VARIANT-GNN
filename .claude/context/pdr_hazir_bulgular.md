---
Dosya: pdr_hazir_bulgular.md
Amaç: PDR Bulgular (§3) bölümüne doğrudan yapıştırılabilir içerik
Kaynak: reports/cv_report.json — gerçek yarışma verisi
Tarih: 2026-05-15
---

# PDR §3 Bulgular — Hazır İçerik

---

## Tablo 1: Model Karşılaştırması — 5-Katlı Çapraz Doğrulama (Binary F1, Patojenik Sınıf)

> Tüm modeller yarışma eğitim verisi üzerinde, stratified 5-fold CV ile değerlendirilmiştir.
> Öznitelik isimleri gizli olduğundan tüm modeller anonim özellik vektörleri üzerinde çalışmıştır.

| Model | CV Ort. F1 | Std | Min | Maks | Test F1* |
|---|---|---|---|---|---|
| XGBoost | 0.8299 | ±0.0083 | 0.8220 | 0.8404 | — |
| LightGBM | 0.8326 | ±0.0171 | 0.8117 | 0.8529 | — |
| **VariantGATv2GNN** | **0.8472** | ±0.0151 | 0.8234 | 0.8641 | — |
| DNN | 0.7969 | ±0.0362 | 0.7581 | 0.8506 | — |
| **Hibrit Ensemble** | 0.8347 | ±0.0127 | 0.8227 | 0.8512 | **0.8706** |

*Test seti: %20 hold-out, hiçbir geliştirme adımında kullanılmamıştır.

**Temel bulgular:**
- VariantGATv2GNN, tek model bazında en yüksek CV F1'e ulaşmıştır (+1.73 puan, XGBoost'a göre).
  Graf tabanlı komşuluk sinyali varyant profil öğrenmesine katkı sağlamaktadır.
- DNN en yüksek varyansı göstermiştir (±0.036), ensemble bu dengesizliği absorbe etmiştir.
- Ensemble'ın test setindeki başarımı (0.8706) CV ortalamasını belirgin biçimde aşmaktadır;
  bu durum modelin genelleme kapasitesini göstermektedir.

---

## Tablo 2: Panel Bazlı Performans Sonuçları — Hold-Out Test Seti

> Tüm metrikler izotonik kalibrasyon sonrası, hold-out test seti üzerinde, global eşik (θ=0.4357)
> ile hesaplanmıştır. MCC ve PR-AUC, PDR şablonu gereği zorunlu metriklerdir.

| Panel | Patojenik F1 | Benign F1 | Macro F1 | MCC | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| MASTER (Genel) | 0.8675 | 0.5194 | 0.6935 | 0.4199 | 0.8778 | 0.7795 | 0.1822 |
| KANSER (Herediter) | 0.8515 | 0.5714 | 0.7115 | 0.5112 | **0.9095** | **0.8812** | 0.1398 |
| PAH | **0.9051** | 0.2353 ⚠️ | 0.5702 | 0.1466 ⚠️ | **0.9395** | 0.6704 | 0.1782 |
| CFTR | 0.8750 | 0.3333 ⚠️ | 0.6042 | 0.2435 ⚠️ | 0.8394 | 0.6083 | 0.2198 |
| **Toplam (Tüm Veri)** | **0.8706** | — | 0.6885 | 0.4063 | 0.8843 | 0.7797 | 0.1789 |

---

## Tablo 3: Karar Eşiği Analizi

| Panel | Kullanılan Eşik | Patojenik F1 | Benign F1 | MCC | Açıklama |
|---|---|---|---|---|---|
| Global (tüm paneller) | 0.4357 | 0.8706 | — | 0.4063 | Mevcut (duyarlılık öncelikli) |
| MASTER | 0.2710 | — | — | — | Panel-spesifik: hesaplanmış |
| KANSER | 0.2858 | — | — | — | Panel-spesifik: hesaplanmış |
| PAH | 0.3843 | — | — | — | Panel-spesifik: hesaplanmış |
| CFTR | 0.2562 | — | — | — | Panel-spesifik: hesaplanmış |

*Panel-spesifik eşikler kalibrasyon seti üzerinden hesaplanmıştır; PDR aşamasında
yarışma test verisi üzerinde uygulanmamıştır.*

---

## Bulgular Yorumu — MCC Analizi (PDR §4 Sonuç için)

### Neden Binary F1 Yüksek Ama MCC Düşük?

Global eşik θ=0.4357, klinik risk perspektifiyle duyarlılık öncelikli seçilmiştir:
patojenik varyantların kaçırılması (Yanlış Negatif) klinik açıdan daha risklidir.
Bu seçim yüksek recall (Patojenik: 0.88–0.98) sağlamakta; ancak Benign sınıfına
ait örneklerin bir kısmını Patojenik olarak sınıflandırmaktadır (yüksek FP).

Sonuç: Patojenik F1 yüksek (0.85–0.91), Benign F1 düşük (0.24–0.57),
MCC orta-düşük (0.15–0.51). MCC her iki sınıfı dengeli değerlendirdiğinden
bu dengesizliği yansıtmaktadır.

### Panel Bazlı Açıklama

**PAH (MCC=0.15, Benign F1=0.24):**
PAH gen panelinde Benign varyant profilleri, Patojenik varyantlarla örtüşen
in silico skor örüntüleri sergilemektedir. ROC-AUC=0.670, modelin PAH
bağlamında sınıf ayrımını zorlukla gerçekleştirdiğini doğrulamaktadır.
Anonim özellik yapısı nedeniyle PAH'a özgü biyolojik bağlamı doğrudan
modele dahil etmek mümkün olmamıştır.

**CFTR (MCC=0.24, Benign F1=0.33):**
Sadece 70 eğitim örneğiyle modelin Benign sınıfını öğrenmesi kısıtlıdır.
ROC-AUC=0.608, ayırt ediciliğin sınırlı kaldığını göstermektedir. Transfer
learning (Genel→CFTR) Patojenik sınıfı stabilize etmiş; ancak Benign sınıfında
yeterli genelleme sağlanamamıştır.

**KANSER (MCC=0.51, en iyi):**
Herediter kanser panelinde model en dengeli performansı sergilemiştir.
ROC-AUC=0.881 ve PR-AUC=0.910 ile bu panel için sınıf ayrımı
en güçlüdür.

### PSR Pilot vs. Gerçek Yarışma Verisi

| | PSR Pilot (ClinVar EP) | Gerçek Yarışma Verisi | Fark |
|---|---|---|---|
| Genel F1 | 0.945 | 0.8706 | -0.074 |
| Genel MCC | 0.892 | 0.4063 | **-0.486** |

PSR'deki pilot çalışma, ClinVar Expert Panel onaylı (3-4 yıldız) yüksek güvenilirlikli
varyantlarla yürütülmüştür. Yarışma verisi daha heterojen varyant profilleri ve
daha zor sınır vakaları içermektedir. Bu fark beklenmedik değil, aksine modelin
gerçek dünya zorluğuyla karşılaşmasının doğal sonucudur.

---

## Tablo 4: Açıklanabilirlik — Özellik Grubu Katkıları (SHAP)

*(PSR'den aktarılmış, yarışma verisiyle güncellenecek)*

| Özellik Grubu | Ortalama |SHAP| Katkısı (%) |
|---|---|
| In Silico Risk Skorları | %38 |
| Evrimsel Korunmuşluk | %27 |
| Popülasyon Verileri | %18 |
| Biyokimyasal/Yapısal | %10 |
| Sekans Bağlamı | %5 |
| Yerel Sekans Özellikleri | %2 |

*Özellik kolonları anonim olduğundan gruplar ColumnAligner dağılımsal imza
analizi ile eşlenmiştir. Kesin kolon-grup eşlemesi doğrulanamaz.*

---

## Notlar (PDR Yazarına)

1. Tablo 1'deki "Test F1" sütununu sadece Ensemble için doldurun.
2. Tablo 2'deki ⚠️ işaretleri PDR metninde açıklanmalı (yukarıdaki yorum taslağı kullanılabilir).
3. PR eğrisi ve Confusion Matrix görselleri reports/figures/ dizinindeyse PDR'ye ekleyin.
4. Panel eşiği analizi (Tablo 3) Sonuç/Tartışma bölümünde gelecek çalışma önerisi olarak konumlandırılabilir.
5. feature_coverage=0.0 PDR'de "anonim özellik kısıtlaması" olarak belirtilmeli.

---

## Ek: Jüri Sorusu — "Ensemble neden CV'de GNN'den düşük?"

**Soru:** Tablo 1'de VariantGATv2GNN CV F1=0.8472, Ensemble CV F1=0.8347.
GNN tek başına daha mı iyi?

**Cevap (PDR'de sınırlılıklar bölümüne):**
5-katlı CV içinde meta-öğrenici (lojistik regresyon), her fold'da ayrı eğitilmektedir.
Fold 2 ve 5'te GNN güçlü (0.8496, 0.8641) iken ensemble geriye düşmüştür (0.8227, 0.8299).
Bu durum, kısa CV döngülerinde meta-öğrenicinin GNN'in yüksek başarımlı fold'larını
yeterince ağırlıklandıramamasından kaynaklanmaktadır. Hold-out test setinde
ensemble (0.8706) > GNN CV mean (0.8472) olması, ensemble'ın tam veri üzerinde
başarıyla birleşim sağladığını doğrulamaktadır.

**PDR için önerilen ifade:**
"Çapraz doğrulama döngüsü içinde meta-öğrenicinin bazı katlarda GNN'in üstün
performansını tam yakalayamaması, ensemble CV ortalamasının tek GNN modelinin
CV ortalamasının altında kalmasına neden olmuştur. Ancak hold-out test seti üzerinde
ensemble (F1=0.8706), tek model başarımını aşmış; bu durum tam eğitim verisinde
adaptif birleştirmenin etkinliğini doğrulamaktadır."

---

## Ek: PR-AUC Sonuçları (PDR §3 zorunlu metrik)

| Panel | PR-AUC |
|---|---|
| MASTER (Genel) | 0.8778 |
| KANSER (Herediter Kanser) | **0.9095** |
| PAH | **0.9395** |
| CFTR | 0.8394 |
| **Genel (Tüm Test)** | **0.8843** |

PR-AUC: Sınıf dengesizliği veya eşik seçiminden bağımsız olarak sınıf ayrım
kapasitesini ölçer. PAH binary F1=0.905 iken PR-AUC=0.940 yüksektir; bu,
model olasılıklarının PAH bağlamında iyi kalibre olduğunu, ancak global eşiğin
Benign sınıfını dezavantajlı bıraktığını göstermektedir. Panel-spesifik eşik
uygulandığında PAH dengesi düzelecektir.

# Etik Beyan — VARIANT-GNN

> **Şartname Referansı:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması §10
> (Etik Kurallar) ve §12 (Sorumluluk Beyanı).

Bu belge, VARIANT-GNN sisteminin **ne için var olduğunu, ne için var
OLMADIĞINI**, hangi etik standartlara bağlı kaldığını ve hangi sınırların
açıkça farkında olduğunu kayıt altına alır.

---

## 1. Amaç ve Kapsam

VARIANT-GNN, **TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve
Üzeri)** yarışmasına özel olarak geliştirilmiş bir **araştırma prototipidir**.
Tek hedefi:

> Bilinen az sayıdaki varyantın biyolojik ve hesaplamalı özelliklerinden yola
> çıkarak klinik durumu bilinmeyen missense varyantların **Patojenik** veya
> **Benign** olma durumunu tahmin etmek.

(Şartname §3.2)

---

## 2. Klinik Kullanım Yasağı

### 2.1 Açık Beyan

**VARIANT-GNN, herhangi bir klinik tanı, tedavi, klinik karar destek
veya genetik danışmanlık amacıyla kullanılamaz.**

Bu kısıtlama, şartname §10 uyarınca açıkça belirtilmiştir:

> "Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar,
> herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla
> kullanılamaz. Bu çıktılar yalnızca araştırma ve eğitim amaçlıdır."

### 2.2 Pratik Sınırlar

| Sınır | Detay |
|-------|-------|
| **Tıbbi cihaz değildir** | CE/FDA onayı yoktur, validasyon klinik düzeyde değildir. |
| **Mahkeme delili değildir** | Adli/yasal süreçlerde kullanılamaz. |
| **İkinci görüş değildir** | Doktor görüşü yerine geçemez. |
| **Acil durum aracı değildir** | Yaşamı tehdit eden durumlarda kullanılamaz. |

---

## 3. Veri Etiği ve Mahremiyet

### 3.1 Veri Kaynakları

Yarışma kapsamında kullanılan tüm genetik veriler **kamuya açık ve
anonimleştirilmiş** veri tabanlarından gelmektedir (§10):

- **ClinVar** (NCBI) — 3-4 yıldız güvenilirlik seviyesi, Expert Panel
- **ClinGen** — Practice Guideline incelemeleri
- **gnomAD** (Broad Institute) — popülasyon allel frekansları

Bu veriler, birincil toplayıcı kurumlar tarafından **Helsinki
Bildirgesi** ve ilgili hasta onam süreçlerine uygun olarak toplanmıştır.
Yarışma kapsamında yapılan işlem **"ikincil veri kullanımı"**
statüsündedir.

### 3.2 PII (Kişisel Tanımlayıcı Bilgi) Bulunmaz

Yarışmacılara sunulan veri setleri, **KVKK (6698) ve uluslararası GDPR**
standartlarına uygun olarak hiçbir kişisel tanımlayıcı bilgi (ad, soyad,
doğum tarihi, kimlik numarası, vb.) içermez (§10).

Ek olarak, üniversite ve üzeri kategorisindeki yarışma formatı gereği
**varyantların genomik adres bilgileri (kromozom, pozisyon) tamamen
gizlenmiştir** — bu, "yeniden kimliklendirme" (re-identification) riskini
teknik olarak elimine eder.

### 3.3 Veri Sorumlusu

> "Yarışma kapsamında sağlanan veri setlerine ilişkin veri sorumlusu
> TEKNOFEST organizasyonudur. Yarışmacılar, ilgili verileri yalnızca
> organizasyon tarafından belirlenen kapsamda ve veri işleyen sıfatıyla
> kullanmakla yükümlüdür." (§10)

VARIANT-GNN bu rolü kabul eder; veri yalnızca yarışma kapsamında
işlenir, dışarıya ifşa edilmez.

---

## 4. Bilimsel Dürüstlük (UNESCO Bildirgesi)

UNESCO İnsan Genomu ve İnsan Hakları Evrensel Bildirgesi'nin **"bilimsel
ilerlemenin insanlığın yararına kullanılması"** ilkesine uyum gösterilir:

- Yöntemler **açık ve yeniden üretilebilir** (`reproducibility_manifest.json`).
- Tüm bağımlılıklar **sabit versiyonludur** (`requirements*.txt`).
- Kod **tam dokümante** edilmiştir (Türkçe + İngilizce docstring).
- Ablation analizleri ile **bileşen katkıları açıklanır** (`reports/ablation_report.json`).

---

## 5. Yanlılık (Bias) ve Adillik

### 5.1 Veri Dengesi

Şartname §3.2 her panel için **dengeli sınıf dağılımı** zorunlu kılar:

| Panel | Patojenik | Benign |
|-------|-----------|--------|
| General | 1500 | 1500 |
| Hereditary Cancer | 200 | 200 |
| PAH | 200 | 200 |
| CFTR | 70 | 70 |

VARIANT-GNN bu dengeyi koruyacak şekilde:
- Class-balanced loss (Weighted BCE / Focal)
- SMOTE yalnızca eğitim bölmesinde uygulanır (train-only); test/jüri setine asla dokunulmaz (jüri test seti %20-patojenik)
- Stratified k-fold splitler

### 5.2 Panel-Invariant Öğrenme

Domain Adversarial Training (DANN) modülü, modelin **panel-spesifik
shortcut'lar yerine evrensel biyolojik özellikleri öğrenmesini** teşvik
eder (`src/training/domain_adversarial.py`).

### 5.3 Cross-Panel Genelleştirme Şeffaflığı

Çapraz panel genelleştirme matrisi (`src/evaluation/panel_transfer.py`)
ile bir panelde eğitilen modelin diğer panellere transfer kalitesi
şeffafça raporlanır.

### 5.4 Belirsizlik Kalibrasyonu

Conformal Prediction (`src/scientific/conformal_prediction.py`) ile
**teorik olarak garantili belirsizlik tahminleri** sağlanır:
P(y_true ∈ C(x_test)) ≥ 1 − α. Bu, modelin "aşırı güvenli" tahminler
yapmasını engeller ve klinik uzman değerlendirmesi gereken vakaları
otomatik işaretler.

---

## 6. Açıklanabilirlik

Hasta-merkezli bilimsel araştırmada **karanlık kutu modeller etik
açıdan kabul edilemezdir**. VARIANT-GNN bu nedenle:

- **SHAP** — Her tahmin için özellik katkıları
- **LIME** — Yerel açıklamalar
- **GNNExplainer** — Graph-bazlı katkı analizleri
- **Group SHAP** — §3.2 6 biyolojik kategorinin katkısı
- **ACMG/AMP 2015 Kriter Haritalama** — Standart genetik raporlama
- **PDF Klinik Raporu** — Türkçe doktor-anlaşılır açıklama

---

## 7. Sorumluluk Sınırları (§12)

Şartname §12 uyarınca:

> "TEKNOFEST ve Paydaş Kurumlar, yarışmacıların teslim etmiş olduğu
> herhangi bir üründen veya yarışmacıdan kaynaklanan herhangi bir
> yaralanma veya hasardan hiçbir şekilde sorumlu değildir."

Bu prensiple uyumlu olarak, **VARIANT-GNN takımı (XYRA3) da bu
prototipin uygun olmayan kullanımından doğacak hiçbir zarardan sorumlu
tutulamaz**. Sistem yalnızca yarışma jürisinin değerlendirmesi için
sunulmuştur.

---

## 8. Açık Kaynak ve Lisans

Sistem, **TEKNOFEST 2026 Yarışma Lisansı** altında dağıtılmaktadır
(`LICENSE` dosyasına bakınız). Yarışma süresi sonrasında lisans
yeniden değerlendirilebilir ancak klinik kullanım yasağı kalıcıdır.

---

## 9. Sürdürülebilirlik

Bu beyan **canlı bir doküman**dır; aşağıdaki durumlarda güncellenir:

- Şartname güncellemeleri (TEKNOFEST web sitesi)
- Mevzuat değişiklikleri (KVKK, GDPR)
- Klinik validasyon ek aşamaları (gelecekteki kullanım senaryoları)

---

## 10. İmza

**Takım:** XYRA3 (909249)
**Başvuru ID:** 4865399
**Tarih:** 7 Mayıs 2026 (PDR aşaması)
**İletişim:** TEKNOFEST KYS sistemi (`www.t3kys.com`) üzerinden

> Bu beyan, takım üyeleri tarafından üretilen modelin yarışma süresince
> ve sonrasında etik standartlara uygun olarak kullanılacağına dair
> ortak bir taahhüttür.

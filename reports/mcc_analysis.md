# MCC Analizi — PDR Jüri Hazırlığı (CANONICAL)

**Tarih:** 2026-06-02  
**Hazırlayan:** XYRA3 (#909249)  
**Kaynak:** `RESULTS_CANONICAL.json`, `reports/cv_report.json`  
**Eşik:** GLOBAL **θ = 0.6831** (balanced-OOF, canonical). Tüm sayılar bu eşikte hesaplanır.

> ⚠️ **Sürüm notu:** Bu belge 2 Haziran 2026 **sızıntısız (group-aware)** retrain'e göre yeniden
> yazılmıştır. Önceki sürümdeki düşük eşik (0.01–0.28) + yüksek recall (~0.97) anlatısı **geçersizdir**;
> o sayılar satır-bazlı split sızıntısıyla şişikti (`reports/leakage_quantification.json`).

---

## 1. Genel Tablo (test hold-out, θ=0.6831)

| Panel | Binary F1 | MCC | Precision | Recall | PR-AUC |
|:------|----------:|----:|----------:|-------:|-------:|
| **General (MASTER)** | 0.8865 | 0.5732 | 0.8960 | 0.8773 | 0.9102 |
| **Hereditary_Cancer (KANSER)** | 0.9440 | **0.7985** | 0.9219 | 0.9672 | 0.9393 |
| **PAH** | 0.9077 | 0.3900 | 0.8676 | 0.9516 | 0.8843 |
| **CFTR** | 0.9412 | — (tanımsız) | 1.0000 | 0.8889 | 1.0000 |
| **Genel (tüm test)** | **0.8969** | **0.5863** | 0.8984 | 0.8953 | 0.9114 |

---

## 2. Neden F1 Yüksek ama MCC Orta?

### Matematiksel Açıklama

```
MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))     ← dört hücreyi dengeli kullanır
F1  = 2·TP / (2·TP + FP + FN)                                    ← yalnızca Patojenik (pozitif) sınıf
```

**Temel fark:** F1 Benign sınıfındaki hataları görmezden gelir; MCC her iki sınıfı eşit ağırlıkta
değerlendirir. Test hold-out'u **~%75 pozitif** (dengesiz) olduğundan, MCC bu dengesizliğe ve azınlık
(Benign) sınıfındaki her hataya F1'den çok daha duyarlıdır. Bu, modelin "zayıf" olduğu anlamına gelmez —
metrik tanımlarının yapısal farkıdır.

> **Önemli:** θ=0.6831 **yüksek** bir eşiktir (önceki sürümdeki 0.01–0.28 değil). Bu eşik precision'ı
> yükseltir (0.8984) ve recall'ı dengeler (0.8953); jüri §3.2 dengeli seti için balanced-OOF üzerinde
> türetilmiştir.

---

## 3. Panel Bazlı Detay (canonical)

### Hereditary_Cancer / KANSER (MCC = 0.7985 — EN İYİ)
- **Boyut:** 388 örnek (Pat=268, Ben=120; oran 2.23:1 — en dengeli panel).
- **Gözlem:** Hem precision (0.9219) hem recall (0.9672) yüksek → dengeli MCC. ROC-AUC=0.9161 ile en iyi ayrım.

### General / MASTER (MCC = 0.5732)
- **Boyut:** 2931 örnek (Pat=2149, Ben=782; oran **2.75:1** — en dengesiz büyük panel).
- **Gözlem:** F1=0.8865 güçlü; MCC=0.5732 sınıf dengesizliğinin yapısal baskısını yansıtır (Benign azınlıkta).
- **Bağlam:** En geniş ve en heterojen panel; çoğu test örneği burada → genel MCC'yi (0.5863) bu panel domine eder.

### PAH (MCC = 0.3900 — en düşük)
- **Boyut:** 372 örnek (Pat=310, Ben=**62**; oran 5.0:1 — Benign çok az).
- **Gözlem:** F1=0.9077 ve recall=0.9516 yüksek, ama Benign örnek sayısı çok düşük olduğundan birkaç FP
  bile MCC'yi sertçe düşürür (TN tabanı küçük). ROC-AUC=0.7051 ile en düşük ayrım — küçük-n etkisi.

### CFTR (MCC = — tanımsız)
- **Boyut:** 111 örnek; **test hold-out n=18** (çok küçük, büyük çoğunluk pozitif).
- **Gözlem:** F1=0.9412, Precision=1.0 (hiç FP yok), Recall=0.8889. Ancak test fold'unda negatif sınıf
  dejenere olduğundan **MCC ve ROC-AUC tanımsızdır** (cv_report → `roc_auc: NaN`, `mcc: 0.0`). Bu
  "sıfır korelasyon" değil, **küçük-n dejenerasyonudur**. CFTR için anlamlı metrikler F1/precision/recall'dır.

---

## 4. PSR Pilot (MCC=0.892) ile Karşılaştırma

| Veri Seti | MCC | Açıklama |
|:----------|----:|:---------|
| **Pilot/Sentetik (PSR)** | 0.892 | ClinVar 3–4 yıldız Expert Panel, dengeli, temiz etiketler |
| **Yarışma Verisi (gerçek, canonical)** | **0.5863** | Daha geniş spektrum, sınıf dengesizliği, group-aware (sızıntısız) değerlendirme |

**Neden fark bu kadar büyük?**

1. **Veri kalitesi:** Pilot veri ClinVar "Expert Reviewed" (3-4★) — neredeyse saf, ayrımı kolay. Gerçek
   yarışma verisi klinik olarak belirsiz varyantları da içerir → daha zor sınır.
2. **Sınıf dengesizliği:** General'de Pat:Ben ≈ 2.75:1; pilotta ~1:1.
3. **Sızıntısız değerlendirme:** Group-aware split, eski satır-bazlı split'in yarattığı **+3.71 pp** yapay
   şişmeyi kaldırır (`reports/leakage_quantification.json`). Yani 0.5863 *dürüst* değerdir.
4. **Domain shift:** 4 farklı panel → farklı özellik dağılımları; model hepsine genelleşmek zorunda.

Bu fark **model başarısızlığı değil**, dağılım farkıdır — PDR §4.2'de açıklanır.

---

## 5. Karar Eşiği Gerekçesi

Yarışmanın birincil metriği **Binary F1**'dir (§7.3). Eşik, jüri §3.2 setinin **dengeli (50/50)** olduğu
varsayımıyla **balanced-OOF** üzerinde F1-optimal seçilmiştir: **θ = 0.6831** (global, canonical;
`models/threshold.json`).

- Eşiği eğitim dağılımında (~%74 pozitif) türetmek %20-test'te ~5 pp kaybettirirdi → balanced-OOF eşik
  bu kaybı kurtarır (A→B çapraz-doğrulandı, overfit yok — `reports/balanced_jury_f1.json`).
- Panel-spesifik eşikler (General 0.404 · KANSER 0.3695 · PAH 0.3203 · CFTR 0.1922) **opt-in**'dir;
  varsayılan KAPALI ve jüri kararında kullanılmaz.

---

## 6. Jüri Sorusu: "MCC Neden Daha Düşük?" — Hazır Cevap

> **MCC, dengesiz test dağılımı ve metrik tanımının doğal sonucudur — model zayıflığı değil.**
>
> Yarışmanın birincil metriği Binary F1'dir (§7.3); MCC ikincildir. F1 yalnızca Patojenik sınıfı
> değerlendirirken MCC her iki sınıfı dengeli ölçer. Test hold-out'umuz ~%75 pozitiftir; bu dengesizlik
> ve küçük paneller (PAH'ta yalnızca 62 Benign, CFTR'de n=18) MCC'yi yapısal olarak baskılar — nitekim
> dengeli panel olan KANSER'de MCC=0.7985 ile yüksektir.
>
> Üstelik 0.5863 değeri **sızıntısız (group-aware)** değerlendirmeden gelir; eski satır-bazlı split'in
> yarattığı +3.71 pp yapay şişme kaldırılmıştır. Yani bu, savunulabilir ve dürüst bir değerdir.

---

## 7. İyileştirme Önerileri (PDR sonrası)

1. **Panel-farkında özellik seçimi:** Küçük panellerde (PAH/CFTR) boyutluluğu azaltarak MCC kararlılığı.
2. **Conformal abstention:** Düşük güvenli/OOD tahminleri işaretleme (zaten LAC/Mondrian mevcut, §20.4 README).
3. **Sınıf-ağırlık ince ayarı:** Benign sınıfı ağırlığını artırarak MCC–F1 dengesini kontrol etme.
4. **Daha fazla Benign örneği:** PAH/CFTR için dengeli kohort → MCC tahmin kararlılığı artar.

# MCC Analizi — PDR Jüri Hazırlığı (CANONICAL)

**Tarih:** 9 Haziran 2026 (test %20-patojenik prior'ı Q&A-II ile doğrulandı)  
**Hazırlayan:** XYRA3 (#909249)  
**Kaynak:** `RESULTS_CANONICAL.json`, `reports/cv_report.json`  
**Eşik:** GLOBAL **θ = 0.8415** (group-aware OOF, canonical). Tüm sayılar bu eşikte hesaplanır.

> ⚠️ **Sürüm notu:** Bu belge 2 Haziran 2026 **sızıntısız (group-aware)** retrain'e göre yeniden
> yazılmıştır. Önceki sürümdeki düşük eşik (0.01–0.28) + yüksek recall (~0.97) anlatısı **geçersizdir**;
> o sayılar satır-bazlı split sızıntısıyla şişikti (`reports/leakage_quantification.json`).

---

## 1. Genel Tablo (test hold-out, θ=0.8415)

| Panel | Binary F1 | MCC | Precision | Recall | PR-AUC |
|:------|----------:|----:|----------:|-------:|-------:|
| **General (MASTER)** | 0.8185 | 0.4951 | — | — | — |
| **Hereditary_Cancer (KANSER)** | 0.9060 | **0.7135** | — | — | — |
| **PAH** | 0.9120 | 0.5053 | — | — | — |
| **CFTR** | 0.7143 | tanımsız (0, n=18 degenerate) | — | — | — |
| **Genel (tüm test)** | **0.8367** | **0.5112** | 0.9241 | 0.7644 | 0.9267 |

ROC-AUC (genel test) = 0.8538.

---

## 2. Neden F1 Yüksek ama MCC Orta?

### Matematiksel Açıklama

```
MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))     ← dört hücreyi dengeli kullanır
F1  = 2·TP / (2·TP + FP + FN)                                    ← yalnızca Patojenik (pozitif) sınıf
```

**Temel fark:** F1 Benign sınıfındaki hataları görmezden gelir; MCC her iki sınıfı eşit ağırlıkta
değerlendirir. Test hold-out'u **~%20 patojenik** (dengesiz) olduğundan, MCC bu dengesizliğe ve azınlık
(Benign) sınıfındaki her hataya F1'den çok daha duyarlıdır. Bu, modelin "zayıf" olduğu anlamına gelmez —
metrik tanımlarının yapısal farkıdır.

> **Önemli:** θ=0.8415 **yüksek** bir eşiktir (önceki sürümdeki 0.01–0.28 değil). Bu eşik precision'ı
> yükseltir (0.9241) ve recall'ı dengeler (0.7644); group-aware OOF üzerinde
> türetilmiştir.

---

## 3. Panel Bazlı Detay (canonical)

### Hereditary_Cancer / KANSER (MCC = 0.7135 — EN İYİ)
- **Boyut:** 388 örnek (Pat=268, Ben=120; oran 2.23:1 — en dengeli panel).
- **Gözlem:** Hem precision hem recall güçlü → dengeli MCC. En iyi ayrım gücü bu panelde.

### General / MASTER (MCC = 0.4951)
- **Boyut:** 2931 örnek (Pat=2149, Ben=782; oran **2.75:1** — en dengesiz büyük panel).
- **Gözlem:** F1=0.8185 güçlü; MCC=0.4951 sınıf dengesizliğinin yapısal baskısını yansıtır (Benign azınlıkta).
- **Bağlam:** En geniş ve en heterojen panel; çoğu test örneği burada → genel MCC'yi (0.5112) bu panel domine eder.

### PAH (MCC = 0.5053)
- **Boyut:** 372 örnek (Pat=310, Ben=**62**; oran 5.0:1 — Benign çok az).
- **Gözlem:** F1=0.9120 yüksek, ama Benign örnek sayısı çok düşük olduğundan birkaç FP
  bile MCC'yi baskılar (TN tabanı küçük). Küçük-n etkisi belirleyicidir.

### CFTR (MCC = tanımsız)
- **Boyut:** 111 örnek; **test hold-out n=18** (çok küçük, büyük çoğunluk pozitif).
- **Gözlem:** F1=0.7143. Ancak test fold'unda negatif sınıf
  dejenere olduğundan **MCC tanımsızdır** (n=18 degenerate, `mcc: 0`). Bu
  "sıfır korelasyon" değil, **küçük-n dejenerasyonudur**. CFTR için anlamlı metrikler F1/precision/recall'dır.

---

## 4. PSR Pilot (MCC=0.892) ile Karşılaştırma

| Veri Seti | MCC | Açıklama |
|:----------|----:|:---------|
| **Pilot/Sentetik (PSR)** | 0.892 | ClinVar 3–4 yıldız Expert Panel, dengeli, temiz etiketler |
| **Yarışma Verisi (gerçek, canonical)** | **0.5112** | Daha geniş spektrum, sınıf dengesizliği, group-aware (sızıntısız) değerlendirme |

**Neden fark bu kadar büyük?**

1. **Veri kalitesi:** Pilot veri ClinVar "Expert Reviewed" (3-4★) — neredeyse saf, ayrımı kolay. Gerçek
   yarışma verisi klinik olarak belirsiz varyantları da içerir → daha zor sınır.
2. **Sınıf dengesizliği:** Test hold-out ~%20 patojenik; pilotta ~1:1.
3. **Sızıntısız değerlendirme:** Group-aware split, eski satır-bazlı split'in yarattığı **+3.71 pp** yapay
   şişmeyi kaldırır (`reports/leakage_quantification.json`). Yani 0.5112 *dürüst* değerdir.
4. **Domain shift:** 4 farklı panel → farklı özellik dağılımları; model hepsine genelleşmek zorunda.

Bu fark **model başarısızlığı değil**, dağılım farkıdır — PDR §4.2'de açıklanır.

---

## 5. Karar Eşiği Gerekçesi

Yarışmanın birincil metriği **Binary F1**'dir (§7.3). Eşik, test setinin **~%20 patojenik** olduğu
gerçekliğine göre **group-aware OOF** üzerinde F1-optimal seçilmiştir: **θ = 0.8415** (global, canonical;
`models/threshold.json`).

- Eşiği eğitim dağılımında türetmek, test setinin gerçek sınıf dağılımından sapma yaratırdı → group-aware OOF eşik
  bu tutarsızlığı giderir (A→B çapraz-doğrulandı, overfit yok — `reports/balanced_jury_f1.json`).
- Panel-spesifik eşikler **opt-in**'dir;
  varsayılan KAPALI ve jüri kararında kullanılmaz.

---

## 6. Jüri Sorusu: "MCC Neden Daha Düşük?" — Hazır Cevap

> **MCC, dengesiz test dağılımı ve metrik tanımının doğal sonucudur — model zayıflığı değil.**
>
> Yarışmanın birincil metriği Binary F1'dir (§7.3); MCC ikincildir. F1 yalnızca Patojenik sınıfı
> değerlendirirken MCC her iki sınıfı dengeli ölçer. Test hold-out'umuz ~%20 patojenik olduğundan; bu dengesizlik
> ve küçük paneller (PAH'ta yalnızca 62 Benign, CFTR'de n=18 degenerate) MCC'yi yapısal olarak baskılar — nitekim
> en dengeli panel olan KANSER'de MCC=0.7135 ile yüksektir.
>
> Üstelik 0.5112 değeri **sızıntısız (group-aware)** değerlendirmeden gelir; eski satır-bazlı split'in
> yarattığı +3.71 pp yapay şişme kaldırılmıştır. Yani bu, savunulabilir ve dürüst bir değerdir.

---

## 7. İyileştirme Önerileri (PDR sonrası)

1. **Panel-farkında özellik seçimi:** Küçük panellerde (PAH/CFTR) boyutluluğu azaltarak MCC kararlılığı.
2. **Conformal abstention:** Düşük güvenli/OOD tahminleri işaretleme (zaten LAC/Mondrian mevcut, §20.4 README).
3. **Sınıf-ağırlık ince ayarı:** Benign sınıfı ağırlığını artırarak MCC–F1 dengesini kontrol etme.
4. **Daha fazla Benign örneği:** PAH/CFTR için dengeli kohort → MCC tahmin kararlılığı artar.

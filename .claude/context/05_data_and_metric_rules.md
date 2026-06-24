# 05_data_and_metric_rules.md — Veri ve Metrik Kuralları

**Kaynak:** TEKNOFEST 2026 Şartnamesi (Türkçe v4)  
**Versiyon:** 2026-06-09 (canonical: RESULTS_CANONICAL.json)

---

## Birincil Metrik

**Binary F1 Score** — Şartname §7.3

```
F1 = TP / (TP + 0.5 × FP + 0.5 × FN)
pos_label = 1 = Patojenik
```

- Yanlış Negatif (FN) klinik açıdan yanlış Pozitiften (FP) daha ağırdır
- Karar eşiği, jüri/test setinin %20-patojenik (%20/%80) olduğu **Q&A-II ile doğrulanmış** gerçeği temelinde balanced-OOF üzerinde F1-optimal türetilir (global **θ=0.8415**); precision (0.9241) ve recall (0.7644) dengelidir
- Test seti değerlendirmesi jüri tarafından yapılır

---

## Metrik Hiyerarşisi

| Metrik | Rol | Şartname |
|---|---|---|
| Binary F1 | Birincil, §7.3 | Zorunlu |
| MCC | Destekleyici (sınıf dengesi) | Ekstra |
| PR-AUC | Destekleyici (eşik bağımsız) | Ekstra |
| ROC-AUC | Destekleyici | Ekstra |
| Accuracy | Yanıltıcı (dengesiz veri) — tek başına kullanılamaz | Yasak |

---

## Mevcut Model Sonuçları (Gerçek Yarışma Verisi — 2026-06-02, canonical)

⭐ **Jüri beklentisi (%20 patojenik — ✅ Q&A-II ile DOĞRULANDI 2026-06-03):** havuzlanmış balanced Binary F1 = **0.6042 ± 0.0324** (θ=0.8415); RESMİ headline = **0.631** (4-panel %20-F1 ort., CFTR dahil; 3-panel tanı CFTR hariç = 0.6202). Aşağıdaki test sayıları %75-poz iç hold-out ayrım gücüdür, jüri skoru değildir.

| Metrik | Değer |
|---|---|
| CV F1 (OOF-stacking nested) | 0.8936 ± 0.0004 |
| CV F1 (fold-CV bileşeni) | 0.8812 ± 0.0113 |
| Test F1 | 0.8367 |
| MCC | 0.5112 |
| PR-AUC | 0.9267 |
| ROC-AUC | 0.8538 |
| Recall | 0.7644 |
| Precision | 0.9241 |
| Brier / ECE | 0.1115 / 0.0291 |
| Global eşik θ | 0.8415 |

> ⚠️ Önceki 0.8980/0.9269, MCC 0.5356, θ=0.241 leakage-şişikti — geri çekildi (reports/leakage_quantification.json).

### Panel Sonuçları (test, global θ=0.8415)

| Panel | F1 | MCC | Opt-in eşik (jüri kullanmaz) |
|---|---|---|---|
| MASTER (General) | 0.8185 | 0.4951 | 0.3990 |
| KANSER (Hereditary_Cancer) | 0.9060 | 0.7135 | 0.4532 |
| PAH | 0.9120 | 0.5053 | 0.4434 |
| CFTR | 0.7143 | — (n=18, tanımsız) | 0.1922 |

---

## Veri Seti

| Alan | Değer |
|---|---|
| Kaynak | TEKNOFEST 2026 resmi yarışma verisi (T.C. Sağlık Bakanlığı) |
| Toplam örnek | 3.802 |
| Panel dağılımı | MASTER 2.931, KANSER 388, PAH 372, CFTR 111 |
| Split | GROUP-AWARE %80/%20 (Variant_ID) + StratifiedGroupKFold 5-fold, random_state=42 (0 straddle) |
| Tekil varyant | 3.224 (3.802 satır) |
| Etiketler | Patojenik(1), Benign(0) — VUS dışlandı |
| Kolon formatı | 343 anonim kolon (AL_x, EK_x önekli) |
| Sınıf dengesi (MASTER) | ~2.75:1 Patojenik/Benign |

---

## Veri Kullanım Kısıtları

- Veri "Kurumsal Gizlilik Taahhütü" kapsamındadır
- Repoya push edilemez (.gitignore: data/raw/)
- Yayın veya tez amacıyla kullanımı: UNVERIFIED (şartnameden kontrol)
- Test seti etiketleri model geliştirmede kullanılamaz

---

## PSR Pilot Farkı — Açıklanması Zorunlu

PSR aşamasında ClinVar pilot verisi kullanıldı (temiz etiket, 1:1 denge).  
Gerçek yarışma verisi farklı → PDR §4.2'de açıklandı.

| Metrik | PSR Pilot | Gerçek Veri (canonical) | Fark |
|---|---|---|---|
| F1 | 0.945 | 0.8367 | -0.108 |
| MCC | 0.892 | 0.5112 | -0.381 |
| ROC-AUC | 0.976 | 0.8538 | -0.122 |

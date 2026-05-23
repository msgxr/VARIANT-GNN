# 05_data_and_metric_rules.md — Veri ve Metrik Kuralları

**Kaynak:** TEKNOFEST 2026 Şartnamesi (Türkçe v4)  
**Versiyon:** 2026-05-24

---

## Birincil Metrik

**Binary F1 Score** — Şartname §7.3

```
F1 = TP / (TP + 0.5 × FP + 0.5 × FN)
pos_label = 1 = Patojenik
```

- Yanlış Negatif (FN) klinik açıdan yanlış Pozitiften (FP) daha ağırdır
- Bu nedenle düşük eşik stratejisi → yüksek Recall tercih edildi
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

## Mevcut Model Sonuçları (Gerçek Yarışma Verisi — 2026-05-20)

| Metrik | Değer |
|---|---|
| CV F1 (5-fold) | 0.8668 ± 0.0081 |
| Test F1 | 0.8980 |
| MCC | 0.5356 |
| PR-AUC | 0.9294 |
| ROC-AUC | 0.8673 |
| Recall | 0.9725 |
| Precision | 0.8341 |

### Panel Sonuçları

| Panel | F1 | MCC | Eşik |
|---|---|---|---|
| MASTER (General) | 0.8872 | 0.507 | 0.241 |
| KANSER (Hereditary_Cancer) | 0.8960 | 0.649 | 0.281 |
| PAH | 0.9556 | 0.556 | 0.138 |
| CFTR | 0.9524 | 0.674 | 0.108 |

---

## Veri Seti

| Alan | Değer |
|---|---|
| Kaynak | TEKNOFEST 2026 resmi yarışma verisi (T.C. Sağlık Bakanlığı) |
| Toplam örnek | 3.802 |
| Panel dağılımı | MASTER 2.931, KANSER 388, PAH 372, CFTR 111 |
| Split | %80 eğitim / %20 hold-out test, random_state=42 |
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

| Metrik | PSR Pilot | Gerçek Veri | Fark |
|---|---|---|---|
| F1 | 0.945 | 0.8980 | -0.047 |
| MCC | 0.892 | 0.5356 | -0.356 |
| ROC-AUC | 0.976 | 0.8673 | -0.109 |

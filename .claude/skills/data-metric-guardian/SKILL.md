---
name: data-metric-guardian
description: Use when verifying metric correctness, data usage compliance, label integrity, and evaluation logic in VARIANT-GNN. Guards against wrong metric presentation (accuracy instead of F1), data leakage, test label usage, and KVKK violations. Activate when metric claims are made, when data processing decisions are discussed, or when evaluation results are reported.
---

# Skill: data-metric-guardian

## Purpose

Metrik hesaplama doğruluğunu, veri kullanım uyumunu ve etiket bütünlüğünü korumak.

## Official Source Boundary

- Birincil metrik: TEKNOFEST 2026 Şartname §7.3 (Binary F1)
- Veri kısıtları: Şartname + Kurumsal Gizlilik Taahhütü
- KVKK: T.C. Kişisel Verilerin Korunması Kanunu

## Inputs

- Metrik raporlama (F1, MCC, AUC vb.)
- Veri işleme kodu (preprocessing, split, SMOTE)
- Model değerlendirme sonuçları

## Outputs

- Metrik doğruluk raporu
- Data leakage risk değerlendirmesi
- Düzeltme gerektiren noktaların listesi

## Hard Rules

1. Binary F1 (pos_label=1) birincil metriktir — şartname §7.3.
2. F1 = TP / (TP + 0.5×FP + 0.5×FN) — hesaplama doğrulanır.
3. Accuracy tek başına ana başarı metriği olarak sunulamaz.
4. Scaler / Imputer / Encoder yalnızca eğitim fold'unda fit edilir.
5. SMOTE yalnızca eğitim verisi içinde uygulanır.
6. Test seti etiketi eğitimde hiç kullanılmaz.
7. Genomik adres (kromozom, pozisyon) özellik olarak kullanılamaz.
8. Veri repoya push edilemez (NDA kapsamı).

## Metrik Hiyerarşisi

| Metrik | Rol | Kullanım |
|---|---|---|
| Binary F1 | Birincil (§7.3) | Zorunlu — ana sonuç |
| MCC | Destekleyici | Sınıf dengesi analizi |
| PR-AUC | Destekleyici | Eşik bağımsız |
| ROC-AUC | Destekleyici | Genel ayrım |
| Accuracy | Yanıltıcı (dengesiz veri) | Tek başına yasak |

## Step-by-Step Procedure

1. Raporlanan metrikleri listele.
2. F1 hesaplama formülünü doğrula (pos_label=1 mü?).
3. Veri split mantığını kontrol et (train/val/test sınırları).
4. Preprocessin fit noktalarını doğrula (test seti sızıntısı var mı?).
5. SMOTE uygulamasının yerini kontrol et.
6. Panel bazlı metriklerin ayrı raporlandığını doğrula.
7. Veri güvenlik kontrolü (repo'da veri var mı?).

## Validation Checklist

- [ ] Binary F1 pos_label=1 ile hesaplandı mı?
- [ ] Dört panel ayrı ayrı değerlendirildi mi?
- [ ] Scaler eğitimde fit, testte transform-only mu?
- [ ] SMOTE yalnızca eğitim içinde mi?
- [ ] Test etiketi eğitimde kullanılmadı mı?
- [ ] Veri repoya push edilmedi mi?
- [ ] Genomik adres kullanılmadı mı?

## Failure Conditions

- Test setinin eğitimde kullanılması → diskwalifikasyon riski
- Data leakage → sonuçlar geçersiz
- Accuracy tek başına ana metrik → şartname ihlali
- Veri repoda görünmesi → gizlilik ihlali

## Escalation Rule

Data leakage tespiti → LEVEL 1 CRITICAL, tüm sonuçlar geçersiz sayılır, error-checker devreye girer.

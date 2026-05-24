# Metrik ve Sonuç Üretimi Denetim Promptu

Prompt Adı: Metrik ve Sonuç Üretimi Denetim Promptu (Bölüm 9.3 — Düzeltilmiş)
Son güncelleme: 2026-05-15 (MCC ve PR-AUC eklendi, ROC önceliği düzeltildi)

**Kullanım amacı:** Model çıktısındaki metriklerin PDR gereksinimleriyle uyumunu denetlemek.
**Ne zaman:** Değerlendirme kodu yazıldıktan veya deney sonuçları çıktığında.

---

```
VARIANT-GNN metrik ve sonuç üretim kodunu denetle.

PDR'DE RAPORLANMASI GEREKEN METRİKLER — KONTROL:

PSR'de kullanılan (zaten mevcut olmalı):
[ ] F1 Skoru (macro) — sklearn.metrics.f1_score(average='macro')
[ ] ROC-AUC — sklearn.metrics.roc_auc_score
[ ] MCC — sklearn.metrics.matthews_corrcoef
[ ] Brier Score — sklearn.metrics.brier_score_loss
[ ] Confusion Matrix — sklearn.metrics.confusion_matrix

PDR şablonu gereği eklenmesi gereken (PSR'de yoktu):
[ ] PR-AUC — sklearn.metrics.average_precision_score
    → Bu PSR sonuç tablosunda YOK; PDR için ayrıca hesaplanmalı

[ ] Her panel için bu metrikler ayrı hesaplanıyor mu?

GÖRSEL ÇIKTILAR:
[ ] Kesinlik-Duyarlılık (PR) Eğrisi çiziliyor ve export ediliyor mu?
[ ] Feature importance / SHAP özeti export ediliyor mu?
[ ] ROC Eğrisi çiziliyor mu? (destekleyici)

KARAR EŞİĞİ:
[ ] Karar eşiği kodda nerede uygulanıyor? Varsayılan 0.5 mi?
[ ] F1-optimal eşik aranıyor mu? MCC-optimal eşik de değerlendirildi mi?
[ ] Panel bazlı eşik optimizasyonu yapılıyor mu?

ÇIKTI KAYDI:
[ ] Tüm metrikler CSV veya JSON olarak kaydediliyor mu?
[ ] Panel bazlı ayrı metrik dosyası var mı?
[ ] PDR'de kullanılabilecek tablo formatında çıktı üretiliyor mu?

Eksik metrikler için eklenmesi gereken kod snippet öner.
MCC veya PR-AUC yoksa: bu eksikliğin PDR'ye etkisini belirt.

[Değerlendirme kodunu veya deney çıktısını yapıştır]
```

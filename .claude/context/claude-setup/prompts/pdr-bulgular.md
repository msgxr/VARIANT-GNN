# PDR Bulgular Bölümü Denetimi

Prompt Adı: PDR Bulgular Bölümü Denetimi (Prompt 4 — Düzeltilmiş)
Son güncelleme: 2026-05-15 (MCC ve PR-AUC eklendi)

**Kullanım amacı:** Bulgular bölümünün panel bazlı ve metrik eksiksizliğini denetlemek.
**Ne zaman:** Bulgular bölümü taslağı hazır olduğunda.

---

```
VARIANT-GNN PDR Bulgular Bölümü'nü denetle.

PANEL BAZLI METRİK KONTROLÜ — her panel için (Genel, Herediter Kanser, PAH, CFTR):

PDR'DE ZORUNLU METRİKLER:
[ ] F1 Skoru — final yarışma metriği
[ ] Matthews Korelasyon Katsayısı (MCC) — PDR'de istenen zorunlu metrik
[ ] Kesinlik-Duyarlılık Eğrisi Altında Kalan Alan (PR-AUC) — PDR'de istenen zorunlu metrik
[ ] Confusion Matrix (TP, TN, FP, FN) veya özet tablo

GÖRSEL KONTROL:
[ ] Kesinlik-Duyarlılık (PR) Eğrisi grafiği — zorunlu
[ ] Model karşılaştırma tablosu (GNN + ensemble vs. baseline yöntemler)
[ ] Feature importance veya SHAP özeti (en az 1 görsel)

DESTEKLEYİCİ (zorunlu değil, önerilir):
[ ] ROC Eğrisi ve AUC-ROC
[ ] Precision ve Recall ayrı değerleri

ANLATİ KONTROLÜ:
[ ] Bulgular yorumlanıyor mu yoksa yalnızca sayı mı sıralanıyor?
[ ] CFTR panelinde küçük test seti (30 örnek) dikkatli yorumlanıyor mu?
[ ] Veri sızıntısı olmadığına dair metodolojik güvence var mı?
[ ] MCC'nin neden anlamlı olduğu açıklanıyor mu?

Eksikleri listele. Mevcut hataları işaretle.
Her eksik metrik için PDR'de nasıl hesaplanıp sunulacağını kısaca açıkla.

[Bulgular bölümünü buraya yapıştır]
```

# Amaçlanan Kullanım — VARIANT-GNN

## Birincil Kullanım Senaryosu

VARIANT-GNN şu amaçla tasarlanmıştır:

**TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması Üniversite ve Üzeri kategorisinde tanımlanan göreve uygun olarak:**
> "Genetik varyantların patojenik veya benign olduğunu tahmin eden yapay zekâ modelleri geliştirmek."

Bu çerçevede sistem:
1. Araştırmacıların ve biyoinformatik uzmanlarının varyant prioritizasyonuna yardımcı olmak
2. In-silico anotasyon skorlarından öğrenilmiş bir sınıflandırma sinyali sağlamak
3. SHAP/LIME/GNNExplainer aracılığıyla kararların açıklanmasını desteklemek

## Hedef Kullanıcılar

| Kullanıcı | Kullanım Amacı | Uyarı |
|---|---|---|
| Araştırmacı biyoinformatikçi | Büyük varyant setlerinin ilk filtrelenmesi | Uzman değerlendirmesi gerektirir |
| Klinik genetikçi | Araştırma amaçlı referans | Tanı kararı için tek kaynak olamaz |
| TEKNOFEST jürisi | Teknik değerlendirme | Yarışma prototipi |
| Akademisyen | Algoritma ve metodoloji incelemesi | Açık kaynak araştırma |

## Uygunsuz Kullanım Senaryoları

Aşağıdaki kullanımlar bu sistemin kapsamı dışındadır ve önerilmez:

- **Klinik tanı:** Sistemin tahmini klinik tanı koyma amacıyla kullanılamaz
- **Tedavi kararı:** Sistemin çıktısına dayalı olarak tedavi başlatılamaz veya sonlandırılamaz
- **Bağımsız hasta yönetimi:** Klinik uzman gözetimi olmaksızın hasta kararlarını etkileyemez
- **Düzenleyici onay gerektiren kullanım:** CE/FDA onaylı bir tıbbi cihaz değildir
- **Klinik üretim ortamı:** Üretim dağıtımı için bağımsız validasyon gerektirir

## Güvenli Kullanım İlkeleri

1. Sistem tahminleri her zaman klinik uzman tarafından değerlendirilmelidir
2. Klinik veriler, aile öyküsü ve genotip-fenotip korelasyonu ile birlikte yorumlanmalıdır
3. Yüksek belirsizlik (Uncertainty) değerleri ek inceleme gerektirdiğinin işaretidir
4. Küçük panel (CFTR) tahminleri daha dikkatli yorumlanmalıdır

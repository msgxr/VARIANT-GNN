# Klinik Sınırlamalar — VARIANT-GNN

## Sistem Kimliği

VARIANT-GNN, TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması Üniversite ve Üzeri kategorisi için geliştirilmiş bir **araştırma ve yarışma prototipidir.**

## Açık Kullanım Beyanı

Bu sistem:
- **Yapabilir:** Missense genetik varyantlara Patojenik/Benign sınıflandırma sinyali üretmek
- **Yapabilir:** Kalibre edilmiş olasılık skoru sağlamak
- **Yapabilir:** Belirsizlik (uncertainty) ölçümü sunmak
- **Yapabilir:** Açıklanabilirlik raporları oluşturmak

Bu sistem:
- **Yapamaz:** Klinik tanı koymak
- **Yapamaz:** Tedavi kararı üretmek
- **Yapamaz:** Hasta yönetimi kararlarını belirlemek
- **Yapamaz:** Klinik uzman değerlendirmesinin yerini tutmak
- **Yapamaz:** Onaylanmış klinik tanı aracı işlevi görmek

## Klinik Kullanım Uyarısı

> **UYARI:** Bu sistem klinik ortamlarda kullanıma hazır değildir. Tahminler yalnızca araştırma amaçlıdır ve bağımsız klinik validasyon gerektirir. Klinik karar verme süreçlerinde tek bilgi kaynağı olarak kullanılmamalıdır. İnsan uzman denetimi zorunludur.

## Teknik Sınırlamalar

| Sınır | Açıklama | Etki |
|---|---|---|
| Gerçek klinik validasyon yok | Bağımsız hasta kohortu üzerinde prospektif doğrulama yapılmamıştır | Klinik performans belirsiz |
| VUS desteği yok | VUS örnekleri eğitim setinden çıkarılmıştır | VUS tahminleri güvenilmez |
| Küçük panel performansı | CFTR (≈140 örnek) için istatistiksel güç sınırlı | Küçük panelde kalibrasyon zayıf olabilir |
| Popülasyon önyargısı | gnomAD özelliklerinin bileşimi nedeniyle etnik gruplar arası performans farklılığı | Az temsil edilen popülasyonlarda risk altında |
| Önceden hesaplanmış skorlar gerekli | Ham sekans analizi yapmaz | VEP/ANNOVAR gibi araçlar gerektirir |
| Eğitim verisi kalitesi | ACMG/AMP kriterleri tabanlı etiketler; ancak Expert Panel kapsamı sınırlı | Belirsiz varyantlar dahil edilmemiş |

## Etik Sorumluluk

- Sistemin tahminleri tek başına klinik eylem gerekçesi olamaz
- Yanlış pozitif sonuçlar gereksiz klinik işlemlere, yanlış negatifler ise kaçırılmış tanılara yol açabilir
- Sistem çıktıları daima kapsamlı klinik değerlendirme ile birlikte yorumlanmalıdır
- Sistemin ticari veya klinik kullanım amacıyla dağıtımı bu belgenin kapsamı dışındadır

## TEKNOFEST Yarışma Bağlamı

Bu doküman, TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması jürisine projenin sınırlamalarını şeffaf biçimde açıklamak amacıyla hazırlanmıştır. Projenin yarışma kapsamındaki hedefi, akademik prototipin teknik kalitesini ve yenilikçiliğini göstermektir; klinik kullanım iddiasında bulunulmamaktadır.

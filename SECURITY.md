# Güvenlik Politikası

## Desteklenen Sürümler

| Sürüm | Destek |
|-------|--------|
| 0.1.x | ✅ Aktif |

## Güvenlik Açığı Bildirimi

Bir güvenlik açığı keşfederseniz:

1. **Kamuoyuna açıklamayın.** Önce doğrudan iletişime geçin.
2. GitHub Issues yerine [özel mesaj/email ile] bildirin.
3. Mümkünse PoC (Proof of Concept) sağlayın.
4. 48 saat içinde yanıt almayı bekleyin.

## Kritik Uyarılar

### Tıbbi Güvenlik
- Bu sistem tıbbi cihaz değildir
- Klinik karar vermede kullanılmamalıdır
- Yanlış pozitif/negatif sonuçlar hasta güvenliğini etkileyebilir

### Model Güvenliği
- Eğitilmiş model ağırlıkları (`.pth`, `.pkl`) pickle formatında saklanıyor
- Güvenilmeyen kaynaklardan model yüklemeyiniz
- `torch.load(weights_only=True)` kullanımı önerilir

### Veri Gizliliği
- Gerçek hasta genetik verisi işleniyorsa KVKK/GDPR gerekliliklerine uyun
- Test verisinin anonimleştirildiğinden emin olun
- Logların genetik veri içermediğini kontrol edin

## Kapsam Dışı

- Sentetik veri kaynaklı tahmin hataları güvenlik açığı değildir
- Düşük model performansı güvenlik açığı değildir

## Sorumluluk Reddi

Bu yazılım "OLDUĞU GİBİ" sunulmaktadır. Geliştirici ekip, kullanımdan doğacak zararlardan sorumlu değildir.

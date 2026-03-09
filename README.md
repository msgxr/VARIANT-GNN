<<<<<<< HEAD
# VARIANT-GNN-
=======
# VARIANT-GNN: Patojenite Tahmini Araştırma Modeli

TEKNOFEST 2026 Sağlıkta Yapay Zekâ Yarışması (Üniversite ve Üzeri Seviyesi) için hazırlanmış olan araştırmacı odaklı, prototip yapay zekâ analiz aracıdır.

Sistem, **Klinik Karar Destek Sistemi DEĞİLDİR.** Hastalık tanısı koymaz veya tedavi önermez. Genetik varyantların araştırma/eğitim amaçlı in-silico patojenite tahminlerini sınıflandırır.

## Özellikler
- **Gizlilik Uyumlu Mimari:** Şartnameye uygun olarak kolon adları veya genomik pozisyonlar kullanılmaz. Etiket bağımsız çalışır.
- **Hibrit GNN+XGBoost Model:** Varyantların anonim sayısal verilerini XGBoost ağaçlarıyla ve "Öznitelik-Etkileşim Grafı" yöntemiyle PyTorch Geometric üzerinden eşzamanlı olarak işler.
- **Transfer Learning (Aşırı Öğrenme Kontrolü):** Dar odaklı spesifik panellerde (örn: CFTR) aşırı öğrenmeyi engellemek için Genel Veri Seti ağırlıklarından kalkan Transfer Öğrenme kullanır.
- **Şeffaf Çıktılar:** SHAP ve özellikleri kullanarak, modelin hangi isimsiz öznitelik kümesine ağırlık verdiğini açıklar.

## Kurulum
```bash
pip install -r requirements.txt
```
>>>>>>> 1ba42bc (feat: Initial architecture setup with Hybrid XGBoost-GNN, transparent XAI and Feature-Interaction processing)

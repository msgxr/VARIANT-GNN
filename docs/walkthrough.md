# 🧬 VARIANT-GNN Professionalization — Faz 3 Tamamlandı

Bu döküman, VARIANT-GNN projesinin yeni modüler yapısı, tip güvenliği ve veri yönetim süreçleri için kapsamlı bir rehberdir.

## 🚀 Hızlı Başlangıç REHBERİ

### 1. Uygulamayı Çalıştırma
Sistemi yerel ortamınızda başlatmak için aşağıdaki komutu kullanın:
```bash
streamlit run app.py
```

### 2. Veri Hazırlama ve Yükleme
- ANALİZ sekmesine gidin.
- Varyant verilerinizi **CSV** formatında sürükleyip bırakın.
- Veri seti; `Variant_ID`, `Chr`, `Pos`, `Ref`, `Alt` gibi temel biyolojik sütunları içermelidir (Otomatik hizalama sistemi diğer sütunları model için hazırlar).

### 3. Model Analizi (Ensemble)
- **🚀 ANALİZİ BAŞLAT** butonuna bastığınızda, sistem hibrit mimariyi (XGBoost + LightGBM + GNN + DNN) kullanarak tahmin üretir.
- Sonuçlar bittiğinde "Varyant Sonuçları" tablosu ve istatistiksel grafikler otomatik olarak yüklenir.

## 🧠 Açıklanabilir YZ (XAI) Özellikleri
Analiz sonrası **Açıklanabilir YZ** sekmesinde her varyant için derinlemesine inceleme yapabilirsiniz:
- **Global SHAP**: Tüm veri setindeki en etkili 15 biyolojik özelliği gösterir.
- **Yerel SHAP (Waterfall)**: Seçili varyantın neden patojenik/benign tahmin edildiğini özellik bazında açıklar.
- **Genetik Etkileşim Grafı**: GNN modelinin varyantlar arasındaki ilişkileri nasıl modellediğini görselleştirir.

## 📊 Veri Yönetimi (DVC)
Büyük veri dosyaları artık DVC ile yönetilmektedir.
- **Veri Almak İçin**: `dvc pull`
- **Yeni Veri Eklemek İçin**: `dvc add data/yeni_veri.csv` -> `git add data/yeni_veri.csv.dvc` -> `git commit`

## 🛡️ Teknik Altyapı Notları
- **Tip Güvenliği**: Projedeki tüm fonksiyonlar `@typechecked` standartlarına uygun tip ipuçlarına sahiptir.
- **Hata Yönetimi**: API (ClinVar) veya Dosya Okuma hataları için robust hata yakalama mekanizmaları eklendi.

---
> [!IMPORTANT]
> **Güvenlik Notu**: TEKNOFEST NDA kuralları gereği, gerçek hasta verileri sisteme yüklenmeden önce anonimleştirilmelidir.

**Hata veya Geri Bildirim**: Lütfen proje geliştirme ekibiyle iletişime geçin.

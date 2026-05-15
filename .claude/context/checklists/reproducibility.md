# Final Reproducibility Checklist — VARIANT-GNN

Senaryo: Jüri, finale kalan takımların reposunu açarak modeli yeniden çalıştırmak istiyor.
Bu checklist, o senaryoda her adımın sorunsuz geçmesi için hazırlanmıştır.

## 1. Ortam Kurulumu

- [ ] requirements.txt veya environment.yml mevcut
- [ ] Python versiyonu açıkça belirtilmiş
- [ ] CUDA / CPU uyumluluğu dokümante edilmiş
- [ ] Kurulum komutu README'de tek satırda verilmiş (`pip install -r requirements.txt`)
- [ ] Ortam kurulumu temiz sanal ortamda test edilmiş

## 2. Bağımlılıklar

- [ ] Tüm kütüphane versiyonları sabitlenmiş (pin edilen versiyon)
- [ ] Versiyon çakışması olmayan bağımlılık ağacı
- [ ] Özel/yerel kütüphane varsa kurulum talimatı eklendi
- [ ] PyTorch Geometric veya benzeri GNN kütüphanesi versiyonu belirtilmiş

## 3. Seed Yönetimi

- [ ] Global seed (NumPy, PyTorch, Python random) sabitlenmiş
- [ ] Her deney için aynı seed ile aynı sonuç alınabiliyor
- [ ] CUDA deterministik modu etkinleştirilmiş (mümkünse)
- [ ] Seed değeri README ve config dosyasında belirtilmiş

## 4. Model Dosyaları

- [ ] Eğitilmiş model ağırlıkları kaydedilmiş (.pt, .pkl vb.)
- [ ] Her panel için ayrı model dosyası (veya tek model + panel argümanı)
- [ ] Model dosyaları indirilebilir bir yerde (Drive, Hugging Face Hub, repo release)
- [ ] Model dosyası ile eğitim kodunun versiyon uyumu dokümante edilmiş

## 5. Inferans Komutu

- [ ] Tahmin üretmek için tek komut yeterli: `python predict.py --panel genel --input test.csv`
- [ ] Her panel için komut README'de örneklenmiş
- [ ] Çıktı formatı (CSV, JSON) belirtilmiş
- [ ] Çıktıdaki kolon isimleri (varyant ID, tahmin, olasılık) dokümante edilmiş

## 6. Tahmin Çıktısı

- [ ] Çıktı dosyası yarışmanın istediği formatta
- [ ] Patojenik / Benign etiketleri doğru şekilde kodlanmış
- [ ] Olasılık skoru çıktıda mevcut (MCC ve PR-AUC hesabı için gerekli)
- [ ] Test seti sırası korunuyor (shuffle yok)

## 7. Log Dosyası

- [ ] Eğitim logu mevcut (epoch başına loss, F1)
- [ ] Her panel için eğitim logu ayrı kaydedilmiş
- [ ] Son epoch F1 değeri log dosyasından doğrulanabiliyor
- [ ] Hata durumunda anlamlı hata mesajı üretiliyor

## 8. README

- [ ] Projenin amacı açık
- [ ] Kurulum → Çalıştırma → Tahmin akışı adım adım yazılı
- [ ] Veri formatı beklentisi açıklanmış
- [ ] Her panel için örnek komut var
- [ ] Beklenen F1, MCC ve PR-AUC değerleri (panel bazlı) belirtilmiş (referans için)

## 9. Örnek Kullanım

- [ ] Küçük örnek veri (dummy) ile hızlı test çalıştırılabiliyor
- [ ] Örnek çıktı repo'da mevcut (beklenen output)
- [ ] `tests/` veya `examples/` klasörü var

## 10. Tek Komutla Çalıştırma

- [ ] `bash run_all.sh` veya `python main.py --all` gibi tek komut her şeyi çalıştırıyor
- [ ] Bu komut test verisi üzerinde tahmin üretiyor
- [ ] Çıktı doğrulama adımı var (F1, MCC ve PR-AUC referans değerleriyle karşılaştırma)
- [ ] Süre tahmini dokümante edilmiş

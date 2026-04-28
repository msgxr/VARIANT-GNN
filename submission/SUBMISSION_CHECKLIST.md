# Teslim Kontrol Listesi — VARIANT-GNN

## PDR (Proje Detay Raporu) — Teslim: 29 Haziran 2026

### Rapor

- [ ] PDR resmi şablonu kullanıldı
- [ ] Takım şeması dolduruldu
- [ ] 10 uluslararası makale özeti yazıldı
- [ ] Veri seti ve etiketler açıklandı (Bölüm 3.1)
- [ ] Veri ön işleme detaylandırıldı (Bölüm 3.3)
- [ ] Sınıf dengesi stratejisi açıklandı (Bölüm 3.5)
- [ ] Deney protokolü ve bölme stratejisi açıklandı (Bölüm 4.1)
- [ ] Metrikler ve panel bazlı sonuçlar raporlandı (Bölüm 4.2)
- [ ] Hata analizi yapıldı (Bölüm 4.3)
- [ ] Açıklanabilirlik yaklaşımı anlatıldı (Bölüm 4.4)
- [ ] Mimari seçim gerekçesi yazıldı (Bölüm 5.1)
- [ ] Ablation sonuçları eklendi (Bölüm 5.2–5.3)
- [ ] Hesaplama kaynakları belirtildi (Bölüm 5.4)

### Teknik Dosyalar

- [ ] `submission/teknofest/jury_predictions.csv` üretildi
- [ ] `submission/teknofest/artifact_manifest.json` üretildi
- [ ] `submission/teknofest/checksums.json` üretildi
- [ ] Model kartı PDF olarak hazırlandı
- [ ] Reproducibility checklist tamamlandı

### Kod Kalitesi

- [ ] `pytest tests/smoke/` geçiyor
- [ ] `pytest tests/unit/` geçiyor
- [ ] `pytest tests/integration/` geçiyor
- [ ] `ruff check src/` geçiyor
- [ ] `mypy src/` geçiyor
- [ ] CI pipeline yeşil

### Veri

- [ ] Gerçek yarışma verisi repoya eklenmedi
- [ ] `data/samples/` örnek veri güncel
- [ ] `data/contracts/` sözleşmeler tamamlandı
- [ ] Panel bazlı dosyalar doğru yerleştirildi

### Güvenlik

- [ ] `.env` veya gizli credential repoda yok
- [ ] Model binary'leri gitignore kapsamında
- [ ] NDA kapsamındaki veri repoda yok

## Reproducibility Checklist

- [ ] `seed=42` tüm bileşenlerde aktif
- [ ] `requirements.txt` sabitlenmiş versiyonlar içeriyor
- [ ] Docker ile ortam yeniden oluşturulabilir
- [ ] README kurulum adımları test edildi
- [ ] `make crossval` örnek veriyle çalışıyor

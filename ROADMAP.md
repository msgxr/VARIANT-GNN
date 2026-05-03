# Yol Haritası — VARIANT-GNN

**Güncelleme tarihi:** Nisan 2026

## P0 — PDR Kritik (Mayıs–Haziran 2026)

Veri dağıtımı (5 Mayıs 2026) sonrası gerçek yarışma verisiyle yapılacak zorunlu çalışmalar:

- [ ] Gerçek yarışma verisiyle tam eğitim pipeline çalıştır
- [ ] Panel bazlı metrik raporlarını gerçek veriyle üret
- [ ] External validation raporunu güçlendir
- [ ] Adversarial validation raporunu belgele
- [ ] Ablation analizi yürüt ve raporla (GNN yok / DNN yok / kalibrasyon yok)
- [ ] Jüri CSV export formatını doğrula ve test et
- [ ] JSON veri sözleşmelerini tamamla (`data/contracts/`)
- [ ] Anonim kolon modunu gerçek veriyle test et
- [ ] Reproducibility checklist'i tamamla
- [ ] PDR raporunu hazırla (teslim: 29 Haziran 2026)

## P1 — Jüri Hazırlık (Haziran–Temmuz 2026)

Final aşamasına hazırlık için geliştirmeler:

- [ ] Artifact manifest ve checksum doğrulamasını aktif hale getir
- [ ] Model registry versiyonlamasını otur
- [ ] `submission/` paket yapısını tamamla
- [ ] docs/clinical/ altındaki tüm klinik uyarı dokümantasyonunu tamamla
- [x] docs/evaluation/evaluation_protocol.md mevcut (güncelle ve PDR ile hizala)
- [x] docs/submission/teknofest_submission.md mevcut (teslim paketine bağla)
- [ ] Streamlit UI klinik uyarı metnini güçlendir
- [ ] Teknik rapor PDF taslağını hazırla
- [ ] LightGBM artifact roundtrip testini ekle (TD-004)
- [ ] Multimodal sekans inference tutarlılığını test et (TD-006)

## P2 — Profesyonellik (Temmuz+ / Finaller)

İleride yapılabilecek iyileştirmeler:

- [ ] VUS (Önemi Belirsiz Varyant) desteği araştır
- [ ] Çok dilli XAI desteği (İngilizce klinik rapor)
- [ ] Streamlit Cloud deployment
- [ ] GitHub Actions workflow'larını güçlendir
- [ ] Tüm notebook'ları temizle ve belgele
- [ ] Config JSON schema doğrulaması ekle (TD-011)
- [ ] MLflow bağımlılığını aktif kullanım veya kaldırma kararı ver (TD-012)

## Kapsam Dışı (Bu Yarışma için)

Aşağıdakiler kasıtlı olarak bu yarışma kapsamı dışındadır:

- Bağımsız klinik validasyon (prospektif kohort çalışması)
- Klinik üretim dağıtımı
- Ham sekans analizi (ANNOVAR/VEP entegrasyonu)
- Yapısal varyant (SV) desteği
- De novo varyant keşfi

# Yol Haritası — VARIANT-GNN

**Güncelleme tarihi:** 24 Mayıs 2026

## P0 — PDR Kritik (Mayıs–Haziran 2026)

- [x] Gerçek yarışma verisiyle tam eğitim pipeline çalıştır (20 Mayıs 2026 — Test F1=0.833)
- [x] Panel bazlı metrik raporlarını gerçek veriyle üret (`reports/cv_report.json`)
- [x] External validation raporunu güçlendir (`reports/external_validation_report.json`)
- [x] Adversarial validation raporunu belgele (AUC≈0.50, tüm paneller)
- [x] Ablation analizi yürüt ve raporla (`reports/ablation_report.json`, 8 konfigürasyon)
- [x] Jüri CSV export formatını doğrula ve test et (7 garantili kolon — predict.py PASSED)
- [x] JSON veri sözleşmelerini tamamla (`data/contracts/`)
- [x] Anonim kolon modunu gerçek veriyle test et (feature_coverage=0.0, beklenen)
- [x] Reproducibility checklist'i tamamla (SHA256, PROVENANCE.json, seed=42)
- [ ] PDR raporunu resmi DOCX şablonuna aktar ve teslim et (29 Haziran 2026)
- [x] SHAP waterfall görseli PDR §2.4'e ekle (§4.4 puanı +1.67) — 24 Mayıs 2026
- [ ] Deney günlüğü tablosu PDR §4.5'e ekle (§4.5 puanı +1.67)
- [x] 5×4 model-panel ablasyon tablosu (§5.1 puanı +1.00) — PDR Tablo 8, 24 Mayıs 2026

## P1 — Jüri Hazırlık (Haziran–Temmuz 2026)

Final aşamasına hazırlık için geliştirmeler:

- [x] Artifact manifest ve checksum doğrulamasını aktif hale getir (`submission/teknofest/artifact_manifest.json`)
- [x] `submission/` paket yapısını tamamla (predict.py, manifest, checksums)
- [x] docs/clinical/ altındaki tüm klinik uyarı dokümantasyonunu tamamla
- [x] docs/evaluation/evaluation_protocol.md mevcut
- [x] docs/submission/teknofest_submission.md mevcut
- [x] LightGBM artifact roundtrip testini ekle (TD-004 — CLOSED)
- [ ] jury_predictions.csv gerçek kör test verisiyle yeniden üret (`submission/predict.py`)
- [ ] Streamlit UI klinik uyarı metnini güçlendir
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

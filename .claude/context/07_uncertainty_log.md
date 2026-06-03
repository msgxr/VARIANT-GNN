# 07_uncertainty_log.md — Belirsizlik Günlüğü

**Versiyon:** 2026-05-24  
**Amaç:** Resmi TEKNOFEST 2026 kaynağında doğrulanamayan bilgileri kayıt altına almak.

---

## Aktif Belirsizlikler

| ID | Konu | Son Kontrol | Netleştirme Yolu |
|---|---|---|---|
| U-001 | PDR sayfa limiti (tam sayı) | 2026-05-24 | PDR şablonu DOCX'ten oku |
| U-002 | Puanlama ağırlıkları (% görev / % sunum) | 2026-05-24 | Şartname §7 tam okuma |
| U-003 | Teslim formatı (PDF mi, DOCX mi?) | 2026-05-24 | Şartname veya sistem duyurusu |
| U-004 | Şeyma'nın GitHub kullanıcı adı ve e-postası | 2026-05-24 | Şeyma'ya sor |
| U-005 | Mezuniyet tarih kısıtı (üniversite kategori uygunluğu) | 2026-05-24 | Şartname §3 |
| U-006 | Veri yayın/tez kullanım hakları | 2026-05-24 | Şartname veya gizlilik taahhütü |
| U-007 | Final değerlendirmesinde sözlü sunum ağırlığı | 2026-05-24 | Şartname §7 |
| U-009 | Submission çıktı dosya formatı (sütun düzeni) | 2026-06-03 | 🟡 KISMEN: Q&A-II **çıktının ikili 0/1 etiket olduğunu** teyit etti (transkript 08:12, 20:44) — olasılık DEĞİL. Ancak tam dosya formatı (kolon düzeni) organizatör tarafından "son aşamada netleşecek, ayrı şartname paylaşılacak" denildi (20:35). Kod=12 kolon (OOD_Score/OOD_Flag kasıtlı; ood_detector.pkl var), CI guard 12'ye hizalı. GERÇEK format sistem duyurusu/Google Groups'tan teyit edilecek; ikili etiket çekirdeği kesin. |

---

## Çözülmüş Belirsizlikler

| ID | Konu | Sonuç | Kaynak |
|---|---|---|---|
| C-001 | GNN mimarisi (SAGEConv mu, GATv2Conv mu?) | GATv2Conv (Brody et al. 2022) | Kod incelemesi, PDR §2.2 |
| C-002 | Birincil metrik | Binary F1 Score, pos_label=1 | Şartname §7.3 |
| C-003 | PDR teslim tarihi | 29 Haziran 2026, 17:00 | Şartname / Takvim |
| C-004 | Veri dağıtım tarihi | 5 Mayıs 2026 | Takvim |
| C-005 | Pilot MCC vs gerçek MCC farkı | Açıklandı — veri kalitesi ve denge farkı | PDR §4.2 |
| C-006 | Gizli test seti dağılımı (50/50 mı, %20-patojenik mi?) | **Test ~%20 patojenik / %80 benign** (eğitim ~%80/%20'nin tersi). Şartnamedeki 50/50 organizatör tarafından eski/hatalı kabul edildi. θ=0.8415 + 0.6202 dayanağı RESMİ. | Q&A-II transkripti (`docs/qa/2026_QA_Universite_transkript.md`, 07:13/23:37/26:23); `01_official_source_map.md` Tier-4 |
| C-007 | Skorlama: per-panel F1 + ortalama; çıktı 0/1; 4 ayrı model | Hepsi doğrulandı (per-panel F1→ortalama 34:10; ikili 0/1 08:12; 4 model 31:30) | Q&A-II transkripti |
| C-008 | Komşu nükleotit/aminoasit (yerel sekans bağlamı) verilecek mi? | **Verilmiyor** — şartname §3.2'deki ±5 nt/±5 aa bilim kurulu kararıyla kaldırıldı | Q&A-II transkripti (21:47, 23:57) |

---

## Günlük Kullanım Talimatı

- Belirsiz bir yarışma kuralıyla karşılaşıldığında bu dosyaya UNVERIFIED olarak eklenir.
- Netleştirildiğinde "Çözülmüş Belirsizlikler" tablosuna taşınır.
- Bu dosyadaki belirsizlikler kesinlikle doğrulanmış bilgi gibi kullanılmaz.

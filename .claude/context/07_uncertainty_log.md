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
| U-008 | Gizli test seti boyutu ve dağılımı | 2026-05-24 | Jüri değerlendirmesiyle ortaya çıkar |

---

## Çözülmüş Belirsizlikler

| ID | Konu | Sonuç | Kaynak |
|---|---|---|---|
| C-001 | GNN mimarisi (SAGEConv mu, GATv2Conv mu?) | GATv2Conv (Brody et al. 2022) | Kod incelemesi, PDR §2.2 |
| C-002 | Birincil metrik | Binary F1 Score, pos_label=1 | Şartname §7.3 |
| C-003 | PDR teslim tarihi | 29 Haziran 2026, 17:00 | Şartname / Takvim |
| C-004 | Veri dağıtım tarihi | 5 Mayıs 2026 | Takvim |
| C-005 | Pilot MCC vs gerçek MCC farkı | Açıklandı — veri kalitesi ve denge farkı | PDR §4.2 |

---

## Günlük Kullanım Talimatı

- Belirsiz bir yarışma kuralıyla karşılaşıldığında bu dosyaya UNVERIFIED olarak eklenir.
- Netleştirildiğinde "Çözülmüş Belirsizlikler" tablosuna taşınır.
- Bu dosyadaki belirsizlikler kesinlikle doğrulanmış bilgi gibi kullanılmaz.

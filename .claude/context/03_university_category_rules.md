# 03_university_category_rules.md — Üniversite Kategorisi Kuralları

**Kaynak:** TEKNOFEST 2026 Şartnamesi (Türkçe v4)  
**Versiyon:** 2026-05-24

---

## Kategori Tanımı

**Üniversite ve Üzeri Seviyesi**  
Görev: Genetik varyantların Patojenik veya Benign olduğunu tahmin eden yapay zeka modeli geliştirme.

> Lise seviyesi farklıdır (EKG/kardiyoloji). Bu proje yalnızca Üniversite ve Üzeri için geçerlidir.

---

## Görev Detayları

| Alan | Değer |
|---|---|
| Giriş | Missense genetik varyant profili (anonim kolon formatı) |
| Çıkış | Binary sınıflandırma: Patojenik(1) / Benign(0) |
| Etiket kaynağı | ClinVar Expert Panel (3–4 yıldız onaylı) |
| Dışlanan | VUS (Önemi Belirsiz Varyant) — sınıflandırılmaz |
| Panel yapısı | MASTER (General), Hereditary_Cancer, PAH, CFTR |

---

## Veri Kısıtları

- Kolon isimleri anonimdir (AL_x, EK_x önekli veya benzeri)
- Genomik adres (kromozom, pozisyon) kullanılamaz
- Harici veri tabanından etiket aramak diskwalifikasyon sebebidir
- Veri "Kurumsal Gizlilik Taahhütü" kapsamındadır

---

## Katılımcı Uygunluğu

- Önlisans, lisans, yüksek lisans, doktora, açık öğretim öğrencileri
- Mezunlar (yakın dönem — UNVERIFIED: mezuniyet tarih kısıtı şartnameden kontrol)
- Danışman yarışmacı sayılmaz, takım büyüklüğüne dahil değil

---

## Diskwalifikasyon Riskleri

| Risk | Sonuç |
|---|---|
| Test etiketlerini eğitimde kullanmak | Diskwalifikasyon |
| Gizlilik taahhütü imzalamamak | Finalden men |
| Genomik adres kullanmak | LEVEL 1 CRITICAL |
| Lise görevi içeriği (EKG vb.) sunmak | Kategori hatası |
| Binary F1 yerine yanlış metrik sunmak | Ciddi puan kaybı |

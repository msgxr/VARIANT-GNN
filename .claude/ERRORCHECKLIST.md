# ERRORCHECKLIST.md — VARIANT-GNN

**Versiyon:** 2026-05-24  
**Kaynak:** TEKNOFEST 2026 Şartnamesi (Türkçe v4) + Proje deneyimi  
**Kullanım:** Her teslim öncesi ve her kritik değişiklik sonrası uygulanır

---

## Kontrol A — Kategori ve Görev Doğruluğu

- [ ] Üniversite ve Üzeri kategorisi için çalışıldığı teyit edildi
- [ ] Lise kategorisi içeriği (EKG, kardiyoloji) karışmadı
- [ ] Görev binary sınıflandırma: Patojenik(1) / Benign(0)
- [ ] VUS etiketleri dışarıda tutuldu, sınıflandırılmadı
- [ ] Dört panel ayrı ayrı raporlandı: MASTER / KANSER / PAH / CFTR

---

## Kontrol B — Metrik Doğruluğu

- [ ] Birincil metrik Binary F1 Score (pos_label=1)
- [ ] F1 = TP / (TP + 0.5×FP + 0.5×FN) formülü doğru uygulandı
- [ ] Accuracy tek başına ana başarı metriği olarak sunulmadı
- [ ] Macro F1 / Weighted F1, Binary F1 yerine kullanılmadı
- [ ] MCC / PR-AUC / ROC-AUC destekleyici metrik olarak sunuldu
- [ ] Her panelin F1 skoru ayrı raporlandı

---

## Kontrol C — Veri Sızıntısı (Data Leakage)

- [ ] Scaler / Imputer / Encoder yalnızca eğitim fold'unda fit edildi
- [ ] SMOTE yalnızca eğitim verisi içinde uygulandı
- [ ] Test seti etiketleri eğitim sırasında hiç kullanılmadı
- [ ] AutoEncoder yalnızca eğitim verisinde fit edildi
- [ ] SelectKBest yalnızca eğitim verisinde fit edildi
- [ ] Kalibrasyon seti eğitim verisinin ayrı %15'lik dilimidir
- [ ] Adversarial validation ROC-AUC ≈ 0.50 (dağılım uyumu teyit edildi)

---

## Kontrol D — Veri ve Gizlilik

- [ ] Yarışma verisi repoya push edilmedi (.gitignore kapsamında)
- [ ] Kurumsal Gizlilik Taahhütü imzalandı
- [ ] KVKK uyumu: kişisel veri işlenmedi
- [ ] Genomik adres (kromozom / pozisyon) özellik olarak kullanılmadı
- [ ] Anonim kolon kısıtı ihlal edilmedi
- [ ] Harici kaynaklardan etiket arama yapılmadı

---

## Kontrol E — Resmi Kaynak Kullanımı

- [ ] Yarışma kuralları yalnızca 2026 resmi şartnamesinden alındı
- [ ] Üçüncü taraf kaynak kullanılmadı
- [ ] Doğrulanamayan bilgiler UNVERIFIED olarak işaretlendi
- [ ] Şartnamede açıkça yazmayan kural kesin hüküm gibi yazılmadı
- [ ] 2024 veya önceki yıl şartnamesi 2026 kuralı gibi sunulmadı

---

## Kontrol F — Rapor Uyumu

- [ ] PDR resmi şablonu kullanıldı (Üniversite ve Üzeri)
- [ ] PSR resmi şablonu kullanıldı (Üniversite ve Üzeri)
- [ ] Lise şablonu kullanılmadı
- [ ] Tüm zorunlu bölümler mevcut
- [ ] Sayfa limiti aşılmadı (şablondan kontrol)
- [ ] Her performans iddiasının kanıtı var (deney çıktısı, log, JSON)
- [ ] PSR → PDR veri farkı (MCC 0.892 → 0.536) açıklandı

---

## Kontrol G — Etik ve Klinik Sınır

- [ ] "Klinik tanı yapabilir" ifadesi yok
- [ ] "Tedavi önerir" ifadesi yok
- [ ] "Hekimin yerini alır" ifadesi yok
- [ ] Etik beyan PDR'de mevcut
- [ ] "Araştırma ve yarışma amacıyla" kısıtı belirtildi

---

## Kontrol H — Tekrar Üretilebilirlik

- [ ] random_state=42 tüm bileşenlerde sabit
- [ ] torch.manual_seed(42) ve np.random.seed(42) set edildi
- [ ] requirements.txt sabitlenmiş versiyonlar içeriyor
- [ ] environment.yml mevcut ve güncel
- [ ] Eğitim tek komutla çalışıyor
- [ ] Tahmin tek komutla çalışıyor
- [ ] 5-seed kararlılık testi yapıldı (std ±0.0013)
- [ ] Model dosyaları (Şeyma makinesinde) yeniden üretilmiş

---

## Kontrol I — Kod Kalitesi

- [ ] src/core/dnn.py vs src/models/dnn_model.py çakışması çözüldü (TD-007)
- [ ] API rest_api.py stub durumu belgelendi
- [ ] Stub/boş dosyalar kasıtlı olduğu belirtildi
- [ ] Tüm import yolları çalışıyor (ilgili ortamda)

---

## Kontrol J — Git Kimliği

- [ ] Push öncesi kimin commit atacağı soruldu
- [ ] git config user.name doğru kişi için ayarlandı
- [ ] git config user.email doğru kişi için ayarlandı
- [ ] git log -1 --pretty=fuller kontrolü yapıldı
- [ ] Author satırı doğru kişiyi gösteriyor
- [ ] Committer satırı doğru kişiyi gösteriyor
- [ ] Claude / Bot / AI / Automation commit kimliğinde görünmüyor

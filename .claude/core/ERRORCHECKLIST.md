# ERRORCHECKLIST.md — VARIANT-GNN

**Versiyon:** 2026-06-09  
**Kullanım:** error-checker skill ve genel hata denetimi için referans  
**Kapsam:** Domain A–J — tam proje denetim alanları

---

## DOMAIN A — Kategori ve Görev Doğruluğu

- [ ] Kategori: Üniversite ve Üzeri (Lise değil, EKG/Kardiyoloji değil)
- [ ] Görev: Patojenik(1) / Benign(0) binary sınıflandırma
- [ ] VUS etiketleri sınıflandırmadan hariç
- [ ] Dört panel ayrı değerlendirildi: MASTER / KANSER / PAH / CFTR
- [ ] Genomik adres (kromozom, pozisyon) özellik olarak kullanılmadı

**FAIL tetikleyici:** Lise şablonu, EKG içeriği, genomik adres, paneller birleştirme

---

## DOMAIN B — Birincil Metrik

- [ ] Binary F1 Score (pos_label=1=Patojenik) birincil metrik olarak raporlandı
- [ ] F1 = TP / (TP + 0.5×FP + 0.5×FN) — formül doğru
- [ ] Accuracy tek başına ana metrik olarak sunulmadı
- [ ] MCC ve PR-AUC destekleyici metrik olarak mevcut
- [ ] Her panel için ayrı F1 skoru raporlandı
- [ ] GLOBAL karar eşiği θ=0.8415 (canonical, models/threshold.json); panel eşikleri opt-in (General 0.3990, KANSER 0.4532, PAH 0.4434, CFTR 0.1922 — jüri kullanmaz)
- [ ] Eşik değerleri uygulandıktan SONRA metrik hesaplandı

**FAIL tetikleyici:** Yanlış F1 formülü, Accuracy birincil, eksik panel metriği

---

## DOMAIN C — Veri Bütünlüğü ve Sızıntı

- [ ] Scaler yalnızca eğitim fold'unda fit edildi
- [ ] Imputer yalnızca eğitim fold'unda fit edildi
- [ ] SelectKBest yalnızca eğitim fold'unda fit edildi
- [ ] AutoEncoder yalnızca eğitim fold'unda fit edildi
- [ ] SMOTE yalnızca eğitim fold'u içinde uygulandı (split sonrası)
- [ ] Kalibrasyon seti eğitim verisinin %15'i (test seti değil)
- [ ] Adversarial Validation ROC-AUC ≈ 0.50 (dağılım sızıntısı yok)
- [ ] Test seti etiketleri eğitimde hiç kullanılmadı

**FAIL tetikleyici:** Herhangi bir preprocessor test verisinde fit, SMOTE split öncesi, test etiketi eğitimde

---

## DOMAIN D — Veri Gizliliği ve Güvenlik

- [ ] Yarışma verisi (.csv, .tsv, .vcf) git reposunda yok
- [ ] .gitignore data/raw/ ve benzeri yolları kapsıyor
- [ ] Kişisel veri (hasta kimliği, kişisel bilgi) yok
- [ ] KVKK uyumu PDR etik beyanında belirtildi
- [ ] Model .pkl/.pt dosyaları repoda yok (Şeyma'nın makinesinde — doğru)
- [ ] .env veya credential dosyası commit edilmedi

**FAIL tetikleyici:** Veri repoda, model repoda (repo boyutu), credential görünür

---

## DOMAIN E — Resmi Kaynak Uyumu

- [ ] Tüm yarışma kuralları TEKNOFEST 2026 Türkçe Şartname v4'ten alındı
- [ ] Üçüncü taraf kaynak kullanılmadı
- [ ] Doğrulanamayan bilgi UNVERIFIED işaretlendi
- [ ] 2024 şartname referansı yok
- [ ] Tüm tarihler 2026 takviminden doğrulandı

**FAIL tetikleyici:** 2024 şartname kullanımı, doğrulanamayan kuralın kesin hüküm gibi sunulması

---

## DOMAIN F — Rapor Uyumu

- [ ] Üniversite PDR şablonu kullanıldı (Lise değil)
- [ ] 5 bölüm ve alt bölümler eksiksiz: Giriş(10) / Yöntem(25) / Bulgular(30) / Sonuç(25) / Kaynakça(10)
- [ ] PDR ≤ 10 sayfa (kapak + içindekiler hariç)
- [ ] Yazı tipi Aptos, 12pt gövde, 14pt başlık
- [ ] Etik beyan mevcut
- [ ] BUG-01..09 hepsi kapatıldı (bkz. MASTER_PLAYBOOK.md §6)
- [ ] PSR↔PDR tutarsızlıkları (GNN adı, MCC farkı) açıklandı

**FAIL tetikleyici:** Açık bug, eksik bölüm, 10 sayfa aşımı, yanlış font

---

## DOMAIN G — Etik ve Klinik Sınır

- [ ] Klinik tanı iddiası yok
- [ ] Tedavi tavsiyesi dili yok
- [ ] "Araştırma ve yarışma amacıyla" sorumluluk reddi mevcut
- [ ] KVKK uyumu beyan edildi
- [ ] "Klinik pratikte kullanılabilir" ifadesi yok

**FAIL tetikleyici:** Herhangi bir klinik kullanım dili

---

## DOMAIN H — Tekrar Üretilebilirlik

- [ ] random_state=42 tüm bileşenlerde sabit
- [ ] torch.manual_seed(42) set
- [ ] np.random.seed(42) set
- [ ] requirements.txt versiyonları pinlenmiş
- [ ] environment.yml eksiksiz
- [ ] Eğitim tek komutla: `python main.py --mode train --config configs/pdr.yaml`
- [ ] Tahmin tek komutla: `python submission/predict.py --input <dosya>`
- [ ] 5-seed stabilite CV F1=0.8738±0.0034 belgelenmiş
- [ ] Model artifact'ları repoda mevcut (<7MB, REPRODUCE.md — jüri veri olmadan tahmin üretebilir)

**FAIL tetikleyici:** Non-deterministik çalışma, eksik requirements, çoklu adım gerektiren çalıştırma

---

## DOMAIN I — Kod Kalitesi

- [ ] Panel-bazlı threshold uygulaması doğru (her panel kendi threshold'unu kullanıyor)
- [ ] GNN grafı cosine k-NN (k=10, eşik=0.3) doğru kurulmuş
- [ ] AutoEncoder DEVRE DIŞI (use_autoencoder=false; doğrula)
- [ ] SelectKBest DEVRE DIŞI (use_feature_selection=false; tam 343 öznitelik korunuyor; doğrula)
- [ ] SMOTE yalnızca eğitim içinde (kod okunarak doğrula, varsayım değil)
- [ ] Stacking meta-learner logistic regression (doğrula)

**RISK tetikleyici:** Herhangi bir sabit değerin kod gerçeğiyle uyuşmaması

---

## DOMAIN J — Git Kimliği

- [ ] Son commit Author: msgxr <mgun345@icloud.com> (bu PC)
- [ ] Author satırında Claude / Bot / AI / Automation görünmüyor
- [ ] `git log -1 --pretty=fuller` doğrulandı
- [ ] Yarışma verisi son commit'te yok

**FAIL tetikleyici:** Yanlış author, bot/AI adı görünmesi

---

## Hızlı Tarama Öncelik Sırası

```
P0 (diskwalifikasyon riski): C (leakage) → D (data privacy) → G (clinical ethics)
P1 (final skoru riski):      B (metrik) → H (reproducibility) → I (kod)
P2 (jüri izlenimi riski):   F (rapor) → A (görev) → E (kaynak) → J (git)
```

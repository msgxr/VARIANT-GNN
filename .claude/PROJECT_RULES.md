# PROJECT_RULES.md — VARIANT-GNN

**Versiyon:** 2026-05-24  
**Kaynak:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması Şartnamesi (Türkçe v4)  
**Geçerlilik:** Bu proje süresince değiştirilemez kurallar

---

## 1. Proje Kimliği

- **Proje Adı:** VARIANT-GNN
- **Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması
- **Kategori:** Üniversite ve Üzeri
- **Görev:** Missense genetik varyantların Patojenik (1) / Benign (0) binary sınıflandırması
- **Takım:** XYRA3 | ID: 909249
- **Git Kimliği:** msgxr <mgun345@icloud.com>

---

## 2. Kaynak Politikası (Değiştirilemez)

1. Tek yetkili kaynak TEKNOFEST 2026 resmi şartnamesi ve resmi şablonlarıdır.
2. Üçüncü taraf blog, forum, sosyal medya, gayriresmi özet kaynak kabul edilmez.
3. Doğrulanamayan bilgi **UNVERIFIED** olarak işaretlenir; hiçbir zaman kesin kural gibi sunulmaz.
4. Şartname, şablon ve web sayfası çelişirse öncelik: Şartname > Şablon > Web Sayfası > Duyuru.
5. Kaynak uydurulmaz; emin olunmayan her şey durdurulur ve sorulur.

---

## 3. Metrik Kuralları (Değiştirilemez)

- **Birincil metrik:** Binary F1 Score (pos_label=1=Patojenik)
- F1 = TP / (TP + 0.5×FP + 0.5×FN)
- Accuracy, macro F1, weighted F1 tek başına ana başarı metriği olarak sunulamaz.
- MCC, PR-AUC, ROC-AUC destekleyici metriklerdir; F1'in yerini tutmaz.
- Test seti etiketi eğitim sırasında kullanılamaz — kullanılması diskwalifiye sebebidir.

---

## 4. Veri Kuralları (Değiştirilemez)

- Yarışma verisi yalnızca eğitim ve değerlendirme amacıyla kullanılır.
- Veri "Kurumsal Gizlilik Taahhütü" kapsamındadır — paylaşılamaz, yayınlanamaz.
- Kolon isimleri anonimdir; genomik adres kullanılamaz.
- KVKK uyumu zorunludur; kişisel veri bulunmaz, işlenemez.
- VUS (Önemi Belirsiz Varyant) etiketleri sınıflandırma dışındadır.

---

## 5. Kod ve Tekrar Üretilebilirlik Kuralları

- Tüm random seed'ler sabittir: random_state=42, torch.manual_seed(42), np.random.seed(42).
- Ortam, requirements.txt ve environment.yml ile tam olarak tanımlanmıştır.
- Eğitim: tek komutla çalışır → `python main.py --mode train --config configs/pdr.yaml`
- Tahmin: tek komutla çalışır → `python submission/predict.py --input <dosya>`
- Jüri kodu yeniden çalıştırabilir; beyan edilen sonuçlar yeniden üretilmelidir.

---

## 6. Etik Sınırlar (Değiştirilemez)

- Sistem yalnızca araştırma ve yarışma amacıyla kullanılır.
- Klinik tanı, tedavi kararı veya hasta yönetimine dair hiçbir iddia yapılamaz.
- "Bu sistem klinik pratikte kullanılabilir" türünde hiçbir ifade yazılamaz.
- Klinik kullanım için bağımsız validasyon, sağlık otoritesi onayı ve etik kurul gereklidir.

---

## 7. Git Kimliği Kuralı (Değiştirilemez)

Projede iki takım üyesi commit atabilir. Her push işleminden önce **kim push atacak** mutlaka sorulur.

### Takım Üyesi Kimlik Tablosu

| Takım Üyesi | GitHub Kullanıcı Adı | E-posta |
|---|---|---|
| Kullanıcı (sen) | `msgxr` | `mgun345@icloud.com` |
| Şeyma (iş arkadaşı) | `cebi101` | `seymanurcebi6@gmail.com` |

### Zorunlu Prosedür

**Push öncesi Claude şunu sorar:**
> "Bu commit hangi takım üyesi adına atılacak? Kimin makinesi kullanılıyor?"

Yanıta göre git config ayarlanır:
```bash
git config user.name "<isim>"
git config user.email "<email>"
```

Commit sonrası doğrulama:
```bash
git log -1 --pretty=fuller
```

- Author satırı doğru kişiyi gösteriyorsa push yapılır.
- Author veya Committer yanlışsa: `git commit --amend --reset-author --no-edit`
- Claude, Bot, AI, Automation gibi isimler commit kimliğinde **kesinlikle görünemez**.
- Kimlik belirsizse push yapılmaz, önce sorulur.

---

## 8. Yasak Eylemler

- PSR pilot sonuçlarını gerçek yarışma verisi sonucu gibi sunmak
- Dört paneli tek homojen blok olarak değerlendirmek (MASTER / KANSER / PAH / CFTR ayrı analiz edilir)
- Eleştirel bulguları yumuşatmak veya olumsuz sonuçları gizlemek
- Kanıtsız performans iddiası yapmak
- Şartnamede açıkça belirtilmemiş kuralı kesin hüküm gibi yazmak
- Geri dönüşü olmayan değişiklikleri (dosya silme, yapısal değişiklik) onaysız uygulamak

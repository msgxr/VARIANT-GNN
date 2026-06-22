# PROJECT_RULES.md — VARIANT-GNN

**Versiyon:** 2026-06-09  
**Kaynak:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması Şartnamesi (Türkçe v4) + PDR Şablonu  
**Geçerlilik:** Bu proje süresince değiştirilemez kurallar

---

## 1. Proje Kimliği

- **Proje Adı:** VARIANT-GNN
- **Yarışma:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması
- **Kategori:** Üniversite ve Üzeri
- **Görev:** Missense genetik varyantların Patojenik (1) / Benign (0) binary sınıflandırması
- **Takım:** XYRA3 | ID: 909249 | Başvuru ID: 5200240
- **Git Kimliği (Bu PC):** msgxr <mgun345@icloud.com>

---

## 2. Kaynak Politikası (Değiştirilemez)

1. Tek yetkili kaynak TEKNOFEST 2026 resmi şartnamesi ve resmi şablonlarıdır.
2. Üçüncü taraf blog, forum, sosyal medya, gayriresmi özet kaynak kabul edilmez.
3. Doğrulanamayan bilgi **UNVERIFIED** olarak işaretlenir; hiçbir zaman kesin kural gibi sunulmaz.
4. Şartname, şablon ve web sayfası çelişirse öncelik: Şartname > Şablon > Web Sayfası > Duyuru.
5. Kaynak uydurulmaz; emin olunmayan her şey durdurulur ve kullanıcıya sorulur.
6. 2024 veya önceki yıl şartnameleri bu proje için otomatik reddedilir.

---

## 3. Metrik Kuralları (Değiştirilemez)

- **Birincil metrik:** Binary F1 Score (pos_label=1=Patojenik) — Şartname §7.3
- F1 = TP / (TP + 0.5×FP + 0.5×FN)
- Accuracy, macro F1, weighted F1 tek başına ana başarı metriği olarak sunulamaz.
- MCC, PR-AUC, ROC-AUC destekleyici metriklerdir; F1'in yerini tutmaz.
- PDR'de F1 + MCC + PR-AUC her panel için ayrı ayrı raporlanmalıdır (PDR Şablonu §3).
- Test seti etiketi eğitim sırasında kullanılamaz — kullanılması diskwalifiye sebebidir.
- **⚠️ RISK:** TÜSEB metrik değişiklik hakkını saklı tutar (§7.5). Duyurular takip edilmeli.

---

## 4. Veri Kuralları (Değiştirilemez)

- Yarışma verisi yalnızca eğitim ve değerlendirme amacıyla kullanılır.
- Veri "Kurumsal Gizlilik Taahhütü" kapsamındadır — paylaşılamaz, yayınlanamaz.
- Kolon isimleri anonimdir; genomik adres (kromozom, pozisyon) kullanılamaz.
- KVKK uyumu zorunludur; kişisel veri bulunmaz, işlenemez.
- VUS (Önemi Belirsiz Varyant) etiketleri sınıflandırma dışındadır.
- Yarışma verisi git reposuna commit edilemez.

---

## 5. Kod ve Tekrar Üretilebilirlik Kuralları

- Tüm random seed'ler sabittir: random_state=42, torch.manual_seed(42), np.random.seed(42).
- Ortam, requirements.txt ve environment.yml ile tam olarak tanımlanmıştır.
- Eğitim: tek komutla çalışır → `python main.py --mode train --config configs/pdr.yaml`
- Tahmin: tek komutla çalışır → `python submission/predict.py --input <dosya>`
- Jüri kodu yeniden çalıştırabilir; beyan edilen sonuçlar yeniden üretilmelidir (§7.5).
- Model artifact'ları (.pkl, .pt): Şeyma'nın makinasında — bu repoda yok (doğru).

---

## 6. Etik Sınırlar (Değiştirilemez)

- Sistem yalnızca araştırma, eğitim ve yarışma amacıyla kullanılır.
- Klinik tanı, tedavi kararı veya hasta yönetimine dair hiçbir iddia yapılamaz.
- "Bu sistem klinik pratikte kullanılabilir" türünde hiçbir ifade yazılamaz.
- Resmi etik sınır (Şartname §10): *"Yarışma kapsamında geliştirilen modeller ve elde edilen çıktılar, herhangi bir klinik tanı, tedavi veya tıbbi karar destek amacıyla kullanılamaz."*

---

## 7. Git Kimliği Kuralı (Değiştirilemez)

Makine kimliği belirler — her push öncesi sormaya gerek yok:

| Makine | git user.name | git user.email |
|---|---|---|
| Bu Windows PC | `msgxr` | `mgun345@icloud.com` |
| Şeyma'nın Mac'i | `cebi101` | `seymanurcebi6@gmail.com` |

### Prosedür (git-identity-guardian skill'i otomatik yürütür)

1. git config user.name ve user.email'i doğru değere ayarla
2. Commit yap
3. `git log -1 --pretty=fuller` ile Author ve Committer satırlarını doğrula
4. Author yanlışsa: `git commit --amend --reset-author --no-edit`
5. Doğru olduğu onaylandıktan sonra push yap

**Kesinlikle görünemez:** Claude / Bot / AI / Automation / Assistant — herhangi bir author/committer satırında.

---

## 8. Yasak Eylemler

- PSR pilot sonuçlarını gerçek yarışma verisi sonucu gibi sunmak
- Dört paneli tek homojen blok olarak değerlendirmek (MASTER / KANSER / PAH / CFTR ayrı)
- Eleştirel bulguları yumuşatmak veya olumsuz sonuçları gizlemek
- Kanıtsız performans iddiası yapmak
- Şartnamede açıkça belirtilmemiş kuralı kesin hüküm gibi yazmak
- Geri dönüşü olmayan değişiklikleri (dosya silme, yapısal değişiklik) onaysız uygulamak
- Klinik kullanım ima eden herhangi bir ifade kullanmak
- Yarışma verisini git reposuna commit etmek

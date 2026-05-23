# 06_team_and_process_rules.md — Takım ve İş Süreci

**Kaynak:** TEKNOFEST 2026 Şartnamesi + Proje iç düzeni  
**Versiyon:** 2026-05-24

---

## Takım Kimliği

| Alan | Değer |
|---|---|
| Takım Adı | XYRA3 |
| Takım ID | 909249 |
| Başvuru ID | 4865399 |
| PSR Puanı | 93/100 |

---

## İş Bölümü

| Görev | Sorumlu |
|---|---|
| Kod geliştirme | Kullanıcı (msgxr) |
| PDR / PSR dokümantasyonu | Kullanıcı (msgxr) |
| Proje yönetimi | Kullanıcı (msgxr) |
| Model eğitimi ve ML pipeline | Şeyma |
| Model artifact üretimi (.pkl, .pt) | Şeyma |
| Git push (her ikisi yapabilir) | Kullanıcı veya Şeyma |

---

## Git Kimlik Protokolü

**Temel kural:** Push öncesi kim push atacak sorulur. Her kişi kendi kimliğiyle commit atar.

### Kullanıcı (msgxr)

```bash
git config user.name "msgxr"
git config user.email "mgun345@icloud.com"
```

### Şeyma

```bash
git config user.name "cebi101"
git config user.email "seymanurcebi6@gmail.com"
```

### Doğrulama (her push öncesi zorunlu)

```bash
git log -1 --pretty=fuller
```

Author ve Committer doğruysa push yapılır. Yanlışsa:

```bash
git commit --amend --reset-author --no-edit
git log -1 --pretty=fuller  # tekrar kontrol
```

### Yasak Commit Kimlikleri

- Claude
- Bot / bot
- AI / ai
- Automation / automation
- Assistant
- Herhangi bir otomasyon adı

---

## Model Dosyası Konumu

Model artifact dosyaları (.pkl, .pt, .joblib) Şeyma'nın makinesinde üretilir.  
Bu repodaki `models/` klasöründe yalnızca JSON config dosyaları bulunur.  
Model dosyaları `.gitignore` kapsamındadır (NDA / boyut kısıtı).

---

## İletişim ve Senkronizasyon

- Model sonuçları Şeyma → kullanıcıya aktarılır
- Kullanıcı PDR/rapor günceller
- Git push: kim push atacaksa kendi kimliğiyle atar

# Proje Durumu — VARIANT-GNN

**Güncelleme tarihi:** Nisan 2026

## Mevcut Aşama

**PDR (Proje Detay Raporu) Geliştirmesi**

PSR aşaması 93/100 puanla geçilmiştir. PDR teslim tarihi: 29 Haziran 2026.

## Olgunluk Seviyesi

**Araştırma ve Yarışma Prototipi** — klinik kullanıma hazır değildir.

| Boyut | Durum | Notlar |
|---|---|---|
| **Model mimarisi** | ✅ Stabil | XGB + LGB + GATv2GNN + DNN, 4 model ensemble |
| **Eğitim pipeline** | ✅ Çalışıyor | 5-fold CV, kalibrasyon, erken durdurma |
| **Inference pipeline** | ✅ Çalışıyor | Batch + tekli tahmin, belirsizlik desteği |
| **Açıklanabilirlik** | ✅ Çalışıyor | SHAP, LIME, GNNExplainer, Türkçe rapor |
| **Panel değerlendirme** | ✅ Çalışıyor | General, Hereditary Cancer, PAH, CFTR |
| **External validation** | ✅ Çalışıyor | `--mode external_val` |
| **Adversarial validation** | ✅ Çalışıyor | `--mode adversarial_val` |
| **Kalibrasyon** | ✅ Çalışıyor | İzotonik Regresyon |
| **MC Dropout belirsizlik** | ✅ Çalışıyor | 30 ileri geçiş |
| **Streamlit UI** | ✅ Çalışıyor | `streamlit run app.py` |
| **CI pipeline** | ✅ Çalışıyor | GitHub Actions: lint, typecheck, test, security |
| **Docker** | ✅ Mevcut | CPU ve GPU destekli |
| **Test altyapısı** | ✅ Temel seviye | Smoke, unit, integration testler |
| **Veri sözleşmeleri** | 🟡 Kısmi | Pydantic şema mevcut; JSON contract'lar eksik |
| **Ablation analizi** | 🟡 Kısmi | Kod hazır; rapor üretilmemiş |
| **Bağımsız klinik validasyon** | ❌ Yok | Araştırma prototipi olarak kasıtlı kapsam dışı |
| **VUS sınıflandırma** | ❌ Yok | Etiketli VUS verisi gerektirir |
| **Deployment (üretim)** | ❌ Yok | Araştırma prototipi; üretim dağıtımı planlanmıyor |

## PDR için Kalan Ana Görevler

1. Ablation analizi raporunu üret ve belgele
2. JSON veri sözleşmelerini tamamla (`data/contracts/`)
3. Panel bazlı metrik raporunu güçlendir
4. External validation raporunu zenginleştir
5. Jüri export formatını doğrula
6. docs/ alt yapısını tamamla (clinical/, evaluation/, submission/)
7. Teknik borç listesini güncelle

## Bilinen Sınırlamalar

- Gerçek yarışma verisi henüz mevcut değildir (veri dağıtımı: 5 Mayıs 2026)
- Klinik validasyon kapsamı dışındadır
- Anonim kolon modu test edilmemiştir
- VUS desteği bulunmamaktadır
- CI bazı ortamlarda PyTorch Geometric kurulum sorunları yaşanabilir

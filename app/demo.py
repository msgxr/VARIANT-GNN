# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
app/demo.py — VARIANT-GNN Streamlit Demo
Kullanım: streamlit run app/demo.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Sayfa yapılandırması ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="VARIANT-GNN | TEKNOFEST 2026",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO = Path(__file__).resolve().parent.parent

# ── Yardımcı: model yükle (önbellekle) ────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_pipeline():
    from src.config import reset_settings, get_settings
    from src.utils.serialization import ModelStore
    reset_settings()
    cfg = get_settings(str(REPO / "configs" / "pdr.yaml"))
    store = ModelStore(cfg.paths.models_dir)
    preproc, ensemble, _calib = store.load_all()
    # KANONİK karar: tek GLOBAL θ (models/threshold.json). Opt-in panel eşikleri
    # (panel_thresholds.json, 0.399 vb.) jüri skorunda KULLANILMAZ.
    theta = store.load_threshold(default=0.8415)
    return preproc, ensemble, theta

@st.cache_data(show_spinner=False)
def load_metrics():
    cv = json.loads((REPO / "reports" / "cv_report.json").read_text())
    conf = json.loads((REPO / "reports" / "conformal_coverage_report.json").read_text())
    bench = json.loads((REPO / "reports" / "benchmark_comparison.json").read_text())
    return cv, conf, bench

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/DNA_Structure%2BKey%2BLabelled.pn_NoBB.png/220px-DNA_Structure%2BKey%2BLabelled.pn_NoBB.png", width=80)
    st.title("VARIANT-GNN")
    st.caption("TEKNOFEST 2026 · Sağlıkta Yapay Zeka")
    st.divider()
    st.markdown("**Takım:** XYRA3  \n**Kategori:** Üniversite ve Üzeri  \n**Görev:** Missense Varyant Patojenite Tahmini")
    st.divider()
    page = st.radio("Sayfa", ["🔬 Tahmin", "📊 Model Performansı", "🛡️ Konformal Garanti", "📚 Literatür Karşılaştırması", "🕸️ GNN Graf Açıklaması"])

# ═══════════════════════════════════════════════════════════════════════════════
# Sayfa 1 — Tahmin
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Tahmin":
    st.title("🧬 Varyant Patojenite Tahmini")
    st.caption("CSV yükleyerek veya örnek veri ile modeli test edin.")

    col1, col2 = st.columns([2, 1])

    with col1:
        upload_mode = st.radio("Veri Kaynağı", ["Örnek Veri", "CSV Yükle"], horizontal=True)

    if upload_mode == "CSV Yükle":
        uploaded = st.file_uploader("Varyant CSV dosyası yükle (etiketsiz, AL_x kolonları)", type="csv")
        if uploaded:
            df_input = pd.read_csv(uploaded)
        else:
            df_input = None
    else:
        sample_path = REPO / "data" / "samples" / "jury_blind_sample.csv"
        if sample_path.exists():
            df_input = pd.read_csv(sample_path)
            st.info(f"Örnek veri: {len(df_input)} varyant — `data/samples/jury_blind_sample.csv`")
        else:
            st.warning("Örnek veri bulunamadı.")
            df_input = None

    if df_input is not None:
        st.dataframe(df_input.head(5), use_container_width=True)

        if st.button("🚀 Tahmin Yap", type="primary"):
            try:
                preproc, ensemble, theta = load_pipeline()
                panel_col = "Panel" if "Panel" in df_input.columns else None

                feature_cols = [c for c in df_input.columns if c not in ["Panel", "label", "Variant_ID"]]
                X = df_input[feature_cols].values.astype(float)

                with st.spinner("Preprocessing ve tahmin..."):
                    X_p = preproc.transform(X)
                    # HybridEnsemble.predict → (labels, proba_2d). predict_proba YOK.
                    _, proba2d = ensemble.predict(X_p, threshold=theta)
                    probas = proba2d[:, 1]

                # KANONİK karar: tek GLOBAL θ (panel-bazlı eşik DEĞİL).
                preds = [1 if prob >= theta else 0 for prob in probas]

                result_df = df_input[["Panel"] if panel_col else []].copy()
                if "Variant_ID" in df_input.columns:
                    result_df.insert(0, "Variant_ID", df_input["Variant_ID"])
                result_df["pathogenic_probability"] = np.round(probas, 4)
                result_df["prediction_label"] = ["Patojenik" if p == 1 else "Benign" for p in preds]
                result_df["confidence"] = ["Yüksek" if abs(p - 0.5) > 0.3 else "Orta" if abs(p - 0.5) > 0.15 else "Düşük" for p in probas]

                st.success(f"✅ {len(result_df)} varyant tahmin edildi")

                pat_count = sum(preds)
                ben_count = len(preds) - pat_count
                m1, m2, m3 = st.columns(3)
                m1.metric("Toplam Varyant", len(preds))
                m2.metric("Patojenik", pat_count, delta=f"%{pat_count/len(preds)*100:.0f}")
                m3.metric("Benign", ben_count, delta=f"%{ben_count/len(preds)*100:.0f}")

                def color_pred(val):
                    if val == "Patojenik":
                        return "background-color: #fecaca; color: #991b1b"
                    return "background-color: #bbf7d0; color: #166534"

                styled = result_df.style.applymap(color_pred, subset=["prediction_label"])
                st.dataframe(styled, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Sonuçları İndir (CSV)", csv_out, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Hata: {e}")
                st.code(str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# Sayfa 2 — Model Performansı
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performansı":
    st.title("📊 Model Performansı")

    try:
        cv, _, _ = load_metrics()

        st.subheader("Genel Test Sonuçları")
        # NOT: cv_report.json'da test_mcc/test_pr_auc TOP-LEVEL DEĞİL; cv['test_metrics']
        # altındadır (eskiden cv['test_mcc'] → KeyError → sayfa boş kalıyordu).
        tm = cv.get("test_metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Binary F1 (birincil §7.3)",
            f"{cv.get('test_binary_f1', tm.get('binary_f1', 0)):.4f}",
            delta="iç hold-out F1=0.8367 @ θ=0.8415",
        )
        m2.metric("MCC", f"{tm.get('mcc', cv.get('test_mcc', 0)):.4f}")
        m3.metric("PR-AUC", f"{tm.get('pr_auc', cv.get('test_pr_auc', 0)):.4f}")
        m4.metric("5-CV F1", f"{cv.get('mean_cv_binary_f1', 0):.4f} ±{cv.get('std_cv_binary_f1', 0):.4f}")

        st.divider()
        st.subheader("Panel Bazlı Sonuçlar")
        st.caption("Karar eşiği: GLOBAL θ=0.8415 (canonical, §3.2). Test prior: %20-patojenik. Resmi jüri skoru (4-panel %20-F1 ort., CFTR dahil): 0.631 (3-panel tanı CFTR hariç: 0.6202). Havuzlanmış: 0.6042±0.0324.")

        panel_data = cv.get("panel_metrics", {})
        if panel_data:
            rows = []
            panel_labels = {
                "General": "MASTER (General)",
                "Hereditary_Cancer": "KANSER (Hereditary Cancer)",
                "PAH": "PAH",
                "CFTR": "CFTR",
            }
            for panel, metrics in panel_data.items():
                rows.append({
                    "Panel": panel_labels.get(panel, panel),
                    "F1": round(metrics.get("binary_f1", 0), 4),
                    "MCC": round(metrics.get("mcc", 0), 4),
                    "PR-AUC": round(metrics.get("pr_auc", 0), 4),
                    "Recall": round(metrics.get("recall", 0), 4),
                    "Eşik θ": round(metrics.get("threshold", 0), 3),
                })
            panel_df = pd.DataFrame(rows)
            st.dataframe(panel_df.style.highlight_max(subset=["F1", "MCC", "PR-AUC"], color="#bbf7d0"), use_container_width=True)
        else:
            # Canonical panel results @ θ=0.8415 (RESULTS_CANONICAL.json)
            data = {
                "Panel": ["MASTER (General)", "KANSER (Hereditary Cancer)", "PAH", "CFTR"],
                "F1": [0.8185, 0.9060, 0.9120, 0.7143],
                "MCC": [0.4951, 0.7135, 0.5053, float("nan")],
                "PR-AUC": [float("nan"), float("nan"), float("nan"), float("nan")],
                "Recall": [float("nan"), float("nan"), float("nan"), float("nan")],
                "Eşik θ": [0.8415, 0.8415, 0.8415, 0.8415],
            }
            df = pd.DataFrame(data)
            st.dataframe(df.style.highlight_max(subset=["F1", "MCC", "PR-AUC", "Recall"], color="#bbf7d0"), use_container_width=True)

        st.divider()
        st.subheader("Rapor Figürleri")
        fig_dir = REPO / "reports" / "figures" / "pdr"
        figs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
        if figs:
            cols = st.columns(2)
            for i, fig_path in enumerate(figs):
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_path.stem, use_container_width=True)
        else:
            st.info("Figür bulunamadı. `python scripts/generate_all_figures.py` çalıştırın.")

    except Exception as e:
        st.error(f"Metrikler yüklenemedi: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Sayfa 3 — Konformal Garanti
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Konformal Garanti":
    st.title("🛡️ Konformal Tahmin Garantileri")
    st.caption("Dağılımdan bağımsız (distribution-free) istatistiksel kapsama garantisi — APS yöntemi")

    try:
        _, conf, _ = load_metrics()

        st.info(
            "**Konformal Tahmin nedir?** Her karar için kümeler üretir. "
            "Hedef kapsamanın *en az* hedef değer kadar gerçekleşmesi matematiksel olarak garanti edilir."
        )

        # GERÇEK JSON şekli: per-level dict'ler conf['levels'] ALTINDA; anahtarlar
        # target / empirical_coverage / abstain_pct / valid_marginal (singleton_pct YOK).
        # Eskiden conf.items() (top-level) iteratlanıp olmayan anahtarlara erişiliyor +
        # 94.3/81.1/18.9 hardcoded yazılıyordu.
        levels = conf.get("levels", {})
        rows = []
        for level, data in levels.items():
            abstain = float(data.get("abstain_pct", 0.0))
            rows.append({
                "Güven Seviyesi": level,
                "Hedef Kapsama": f"%{float(data.get('target', 0)) * 100:.0f}",
                "Gerçekleşen Kapsama": f"%{float(data.get('empirical_coverage', 0)) * 100:.1f}",
                "Garanti ✓/✗": "✅ GEÇERLİ" if data.get("valid_marginal") else "❌ GEÇERSİZ",
                "Kesin Tahmin (Singleton)": f"%{max(0.0, 100.0 - abstain):.1f}",
                "Çift Küme (Abstain)": f"%{abstain:.1f}",
            })
        df_conf = pd.DataFrame(rows)

        def color_valid(val):
            return "color: #166534; font-weight: bold" if "GEÇERLİ" in str(val) else "color: #991b1b"

        st.dataframe(
            df_conf.style.applymap(color_valid, subset=["Garanti ✓/✗"]),
            use_container_width=True, hide_index=True
        )

        st.divider()
        st.subheader("Yorumlama")
        # DİNAMİK (hardcoded değerler kaldırıldı) — doğrudan levels'tan.
        cols = st.columns(max(1, len(levels)))
        for col, (level, data) in zip(cols, levels.items()):
            tgt = float(data.get("target", 0)) * 100
            emp = float(data.get("empirical_coverage", 0)) * 100
            valid = data.get("valid_marginal")
            col.metric(
                f"{level} Hedef",
                f"%{emp:.1f} Gerçekleşen",
                delta=f"{emp - tgt:+.1f}pp {'✅' if valid else '⚠️'}",
            )

        caveats = conf.get("honest_caveats")
        if caveats:
            if isinstance(caveats, (list, tuple)):
                st.markdown("**Dürüst notlar:**\n" + "\n".join(f"- {c}" for c in caveats))
            else:
                st.markdown(f"**Dürüst notlar:** {caveats}")

    except Exception as e:
        st.error(f"Konformal sonuçlar yüklenemedi: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Sayfa 4 — Literatür Karşılaştırması
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Literatür Karşılaştırması":
    st.title("📚 Literatür Karşılaştırması")
    st.caption("VARIANT-GNN vs REVEL, CADD, EVE, MutPred2, ClinPred")

    try:
        _, _, bench = load_metrics()

        rows = []
        for m in bench["models"]:
            rows.append({
                "Model": m["name"].replace("\n", " "),
                "Binary F1": m["f1"],
                "MCC": m["mcc"],
                "PR-AUC": m["pr_auc"],
                "Notlar": m.get("note", ""),
                "Bizim Modelimiz": "★" if m.get("ours") else "†",
            })
        df_bench = pd.DataFrame(rows).sort_values("Binary F1", ascending=False)

        def highlight_ours(row):
            if row["Bizim Modelimiz"] == "★":
                return ["background-color: #dcfce7; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_bench.style.apply(highlight_ours, axis=1),
            use_container_width=True, hide_index=True
        )

        st.caption("★ = TEKNOFEST 2026 gerçek verisi  |  † = Literatür değeri (farklı test seti)")
        st.divider()

        bench_fig = REPO / "reports" / "figures" / "pdr" / "13_benchmark_comparison.png"
        if bench_fig.exists():
            st.image(str(bench_fig), use_container_width=True)

        st.info(
            "⚠️ Literatür değerleri farklı test veri setlerinde ölçülmüştür. "
            "VARIANT-GNN değerleri TEKNOFEST 2026 resmi eğitim verisinin "
            "hold-out test bölümünde (%20) elde edilmiştir."
        )

    except Exception as e:
        st.error(f"Benchmark verisi yüklenemedi: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Sayfa 5 — GNN Graf Açıklaması (pyvis interaktif)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ GNN Graf Açıklaması":
    st.title("🕸️ GNN Komşuluk Grafiği — İnteraktif Açıklanabilirlik")
    st.caption("GATv2Conv modelinin en yüksek ağırlık verdiği komşu varyantlar (pyvis)")

    try:
        from pyvis.network import Network
        import tempfile, os

        st.info(
            "**Nasıl çalışır?** Cosine k-NN grafiğinde her varyant bir düğümdür. "
            "GNNExplainer'ın kenar maskeleri, hangi komşuların tahmine en çok katkı "
            "sağladığını gösterir. Kırmızı = patojenik, yeşil = benign."
        )

        n_nodes = st.slider("Gösterilecek varyant sayısı", 5, 30, 12)
        k_neighbors = st.slider("k-NN komşu sayısı", 3, 8, 5)

        if st.button("🔍 Graf Oluştur", type="primary"):
            with st.spinner("Graf hesaplanıyor..."):
                preproc, ensemble, theta = load_pipeline()
                import numpy as np
                rng = np.random.default_rng(42)

                # Demo: rastgele örnek nokta bulutu (gerçek veri yüklenemiyorsa)
                sample_path = REPO / "data" / "samples" / "jury_blind_sample.csv"
                if sample_path.exists():
                    df_demo = pd.read_csv(sample_path)
                    feat_cols = [c for c in df_demo.columns
                                 if c not in ["Panel", "label", "Variant_ID", "Label"]]
                    X_demo = df_demo[feat_cols].values.astype(float)
                    var_ids = (df_demo["Variant_ID"].tolist()
                               if "Variant_ID" in df_demo.columns
                               else [f"VAR_{i}" for i in range(len(df_demo))])
                else:
                    X_demo = rng.standard_normal((n_nodes, 50))
                    var_ids = [f"VAR_{i:03d}" for i in range(n_nodes)]

                X_proc = preproc.transform(X_demo[:n_nodes])
                probas = ensemble.predict(X_proc)[1][:, 1]

                # Cosine similarity graf
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(X_proc)
                np.fill_diagonal(sim_matrix, 0)

                net = Network(height="520px", width="100%",
                              bgcolor="#0f1117", font_color="white",
                              notebook=False, directed=False)
                net.set_options("""
                {
                  "physics": {"stabilization": {"iterations": 80}},
                  "nodes": {"borderWidth": 2, "size": 20},
                  "edges": {"smooth": {"type": "continuous"}}
                }
                """)

                for i, (vid, prob) in enumerate(zip(var_ids[:n_nodes], probas)):
                    color = "#ef4444" if prob >= theta else "#22c55e"  # kanonik global θ (0.5 değil)
                    label_str = "P" if prob >= theta else "B"
                    net.add_node(i, label=f"{label_str}\n{prob:.2f}",
                                 title=f"{vid}\nP(patojenik)={prob:.3f}",
                                 color=color, size=18 + int(prob * 10))

                top_k = min(k_neighbors, n_nodes - 1)
                for i in range(n_nodes):
                    neighbors = np.argsort(sim_matrix[i])[::-1][:top_k]
                    for j in neighbors:
                        if i < j:
                            weight = float(sim_matrix[i, j])
                            net.add_edge(i, int(j),
                                         value=weight,
                                         title=f"cosine={weight:.3f}",
                                         color="#6b7280" if weight < 0.7 else "#a78bfa")

                with tempfile.NamedTemporaryFile(suffix=".html",
                                                  delete=False, mode="w") as f:
                    net.save_graph(f.name)
                    html_path = f.name

                with open(html_path, "r") as f:
                    html_content = f.read()
                os.unlink(html_path)

                st.components.v1.html(html_content, height=540, scrolling=False)

                col1, col2 = st.columns(2)
                col1.metric("Patojenik (kırmızı)", sum(1 for p in probas if p >= 0.5))
                col2.metric("Benign (yeşil)", sum(1 for p in probas if p < 0.5))
                st.caption("Düğüm boyutu patojenik olasılıkla orantılı. "
                           "Mor kenar = yüksek benzerlik (cosine > 0.7).")

    except ImportError:
        st.error("pyvis kütüphanesi gerekli: `pip install pyvis`")
    except Exception as e:
        st.error(f"Graf oluşturulamadı: {e}")
        st.code(str(e))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("VARIANT-GNN · Takım XYRA3 · TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması · Yalnızca araştırma ve eğitim amaçlıdır · Resmi jüri skoru: 0.631 (4-panel, CFTR dahil; 3-panel tanı 0.6202; %20-patojenik test prior)")

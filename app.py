import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.config import Config
from src.data_processing import TabularGraphPreprocessor
from src.models import FeatureGNN, VariantHybridModel, VariantDNN
from src.utils import ModelSerializer
from src.train import evaluate_gnn_epoch
from src.explain import FeatureExplainer

# Sayfa Yapılandırması
st.set_page_config(
    page_title="VARIANT-GNN Analiz Paneli",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS: Temel stil iyileştirmeleri
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .pathogenic-badge {
        background-color: #c0392b;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .benign-badge {
        background-color: #27ae60;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .section-header {
        border-left: 4px solid #2d6a9f;
        padding-left: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Modelleri ve Preprocessor'ı yükleyen (önbelleğe alan) yardımcı fonksiyon."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_model = VariantHybridModel(weights=Config.ENSEMBLE_WEIGHTS)
    hybrid_model.gnn = FeatureGNN(
        in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2
    ).to(device)

    try:
        preprocessor = ModelSerializer.load_models(hybrid_model)

        # DNN — gerçek boyutu ölçmek için dummy 1-satırlı transform kullan
        dummy_in = pd.DataFrame(np.zeros((1, 34)))
        actual_dim = preprocessor.transform(dummy_in).shape[1]
        hybrid_model.dnn = VariantDNN(input_dim=actual_dim, num_classes=2).to(device)

        # DNN ağırlıklarını yükle (varsa)
        if os.path.exists(Config.DNN_MODEL_PATH):
            hybrid_model.dnn.load_state_dict(
                torch.load(Config.DNN_MODEL_PATH, map_location=device)
            )

        return hybrid_model, preprocessor, device
    except Exception as e:
        st.error(f"❌ Modeller yüklenemedi: {e}\n\nLütfen terminalde şunu çalıştırın: `python main.py --mode train`")
        return None, None, device


def run_inference(model, preprocessor, df_features, device):
    """
    Yüklenen özellik DataFrame'inden ensemble tahmin üretir.
    Returns:
        df_result: tahmin + risk skoru eklenmiş DataFrame
        X_scaled:  ölçeklendirilmiş numpy array (XAI için)
        ensemble_probs: numpy array (XAI için)
    """
    X_scaled = preprocessor.transform(df_features)
    graphs   = [preprocessor.row_to_graph(row) for row in X_scaled]
    loader   = DataLoader(graphs, batch_size=32, shuffle=False)

    # Model tahminleri
    xgb_probs = model.predict_xgb_proba(X_scaled)

    _, _, gnn_probs_list = evaluate_gnn_epoch(model.gnn.to(device), loader, device)
    gnn_probs = np.array(gnn_probs_list)

    model.dnn.eval()
    with torch.no_grad():
        dnn_out   = model.dnn(torch.FloatTensor(X_scaled).to(device))
        dnn_probs = torch.softmax(dnn_out, dim=1).cpu().numpy()

    ensemble_probs = model.predict_ensemble(xgb_probs, gnn_probs, dnn_probs)
    preds          = np.argmax(ensemble_probs, axis=1)
    risk_scores    = model.get_clinical_risk_score(ensemble_probs)

    df_result = df_features.copy()
    df_result['Tahmin']                = ['🔴 Patojenik' if p == 1 else '🟢 Benign' for p in preds]
    df_result['Klinik Risk Skoru (%)'] = risk_scores.round(2)
    df_result['Güven Skoru (%)']       = (np.max(ensemble_probs, axis=1) * 100).round(2)

    return df_result, X_scaled, ensemble_probs


def main():
    # ──────────────────────────────────────────────
    # Başlık
    # ──────────────────────────────────────────────
    st.title("🧬 VARIANT-GNN: Genetik Varyant Patojenite Analizi")
    st.markdown("""
    Bu arayüz, eğitilmiş **VARIANT-GNN** modellerini kullanarak genomik varyantların
    patojenite potansiyelini etkileşimli olarak analiz eder.

    > ⚠️ *Bu sistem yalnızca araştırma ve eğitim amaçlıdır. Klinik karar için yeterli değildir.*
    """)

    # ──────────────────────────────────────────────
    # Sidebar
    # ──────────────────────────────────────────────
    st.sidebar.header("📊 Sistem Bilgisi")
    st.sidebar.info("🤖 Model: Hybrid XGBoost + Feature-GNN + DNN\n\n📐 Ensemble Ağırlıkları (XGB/GNN/DNN): {}".format(Config.ENSEMBLE_WEIGHTS))
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ XAI Seçenekleri")
    show_shap_global  = st.sidebar.checkbox("🌍 Global SHAP Özet", value=True)
    show_shap_local   = st.sidebar.checkbox("🔍 Yerel SHAP Waterfall", value=True)
    show_lime         = st.sidebar.checkbox("🟡 LIME Açıklaması", value=True)
    show_gnn_xai      = st.sidebar.checkbox("🕸️ GNN Graf Açıklanabilirliği", value=False)
    selected_variant  = st.sidebar.number_input("Yerel XAI için Varyant No:", min_value=0, value=0, step=1)

    # ──────────────────────────────────────────────
    # Model Yükleme
    # ──────────────────────────────────────────────
    model, preprocessor, device = load_models()

    if model is None or preprocessor is None:
        st.stop()

    # ──────────────────────────────────────────────
    # Veri Yükleme
    # ──────────────────────────────────────────────
    st.subheader("📂 Veri Yükleme")
    uploaded_file = st.file_uploader(
        "Varyant özelliklerini içeren CSV dosyasını yükleyin",
        type=["csv"],
        help="Sütunlar: sayısal özellikler (Label sütunu opsiyonel — varsa görmezden gelinir)"
    )

    if uploaded_file is None:
        st.info("💡 CSV dosyası yükleyin. `data/` klasöründe örnek formatı inceleyebilirsiniz.")
        return

    df = pd.read_csv(uploaded_file)

    # Label sütununu çıkar (varsa)
    label_col = None
    if 'Label' in df.columns:
        label_col = df['Label'].copy()
        df = df.drop(columns=['Label'])

    # Sayısal olmayan sütunları çıkar (örn. Variant_ID gibi ID sütunları)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.caption(f"ℹ️ Sayısal olmayan sütunlar analiz dışı bırakıldı: {', '.join(non_numeric_cols)}")
        df = df.drop(columns=non_numeric_cols)


    st.write("**Yüklenen Veri Önizlemesi (İlk 5 Satır):**")
    st.dataframe(df.head())
    st.caption(f"📋 {len(df)} varyant | {df.shape[1]} özellik")

    if st.button("🚀 Analizi Başlat", type="primary"):
        with st.spinner("⏳ Yapay zeka modelleri varyantları işliyor..."):
            try:
                df_result, X_scaled, ensemble_probs = run_inference(model, preprocessor, df, device)
            except Exception as e:
                st.error(f"❌ Analiz sırasında hata: {e}")
                st.stop()

        st.success("✅ Analiz Tamamlandı!")

        # ──────────────────────────────────────────────
        # Özet Metrikler
        # ──────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Genel Özet")
        preds_raw = (np.argmax(ensemble_probs, axis=1))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Patojenik", int((preds_raw == 1).sum()))
        with col2:
            st.metric("🟢 Benign", int((preds_raw == 0).sum()))
        with col3:
            st.metric("📊 Ort. Risk Skoru", f"{df_result['Klinik Risk Skoru (%)'].mean():.1f}%")
        with col4:
            st.metric("🎯 Ort. Güven", f"{df_result['Güven Skoru (%)'].mean():.1f}%")

        # ──────────────────────────────────────────────
        # Sonuç Tablosu + İndirme
        # ──────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Detaylı Sonuçlar ve Klinik Karar Destek")

        priority_cols = ['Tahmin', 'Klinik Risk Skoru (%)', 'Güven Skoru (%)']
        other_cols    = [c for c in df_result.columns if c not in priority_cols]
        st.dataframe(
            df_result[priority_cols + other_cols],
            use_container_width=True,
            height=350
        )

        csv_data = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Sonuçları İndir (.csv)",
            csv_data,
            "analiz_sonuclari.csv",
            "text/csv"
        )

        # ──────────────────────────────────────────────
        # XAI Bölümü
        # ──────────────────────────────────────────────
        st.divider()
        st.subheader("🧠 Açıklanabilir Yapay Zeka (XAI) Paneli")

        # Gerçek sütun isimlerini kullan — AutoEncoder gizli özellikleri sonuna 'AE_Latent_N' şeklinde eklenir
        original_cols = list(df.columns)
        # df_dummy = generate_dummy_data(100, 34)  # 34 yeni özellik- len(original_cols) # This line was not added as it introduces an undefined function and is not part of the original code's logic for dummy data generation.
        ae_dim = X_scaled.shape[1] - len(original_cols)
        ae_cols = [f"AE_Latent_{i}" for i in range(ae_dim)] if ae_dim > 0 else []
        feat_names = original_cols + ae_cols

        explainer  = FeatureExplainer(
            xgb_model=model.xgb,
            feature_names=feat_names,
            training_data=X_scaled
        )

        # Seçili varyant indeks kontrolü
        var_idx = min(int(selected_variant), len(X_scaled) - 1)

        # ── Global SHAP ──
        if show_shap_global:
            st.markdown("#### 🌍 Global SHAP — Modelin Genel Karar Haritası")
            st.caption("Tüm varyantlar üzerinde hangi özellikler kararları şekillendiriyor?")
            with st.spinner("SHAP hesaplanıyor..."):
                shap_path = "/tmp/variant_shap_summary.png"
                explainer.plot_summary(X_scaled, filename=shap_path)
                if os.path.exists(shap_path):
                    st.image(shap_path, use_column_width=True)

            # Top features tablo
            top_feats = explainer.get_top_features(X_scaled, top_n=10)
            feat_df = pd.DataFrame(top_feats, columns=["Özellik", "Ort. SHAP Değeri"])
            st.dataframe(feat_df, use_container_width=True)

        st.markdown("---")

        # ── Yerel SHAP ──
        if show_shap_local:
            st.markdown(f"#### 🔍 Yerel SHAP — Varyant #{var_idx} için Karar Açıklaması")
            st.caption("Bu varyant için hangi özellikler patojenik/benign kararına yol açtı?")
            with st.spinner("Waterfall hesaplanıyor..."):
                wf_path = "/tmp/variant_shap_waterfall.png"
                try:
                    explainer.plot_waterfall(X_scaled[var_idx], filename=wf_path)
                    if os.path.exists(wf_path):
                        st.image(wf_path, use_column_width=True)
                except Exception as e:
                    st.warning(f"Waterfall grafiği oluşturulamadı: {e}")

        st.markdown("---")

        # ── LIME ──
        if show_lime:
            st.markdown(f"#### 🟡 LIME — Varyant #{var_idx} Yerel Lineer Açıklama")
            st.caption("Seçili varyant çevresinde modelin nasıl davrandığını gösterir.")
            with st.spinner("LIME hesaplanıyor (bu 10-30 saniye sürebilir)..."):
                lime_html = "/tmp/variant_lime.html"
                lime_bar  = "/tmp/variant_lime_bar.png"
                try:
                    # Önce bar grafiği üret (LIME tek seferde 500 örnekle çalışır)
                    explainer.plot_lime_bar(
                        X_reference=X_scaled,
                        instance_index=var_idx,
                        filename=lime_bar
                    )
                    # HTML için aynı iç explainer nesnesini yeniden kullan (çift hesap yok)
                    explainer.explain_with_lime(
                        X_reference=X_scaled,
                        instance_index=var_idx,
                        feature_names=feat_names,
                        output_filename=lime_html
                    )

                    tab1, tab2 = st.tabs(["📊 Bar Grafiği", "🌐 Etkileşimli HTML"])
                    with tab1:
                        if os.path.exists(lime_bar):
                            st.image(lime_bar, use_column_width=True)
                    with tab2:
                        if os.path.exists(lime_html):
                            html_content = open(lime_html, "r", encoding="utf-8").read()
                            st.components.v1.html(html_content, height=450, scrolling=True)
                except Exception as e:
                    st.warning(f"⚠️ LIME raporu oluşturulamadı: {e}")

        st.markdown("---")

        # ── GNN Graph Explainability ──
        if show_gnn_xai:
            st.markdown(f"#### 🕸️ GNN Graf Açıklanabilirliği — Varyant #{var_idx}")
            st.caption("Hangi özellik bağlantıları (kenarlar) GNN kararını etkiledi?")
            with st.spinner("GNNExplainer çalışıyor (bu dakikalar sürebilir)..."):
                gnn_xai_path = "/tmp/variant_gnn_xai.png"
                try:
                    graph_data = preprocessor.row_to_graph(X_scaled[var_idx])
                    explainer.plot_gnn_explanation(
                        gnn_model=model.gnn,
                        graph_data=graph_data,
                        device=str(device),
                        feature_names=feat_names,
                        filename=gnn_xai_path
                    )
                    if os.path.exists(gnn_xai_path):
                        st.image(gnn_xai_path, use_column_width=True)
                    else:
                        st.info("GNN açıklaması üretilemedi. torch_geometric sürümünüzü kontrol edin.")
                except Exception as e:
                    st.warning(f"⚠️ GNN XAI hatası: {e}")


if __name__ == "__main__":
    main()

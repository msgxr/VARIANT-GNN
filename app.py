import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.config import Config
from src.data_processing import TabularGraphPreprocessor
from src.models import FeatureGNN, VariantHybridModel, VariantDNN
from src.utils import ModelSerializer
from src.train import evaluate_gnn_epoch
from src.explain import FeatureExplainer

# Sayfa Yapılandırması
st.set_page_config(page_title="VARIANT-GNN Analiz Paneli", layout="wide")

def load_models():
    """Modelleri ve Preprocessor'ı yükleyen yardımcı fonksiyon"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_model = VariantHybridModel(weights=Config.ENSEMBLE_WEIGHTS)
    hybrid_model.gnn = FeatureGNN(in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2).to(device)
    hybrid_model.dnn = VariantDNN(input_dim=Config.GNN_HIDDEN_DIM, num_classes=2).to(device) # input_dim train sonrası preprocessor'dan gelir
    
    try:
        preprocessor = ModelSerializer.load_models(hybrid_model)
        # DNN input_dim'i preprocessor'dan sonra güncelle
        input_dim = preprocessor.transform(pd.DataFrame(np.zeros((1, 40)))).shape[1]
        hybrid_model.dnn = VariantDNN(input_dim=input_dim).to(device)
        ModelSerializer.load_models(hybrid_model) # Tekrar yükle (DNN ağırlıkları için)
        return hybrid_model, preprocessor, device
    except Exception as e:
        st.error(f"Modeller yüklenemedi: {e}. Lütfen önce '--mode train' çalıştırın.")
        return None, None, device

def main():
    st.title("🧬 VARIANT-GNN: Genetik Varyant Patojenite Analizi")
    st.markdown("""
    Bu arayüz, eğitilmiş **VARIANT-GNN** modellerini kullanarak genomik varyantların patojenite potansiyelini 
    etkileşimli olarak analiz etmenizi sağlar. 
    *Uyarı: Bu sistem yalnızca araştırma ve eğitim amaçlıdır.*
    """)

    # Sidebar: Bilgi ve Ayarlar
    st.sidebar.header("📊 Sistem Bilgisi")
    st.sidebar.info("Model: Hybrid XGBoost + Feature-GNN + DNN")
    st.sidebar.write(f"Ağırlıklar (XGB/GNN/DNN): {Config.ENSEMBLE_WEIGHTS}")
    
    # Modelleri Yükle
    model, preprocessor, device = load_models()
    
    if model and preprocessor:
        st.subheader("📂 Veri Yükleme")
        uploaded_file = st.file_uploader("Varyant özelliklerini içeren CSV dosyasını yükleyin", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Yüklenen Veri Önizleme (İlk 5 Satır):")
            st.dataframe(df.head())

            if st.button("Analizi Başlat"):
                with st.spinner("Yapay zeka modelleri varyantları işliyor..."):
                    # 1. Ön İşleme
                    X_scaled = preprocessor.transform(df)
                    graphs = [preprocessor.row_to_graph(row) for row in X_scaled]
                    loader = DataLoader(graphs, batch_size=32, shuffle=False)
                    
                    # 2. Tahminler
                    xgb_probs = model.predict_xgb_proba(X_scaled)
                    _, _, gnn_probs_list = evaluate_gnn_epoch(model.gnn.to(device), loader, device)
                    gnn_probs = np.array(gnn_probs_list)
                    
                    # DNN Probs
                    model.dnn.eval()
                    with torch.no_grad():
                        dnn_out = model.dnn(torch.FloatTensor(X_scaled).to(device))
                        dnn_probs = torch.softmax(dnn_out, dim=1).cpu().numpy()
                    
                    ensemble_probs = model.predict_ensemble(xgb_probs, gnn_probs, dnn_probs)
                    preds = np.argmax(ensemble_probs, axis=1)
                    risk_scores = model.get_clinical_risk_score(ensemble_probs)

                    # 3. Sonuçları Tabloya Ekle
                    df['Tahmin'] = ['Patojenik' if p == 1 else 'Benign' for p in preds]
                    df['Klinik Risk Skoru (%)'] = risk_scores.round(2)
                    
                    # Renklendirme ve Görselleştirme
                    st.success("Analiz Tamamlandı!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📈 Özet İstatistikler")
                        st.write(df['Tahmin'].value_counts())
                        
                    with col2:
                        st.subheader("📥 Çıktı Dosyası")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Sonuçları İndir (.csv)", csv, "analiz_sonuclari.csv", "text/csv")

                    st.divider()
                    st.subheader("🔍 Detaylı Analiz ve Klinik Karar Destek")
                    st.dataframe(df[['Tahmin', 'Klinik Risk Skoru (%)'] + [c for c in df.columns if c not in ['Tahmin', 'Klinik Risk Skoru (%)']]])

                    # 4. Şeffaflık (XAI)
                    st.divider()
                    st.subheader("🧠 Çok Kanallı Açıklanabilir Yapay Zeka (XAI)")
                    
                    xai_col1, xai_col2 = st.columns(2)
                    
                    with xai_col1:
                        st.write("**SHAP (Küresel Önem):**")
                        explainer = FeatureExplainer(model.xgb, feature_names=df.drop(columns=['Tahmin', 'Klinik Risk Skoru (%)']).columns.tolist())
                        explainer.plot_summary(X_scaled, filename="temp_shap.png")
                        st.image("temp_shap.png")
                        os.remove("temp_shap.png")

                    with xai_col2:
                        st.write("**LIME (Yerel Tahmin Açıklaması - İlk Varyant):**")
                        try:
                            # LIME genelde biraz daha yavaştır
                            explainer.explain_with_lime(X_scaled, feature_names=df.drop(columns=['Tahmin', 'Klinik Risk Skoru (%)']).columns.tolist(), output_filename="temp_lime.html")
                            st.components.v1.html(open("temp_lime.html", "r").read(), height=400, scrolling=True)
                            os.remove("temp_lime.html")
                        except Exception as e:
                            st.warning(f"LIME raporu oluşturulamadı: {e}")

if __name__ == "__main__":
    main()

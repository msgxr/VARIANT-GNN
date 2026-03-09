import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.config import Config
from src.data_processing import TabularGraphPreprocessor, generate_dummy_data # Note: moved dummy to processing or keep it here
from src.models import FeatureGNN, VariantHybridModel
from src.train import ModelTrainer, evaluate_gnn_epoch
from src.evaluate import evaluate_predictions, plot_confusion_matrix
from src.explain import FeatureExplainer
from src.tune import ModelTuner
from src.utils import ModelSerializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data():
    """Mock veri üreten jenerik yükleyici (Şartname gerçeğine en yakın simulasyon)"""
    df = generate_dummy_data(n_samples=2500, n_features=40)
    df_labels_mapped = df.copy()
    df_labels_mapped['Label'] = df_labels_mapped['Label'].map({'Pathogenic': 1, 'Benign': 0})
    y = df_labels_mapped['Label'].values
    X_df = df_labels_mapped.drop(columns=['Label'])
    return X_df, y

def main():
    parser = argparse.ArgumentParser(description="VARIANT-GNN: Advanced Graph-based Variant Pathogenicity Prediction")
    parser.add_argument("--mode", type=str, choices=["train", "tune", "eval", "predict"], default="train",
                        help="Sistemi hangi modda çalıştıracağınızı belirler. (train/tune/eval/predict)")
    parser.add_argument("--test_file", type=str, default="test_variants.csv",
                        help="Tahmin edilecek CSV dosyasının harici yolu (--mode predict için)")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.create_dirs()

    logging.info("="*60)
    logging.info(f"🚀 VARIANT-GNN BAŞLATILDI | MOD: {args.mode.upper()} 🚀")
    logging.info("="*60)

    X_df, y = get_data()
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)

    if args.mode == "tune":
        logging.info("🔍 Hiperparametre Optimizasyonu (Optuna) Başlatılıyor...")
        preprocessor = TabularGraphPreprocessor(corr_threshold=Config.CORR_THRESHOLD, use_autoencoder=False, device=device)
        X_train_scaled = preprocessor.fit_transform(X_train_df)
        
        tuner = ModelTuner(X_train_scaled, y_train, n_trials=10) # Hız için 10, artırılabilir
        best_xgb_params = tuner.optimize_xgboost()
        logging.info(f"Tuning işlemi bitti. En iyi XGB parametreleri: {best_xgb_params}")
        # Not: Config.XGB_PARAMS güncellenerek eğitim aşamasında kullanılabilir.

    elif args.mode == "train":
        logging.info("🏋️ Modeller Eğitiliyor (AutoEncoder + GNN + XGBoost)...")
        
        # 1. Feature Engineering (AutoEncoder dahil)
        preprocessor = TabularGraphPreprocessor(
            corr_threshold=Config.CORR_THRESHOLD, 
            use_autoencoder=True, 
            encoding_dim=16, 
            device=device
        )
        X_train_scaled = preprocessor.fit_transform(X_train_df)
        X_test_scaled = preprocessor.transform(X_test_df)
        
        # 2. PyTorch DataLoaderlar
        train_graphs = [preprocessor.row_to_graph(row, label=l) for row, l in zip(X_train_scaled, y_train)]
        test_graphs = [preprocessor.row_to_graph(row, label=l) for row, l in zip(X_test_scaled, y_test)]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
        # 3. XGBoost Dalı
        logging.info("XGBoost Modeli Eğitiliyor...")
        hybrid_model = VariantHybridModel(xgb_params=Config.XGB_PARAMS, gnn_weight=Config.GNN_ENSEMBLE_WEIGHT)
        hybrid_model.fit_xgb(X_train_scaled, y_train)
        
        # 4. GNN Dalı
        logging.info("GNN Modeli Eğitiliyor...")
        trainer = ModelTrainer(device=device)
        gnn_in_channels = X_train_scaled.shape[1]
        gnn_model = FeatureGNN(in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2).to(device)
        
        # GNN Node Özellik boyutu 1, ama kanal sayısı feature sayısı değil o yüzden in_channels=1
        trained_gnn, best_acc = trainer.train_gnn(train_loader, test_loader, model=gnn_model, epochs=Config.GNN_EPOCHS, lr=Config.GNN_LR)
        hybrid_model.gnn = trained_gnn
        
        # 5. Kayıt
        ModelSerializer.save_models(hybrid_model, preprocessor)
        
        # 6. Evalüasyon
        logging.info("✅ Eğitim Tamamlandı. Sonuçlar Hesaplanıyor...")
        xgb_probs = hybrid_model.predict_xgb_proba(X_test_scaled)
        _, _, gnn_probs_list = evaluate_gnn_epoch(trained_gnn, test_loader, device)
        gnn_probs = np.array(gnn_probs_list)
        
        ensemble_probs = hybrid_model.predict_ensemble(xgb_probs, gnn_probs)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        evaluate_predictions(y_test, ensemble_preds, ensemble_probs)
        plot_confusion_matrix(y_test, ensemble_preds, filename=os.path.join(Config.REPORTS_DIR, "train_confusion_matrix.png"))
        
        # XAI
        explainer = FeatureExplainer(hybrid_model.xgb)
        explainer.plot_summary(X_test_scaled, filename=os.path.join(Config.REPORTS_DIR, "shap_summary.png"))
        logging.info(f"Raporlar ve açıklamalar kaydedildi: {Config.REPORTS_DIR}")

    elif args.mode == "eval":
        logging.info("Analiz ve Prediksiyon moduna geçiliyor (Sadece Raporlama)...")
        try:
            hybrid_model = VariantHybridModel(gnn_weight=Config.GNN_ENSEMBLE_WEIGHT)
            hybrid_model.gnn = FeatureGNN(in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2).to(device)
            preprocessor = ModelSerializer.load_models(hybrid_model)
            logging.info("Modeller başarıyla yüklendi. (Eval Mode)")
        except FileNotFoundError:
            logging.error("Önce '--mode train' ile model üretmelisiniz!")
            
    elif args.mode == "predict":
        logging.info(f"Kör Tahmin (Inference) modu başlatıldı. Veri: {args.test_file}")
        try:
            hybrid_model = VariantHybridModel(gnn_weight=Config.GNN_ENSEMBLE_WEIGHT)
            hybrid_model.gnn = FeatureGNN(in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2).to(device)
            preprocessor = ModelSerializer.load_models(hybrid_model)
            
            # Yarışma formatındaki features dosyasının okunması (Dummy data örneği)
            # Normalde: inference_df = pd.read_csv(args.test_file)
            inference_df = generate_dummy_data(n_samples=500, n_features=40)
            X_infer = inference_df.drop(columns=['Label'], errors='ignore')
            
            X_infer_scaled = preprocessor.transform(X_infer)
            infer_graphs = [preprocessor.row_to_graph(row) for row in X_infer_scaled]
            infer_loader = DataLoader(infer_graphs, batch_size=32, shuffle=False)
            
            # Tahminler
            xgb_probs = hybrid_model.predict_xgb_proba(X_infer_scaled)
            _, _, gnn_probs_list = evaluate_gnn_epoch(hybrid_model.gnn.to(device), infer_loader, device)
            gnn_probs = np.array(gnn_probs_list)
            
            ensemble_probs = hybrid_model.predict_ensemble(xgb_probs, gnn_probs)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            # Submission Dosyasının Oluşturulması
            submission = pd.DataFrame({
                'Variant_ID': [f"VAR_{i}" for i in range(len(ensemble_preds))],
                'Prediction': ['Pathogenic' if p == 1 else 'Benign' for p in ensemble_preds],
                'Confidence': np.max(ensemble_probs, axis=1)
            })
            
            sub_path = os.path.join(Config.REPORTS_DIR, 'submission.csv')
            submission.to_csv(sub_path, index=False)
            logging.info(f"✅ Tahminler tamamlandı. Teslim dosyası hazırlandı: {sub_path}")
            
        except FileNotFoundError:
            logging.error("Lütfen önce modeli eğitin (--mode train)")

if __name__ == "__main__":
    main()

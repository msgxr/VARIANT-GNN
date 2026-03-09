import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.config import Config
from src.models import FeatureGNN, VariantHybridModel, VariantDNN
from src.train import ModelTrainer, evaluate_gnn_epoch
from src.data_processing import TabularGraphPreprocessor, generate_dummy_data, load_and_prepare_data, plot_performance_curves
from src.evaluate import evaluate_predictions, plot_confusion_matrix
from src.explain import FeatureExplainer
from src.tune import ModelTuner
from src.utils import ModelSerializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_data(data_path=None):
    """
    Veri yükleyici:
    - Eğer gerçek bir CSV yolu verilirse onu kullanır.
    - Yoksa sentetik dummy veri üretir (geliştirme/test için).
    """
    if data_path and os.path.exists(data_path):
        logging.info(f"Gerçek veri seti yükleniyor: {data_path}")
        X_df, y = load_and_prepare_data(data_path)
        if y is None:
            raise ValueError("Etiketli veri seti bekleniyor. CSV dosyasında 'Label' sütunu bulunamadı.")
        return X_df, y
    else:
        logging.warning("Gerçek veri seti bulunamadı. Sentetik veri ile devam ediliyor (40 özellik, 2500 örnek).")
        df = generate_dummy_data(n_samples=2500, n_features=40)
        df_labels_mapped = df.copy()
        df_labels_mapped['Label'] = df_labels_mapped['Label'].map({'Pathogenic': 1, 'Benign': 0})
        y = df_labels_mapped['Label'].values
        X_df = df_labels_mapped.drop(columns=['Label'])
        return X_df, y


def build_dnn_loaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size=32):
    """DNN için düz tabular TensorDataset loader'ları oluşturur."""
    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)
    )
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    )


def get_ensemble_probs(hybrid_model, X_scaled, gnn_loader, dnn_model, device):
    """
    XGBoost + GNN + DNN'den olasılık vektörlerini toplar ve ensemble tahmin üretir.
    Returns:
        ensemble_probs (np.array), preds (np.array), risk_scores (np.array)
    """
    xgb_probs = hybrid_model.predict_xgb_proba(X_scaled)

    _, _, gnn_probs_list = evaluate_gnn_epoch(hybrid_model.gnn.to(device), gnn_loader, device)
    gnn_probs = np.array(gnn_probs_list)

    dnn_model.eval()
    with torch.no_grad():
        dnn_out = dnn_model(torch.FloatTensor(X_scaled).to(device))
        dnn_probs = torch.softmax(dnn_out, dim=1).cpu().numpy()

    ensemble_probs = hybrid_model.predict_ensemble(xgb_probs, gnn_probs, dnn_probs)
    preds = np.argmax(ensemble_probs, axis=1)
    risk_scores = hybrid_model.get_clinical_risk_score(ensemble_probs)
    return ensemble_probs, preds, risk_scores


def main():
    parser = argparse.ArgumentParser(description="VARIANT-GNN: Advanced Graph-based Variant Pathogenicity Prediction")
    parser.add_argument("--mode", type=str,
                        choices=["train", "tune", "eval", "predict", "crossval"],
                        default="train",
                        help="Sistemi hangi modda çalıştıracağınızı belirler.")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Eğitim verisi CSV dosyası yolu (opsiyonel). Belirtilmezse sentetik veri kullanılır.")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Tahmin edilecek CSV dosyasının yolu (--mode predict için).")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.create_dirs()

    logging.info("=" * 60)
    logging.info(f"🚀 VARIANT-GNN BAŞLATILDI | MOD: {args.mode.upper()} | CİHAZ: {device} 🚀")
    logging.info("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # TUNE MODU
    # ─────────────────────────────────────────────────────────────
    if args.mode == "tune":
        logging.info("🔍 Hiperparametre Optimizasyonu (Optuna) Başlatılıyor...")
        X_df, y = get_data(args.data_file)
        X_train_df, _, y_train, _ = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)

        preprocessor = TabularGraphPreprocessor(
            corr_threshold=Config.CORR_THRESHOLD, use_autoencoder=False, device=device
        )
        X_train_scaled, y_train = preprocessor.fit_transform(X_train_df, label_y=y_train)

        tuner = ModelTuner(X_train_scaled, y_train, n_trials=20)
        best_xgb_params = tuner.optimize_xgboost()
        logging.info(f"✅ Tuning tamamlandı. En iyi XGB parametreleri: {best_xgb_params}")

    # ─────────────────────────────────────────────────────────────
    # TRAIN MODU
    # ─────────────────────────────────────────────────────────────
    elif args.mode == "train":
        logging.info("🏋️ Modeller Eğitiliyor (AutoEncoder + GNN + XGBoost + DNN)...")

        X_df, y = get_data(args.data_file)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, stratify=y, random_state=42
        )

        # 1. Feature Engineering (AutoEncoder + SMOTE + Graf oluşturma)
        preprocessor = TabularGraphPreprocessor(
            corr_threshold=Config.CORR_THRESHOLD,
            use_autoencoder=True,
            encoding_dim=16,
            device=device
        )
        X_train_scaled, y_train = preprocessor.fit_transform(X_train_df, label_y=y_train)
        X_test_scaled  = preprocessor.transform(X_test_df)

        # 2. PyTorch Geometric DataLoaderlar
        train_graphs = [preprocessor.row_to_graph(row, label=int(l)) for row, l in zip(X_train_scaled, y_train)]
        test_graphs  = [preprocessor.row_to_graph(row, label=int(l)) for row, l in zip(X_test_scaled, y_test)]

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

        # 3. XGBoost (Model 1)
        logging.info("XGBoost Modeli Eğitiliyor...")
        hybrid_model = VariantHybridModel(xgb_params=Config.XGB_PARAMS, weights=Config.ENSEMBLE_WEIGHTS)
        hybrid_model.fit_xgb(X_train_scaled, y_train)

        # 4. GNN (Model 2) ve DNN (Model 3)
        trainer = ModelTrainer(device=device)

        gnn_model = FeatureGNN(
            in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2
        ).to(device)
        trained_gnn, gnn_acc = trainer.train_gnn(
            train_loader, test_loader, model=gnn_model,
            epochs=Config.GNN_EPOCHS, lr=Config.GNN_LR
        )
        hybrid_model.gnn = trained_gnn

        dnn_model = VariantDNN(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=Config.DNN_HIDDEN_DIM,
            num_classes=2
        ).to(device)
        train_dnn_loader, test_dnn_loader = build_dnn_loaders(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        trained_dnn, dnn_acc = trainer.train_dnn(
            train_dnn_loader, test_dnn_loader, model=dnn_model,
            epochs=Config.DNN_EPOCHS, lr=Config.DNN_LR
        )
        hybrid_model.dnn = trained_dnn

        # 5. Model Kayıt
        ModelSerializer.save_models(hybrid_model, preprocessor)
        logging.info(f"✅ Modeller kaydedildi → {Config.MODELS_DIR}")

        # 6. Değerlendirme
        logging.info("📊 Test Seti Değerlendirmesi...")
        ensemble_probs, ensemble_preds, risk_scores = get_ensemble_probs(
            hybrid_model, X_test_scaled, test_loader, trained_dnn, device
        )

        evaluate_predictions(y_test, ensemble_preds, ensemble_probs)
        plot_confusion_matrix(
            y_test, ensemble_preds,
            title="Ensemble Model - Test Confusion Matrix",
            filename=os.path.join(Config.REPORTS_DIR, "train_confusion_matrix.png")
        )
        plot_performance_curves(y_test, ensemble_probs[:, 1], Config.REPORTS_DIR)

        # 7. XAI Raporları
        logging.info("🧠 XAI Açıklamaları Üretiliyor...")
        feat_names = [f"Anonim_Oznitelik_{i}" for i in range(X_test_scaled.shape[1])]
        explainer = FeatureExplainer(
            xgb_model=hybrid_model.xgb,
            feature_names=feat_names,
            training_data=X_train_scaled
        )
        explainer.plot_summary(
            X_test_scaled,
            filename=os.path.join(Config.REPORTS_DIR, "shap_summary.png")
        )
        explainer.plot_waterfall(
            X_test_scaled[0],
            filename=os.path.join(Config.REPORTS_DIR, "shap_waterfall.png")
        )
        top_feats = explainer.get_top_features(X_test_scaled, top_n=10)
        logging.info("🔝 En Önemli 10 Özellik (SHAP):")
        for feat, score in top_feats:
            logging.info(f"    {feat}: {score:.4f}")

        logging.info(f"✅ Raporlar kaydedildi → {Config.REPORTS_DIR}")

    # ─────────────────────────────────────────────────────────────
    # CROSSVAL MODU
    # ─────────────────────────────────────────────────────────────
    elif args.mode == "crossval":
        logging.info("🔄 Stratified K-Fold Cross-Validation Moduna Geçiliyor...")

        X_df, y = get_data(args.data_file)

        preprocessor = TabularGraphPreprocessor(
            corr_threshold=Config.CORR_THRESHOLD,
            use_autoencoder=True,
            encoding_dim=16,
            device=device
        )
        X_scaled, y = preprocessor.fit_transform(X_df, label_y=y)
        all_graphs = [preprocessor.row_to_graph(row, label=int(l)) for row, l in zip(X_scaled, y)]

        trainer = ModelTrainer(device=device)
        cv_results = trainer.cross_validate_gnn(
            all_graphs=all_graphs,
            all_labels=list(y),
            n_splits=5,
            hidden_dim=Config.GNN_HIDDEN_DIM,
            epochs=Config.GNN_EPOCHS,
            lr=Config.GNN_LR
        )
        logging.info(f"📊 Cross-Validation Sonucu: {cv_results}")

    # ─────────────────────────────────────────────────────────────
    # EVAL MODU
    # ─────────────────────────────────────────────────────────────
    elif args.mode == "eval":
        logging.info("📈 Eval Modu — Kaydedilmiş Modeller Yükleniyor...")
        X_df, y = get_data(args.data_file)
        _, X_test_df, _, y_test = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)

        try:
            # Modelleri yükle
            hybrid_model = VariantHybridModel(weights=Config.ENSEMBLE_WEIGHTS)
            hybrid_model.gnn = FeatureGNN(
                in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2
            ).to(device)
            preprocessor = ModelSerializer.load_models(hybrid_model)

            X_test_scaled = preprocessor.transform(X_test_df)
            test_graphs   = [preprocessor.row_to_graph(row, label=int(l)) for row, l in zip(X_test_scaled, y_test)]
            test_loader   = DataLoader(test_graphs, batch_size=32, shuffle=False)

            # DNN yükleme
            dnn_model = VariantDNN(input_dim=X_test_scaled.shape[1], num_classes=2).to(device)
            dnn_path = Config.DNN_MODEL_PATH
            if os.path.exists(dnn_path):
                dnn_model.load_state_dict(torch.load(dnn_path, map_location=device))
            hybrid_model.dnn = dnn_model

            ensemble_probs, ensemble_preds, _ = get_ensemble_probs(
                hybrid_model, X_test_scaled, test_loader, dnn_model, device
            )

            evaluate_predictions(y_test, ensemble_preds, ensemble_probs)
            plot_confusion_matrix(
                y_test, ensemble_preds,
                title="Eval Mode - Test Confusion Matrix",
                filename=os.path.join(Config.REPORTS_DIR, "eval_confusion_matrix.png")
            )
            plot_performance_curves(y_test, ensemble_probs[:, 1], Config.REPORTS_DIR)
            logging.info(f"✅ Eval tamamlandı. Raporlar → {Config.REPORTS_DIR}")

        except FileNotFoundError:
            logging.error("❌ Kaydedilmiş model bulunamadı. Lütfen önce '--mode train' çalıştırın.")

    # ─────────────────────────────────────────────────────────────
    # PREDICT MODU
    # ─────────────────────────────────────────────────────────────
    elif args.mode == "predict":
        test_file = args.test_file
        logging.info(f"🔮 Kör Tahmin (Inference) Modu | Veri: {test_file or 'sentetik'}")

        try:
            hybrid_model = VariantHybridModel(weights=Config.ENSEMBLE_WEIGHTS)
            hybrid_model.gnn = FeatureGNN(
                in_channels=1, hidden_dim=Config.GNN_HIDDEN_DIM, num_classes=2
            ).to(device)
            preprocessor = ModelSerializer.load_models(hybrid_model)

            # Gerçek CSV veya dummy veri
            if test_file and os.path.exists(test_file):
                inference_df, _ = load_and_prepare_data(test_file)
            else:
                logging.warning("Test dosyası bulunamadı. Sentetik veri kullanılıyor.")
                raw_df = generate_dummy_data(n_samples=500, n_features=40)
                inference_df = raw_df.drop(columns=['Label'], errors='ignore')

            X_infer_scaled = preprocessor.transform(inference_df)
            infer_graphs   = [preprocessor.row_to_graph(row) for row in X_infer_scaled]
            infer_loader   = DataLoader(infer_graphs, batch_size=32, shuffle=False)

            # DNN yükleme
            dnn_model = VariantDNN(input_dim=X_infer_scaled.shape[1], num_classes=2).to(device)
            if os.path.exists(Config.DNN_MODEL_PATH):
                dnn_model.load_state_dict(torch.load(Config.DNN_MODEL_PATH, map_location=device))
            hybrid_model.dnn = dnn_model

            ensemble_probs, ensemble_preds, risk_scores = get_ensemble_probs(
                hybrid_model, X_infer_scaled, infer_loader, dnn_model, device
            )

            # Submission dosyası
            submission = pd.DataFrame({
                'Variant_ID':  [f"VAR_{i:05d}" for i in range(len(ensemble_preds))],
                'Prediction':  ['Pathogenic' if p == 1 else 'Benign' for p in ensemble_preds],
                'Risk_Score':  risk_scores.round(2),
                'Confidence':  np.max(ensemble_probs, axis=1).round(4)
            })

            sub_path = os.path.join(Config.REPORTS_DIR, 'submission.csv')
            submission.to_csv(sub_path, index=False)
            logging.info(f"✅ Tahminler tamamlandı ({len(submission)} varyant).")
            logging.info(f"   Patojenik: {(ensemble_preds == 1).sum()} | Benign: {(ensemble_preds == 0).sum()}")
            logging.info(f"   Teslim dosyası: {sub_path}")

        except FileNotFoundError:
            logging.error("❌ Kaydedilmiş model bulunamadı. Lütfen önce '--mode train' çalıştırın.")


if __name__ == "__main__":
    main()

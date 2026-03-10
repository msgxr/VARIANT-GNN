import logging
import os
import sys

import joblib
import torch

from .config import Config


class ModelSerializer:
    """Araçların seri hale getirilip (kayıt) geri yüklenmesi içindir."""

    @staticmethod
    def save_models(hybrid_model, preprocessor):
        logging.info("Modeller artifacts (kayıt) klasörüne yazılıyor...")
        Config.create_dirs()

        # XGBoost Kaydı
        if hybrid_model.xgb is not None:
            hybrid_model.xgb.save_model(Config.XGB_MODEL_PATH)
            logging.info(f"  XGBoost → {Config.XGB_MODEL_PATH}")

        # GNN Kaydı
        if hybrid_model.gnn is not None:
            torch.save(hybrid_model.gnn.state_dict(), Config.GNN_MODEL_PATH)
            logging.info(f"  GNN     → {Config.GNN_MODEL_PATH}")

        # DNN Kaydı
        if hybrid_model.dnn is not None:
            torch.save(hybrid_model.dnn.state_dict(), Config.DNN_MODEL_PATH)
            logging.info(f"  DNN     → {Config.DNN_MODEL_PATH}")

        # AutoEncoder Kaydı — preprocessor ile birlikte tutarlılık için ayrıca kaydedilir
        if hasattr(preprocessor, 'autoencoder') and preprocessor.autoencoder is not None:
            torch.save(preprocessor.autoencoder.state_dict(), Config.AUTOENCODER_PATH)
            logging.info(f"  AutoEncoder → {Config.AUTOENCODER_PATH}")

        # Preprocessor (Scaler + Imputer + Graph bilgisi) Kaydı
        joblib.dump(preprocessor, Config.SCALER_PATH)
        logging.info(f"  Preprocessor → {Config.SCALER_PATH}")
        logging.info("✅ Tüm model ağırlıkları ve scaler dosyaları kaydedildi.")

    @staticmethod
    def load_models(hybrid_model):
        logging.info("Modeller kayıtlı kaynaklardan geri yükleniyor...")

        # XGBoost Yükleme
        if hybrid_model.xgb is not None and os.path.exists(Config.XGB_MODEL_PATH):
            hybrid_model.xgb.load_model(Config.XGB_MODEL_PATH)
            logging.info(f"  XGBoost yüklendi ← {Config.XGB_MODEL_PATH}")

        # GNN Yükleme
        if hybrid_model.gnn is not None and os.path.exists(Config.GNN_MODEL_PATH):
            device = next(hybrid_model.gnn.parameters()).device
            state_dict = torch.load(Config.GNN_MODEL_PATH, map_location=device, weights_only=True)
            hybrid_model.gnn.load_state_dict(state_dict)
            logging.info(f"  GNN yüklendi ← {Config.GNN_MODEL_PATH}")

        # DNN Yükleme
        if hybrid_model.dnn is not None and os.path.exists(Config.DNN_MODEL_PATH):
            device = next(hybrid_model.dnn.parameters()).device
            state_dict = torch.load(Config.DNN_MODEL_PATH, map_location=device, weights_only=True)
            hybrid_model.dnn.load_state_dict(state_dict)
            logging.info(f"  DNN yüklendi ← {Config.DNN_MODEL_PATH}")

        # Preprocessor Yükleme
        preprocessor = joblib.load(Config.SCALER_PATH)

        # AutoEncoder Yükleme — preprocessor içinde autoencoder varsa ağırlıklarını geri yükle
        if (hasattr(preprocessor, 'autoencoder') and
                preprocessor.autoencoder is not None and
                os.path.exists(Config.AUTOENCODER_PATH)):
            device_str = getattr(preprocessor, 'device', 'cpu')
            preprocessor.autoencoder.load_state_dict(
                torch.load(Config.AUTOENCODER_PATH, map_location=device_str, weights_only=True)
            )
            preprocessor.autoencoder.eval()
            logging.info(f"  AutoEncoder yüklendi ← {Config.AUTOENCODER_PATH}")

        logging.info("✅ Tüm modeller başarıyla yüklendi.")
        return preprocessor


def setup_logging(level=logging.INFO, log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger('variant_gnn')

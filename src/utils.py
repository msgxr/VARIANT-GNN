import torch
import joblib
import logging
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
            
        # GNN Kaydı
        if hybrid_model.gnn is not None:
            torch.save(hybrid_model.gnn.state_dict(), Config.GNN_MODEL_PATH)
            
        # AutoEncoder ve Preprocessor Kaydı
        joblib.dump(preprocessor, Config.SCALER_PATH)
        logging.info("✅ Tüm model ağırlıkları ve scaler dosyaları kaydedildi.")

    @staticmethod
    def load_models(hybrid_model):
        logging.info("Modeller kayıtlı kaynaklardan geri yükleniyor...")
        if hybrid_model.xgb is not None:
            hybrid_model.xgb.load_model(Config.XGB_MODEL_PATH)
            
        if hybrid_model.gnn is not None:
            # CPU/GPU aktarımıyla weight loading
            device = next(hybrid_model.gnn.parameters()).device
            state_dict = torch.load(Config.GNN_MODEL_PATH, map_location=device)
            hybrid_model.gnn.load_state_dict(state_dict)
            
        preprocessor = joblib.load(Config.SCALER_PATH)
        return preprocessor

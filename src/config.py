import os

class Config:
    """Merkezi Konfigürasyon Yönetim Dosyası"""
    
    # Dizin Ayarları
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Model Kayıt Yolları
    XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.json')
    GNN_MODEL_PATH = os.path.join(MODELS_DIR, 'gnn_model.pth')
    DNN_MODEL_PATH = os.path.join(MODELS_DIR, 'dnn_model.pth')
    AUTOENCODER_PATH = os.path.join(MODELS_DIR, 'autoencoder.pth')
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    # GNN Parametreleri
    GNN_HIDDEN_DIM = 64
    GNN_EPOCHS = 20
    GNN_LR = 0.005
    GNN_WEIGHT_DECAY = 1e-4
    CORR_THRESHOLD = 0.20 # Graf oluşturmada kullanılacak korelasyon eşiği
    
    # XGBoost Varsayılan Parametreleri (Tuning Öncesi)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'n_estimators': 150,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Ensemble Ağırlıkları [XGB, GNN, DNN]
    ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]
    
    # DNN Parametreleri
    DNN_HIDDEN_DIM = 128
    DNN_EPOCHS = 20
    DNN_LR = 0.001
    
    @staticmethod
    def create_dirs():
        """Kayıt klasörlerini oluşturur"""
        for d in [Config.DATA_DIR, Config.MODELS_DIR, Config.REPORTS_DIR]:
            os.makedirs(d, exist_ok=True)

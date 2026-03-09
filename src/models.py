import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

class FeatureGNN(nn.Module):
    """
    PyTorch Geometric kullanılarak hazırlanan Intra-Variant/Feature-Interaction 
    Grafik Sinir Ağı Mimarisi.
    Anonim verilerdeki her bir sütunu bir Node (düğüm) kabul eder.
    Gizli örüntüleri öğrenmek amacıyla tasarlanmıştır.
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_classes=2, use_gat=False):
        super(FeatureGNN, self).__init__()
        self.use_gat = use_gat
        
        if use_gat:
            self.conv1 = GATConv(in_channels, hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim * 2, heads=4, concat=False)
            self.conv3 = GATConv(hidden_dim * 2, hidden_dim, heads=4, concat=False)
        else:
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.conv3 = GCNConv(hidden_dim * 2, hidden_dim)
            
        # Classifier Head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if self.use_gat:
            # GATConv genellikle edge_weight beklemez.
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
        else:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = F.relu(self.conv3(x, edge_index, edge_weight))
            
        # Readout tabakası (Bir varyanttaki tüm düğümlerin tek bir temsilcisine dönüştürülmesi)
        x = global_mean_pool(x, batch)
        
        # Son Sınıflandırma Katmanı
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

class VariantDNN(nn.Module):
    """
    Kullanıcı vizyonundaki 'Model 3: Deep Neural Network' bileşeni.
    Tablo verileri üzerinde derinlemesine özellik analizi yapar.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(VariantDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class VariantHybridModel:
    """
    XGBoost, GNN ve DNN'i birleştiren Multi-Modal Meta Ensemble sınıfı.
    Geleneksel tablo modelleyici gücüyle derin grafik gösterim 
    ve klasik derin öğrenimi hibridize eder.
    """
    def __init__(self, gnn_model=None, dnn_model=None, xgb_params=None, weights=None):
        self.gnn = gnn_model
        self.dnn = dnn_model
        
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'use_label_encoder': False,
                'n_estimators': 150
            }
        else:
            self.xgb_params = xgb_params
            
        self.xgb = xgb.XGBClassifier(**self.xgb_params)
        
        # Ensemble ağırlıkları: [XGB, GNN, DNN]
        self.weights = weights if weights is not None else [0.4, 0.4, 0.2]

    def fit_xgb(self, X_train_scaled, y_train):
        """XGBoost eğitim fonksiyonu"""
        self.xgb.fit(X_train_scaled, y_train)

    def predict_xgb_proba(self, X_scaled):
        """XGBoost Olasılık Tahmini"""
        return self.xgb.predict_proba(X_scaled) 

    def predict_ensemble(self, xgb_probs, gnn_probs, dnn_probs):
        """
        Üç modelin olasılık tahminlerini ağırlıklı olarak birleştirir.
        Kullanıcı vizyonundaki 'Multi-Modal AI' yaklaşımını uygular.
        """
        combined_probs = (self.weights[0] * xgb_probs) + \
                         (self.weights[1] * gnn_probs) + \
                         (self.weights[2] * dnn_probs)
        return combined_probs

    def get_clinical_risk_score(self, combined_probs):
        """
        Olasılığı 0-100 arası bir Klinik Risk Skoruna dönüştürür.
        """
        # Patojenik sınıf (indeks 1) olasılığını al
        risk_score = combined_probs[:, 1] * 100
        return risk_score

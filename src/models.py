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

class VariantHybridModel:
    """
    XGBoost ve GNN'i birleştiren Meta Ensemble sınıfı.
    Geleneksel tablo modelleyici gücüyle derin grafik gösterim öğrenimlerini hibridize eder.
    """
    def __init__(self, gnn_model=None, xgb_params=None, gnn_weight=0.5):
        self.gnn = gnn_model
        
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
        self.gnn_weight = gnn_weight # Ensemble tahmini için GNN ağırlığı

    def fit_xgb(self, X_train_scaled, y_train):
        """XGBoost eğitim fonksiyonu"""
        self.xgb.fit(X_train_scaled, y_train)

    def predict_xgb_proba(self, X_scaled):
        """XGBoost Olasılık Tahmini"""
        return self.xgb.predict_proba(X_scaled) 

    def predict_ensemble(self, xgb_probs, gnn_probs_normalized):
        """
        İki modelin olasılık tahminlerini normalize ederek birleştirir.
        Tahmin: 1 (Pathogenic), 0 (Benign)
        gnn_probs_normalized: (N, 2) softmax olasılıkları.
        xgb_probs: (N, 2) olasılıkları.
        """
        combined_probs = (self.gnn_weight * gnn_probs_normalized) + ((1 - self.gnn_weight) * xgb_probs)
        return combined_probs

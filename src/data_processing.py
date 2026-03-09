import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data
import logging
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

class TabularAutoEncoder(nn.Module):
    """
    Boyut indirgeme ve gizli özellik sentezi için kullanılan 
    basit bir AutoEncoder (Otokodlayıcı) Ağı.
    Gizli özellikleri daha sıkıştırılmış manifoldda öğrenerek GNN'e yardımcı olur.
    """
    def __init__(self, input_dim, encoding_dim=16):
        super(TabularAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, encoding_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class TabularGraphPreprocessor:
    """
    Şartnameye göre veri özellik isimleri (kolon isimleri) ve 
    genomik adresler anonim/gizli verileceği için;
    Varyant özelliklerini doğrusal öznitelikler olarak scale ederken,
    aynı zamanda bu özniteliklerin kendi aralarındaki istatistiksel 
    ilişkilerine dayalı bir "Öznitelik-Etkileşim Grafı" (Feature-Interaction Graph)
    oluşturur. AutoEncoder ile gizli özellikleri zenginleştirir.
    """
    def __init__(self, corr_threshold=0.25, use_autoencoder=True, encoding_dim=16, device='cpu'):
        # Aykırı değerlere (outliers) karşı standart StandardScaler yerine RobustScaler tercih edilir.
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.corr_threshold = corr_threshold
        self.use_autoencoder = use_autoencoder
        self.encoding_dim = encoding_dim
        self.device = device
        
        self.edge_index = None
        self.edge_attr = None
        self.autoencoder = None
        self.is_fitted = False

    def _train_autoencoder(self, X_scaled):
        """AutoEncoder'ı maskelenmiş veriler ile gizli (latent) özellik çıkartmak için eğitir"""
        input_dim = X_scaled.shape[1]
        self.autoencoder = TabularAutoEncoder(input_dim, self.encoding_dim).to(self.device)
        
        tensor_X = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(tensor_X, tensor_X)
        loader = TorchDataLoader(dataset, batch_size=64, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.autoencoder.train()
        
        logging.info("AutoEncoder eğitimi başlatılıyor (10 epoch) ...")
        for epoch in range(10):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                _, decoded = self.autoencoder(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        logging.info(f"AutoEncoder Loss: {total_loss / len(loader):.4f}")

    def fit_transform(self, X_df, label_y=None):
        """
        Model eğitimi için veri setindeki özellik istatistiklerini hesaplar.
        Talebe göre AutoEncoder eğitimini yapar ve Graf bağlantılarını dizer.
        """
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # SMOTE ile veri dengeleme (Yalnızca eğitim modunda ve label varsa)
        # Not: Etiketlerin 0/1 olması gerekir.
        if label_y is not None:
            logging.info("SMOTE ile veri dengeleme (Data Balancing) uygulanıyor...")
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_scaled, label_y)
            X_scaled, label_y = X_balanced, y_balanced

        if self.use_autoencoder:
            self._train_autoencoder(X_scaled)
            self.autoencoder.eval()
            with torch.no_grad():
                tensor_X = torch.FloatTensor(X_scaled).to(self.device)
                encoded_feats, _ = self.autoencoder(tensor_X)
                # Orijinal özellikler ile AutoEncoder özelliklerini birleştiriyoruz
                X_scaled = np.hstack((X_scaled, encoded_feats.cpu().numpy()))

        # GNN için Öznitelikler arası kovaryans matrisi oluşturma
        corr_matrix = np.corrcoef(X_scaled, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        
        num_features = X_scaled.shape[1]
        edges = []
        edge_weights = []
        
        for i in range(num_features):
            for j in range(num_features):
                if i != j and abs(corr_matrix[i, j]) >= self.corr_threshold:
                    edges.append([i, j])
                    edge_weights.append(abs(corr_matrix[i, j]))

        if len(edges) > 0:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # Yedek boş grafik yapısı
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0,), dtype=torch.float)
        
        self.is_fitted = True
        return X_scaled

    def transform(self, X_df):
        """
        Test ve validasyon testleri için (data leakage önlenerek) ölçeklendirme uygular.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor öncelikle fit_transform ile eğitilmelidir.")
        
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        if self.use_autoencoder and self.autoencoder is not None:
            self.autoencoder.eval()
            with torch.no_grad():
                tensor_X = torch.FloatTensor(X_scaled).to(self.device)
                encoded_feats, _ = self.autoencoder(tensor_X)
                X_scaled = np.hstack((X_scaled, encoded_feats.cpu().numpy()))
                
        return X_scaled

    def row_to_graph(self, x_row, label=None):
        """
        Tek bir varyantın sayısal öznitelik vektörünü PyTorch Geometric Data nesnesine dönüştürür.
        Böylece özellikler graph düğümleri (nodes) olarak temsili edilir.
        """
        # Node özellikleri: [Özellik Sayısı, 1] boyutu
        x_tensor = torch.tensor(x_row, dtype=torch.float).unsqueeze(1)
        y_tensor = torch.tensor([label], dtype=torch.long) if label is not None else None
        
        # Tüm varyantlar, öznitelik topolojisi benzer olduğu için aynı edge_index ve edge_weight'i kullanır.
        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y_tensor)

def load_and_prepare_data(csv_path, target_col='Label', separator=','):
    """
    Yarışma formatındaki potansiyel csv veya excel verilerini yükler.
    Label sütunu: 'Pathogenic' -> 1, 'Benign' -> 0 şeklinde dönüştürülür.
    """
    df = pd.read_csv(csv_path, sep=separator)
    
    # Hedef değişkende standardize (Patojenik=1, Benign=0)
    # Şartnamede Likely Pathogenic -> Pathogenic, Likely Benign -> Benign olacağı belirtilmiştir.
    if target_col in df.columns.tolist():
        df[target_col] = df[target_col].astype(str).str.lower()
        map_labels = {
            'pathogenic': 1, 'likely pathogenic': 1,
            'benign': 0, 'likely benign': 0, 
            '1': 1, '1.0': 1, '0': 0, '0.0':0
        }
        df['target'] = df[target_col].map(map_labels)
        X_df = df.drop(columns=[target_col, 'target'])
        y = df['target'].values
    else:
        # Test seti (etiketsiz veri)
        X_df = df
        y = None
        
    return X_df, y

def generate_dummy_data(n_samples=1000, n_features=30):
    """Eğitim ve test için sentetik varyant verisi üretir."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Patojenite etkileyen rastgele bir doğrusal ağırlık kombinasyonu
    logits = 2.5 * X[:, 0] - 1.2 * X[:, 5] + 0.8 * X[:, 12]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    
    df = pd.DataFrame(X, columns=[f"Anonim_Kolon_{i}" for i in range(n_features)])
    df['Label'] = y
    df['Label'] = df['Label'].map({1: 'Pathogenic', 0: 'Benign'})
    return df

def plot_performance_curves(y_true, y_probs, output_dir):
    """
    Roc-AUC ve Precision-Recall eğrilerini profesyonel formata üretir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_curves.png"))
    plt.close()


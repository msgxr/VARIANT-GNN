import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import copy
from .models import FeatureGNN, VariantHybridModel, VariantDNN

def train_gnn_epoch(model, loader, optimizer, device):
    """
    GNN için tek bir epoch boyunca modelin eğitilmesi.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # Patojenik / Benign etiketleri loss (Kayıp) optimizasyonu
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_gnn_epoch(model, loader, device):
    """
    Validation / Test verisi üzerinden model metriklerinin alınması.
    """
    model.eval()
    correct = 0
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return correct / len(loader.dataset), all_preds, all_probs

class ModelTrainer:
    """
    Bağımsız GNN eğitimi ve Cross-Validation yönetiminden sorumlu eğitici sınıfı.
    Ayrıca Kanser ve CFTR panellerindeki aşırı öğrenmeye (overfitting) karşı
    küçük panellerde Transfer Learning ve Feature Extraction altyapısını kurar.
    """
    def __init__(self, device='cpu'):
        self.device = device

    def train_gnn(self, train_loader, val_loader, model=None, epochs=20, lr=0.001):
        if model is None:
            model = FeatureGNN(in_channels=1, hidden_dim=64, num_classes=2).to(self.device)
            
        # Düzenlileştirme (Weight Decay) ve Öğrenim oranı ile overfitting baskısı
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        best_acc = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(1, epochs + 1):
            loss = train_gnn_epoch(model, train_loader, optimizer, self.device)
            val_acc, _, _ = evaluate_gnn_epoch(model, val_loader, self.device)
            
            # Eğitim döngüsünde anlık validasyon modelini saklama
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_model_wts)
        return model, best_acc

    def fine_tune_for_panel(self, pretrained_model, panel_train_loader, epochs=5, lr=0.0001):
        """
        Şartnamede küçük olacağı belirtilen özel paneller (örn. CFTR: 70/70 varyant)
        bu fonksiyondan geçirilerek ezberlemeden (Transfer Learning ile) fine-tune edilir.
        """
        pretrained_model.train()
        # Çok düşük öğrenme oranı ile ince ayar (Fine tuning)
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(1, epochs + 1):
            train_gnn_epoch(pretrained_model, panel_train_loader, optimizer, self.device)
            
        return pretrained_model
    def train_dnn(self, train_loader, val_loader, model, epochs=20, lr=0.001):
        """
        Kullanıcı vizyonundaki Model 3 (DNN) eğitimi.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(1, epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

            # Eval
            model.eval()
            correct = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    out = model(X_batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == y_batch).sum())
            
            acc = correct / len(val_loader.dataset)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        return model, best_acc

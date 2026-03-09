import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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
            if hasattr(data, 'y') and data.y is not None:
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

    def fine_tune_for_panel(self, pretrained_model, panel_train_loader, epochs=5, lr=0.0001):
        """
        Şartnamede küçük olacağı belirtilen özel paneller (örn. CFTR: 70/70 varyant)
        bu fonksiyondan geçirilerek ezberlemeden (Transfer Learning ile) fine-tune edilir.
        """
        pretrained_model.train()
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(1, epochs + 1):
            train_gnn_epoch(pretrained_model, panel_train_loader, optimizer, self.device)
        return pretrained_model

    def train_gnn(self, train_loader, val_loader, model=None, epochs=20, lr=0.001):
        if model is None:
            model = FeatureGNN(in_channels=1, hidden_dim=64, num_classes=2).to(self.device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # DÜZELTME (P0): Model seçimi val_acc yerine val_f1_macro ile yapılıyor
        # Sebep: Ana yarışma metriği F1-Macro; accuracy ile seçim metrik tutarsızlığı yaratır
        best_f1 = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        logging.info(f"GNN Eğitimi başlatıldı | Epoch: {epochs} | LR: {lr} | Model seçim metriği: F1-Macro")
        for epoch in range(1, epochs + 1):
            loss = train_gnn_epoch(model, train_loader, optimizer, self.device)
            val_acc, val_preds, _ = evaluate_gnn_epoch(model, val_loader, self.device)
            val_labels = [d.y.item() for batch in val_loader for d in batch.to_data_list() if d.y is not None]
            # F1 score hesapla (model seçim metriği)
            try:
                val_f1 = f1_score(val_labels, val_preds[:len(val_labels)], average='macro', zero_division=0)
            except Exception:
                val_f1 = 0.0

            if epoch % 5 == 0 or epoch == epochs:
                logging.info(f"  [GNN] Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-Macro: {val_f1:.4f} {'⭐' if val_f1 > best_f1 else ''}")
            
            # F1-Macro ile model seçimi (val_acc değil)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_model_wts)
        logging.info(f"GNN Eğitimi tamamlandı | En iyi Val F1-Macro: {best_f1:.4f}")
        return model, best_f1

    def train_dnn(self, train_loader, val_loader, model, epochs=20, lr=0.001):
        """
        Kullanıcı vizyonundaki Model 3 (DNN) eğitimi.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        # DÜZELTME (P0): DNN de val_f1_macro ile model seçimi
        best_f1 = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        logging.info(f"DNN Eğitimi başlatıldı | Epoch: {epochs} | LR: {lr} | Model seçim metriği: F1-Macro")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Eval
            model.eval()
            correct = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    out = model(X_batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == y_batch).sum())
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            acc = correct / len(val_loader.dataset)
            try:
                val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            except Exception:
                val_f1 = 0.0

            if epoch % 5 == 0 or epoch == epochs:
                logging.info(f"  [DNN] Epoch {epoch:03d}/{epochs} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f} | Val F1-Macro: {val_f1:.4f} {'⭐' if val_f1 > best_f1 else ''}")

            # F1-Macro ile model seçimi (val_acc değil)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        logging.info(f"DNN Eğitimi tamamlandı | En iyi Val F1-Macro: {best_f1:.4f}")
        return model, best_f1

    def cross_validate_gnn(self, all_graphs, all_labels, n_splits=5,
                           hidden_dim=64, epochs=20, lr=0.001, batch_size=32):
        """
        StratifiedKFold Cross-Validation ile GNN'in genelleme gücünü ölçer.
        Her fold için F1 skoru raporlanır.
        
        Returns:
            dict: fold bazlı ve ortalama metrikler
        """
        all_labels_arr = np.array(all_labels)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_results = []
        logging.info(f"Stratified {n_splits}-Fold Cross-Validation başlatılıyor...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_graphs, all_labels_arr), 1):
            train_graphs_fold = [all_graphs[i] for i in train_idx]
            val_graphs_fold   = [all_graphs[i] for i in val_idx]

            train_loader = DataLoader(train_graphs_fold, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_graphs_fold,   batch_size=batch_size, shuffle=False)

            gnn_model = FeatureGNN(in_channels=1, hidden_dim=hidden_dim, num_classes=2).to(self.device)
            optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=1e-4)

            for epoch in range(1, epochs + 1):
                train_gnn_epoch(gnn_model, train_loader, optimizer, self.device)

            _, preds, _ = evaluate_gnn_epoch(gnn_model, val_loader, self.device)
            val_labels_fold = all_labels_arr[val_idx]

            fold_f1 = f1_score(val_labels_fold, preds[:len(val_labels_fold)], average='macro', zero_division=0)
            fold_results.append(fold_f1)
            logging.info(f"  Fold {fold}/{n_splits} | Val F1 (Macro): {fold_f1:.4f}")

        mean_f1 = np.mean(fold_results)
        std_f1  = np.std(fold_results)
        logging.info(f"Cross-Validation Tamamlandı | Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        return {
            "fold_f1_scores": fold_results,
            "mean_f1": mean_f1,
            "std_f1": std_f1
        }

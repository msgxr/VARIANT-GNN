# src/training/cross_val.py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# Not: src modüllerinin config'den ve veri ön işleme modülünden haberdar olması gerekiyor
# Burada uygun importları varsayarak veya ekleyerek yapıyı kuruyoruz
from src.data_processing import TabularGraphPreprocessor

def leakage_free_cross_validate(raw_X_df, raw_y, config, device, n_splits=5):
    """
    Her fold için ayrı TabularGraphPreprocessor fit eder.
    Leakage riski tamamen ortadan kaldırılmıştır.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_X_df, raw_y), 1):
        X_train_fold = raw_X_df.iloc[train_idx]
        X_val_fold   = raw_X_df.iloc[val_idx]
        y_train_fold = raw_y[train_idx]
        y_val_fold   = raw_y[val_idx]
        
        # HER FOLD İÇİN YENİ PREPROCESSOR — leakage yok
        preprocessor = TabularGraphPreprocessor(
            corr_threshold=config.CORR_THRESHOLD,
            use_autoencoder=False,  # CV'de AE kapatılır hız için
            device=device
        )
        X_tr, y_tr = preprocessor.fit_transform(X_train_fold, label_y=y_train_fold)
        X_val      = preprocessor.transform(X_val_fold)
        
        # GNN grafları fold'a özgü preprocessor ile
        train_graphs = [preprocessor.row_to_graph(r, int(l)) for r, l in zip(X_tr, y_tr)]
        val_graphs   = [preprocessor.row_to_graph(r, int(l)) for r, l in zip(X_val, y_val_fold)]
        
        # Not: GNN Modeli Eğitimi için ModelTrainer burada dışarıdan enjekte edilmeli 
        # veya burada doğrudan çalıştırılmalıdır (main.py'den uyarlanacak).
        # Bu taslak şablon f1_score hesaplama mantığı kurar.
        
        # FIXME: Aşağıdaki kısım asıl eğitim logic'i ile birleştirilmelidir
        # preds = ...
        # fold_f1 = f1_score(y_val_fold, preds, average='macro')
        # fold_results.append(fold_f1)
        pass # Placeholder for actual training loop adapted in main.py
    
    return {'fold_f1': fold_results, 'mean_f1': np.mean(fold_results) if fold_results else 0.0, 'std_f1': np.std(fold_results) if fold_results else 0.0}

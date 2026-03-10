import logging

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTuner:
    """
    Optuna kullanılarak XGBoost ve GNN mimarileri için
    otomatik hiperparametre optimizasyonu yapar.
    """
    def __init__(self, X_train, y_train, n_trials=30):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials

    def optimize_xgboost(self):
        """XGBoost için Optuna optimizasyon hedef fonksiyonu."""
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'use_label_encoder': False,
                'n_jobs': -1,
                'random_state': 42
            }
            
            # Stratified K-Fold Cross Validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
                
                model = xgb.XGBClassifier(**param)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                preds = model.predict(X_val)
                # Yarışma formatı macro F1 skorunu maksimize etme üzerine kurulu
                score = f1_score(y_val, preds, average='macro')
                cv_scores.append(score)
                
            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logging.info(f"🏆 En İyi XGBoost F1 Skoru: {study.best_value:.4f}")
        logging.info(f"📊 En İyi Parametreler: {study.best_params}")
        
        return study.best_params

# GNN için optimizasyon ilerleyen fazlarda eklenebilir, 
# şimdilik XGBoost üzerindeki karmaşıklığı çözmeye odaklanılmıştır.

# src/models/calibration.py
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class EnsembleCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        self._fitted = False
    
    def fit(self, probs_val, y_val):
        """Validasyon seti üzerinde kalibrasyon öğren"""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # platt
            self.calibrator = LogisticRegression(C=1.0)
        
        self.calibrator.fit(probs_val[:, 1].reshape(-1, 1), y_val)
        self._fitted = True
    
    def calibrate(self, probs):
        """Kalibre edilmiş olasılıklar döndür"""
        if not self._fitted:
            return probs
        cal_probs = self.calibrator.predict(probs[:, 1].reshape(-1, 1))
        return np.column_stack([1 - cal_probs, cal_probs])
    
    def get_risk_score(self, probs):
        """Kalibre edilmiş risk skoru (0-100)"""
        cal = self.calibrate(probs)
        return cal[:, 1] * 100  # Gerçek kalibrasyon sonrası
    
    def evaluate_calibration(self, probs, y_true):
        """ECE, Brier score, kalibrasyon eğrisi"""
        brier = brier_score_loss(y_true, probs[:, 1])
        frac_pos, mean_pred = calibration_curve(y_true, probs[:, 1], n_bins=10)
        ece = np.mean(np.abs(frac_pos - mean_pred))
        return {'brier_score': brier, 'ece': ece}

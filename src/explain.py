import shap
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os

class FeatureExplainer:
    """
    Açıklanabilir Yapay Zekâ (XAI) Modülü.
    Buradaki XAI kullanımı, modelin biyolojik nedensellik vadetmesinden ziyade 
    (Etik kural gereği kolon isimleri zaten maskelidir),
    siyah-kutu algoritmasının şeffaflığını artırmak amaçlıdır.
    Modelin hangi isimsiz değişkenlere ('Anonim_Feature_3' gibi) 
    karar ağırlığı verdiğini haritalar.
    """
    def __init__(self, xgb_model, feature_names=None):
        self.explainer = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names

    def explain_instance(self, x_instance):
        """Tek bir varyantın tahmin kararına etki eden isimsiz ağırlık grafiğini üretir."""
        shap_values = self.explainer.shap_values(x_instance)
        return shap_values
        
    def plot_summary(self, X_sample, filename="shap_summary.png"):
        """Genel model karar kümesinin öznitelik ağırlık dökümünü çıkarır."""
        shap_values = self.explainer.shap_values(X_sample)
        plt.figure(figsize=(10, 6))
        
        # Etiketler anonim ise jenerik isimler ver.
        if self.feature_names is None:
            self.feature_names = [f"Anonim_Oznitelik_{i}" for i in range(X_sample.shape[1])]
            
        # show=False ile ortamda pop-up grafik cizmek yerine sadece PNG ciktisi uretir.
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("Anonim Özniteliklerin Karara Etki Haritası (Şeffaflık Raporu)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

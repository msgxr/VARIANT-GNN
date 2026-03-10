import matplotlib
import numpy as np
import shap

matplotlib.use('Agg')  # GUI olmayan ortamlar için
import os
import warnings

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    GNN_EXPLAINER_AVAILABLE = True
except ImportError:
    GNN_EXPLAINER_AVAILABLE = False


class FeatureExplainer:
    """
    Açıklanabilir Yapay Zekâ (XAI) Modülü.
    Üç farklı açıklama tekniği entegre edilmiştir:
    1. SHAP (TreeExplainer) - XGBoost için global ve yerel öznitelik önem haritası
    2. LIME - Tabular Explainer - Tek tahmin için yerel açıklama
    3. GNNExplainer - GNN katmanları için graf açıklanabilirliği
    
    Not: Kolon isimleri anonim/gizli olduğundan açıklamalar
    'Anonim_Oznitelik_N' formatında sunulur (şeffaflık raporu).
    """

    def __init__(self, xgb_model, feature_names=None, training_data=None):
        """
        Args:
            xgb_model: Eğitilmiş XGBoost modeli
            feature_names: Özellik isimleri listesi (anonim olabilir)
            training_data: LIME için referans eğitim verisi (numpy array)
        """
        self.xgb_model = xgb_model
        self.shap_explainer = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names
        self.training_data = training_data
        self._lime_explainer = None

    def _get_feature_names(self, n_features):
        """Anonim özellik isimlerini döndürür."""
        if self.feature_names is not None and len(self.feature_names) == n_features:
            return list(self.feature_names)
        return [f"Anonim_Oznitelik_{i}" for i in range(n_features)]

    def _get_lime_explainer(self, X_reference):
        """LIME explainer'ı lazy yüklemesi ile oluşturur."""
        if self._lime_explainer is None:
            ref_data = X_reference if self.training_data is None else self.training_data
            feat_names = self._get_feature_names(ref_data.shape[1])
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=ref_data,
                feature_names=feat_names,
                class_names=['Benign', 'Pathogenic'],
                mode='classification',
                random_state=42
            )
        return self._lime_explainer

    # ──────────────────────────────────────────────
    # SHAP Metodları
    # ──────────────────────────────────────────────

    def explain_instance(self, x_instance):
        """
        Tek bir varyantın tahmin kararına etki eden SHAP değerlerini döndürür.
        Returns:
            shap_values: numpy array, her özelliğin katkısı
        """
        shap_values = self.shap_explainer.shap_values(x_instance)
        return shap_values

    def plot_summary(self, X_sample, filename="shap_summary.png"):
        """
        Genel model karar kümesinin öznitelik ağırlık dökümünü üretir (beeswarm).
        """
        shap_values = self.shap_explainer.shap_values(X_sample)
        n_features = X_sample.shape[1]
        feat_names = self._get_feature_names(n_features)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False)
        plt.title("Anonim Özniteliklerin Karara Etki Haritası (Şeffaflık Raporu)")
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()

    def plot_waterfall(self, x_instance, filename="shap_waterfall.png"):
        """
        Tek bir varyant için SHAP waterfall grafiği üretir.
        Hangi özelliğin tahmini ne yönde etkilediğini gösterir.
        """
        shap_values = self.shap_explainer(x_instance.reshape(1, -1))
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(shap_values[0], show=False)
        plt.title("Varyant Karar Açıklaması (Waterfall)")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()

    def get_top_features(self, X_sample, top_n=10):
        """
        En etkili N özelliği ve ortalama SHAP değerlerini döndürür.
        Returns:
            list of (feature_name, mean_abs_shap) tuples, azalan sırada
        """
        shap_values = self.shap_explainer.shap_values(X_sample)
        # binary: shap_values shape (n,) veya (n, n_features)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1])  # Pathogenic sınıfı
        else:
            sv = np.array(shap_values)
        mean_abs = np.mean(np.abs(sv), axis=0)
        feat_names = self._get_feature_names(len(mean_abs))
        ranked = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ──────────────────────────────────────────────
    # LIME Metodları
    # ──────────────────────────────────────────────

    def explain_with_lime(self, X_reference, instance_index=0,
                          feature_names=None, output_filename="lime_explanation.html"):
        """
        Belirtilen varyant için LIME açıklaması üretir ve HTML dosyasına kaydeder.
        Args:
            X_reference: Tüm analiz verisi (numpy array) — LIME referans için
            instance_index: Açıklanacak varyantın satır indeksi (default: 0)
            feature_names: Özellik isimleri (opsiyonel)
            output_filename: HTML çıktı dosya yolu
        """
        if feature_names is not None:
            self.feature_names = feature_names

        lime_exp = self._get_lime_explainer(X_reference)

        # Tahmin fonksiyonu olarak XGBoost predict_proba kullanılır
        predict_fn = self.xgb_model.predict_proba
        instance = X_reference[instance_index]

        explanation = lime_exp.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=min(15, X_reference.shape[1]),
            num_samples=500
        )

        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else ".", exist_ok=True)
        explanation.save_to_file(output_filename)
        return explanation

    def plot_lime_bar(self, X_reference, instance_index=0, filename="lime_bar.png"):
        """
        LIME açıklamasını PNG bar grafiği olarak kaydeder.
        """
        lime_exp = self._get_lime_explainer(X_reference)
        predict_fn = self.xgb_model.predict_proba
        instance = X_reference[instance_index]

        explanation = lime_exp.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=min(15, X_reference.shape[1]),
            num_samples=500
        )

        fig = explanation.as_pyplot_figure()
        fig.suptitle(f"LIME Açıklaması — Varyant #{instance_index}", fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()

    # ──────────────────────────────────────────────
    # GNN Graph Explainability
    # ──────────────────────────────────────────────

    def explain_gnn_instance(self, gnn_model, graph_data, device='cpu', top_k_edges=10):
        """
        GNNExplainer ile bir varyant grafiğinin hangi özellik bağlantılarının
        (edge) karar için kritik olduğunu açıklar.
        
        Args:
            gnn_model: Eğitilmiş FeatureGNN modeli
            graph_data: PyTorch Geometric Data nesnesi (tek varyant)
            device: 'cpu' veya 'cuda'
            top_k_edges: Görselleştirilecek en önemli kenar sayısı
        Returns:
            explanation dict: node_mask, edge_mask değerleri
        """
        if not GNN_EXPLAINER_AVAILABLE:
            return {"error": "torch_geometric.explain bulunamadı. Lütfen torch_geometric güncelleyin."}

        try:
            import torch
            gnn_model = gnn_model.to(device)
            gnn_model.eval()

            explainer = Explainer(
                model=gnn_model,
                algorithm=GNNExplainer(epochs=100),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='logits',
                )
            )

            graph_data = graph_data.to(device)
            explanation = explainer(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                batch=torch.zeros(graph_data.x.size(0), dtype=torch.long).to(device)
            )

            return {
                "node_mask": explanation.node_mask.cpu().detach().numpy() if explanation.node_mask is not None else None,
                "edge_mask": explanation.edge_mask.cpu().detach().numpy() if explanation.edge_mask is not None else None,
            }

        except Exception as e:
            return {"error": str(e)}

    def plot_gnn_explanation(self, gnn_model, graph_data, device='cpu',
                             feature_names=None, filename="gnn_explanation.png"):
        """
        GNNExplainer sonuçlarını özellik önem çubuğu olarak görselleştirir.
        """
        result = self.explain_gnn_instance(gnn_model, graph_data, device)

        if "error" in result:
            print(f"[GNN XAI] Uyarı: {result['error']}")
            return

        node_mask = result.get("node_mask")
        if node_mask is None:
            print("[GNN XAI] node_mask bulunamadı.")
            return

        importances = node_mask.flatten()
        feat_names = self._get_feature_names(len(importances))

        sorted_idx = np.argsort(importances)[::-1][:20]
        top_feats = [feat_names[i] for i in sorted_idx]
        top_vals = importances[sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(top_feats[::-1], top_vals[::-1], color='steelblue')
        plt.xlabel("GNN Düğüm Önemi (Node Mask)")
        plt.title("GNN Graf Açıklanabilirlik Raporu (Top-20 Özellik)")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()

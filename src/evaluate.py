import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Şartnameye göre yarışmanın "Ana Değerlendirme Metriği"
    izlenecektir (F1-Score). 
    Diğer metrikler araştırmacıya istatistiksel destek bağlamında üretilir.
    """
    # Yarışma F1 için genellikle 'macro' veya pozitif sınıfın direkt skorunu hedef alır.
    # %50-%50 sınıflar dengeli, bu nedenle macro kullanılabilir.
    f1 = f1_score(y_true, y_pred, average='macro') 
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            pass

    print("--- FİNAL DEĞERLENDİRME METRİKLERİ ---")
    print(f"[ANA METRİK] F1 Score (Macro)  : {f1:.4f}")
    print(f"[YARDIMCI ]  Precision (Macro) : {prec:.4f}")
    print(f"[YARDIMCI ]  Recall (Macro)    : {rec:.4f}")
    if auc is not None:
        print(f"[YARDIMCI ]  ROC-AUC           : {auc:.4f}")

    return {
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Sınıflandırma hatalarını incelemek için karmaşıklık matrisi görsel çizimi
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign (0)', 'Pathogenic (1)'],
                yticklabels=['Benign (0)', 'Pathogenic (1)'])
    plt.title(title)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek (Ground Truth)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

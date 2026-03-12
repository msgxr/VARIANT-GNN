# 🚀 Google Colab'da VARIANT-GNN Eğitim Rehberi

## 📋 **Ön Hazırlık Checklist**
- [ ] Google Colab Pro/Pro+ hesabı (önerilen)
- [ ] Google Drive'da 2-3GB boş alan
- [ ] GitHub'dan projeye erişim
- [ ] 20.000 varyant veri seti hazır

---

## 🔧 **1. Colab Notebook Kurulumu**

### Step 1: Yeni Notebook Oluştur
```python
# İlk cell: Runtime check
import sys
print(f"Python version: {sys.version}")

# GPU kontrolü
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Step 2: Runtime Ayarları
```
Runtime > Change Runtime Type > Hardware Accelerator: GPU (T4)
```

---

## 📦 **2. Kütüphane Kurulumu**

### Cell 1: Sistem Güncellemeleri
```python
# Sistem paketlerini güncelle
!apt-get update -qq
!apt-get install -y -qq software-properties-common
```

### Cell 2: PyTorch ve PyTorch Geometric
```python
# PyTorch kurulumu (CUDA 12.1 uyumlu)
!pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric ecosystem
!pip install torch-geometric
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Verificatio check
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
```

### Cell 3: ML ve Visualization Kütüphaneleri
```python
# Temel ML kütüphaneleri
!pip install xgboost==2.0.3
!pip install scikit-learn==1.4.0
!pip install pandas==2.2.0
!pip install numpy==1.26.0

# Explainable AI
!pip install shap==0.45.0 
!pip install lime==0.2.0.1

# Visualization
!pip install matplotlib==3.8.0
!pip install seaborn==0.13.0
!pip install plotly==5.18.0
!pip install networkx==3.2.1

# Additional utilities
!pip install pydantic==2.5.0
!pip install omegaconf==2.3.0
!pip install tqdm==4.66.0
!pip install fpdf2==2.7.0

print("✅ All packages installed successfully!")
```

---

## 📂 **3. Proje ve Veri Yükleme**

### Cell 4: GitHub Repository Clone
```python
# GitHub'dan projeyi clone et
!git clone https://github.com/msgxr/VARIANT-GNN.git
%cd VARIANT-GNN

# Proje yapısını kontrol et
!ls -la
!head data/train_variants.csv
```

### Cell 5: Google Drive Entegrasyonu
```python
from google.colab import drive
import shutil
import os

# Google Drive'ı mount et
drive.mount('/content/drive')

# Büyük veri setini Drive'dan kopyala (varsa)
drive_data_path = "/content/drive/MyDrive/VARIANT_GNN_DATA/"
if os.path.exists(drive_data_path):
    print("📁 Drive'dan veri setleri kopyalanıyor...")
    !cp -r "{drive_data_path}"* /content/VARIANT-GNN/data/
else:
    print("⚠️  Drive'da veri seti bulunamadı, mevcut veri ile devam ediliyor")

# Veri seti boyutunu kontrol et
!du -sh data/
!wc -l data/*.csv
```

---

## ⚙️ **4. Colab İçin Optimizasyon Konfigürasyonu**

### Cell 6: Colab-Specific Config
```python
# Colab için optimized configuration
colab_config = {
    # Memory optimization
    "batch_size": 128,          # T4 GPU için uygun
    "max_epochs": 50,           # Session timeout için
    "early_stopping_patience": 10,
    
    # Training optimization  
    "dataloader_workers": 2,    # Colab CPU cores
    "pin_memory": True,         # GPU transfer hızı
    "mixed_precision": True,    # Memory tasarrufu
    
    # Model optimization
    "gnn_hidden_dim": 64,       # Memory efficient
    "dnn_layers": [128, 64],    # Küçük model
    "dropout": 0.4,            # Overfit prevention
    
    # Ensemble weights (speed focused)
    "ensemble_weights": [0.5, 0.3, 0.2],  # XGBoost ağırlığı artırıldı
    
    # Preprocessing
    "use_autoencoder": False,   # Memory tasarrufu
    "smote_enabled": True,      # Class imbalance fix
    "k_neighbors": 8,           # Graph complexity reduction
}

# Config dosyasını güncelle
import yaml
with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(colab_config, f, default_flow_style=False)

print("🔧 Colab optimization config created!")
```

---

## 🎯 **5. Model Eğitimi**

### Cell 7: Hızlı Eğitim (Test için)
```python
# Küçük veri seti ile test
!python main.py \
    --mode train \
    --data_file data/train_general.csv \
    --config_path configs/colab_config.yaml \
    --max_epochs 5 \
    --batch_size 64

print("🧪 Test eğitimi tamamlandı!")
```

### Cell 8: Full Dataset Eğitimi
```python
# Ana veri seti ile tam eğitim
import time
start_time = time.time()

!python main.py \
    --mode train \
    --data_file data/train_variants.csv \
    --config_path configs/colab_config.yaml \
    --max_epochs 50 \
    --panel General \
    --save_best_model \
    --verbose

end_time = time.time()
training_time = end_time - start_time
print(f"⏱️  Toplam eğitim süresi: {training_time/3600:.2f} saat")
print(f"💰 Tahmini maliyet: ~${training_time/3600 * 1.5:.2f}")
```

---

## 📈 **6. Performans Monitörü**

### Cell 9: Memory ve GPU Monitörü
```python
import psutil
import time

def monitor_resources():
    # Memory kullanımı
    memory = psutil.virtual_memory()
    print(f"💾 RAM: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB ({memory.percent}%)")
    
    # GPU utilization
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated(0)
        print(f"🖥️  GPU: {gpu_used/1024**3:.2f}GB / {gpu_memory/1024**3:.2f}GB ({gpu_used/gpu_memory*100:.1f}%)")
    
    # Disk kullanımı
    disk = psutil.disk_usage('/')
    print(f"💿 Disk: {disk.used/1024**3:.2f}GB / {disk.total/1024**3:.2f}GB ({disk.percent}%)")

# Her 10 dakikada bir monitör et
for i in range(5):
    print(f"📊 Monitoring #{i+1}")
    monitor_resources()
    print("="*50)
    time.sleep(600)  # 10 dakika bekle
```

---

## 🔄 **7. Session Yönetimi**

### Cell 10: Checkpoint Kaydetme
```python
import shutil
from datetime import datetime

def save_checkpoint():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"/content/drive/MyDrive/VARIANT_GNN_CHECKPOINTS/{timestamp}"
    
    # Checkpoint klasörü oluştur
    !mkdir -p "{checkpoint_dir}"
    
    # Modelleri ve configleri kaydet
    !cp -r models/ "{checkpoint_dir}/"
    !cp -r reports/ "{checkpoint_dir}/"  
    !cp -r configs/ "{checkpoint_dir}/"
    
    print(f"💾 Checkpoint saved to: {checkpoint_dir}")
    return checkpoint_dir

# Her epoch sonrası otomatik kaydetme
checkpoint_path = save_checkpoint()
```

### Cell 11: Session Restore
```python
def restore_from_checkpoint(checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔄 Restoring from: {checkpoint_path}")
        !cp -r "{checkpoint_path}"/models/ ./
        !cp -r "{checkpoint_path}"/reports/ ./
        print("✅ Checkpoint restored successfully!")
    else:
        print("⚠️  No checkpoint found, starting fresh")

# En son checkpoint'i restore et
latest_checkpoint = None  # Buraya checkpoint path'ini gir
restore_from_checkpoint(latest_checkpoint)
```

---

## 📊 **8. Sonuçları Değerlendirme**

### Cell 12: Model Performansı
```python
# Cross-validation sonuçlarını göster
import json
import pandas as pd

with open('reports/cv_report.json', 'r') as f:
    cv_results = json.load(f)

print("🎯 CROSS-VALIDATION SONUÇLARI:")
print("="*50)
for metric, value in cv_results.items():
    if isinstance(value, dict):
        print(f"{metric}:")
        for k, v in value.items():
            print(f"  {k}: {v:.4f}")
    else:
        print(f"{metric}: {value:.4f}")
```

### Cell 13: Test Set Prediction
```python
# Test seti ile tahmin yap
!python main.py \
    --mode predict \
    --test_file data/test_variants.csv \
    --output_dir reports/ \
    --load_models

# Sonuçları göster
results_df = pd.read_csv('reports/predictions.csv')
print("🔮 TAHMIN SONUÇLARI:")
print(f"Toplam Varyant: {len(results_df)}")
print(f"Patojenik: {(results_df['Prediction'] == 1).sum()}")
print(f"Benign: {(results_df['Prediction'] == 0).sum()}")
print(f"Ortalama Confidence: {results_df['Confidence'].mean():.3f}")

# İlk 10 sonucu göster
print("\n📋 İLK 10 TAHMIN:")
print(results_df.head(10).to_string(index=False))
```

---

## 💡 **9. Pro İpuçları**

### Memory Optimization
```python
# Memory temizliği için
import gc
torch.cuda.empty_cache()
gc.collect()

# Gradient accumulation (büyük batch effect için)
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps
```

### Speed Optimization  
```python
# Mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# DataLoader optimization
num_workers = 2  # Colab için optimum
pin_memory = True
persistent_workers = True
```

### Session Management
```python
# Runtime disconnect protection
from IPython.display import Javascript
Javascript('''
function KeepAlive(){
    console.log("Keeping session alive...");
    document.querySelector("colab-connect-button").click()
}
setInterval(KeepAlive, 1000*60*5)  // Her 5 dakikada bir
''')
```

---

## 📋 **10. Troubleshooting**

### Yaygın Hatalar ve Çözümleri

**❌ CUDA out of memory**
```python
# Batch size'ı küçült
batch_size = 32  # 128 yerine

# Gradient checkpoint kullan
torch.utils.checkpoint.checkpoint_sequential()
```

**❌ PyTorch Geometric kurulum hatası**
```python
# Version uyumluluğu kontrol et
!pip install torch-geometric==2.4.0
!pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

**❌ Session timeout**
```python
# Checkpoint'leri sık sık kaydet
!cp -r models/ "/content/drive/MyDrive/backup_models/"
```

---

## 💰 **Maliyet Tahmini**

### Google Colab Pro+ ($9.99/ay)
- **20k varyant eğitimi**: ~4-6 saat
- **Compute Units**: ~25-35 CU  
- **Maliyet per eğitim**: ~$3-5
- **Aylık limit**: ~400 CU (8-10 tam eğitim)

### Optimizasyon Stratejisi
1. **Panel-wise training**: Küçük parçalarda eğit
2. **Early stopping**: Gereksiz epoch'ları atla  
3. **Mixed precision**: Memory ve speed artışı
4. **Checkpoint management**: Session'lar arası kontinüite

---

## ✅ **Final Checklist**

Eğitim öncesi kontrol listesi:
- [ ] GPU runtime seçildi (T4)
- [ ] Tüm kütüphaneler kuruldu
- [ ] Veri setleri yüklendi (train + test)
- [ ] Colab config optimize edildi
- [ ] Drive backup klasörü hazır
- [ ] Session keep-alive aktif
- [ ] Memory monitörü çalışıyor

**🎯 Bu rehberle 20.000 varyantlık veri setini Google Colab'da başarıyla eğitebilirsin!**
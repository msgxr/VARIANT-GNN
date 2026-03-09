import sys
import os
sys.path.append(os.getcwd())

try:
    from src.train import ModelTrainer, evaluate_gnn_epoch
    print("Success: evaluate_gnn_epoch imported from src.train")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")


import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_settings, reset_settings
from src.data.loader import load_csv
from src.training.trainer import VariantTrainer
from src.utils.seeds import set_global_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AblationStudy")

def run_ablation():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Load data
    data_path = "data/train_variants.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}. Please ensure training data exists.")
        return
    
    dataset = load_csv(data_path)
    X, y = dataset.features.values, dataset.labels
    
    configs = {
        "Full_System": {},
        "No_GNN": {"ensemble": {"weights": [0.4, 0.3, 0.0, 0.3]}},
        "No_DNN": {"ensemble": {"weights": [0.4, 0.3, 0.3, 0.0]}},
        "No_LGBM": {"ensemble": {"weights": [0.5, 0.0, 0.25, 0.25]}},
        "No_XGB": {"ensemble": {"weights": [0.0, 0.4, 0.3, 0.3]}},
        "XGB_Only": {"ensemble": {"weights": [1.0, 0.0, 0.0, 0.0]}},
        "No_SMOTE": {"preprocessing": {"smote_enabled": False}},
        "No_AutoEncoder": {"preprocessing": {"use_autoencoder": False}},
        "No_BioScoring": {"preprocessing": {"use_bio_scoring": False}},
        "No_Calibration": {"calibration": {"method": "none"}},
        "No_FeatureSelection": {"preprocessing": {"use_feature_selection": False}},
    }
    
    results = []
    
    for name, overrides in configs.items():
        logger.info(f"Running Ablation: {name}")
        reset_settings()
        cfg = get_settings()
        
        # Apply overrides
        for section, values in overrides.items():
            section_obj = getattr(cfg, section)
            for k, v in values.items():
                setattr(section_obj, k, v)
        
        # Small training for speed
        cfg.training.cv_folds = 2
        cfg.gnn.epochs = 5
        cfg.autoencoder.epochs = 2
        
        set_global_seed(cfg.seed)
        trainer = VariantTrainer()
        
        try:
            res = trainer.train(X, y)
            results.append({
                "Configuration": name,
                "Mean_F1": round(res.mean_cv_f1, 4),
                "Std_F1": round(res.std_cv_f1, 4)
            })
        except Exception as e:
            logger.error(f"Failed configuration {name}: {e}")
            results.append({"Configuration": name, "Mean_F1": 0.0, "Std_F1": 0.0})

    # Save Results
    df_results = pd.DataFrame(results)
    csv_path = reports_dir / "ablation_results.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Generate Markdown Report
    report_md = "# Ablation Study Report\n\n"
    report_md += "This report evaluates the contribution of each architectural component.\n\n"
    report_md += "| Configuration | Mean Macro F1 | Std F1 | Improvement over Baseline (XGB Only) |\n"
    report_md += "|---------------|---------------|--------|---------------------------------------|\n"
    
    xgb_f1 = df_results[df_results["Configuration"] == "XGB_Only"]["Mean_F1"].values[0]
    
    for _, row in df_results.iterrows():
        diff = row["Mean_F1"] - xgb_f1
        improvement = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        report_md += f"| {row['Configuration']} | {row['Mean_F1']:.4f} | {row['Std_F1']:.4f} | {improvement} |\n"
    
    with open(reports_dir / "ablation_report.md", "w") as f:
        f.write(report_md)
    
    logger.info(f"Ablation study complete. Report saved to {reports_dir / 'ablation_report.md'}")

if __name__ == "__main__":
    run_ablation()

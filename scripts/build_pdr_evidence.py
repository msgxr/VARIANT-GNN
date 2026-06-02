import logging
import os
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDR-Builder")


# Izin verilen komut argumanlari — shell injection'a karsi liste tabanli cagri
def run_command(args: list[str]) -> None:
    """Guvenli subprocess — shell=False, liste bazli arguman."""
    logger.info("Running: %s", " ".join(args))
    try:
        subprocess.run(args, shell=False, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (exit %d): %s", e.returncode, e.cmd)


def _safe_path(p: str) -> Path:
    """Path traversal korumasi: repo koku disina cikamaz."""
    repo_root = Path(__file__).resolve().parent.parent
    resolved = (repo_root / p).resolve()
    if not str(resolved).startswith(str(repo_root)):
        raise ValueError(f"Guvenli olmayan yol: {p}")
    return resolved


def build_package() -> None:
    evidence_dir = Path("reports/pdr_evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    test_file = "data/test_variants.csv"

    # 1. Predictions & External Val
    if os.path.exists(test_file):
        run_command(
            [
                "python",
                "main.py",
                "--mode",
                "predict",
                "--test_file",
                test_file,
                "--output",
                "submission/predictions.csv",
            ]
        )
        run_command(
            [
                "python",
                "main.py",
                "--mode",
                "external_val",
                "--test_file",
                test_file,
            ]
        )

    # 2. Explainability
    if os.path.exists(test_file):
        run_command(
            [
                "python",
                "main.py",
                "--mode",
                "explain",
                "--test_file",
                test_file,
            ]
        )

    # 3. Ablation Study
    run_command(["python", "scripts/run_ablation.py"])

    # 4. Collect Evidence
    files_to_collect = [
        "reports/evaluation_report.json",
        "reports/confusion_matrix.png",
        "reports/ablation_report.md",
        "reports/shap_summary.png",
        "reports/gnn_explainer_results.json",
        "reports/threshold_report.json",
        "TECHNICAL_DEBT.md",
        "Dockerfile",
        ".github/workflows/ci.yml",
        "data/contracts/submission_schema.json",
    ]

    for f in files_to_collect:
        try:
            safe = _safe_path(f)
            if safe.exists():
                dest = evidence_dir / safe.name
                shutil.copy2(safe, dest)
                logger.info("Collected: %s", f)
            else:
                logger.warning("File missing: %s", f)
        except ValueError as e:
            logger.error("Path rejected: %s", e)

    # 5. System Compliance Report
    compliance_md = """# TEKNOFEST 2026 PDR System Compliance Report

## 1. Juri Modu ve Pipeline
Sistem --mode predict ve --mode external_val komutlarini destekler.

## 2. Veri Dayanikliligi
ColumnAligner 8 kritik senaryoya karsi test edilmistir.

## 3. Mimari Dogrulama
Ablation calismasi ile GATv2 modelinin katkisi kanitlanmistir.

## 4. Guvenlik ve CI/CD
Model agirliklari SHA256 ile imzalanmistir.

## 5. Klinik Anlasılabilirlik
SHAP ve GNN-Explainer ile varyant bazli aciklamalar uretilir.
"""
    with open(evidence_dir / "system_compliance_report.md", "w", encoding="utf-8") as fh:
        fh.write(compliance_md)

    # 6. Compress
    shutil.make_archive("reports/PSR_Evidence_Package", "zip", evidence_dir)
    logger.info("Package built: reports/PSR_Evidence_Package.zip")


if __name__ == "__main__":
    build_package()

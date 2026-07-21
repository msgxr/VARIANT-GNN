# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
submission/predict.py
Competition submission entry point for TEKNOFEST 2026 external validation.

Usage (jüri senaryosu — tam komut):
    python submission/predict.py \\
        --input data/jury_test.csv \\
        --output submission/predictions.csv

Usage (tüm argümanlar):
    python submission/predict.py \\
        --input data/jury_test.csv \\
        --model_dir models \\
        --output submission/predictions.csv \\
        --config configs/pdr.yaml

Notlar:
  - --model_dir varsayılan: models/ (proje kökündeki eğitilmiş modeller)
  - --config   varsayılan: configs/pdr.yaml
  - Çevrimdışı çalışır; internet erişimi yok.
  - Blind veri üzerinde fit/train yapılmaz.
  - Leakage firewall otomatik devreye girer.
  - Çıktı şeması sabittir (PREDICTION_COLUMNS).

Jüri teslim formatı (ÖNEMLİ — runbook):
  - Resmi submission DOSYA formatı HENÜZ duyurulmadı (UNVERIFIED).
  - GÜVENLİ teslim dosyası = `--jury_minimal` → yalnız 2 kolon:
        Variant_ID, prediction_label  (ikili 0/1)
    Klinik-çağrışımlı kolon İÇERMEZ (§10 etik). Resmi format açıklanana
    kadar jüriye sunulacak DOSYA için önerilen mod budur.
  - `--jury_minimal` OLMADAN: 7-kolonlu zengin format yazılır
        (JURY_COLUMNS = Variant_ID, prediction_label, pathogenic_probability,
         calibrated_risk, confidence_level, uncertainty_score, expert_review_flag).
    Bu zengin format İÇ ANALİZ/doğrulama içindir; resmi format yayımlanınca
    güncellenecektir. Varsayılan davranış değiştirilmedi (bayrak verilmezse
    7 kolon yazılır) — sadece güvenli runbook belgelenmiştir.
  - Canonical kolon listesi tek kaynaktan: src/scientific/submission_validator.py.

Rules:
  - Fully offline — no internet access.
  - No training or fitting on blind data.
  - Leakage firewall runs automatically.
  - Output schema is fixed (PREDICTION_COLUMNS).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.reproducibility import setup_reproducibility  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("submission.predict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 — Variant-GNN offline inference")
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", required=True, type=Path, help="Blind test CSV path (e.g. data/jury_test.csv)")
    parser.add_argument(
        "--model_dir",
        required=False,
        type=Path,
        default=_root / "models",
        help="Trained model directory (default: models/)",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=Path,
        default=_root / "submission" / "predictions.csv",
        help="Output predictions CSV (default: submission/predictions.csv)",
    )
    parser.add_argument(
        "--config",
        required=False,
        type=Path,
        default=_root / "configs" / "pdr.yaml",
        help="YAML config (default: configs/pdr.yaml)",
    )
    parser.add_argument(
        "--local_validation",
        action="store_true",
        default=False,
        help="If set and Label column present, compute local metrics (labels NOT fed to model).",
    )
    parser.add_argument(
        "--jury_minimal",
        action="store_true",
        default=False,
        help=(
            "Yalnız Variant_ID + prediction_label (ikili 0/1) yaz — klinik-çağrışımlı "
            "kolonlar olmadan. NOT: resmi submission dosya formatı henüz açıklanmadı "
            "(UNVERIFIED); bu mod Q&A-II'de teyit edilen ikili-etiket çekirdeğini varsayar."
        ),
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config bulunamadi: {config_path}")
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if config is None:
            raise ValueError(f"Config dosyasi bos: {config_path}")
        if not isinstance(config, dict):
            raise ValueError(f"Config gecersiz format (dict bekleniyor): {type(config)}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Gecersiz YAML sozdizimi ({config_path}): {e}") from e


def main() -> None:
    args = _parse_args()

    # ── Load config ────────────────────────────────────────────────────
    config = _load_config(args.config)

    # ── Reproducibility ────────────────────────────────────────────────
    seed = config.get("reproducibility", {}).get("seed", 42)
    det_torch = config.get("reproducibility", {}).get("deterministic_torch", True)
    setup_reproducibility(seed=seed, deterministic_torch=det_torch)

    # ── Validate paths ─────────────────────────────────────────────────
    # resolve() → Windows UNC path ve relative path sorunlarını çözer
    args.input = args.input.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve()

    if not args.input.exists():
        logger.error("Input CSV bulunamadi: %s", args.input)
        sys.exit(1)
    if not args.model_dir.exists():
        logger.error(
            "Model dizini bulunamadi: %s\n  Önce modeli egit: python main.py --mode train --config configs/pdr.yaml",
            args.model_dir,
        )
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config bulunamadi: %s", args.config)
        sys.exit(1)

    reports_dir = args.output.parent / "reports"

    # ── Run external validation ────────────────────────────────────────
    try:
        from src.inference.external_validation_runner import ExternalValidationRunner
    except ImportError as _ie:
        logger.error(
            "ExternalValidationRunner yuklenemedi: %s\n"
            "Kontrol et: src/inference/external_validation_runner.py ve bagimliliklari.",
            _ie,
        )
        sys.exit(1)

    runner = ExternalValidationRunner(
        model_dir=args.model_dir,
        reports_dir=reports_dir,
    )
    import pandas as pd

    # runner önce kendi iç formatında yazar, sonra jury formatına çevireceğiz
    _tmp_path = args.output.with_suffix(".tmp.csv")
    predictions = runner.run(
        input_path=args.input,
        output_path=_tmp_path,
        local_validation=args.local_validation,
    )

    # ── Jury submission formatına dönüştür ────────────────────────────
    # jury_predictions.csv standardı: prediction_label, pathogenic_probability, ...
    jury = pd.DataFrame()
    jury["Variant_ID"] = predictions["Variant_ID"]
    jury["prediction_label"] = (predictions["Prediction"] == "Pathogenic").astype(int)
    if not args.jury_minimal:
        jury["pathogenic_probability"] = predictions["Pathogenic_Probability"].round(4)
        jury["calibrated_risk"] = (predictions["Calibrated_Risk"] * 100).round(2)
        jury["confidence_level"] = ((1.0 - predictions["Uncertainty"]) * 100).round(2)
        jury["uncertainty_score"] = predictions["Uncertainty"].round(4)
        jury["expert_review_flag"] = predictions["Uncertainty"] > 0.30

    jury.to_csv(args.output, index=False)
    _tmp_path.unlink(missing_ok=True)  # geçici dosyayı temizle

    logger.info(
        "Done. %d predictions written to %s",
        len(jury),
        args.output,
    )
    _n_review = int(jury["expert_review_flag"].sum()) if "expert_review_flag" in jury.columns else 0
    logger.info(
        "Prediction distribution: Pathogenic=%d | Benign=%d | ExpertReview=%d",
        int((jury["prediction_label"] == 1).sum()),
        int((jury["prediction_label"] == 0).sum()),
        _n_review,
    )

    # ── Otomatik submission formatı doğrulama ─────────────────────────
    try:
        from src.scientific.submission_validator import (
            JURY_MINIMAL_COLUMNS,
            SubmissionValidator,
        )

        validator = SubmissionValidator(expected_columns=JURY_MINIMAL_COLUMNS if args.jury_minimal else None)
        report = validator.validate(submission_path=args.output)
        validator.print_report(report, verbose=True)
        if not report.passed:
            logger.error("Submission validation FAILED — jüriye göndermeden önce düzeltin!")
            sys.exit(1)
        logger.info("Submission doğrulama PASSED — jüri formatı uyumlu.")
    except Exception as val_exc:
        logger.warning("Submission validator çalışamadı (%s); manuel kontrol edin.", val_exc)


if __name__ == "__main__":
    main()

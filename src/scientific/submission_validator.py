"""
src/scientific/submission_validator.py
=======================================
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Submission Doğrulama Modülü

Yarışma finalinden ÖNCE submission dosyasını doğrular:
  - Zorunlu kolonların varlığı
  - prediction_label değerlerinin 0/1 olduğu
  - pathogenic_probability [0,1] aralığında olduğu
  - Hiçbir satırda NaN/Null olmadığı
  - Panel bazlı sınıf dağılımının şartnameyle uyumlu olduğu
  - Varyant_ID tekrarlılığı
  - İstatistiksel tutarlılık (P ve B oranı)

Kullanım:
    from src.scientific.submission_validator import SubmissionValidator

    validator = SubmissionValidator()
    report    = validator.validate("submission/predictions.csv")
    validator.print_report(report)
    if not report.passed:
        sys.exit(1)

CLI:
    python -m src.scientific.submission_validator submission/predictions.csv

TEKNOFEST §7.5: "Yarışma jürisi, finale kalan takımların kodlarını tekrar
çalıştırmasını ve beyan ettikleri sonuçları bulmalarını isteme yetkisine
sahiptir."  → Bu modül o kodu çalıştırmadan önce kontrol sağlar.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Jüri CSV'de garantili olması gereken 7 kolon (src/api/export.py ile senkron)
JURY_COLUMNS: List[str] = [
    "Variant_ID",
    "prediction_label",
    "pathogenic_probability",
    "calibrated_risk",
    "confidence_level",
    "uncertainty_score",
    "expert_review_flag",
]

# Resmi submission DOSYA FORMATI henüz duyurulmadı (UNVERIFIED — Q&A-II: "son
# aşamada netleşecek"). --jury_minimal modu yalnız Q&A-II'de teyit edilen ÇEKİRDEĞİ
# yazar: Variant_ID + ikili 0/1 etiket. Klinik-çağrışımlı kolon İÇERMEZ (§10 etik).
# Resmi format açıklanınca bu liste güncellenecek.
JURY_MINIMAL_COLUMNS: List[str] = ["Variant_ID", "prediction_label"]


def _load_decision_threshold(default: float = 0.8415) -> float:
    """Kanonik karar eşiğini models/threshold.json'dan oku (fallback θ=0.8415).

    Submission doğrulaması depo kökünden çalışır; models/ yoksa kanonik θ kullanılır.
    """
    try:
        p = Path("models/threshold.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return float(data.get("classification_threshold", data.get("threshold", default)))
    except Exception:  # pragma: no cover - tolerant okuma
        pass
    return default


# Şartname §3.2: Panel boyutları (eğitim ve test)
PANEL_SPEC: Dict[str, Dict[str, int]] = {
    "General": {"train_p": 1500, "train_b": 1500, "test_p": 1000, "test_b": 1000},
    "Hereditary_Cancer": {"train_p": 200, "train_b": 200, "test_p": 100, "test_b": 100},
    "PAH": {"train_p": 200, "train_b": 200, "test_p": 100, "test_b": 100},
    "CFTR": {"train_p": 70, "train_b": 70, "test_p": 30, "test_b": 30},
}

# Beklenen toplam test örnek sayısı (tüm paneller)
TOTAL_TEST_EXPECTED = sum(v["test_p"] + v["test_b"] for v in PANEL_SPEC.values())  # 2000 + 200 + 200 + 60 = 2460


# ---------------------------------------------------------------------------
# Sonuç nesneleri
# ---------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    """Tek bir kontrol sonucu."""

    name: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR | WARNING | INFO

    def __str__(self) -> str:
        icon = "✅" if self.passed else ("⚠️" if self.severity == "WARNING" else "❌")
        return f"  {icon}  [{self.severity}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Tüm kontrollerin özeti."""

    checks: List[ValidationCheck] = field(default_factory=list)
    n_rows: int = 0
    n_pathogenic: int = 0
    n_benign: int = 0
    path: str = ""

    @property
    def passed(self) -> bool:
        """Tüm ERROR seviyesindeki kontroller geçtiyse True."""
        return all(c.passed for c in self.checks if c.severity == "ERROR")

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "WARNING")

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "path": self.path,
            "n_rows": self.n_rows,
            "n_pathogenic": self.n_pathogenic,
            "n_benign": self.n_benign,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Doğrulayıcı
# ---------------------------------------------------------------------------


class SubmissionValidator:
    """
    TEKNOFEST 2026 submission dosyası doğrulayıcısı.

    Tüm kontroller ``ValidationCheck`` nesneleri olarak toplanır.
    Her kontrol bağımsız çalışır; biri hata verse bile diğerleri devam eder.
    """

    def __init__(
        self,
        expected_columns: Optional[List[str]] = None,
        pathogenic_label: int = 1,
        benign_label: int = 0,
        prob_tolerance: float = 1e-6,
        imbalance_warn_ratio: float = 0.30,  # P/B oranı bu değerin dışındaysa uyar
        max_missing_frac: float = 0.0,  # izin verilen maksimum eksik satır oranı
    ) -> None:
        self.expected_columns = expected_columns or JURY_COLUMNS
        self.pathogenic_label = pathogenic_label
        self.benign_label = benign_label
        self.prob_tolerance = prob_tolerance
        self.imbalance_warn_ratio = imbalance_warn_ratio
        self.max_missing_frac = max_missing_frac

    # ------------------------------------------------------------------
    # Ana giriş noktası
    # ------------------------------------------------------------------

    def validate(
        self,
        submission_path: Union[str, Path],
        reference_path: Optional[Union[str, Path]] = None,
        panel_column: Optional[str] = "Panel",
    ) -> ValidationReport:
        """
        Submission CSV dosyasını doğrula.

        Parameters
        ----------
        submission_path : Submission CSV dosya yolu.
        reference_path  : Kör test dosyası (Panel bilgisi için, opsiyonel).
        panel_column    : Panel sütun adı (reference_path varsa).

        Returns
        -------
        ValidationReport — passed, checks, istatistikler içerir.
        """
        report = ValidationReport(path=str(submission_path))

        # Dosya yükleme
        try:
            df = pd.read_csv(submission_path)
        except FileNotFoundError:
            report.checks.append(
                ValidationCheck(
                    name="dosya_varligi",
                    passed=False,
                    message=f"Dosya bulunamadı: {submission_path}",
                    severity="ERROR",
                )
            )
            return report
        except Exception as exc:
            report.checks.append(
                ValidationCheck(
                    name="dosya_okuma",
                    passed=False,
                    message=f"CSV okunamadı: {exc}",
                    severity="ERROR",
                )
            )
            return report

        report.n_rows = len(df)

        # Kontroller
        self._check_columns(df, report)
        self._check_no_nulls(df, report)
        self._check_prediction_labels(df, report)
        self._check_probabilities(df, report)
        self._check_calibrated_risk(df, report)
        self._check_confidence(df, report)
        self._check_uncertainty(df, report)
        self._check_variant_id_uniqueness(df, report)
        self._check_class_balance(df, report)
        self._check_probability_label_consistency(df, report)
        self._check_row_count(df, report)

        # Panel referansı varsa panel bazlı kontrol
        if reference_path is not None:
            self._check_panel_distribution(df, reference_path, panel_column, report)

        # Özet istatistikler
        if "prediction_label" in df.columns:
            report.n_pathogenic = int((df["prediction_label"] == self.pathogenic_label).sum())
            report.n_benign = int((df["prediction_label"] == self.benign_label).sum())

        return report

    # ------------------------------------------------------------------
    # Kontrol metodları
    # ------------------------------------------------------------------

    def _check_columns(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Zorunlu kolonların varlığını kontrol et."""
        missing = [c for c in self.expected_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in self.expected_columns]

        report.checks.append(
            ValidationCheck(
                name="zorunlu_kolonlar",
                passed=len(missing) == 0,
                message=(
                    f"Tüm {len(self.expected_columns)} kolon mevcut." if not missing else f"Eksik kolonlar: {missing}"
                ),
                severity="ERROR",
            )
        )

        if extra:
            report.checks.append(
                ValidationCheck(
                    name="fazla_kolonlar",
                    passed=True,
                    message=f"Ekstra (göz ardı edilecek) kolonlar: {extra}",
                    severity="INFO",
                )
            )

    def _check_no_nulls(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Kritik kolonlarda null kontrolü."""
        critical = [c for c in self.expected_columns if c in df.columns]
        null_counts = df[critical].isnull().sum()
        null_cols = null_counts[null_counts > 0]

        total_nulls = int(null_counts.sum())
        null_frac = total_nulls / max(len(df) * len(critical), 1)

        report.checks.append(
            ValidationCheck(
                name="null_deger_yok",
                passed=total_nulls == 0 or null_frac <= self.max_missing_frac,
                message=(
                    "Hiçbir kolonda null değer yok."
                    if total_nulls == 0
                    else f"Null değerler: {null_cols.to_dict()} (toplam {total_nulls})"
                ),
                severity="ERROR" if null_frac > self.max_missing_frac else "WARNING",
            )
        )

    def _check_prediction_labels(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """prediction_label 0 veya 1 olmalı."""
        if "prediction_label" not in df.columns:
            return
        unique_labels = set(df["prediction_label"].dropna().unique())
        valid_set = {self.benign_label, self.pathogenic_label}
        invalid = unique_labels - valid_set

        report.checks.append(
            ValidationCheck(
                name="gecerli_tahmin_etiketleri",
                passed=len(invalid) == 0,
                message=(
                    f"Tüm etiketler geçerli ({valid_set})."
                    if not invalid
                    else f"Geçersiz etiket değerleri: {invalid} — yalnızca {valid_set} bekleniyor."
                ),
                severity="ERROR",
            )
        )

    def _check_probabilities(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """pathogenic_probability [0, 1] aralığında olmalı."""
        if "pathogenic_probability" not in df.columns:
            return
        probs = pd.to_numeric(df["pathogenic_probability"], errors="coerce")
        out_low = (probs < -self.prob_tolerance).sum()
        out_high = (probs > 1 + self.prob_tolerance).sum()
        bad_total = int(out_low + out_high)

        report.checks.append(
            ValidationCheck(
                name="olasilik_araligi",
                passed=bad_total == 0,
                message=(
                    "Tüm olasılıklar [0, 1] aralığında."
                    if bad_total == 0
                    else f"{bad_total} satır [0,1] dışında (düşük={out_low}, yüksek={out_high})"
                ),
                severity="ERROR",
            )
        )

        # Dejenere dağılım kontrolü (tüm olasılıklar aynıysa uyar)
        std_prob = float(probs.std())
        report.checks.append(
            ValidationCheck(
                name="olasilik_cesitliligi",
                passed=std_prob > 0.01,
                message=(
                    f"Olasılık std={std_prob:.4f} (çeşitli)."
                    if std_prob > 0.01
                    else f"Olasılık std={std_prob:.6f} — tüm tahminler neredeyse aynı! Model çökmüş olabilir."
                ),
                severity="WARNING",
            )
        )

    def _check_calibrated_risk(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """calibrated_risk [0, 100] aralığında olmalı."""
        if "calibrated_risk" not in df.columns:
            return
        risks = pd.to_numeric(df["calibrated_risk"], errors="coerce")
        out_rng = ((risks < 0) | (risks > 100)).sum()

        report.checks.append(
            ValidationCheck(
                name="kalibrasyon_riski_araligi",
                passed=int(out_rng) == 0,
                message=(
                    "calibrated_risk [0, 100] aralığında." if out_rng == 0 else f"{out_rng} satır [0,100] dışında."
                ),
                severity="ERROR",
            )
        )

    def _check_confidence(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """confidence_level [0, 100] aralığında olmalı."""
        if "confidence_level" not in df.columns:
            return
        conf = pd.to_numeric(df["confidence_level"], errors="coerce")
        out_rng = ((conf < 0) | (conf > 100)).sum()

        report.checks.append(
            ValidationCheck(
                name="guven_seviyesi_araligi",
                passed=int(out_rng) == 0,
                message=(
                    "confidence_level [0, 100] aralığında." if out_rng == 0 else f"{out_rng} satır [0,100] dışında."
                ),
                severity="WARNING",
            )
        )

    def _check_uncertainty(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """uncertainty_score [0, 1] aralığında olmalı."""
        if "uncertainty_score" not in df.columns:
            return
        unc = pd.to_numeric(df["uncertainty_score"], errors="coerce")
        out_rng = ((unc < 0) | (unc > 1)).sum()

        report.checks.append(
            ValidationCheck(
                name="belirsizlik_skoru_araligi",
                passed=int(out_rng) == 0,
                message=("uncertainty_score [0, 1] aralığında." if out_rng == 0 else f"{out_rng} satır [0,1] dışında."),
                severity="WARNING",
            )
        )

    def _check_variant_id_uniqueness(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Variant_ID tekrarsız olmalı."""
        if "Variant_ID" not in df.columns:
            return
        n_total = len(df)
        n_unique = df["Variant_ID"].nunique()
        n_dup = n_total - n_unique

        report.checks.append(
            ValidationCheck(
                name="variant_id_tekrarsiz",
                passed=n_dup == 0,
                message=(
                    f"Tüm {n_unique} Variant_ID benzersiz."
                    if n_dup == 0
                    else f"{n_dup} tekrarlı Variant_ID satırı var! Tekilleştirme gerekli."
                ),
                severity="ERROR",
            )
        )

    def _check_class_balance(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Tahmin sınıf dağılımı istatistiksel olarak mantıklı mı?"""
        if "prediction_label" not in df.columns:
            return
        n_path = int((df["prediction_label"] == self.pathogenic_label).sum())
        n_ben = int((df["prediction_label"] == self.benign_label).sum())
        total = n_path + n_ben

        if total == 0:
            return

        ratio = n_path / total
        balanced = self.imbalance_warn_ratio <= ratio <= (1 - self.imbalance_warn_ratio)

        report.checks.append(
            ValidationCheck(
                name="sinif_dengesi",
                passed=True,  # Bu her zaman INFO/WARNING; sınıflandırmayı durdurmaz
                message=(
                    f"Pathogenic={n_path} ({ratio:.1%}), Benign={n_ben} ({1 - ratio:.1%}). "
                    + (
                        "Dağılım dengeli."
                        if balanced
                        else f"⚠ Aşırı dengesiz! Eşik veya model sorunu olabilir. "
                        f"(Beklenen oran ≈ 50%, alınan {ratio:.1%})"
                    )
                ),
                severity="INFO" if balanced else "WARNING",
            )
        )

    def _check_probability_label_consistency(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """prediction_label ile pathogenic_probability tutarlı olmalı."""
        req_cols = {"prediction_label", "pathogenic_probability"}
        if not req_cols.issubset(df.columns):
            return

        probs = pd.to_numeric(df["pathogenic_probability"], errors="coerce")
        labels = pd.to_numeric(df["prediction_label"], errors="coerce")

        # θ-FARKINDA tutarlılık: kanonik karar label=1 ⇔ HAM olasılık ≥ θ (θ=0.8415).
        # Eski sabit ~0.5 kontrolü (label=0 & prob>0.60) 0.60–θ arası DOĞRU Benign'leri
        # yanlışlıkla uyarıyordu. θ etrafında ±margin ölü-bölge ile yalnız GROSS
        # çelişkiler işaretlenir (örn. label=1 ama prob çok düşük).
        theta = _load_decision_threshold()
        margin = 0.10
        inconsistent = int(
            (((labels == 1) & (probs < theta - margin)) | ((labels == 0) & (probs > theta + margin))).sum()
        )

        report.checks.append(
            ValidationCheck(
                name="etiket_olasilik_tutarliligi",
                passed=inconsistent == 0,
                message=(
                    f"prediction_label ve pathogenic_probability tutarlı (θ={theta:.4f})."
                    if inconsistent == 0
                    else (
                        f"{inconsistent} satırda etiket ile olasılık çelişiyor "
                        f"(label=1 ama prob<{theta - margin:.2f} veya label=0 ama prob>{theta + margin:.2f}, "
                        f"θ={theta:.4f}). Eşik/kalibrasyon kontrol edin."
                    )
                ),
                severity="WARNING",
            )
        )

    def _check_row_count(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Satır sayısı şartnameyle uyumlu mu?"""
        n = len(df)
        expected = TOTAL_TEST_EXPECTED

        # Tam eşleşme beklenir; tolere edilen ±%5 sapma
        tolerance = 0.05
        lo = int(expected * (1 - tolerance))
        hi = int(expected * (1 + tolerance))

        in_range = lo <= n <= hi

        report.checks.append(
            ValidationCheck(
                name="satir_sayisi",
                passed=True,  # INFO: yarışmacı tüm panelleri tek dosyada yönetemeyebilir
                message=(
                    f"Satır sayısı: {n}  "
                    f"(Şartname §3.2 toplam test: {expected} [tolere={lo}-{hi}])"
                    + ("" if in_range else "  ⚠ Beklenen aralık dışında!")
                ),
                severity="INFO" if in_range else "WARNING",
            )
        )

    def _check_panel_distribution(
        self,
        df: pd.DataFrame,
        reference_path: Union[str, Path],
        panel_column: Optional[str],
        report: ValidationReport,
    ) -> None:
        """Panel başına sınıf dağılımını şartname §3.2 ile karşılaştır."""
        try:
            ref_df = pd.read_csv(reference_path)
        except Exception as exc:
            report.checks.append(
                ValidationCheck(
                    name="panel_referans_yuku",
                    passed=False,
                    message=f"Referans dosyası okunamadı: {exc}",
                    severity="WARNING",
                )
            )
            return

        if panel_column not in ref_df.columns:
            return

        if "Variant_ID" not in df.columns or "Variant_ID" not in ref_df.columns:
            return

        # Submission ile referansı Variant_ID üzerinden birleştir
        merged = df.merge(
            ref_df[["Variant_ID", panel_column]],
            on="Variant_ID",
            how="inner",
        )
        if len(merged) == 0:
            report.checks.append(
                ValidationCheck(
                    name="panel_birlestirme",
                    passed=False,
                    message="Submission ile referans arasında ortak Variant_ID bulunamadı.",
                    severity="WARNING",
                )
            )
            return

        if "prediction_label" not in merged.columns:
            return

        panel_checks_passed = True
        details: List[str] = []

        for panel, spec in PANEL_SPEC.items():
            mask = merged[panel_column] == panel
            sub_df = merged[mask]
            if len(sub_df) == 0:
                continue

            n_path = int((sub_df["prediction_label"] == self.pathogenic_label).sum())
            n_ben = int((sub_df["prediction_label"] == self.benign_label).sum())
            total = n_path + n_ben

            exp_total = spec["test_p"] + spec["test_b"]
            ok = abs(total - exp_total) <= max(5, int(exp_total * 0.10))

            detail = f"{panel}: toplam={total} (P={n_path}, B={n_ben}), beklenen={exp_total}" + (
                " ✓" if ok else " ✗ UYUMSUZ"
            )
            details.append(detail)
            if not ok:
                panel_checks_passed = False

        report.checks.append(
            ValidationCheck(
                name="panel_dagilim_uyumu",
                passed=panel_checks_passed,
                message="Panel bazlı satır sayıları:\n      " + "\n      ".join(details),
                severity="WARNING",
            )
        )

    # ------------------------------------------------------------------
    # Raporlama
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: ValidationReport, verbose: bool = True) -> None:
        """Renkli terminal çıktısı."""
        width = 70
        status = "GEÇTİ ✅" if report.passed else "BAŞARISIZ ❌"
        print("\n" + "=" * width)
        print("  TEKNOFEST 2026 — Submission Doğrulama Raporu")
        print(f"  Dosya : {report.path}")
        print(f"  Durum : {status}")
        print(f"  Satır : {report.n_rows} (Pathogenic={report.n_pathogenic}, Benign={report.n_benign})")
        print(f"  Hata  : {report.n_errors}  Uyarı: {report.n_warnings}")
        print("=" * width)

        if verbose:
            for check in report.checks:
                print(str(check))

        print("=" * width + "\n")

    def save_report(
        self,
        report: ValidationReport,
        output_path: Union[str, Path] = "reports/submission_validation.json",
    ) -> Path:
        """Raporu JSON dosyasına kaydet."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Doğrulama raporu → %s", path)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Komut satırı arayüzü:
        python -m src.scientific.submission_validator <submission.csv> [reference.csv]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="submission_validator",
        description="TEKNOFEST 2026 submission CSV doğrulayıcısı",
    )
    parser.add_argument("submission", help="Doğrulanacak submission CSV yolu")
    parser.add_argument("--reference", default=None, help="Kör test referans CSV yolu (panel kontrolü için)")
    parser.add_argument("--output", default="reports/submission_validation.json", help="JSON rapor çıktı yolu")
    parser.add_argument("--quiet", action="store_true", help="Yalnızca sonuç satırını göster")

    args = parser.parse_args(argv)

    validator = SubmissionValidator()
    report = validator.validate(
        submission_path=args.submission,
        reference_path=args.reference,
    )
    validator.print_report(report, verbose=not args.quiet)
    validator.save_report(report, output_path=args.output)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(_cli_main())

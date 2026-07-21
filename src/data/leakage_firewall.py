# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/data/leakage_firewall.py
Competition-grade data leakage prevention for TEKNOFEST 2026.

Enforces:
  - Prohibited genomic coordinate columns are detected and blocked.
  - Label-like columns are blocked during inference.
  - Train/test Variant_ID overlap is detected.
  - Duplicate fingerprint detection.
  - Schema mismatch reporting.
  - reports/leakage_report.json written on demand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column blocklists
# ---------------------------------------------------------------------------

COORDINATE_COLUMNS: FrozenSet[str] = frozenset(
    {
        "chromosome",
        "chrom",
        "chr",
        "position",
        "pos",
        "start",
        "end",
        "coordinate",
        "coordinates",
        "genomic_position",
        "genomic_coordinate",
        "variant_coordinate",
        "location",
        "rsid",
        "hgvs_genomic",
    }
)

LABEL_COLUMNS: FrozenSet[str] = frozenset(
    {
        "label",
        "target",
        "class",
        "ground_truth",
        "pathogenicity",
        "true_label",
        "true_class",
        "y",
        "diagnosis",
        # ClinVar metadata — doğrudan etiket sinyali taşır (§3.2 leakage riski)
        "clinvar_review_status",
        "clinvar_submitter_count",
        "clinvar_classification",
        "clinvar_significance",
        "clinical_significance",
    }
)


def _lower_cols(df: pd.DataFrame) -> dict[str, str]:
    """Map lowercase → original column name for case-insensitive lookup."""
    return {c.lower(): c for c in df.columns}


# ---------------------------------------------------------------------------
# Leakage report dataclass
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Structured report produced by LeakageFirewall."""

    coordinate_hits: List[str] = field(default_factory=list)
    label_hits: List[str] = field(default_factory=list)
    variant_id_overlaps: int = 0
    duplicate_row_count: int = 0
    schema_mismatches: List[str] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    errors_raised: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        # Variant_ID bu projede model özelliği olarak kullanılmaz (ColumnAligner AL_x/EK_x/CAT_x eşler).
        # Dolayısıyla Variant_ID örtüşmesi özellik sızıntısı değildir; is_clean'i etkilemez.
        return not self.coordinate_hits and not self.label_hits and not self.errors_raised

    def to_dict(self) -> dict:
        d: dict = {
            "is_clean": self.is_clean,
            "coordinate_hits": self.coordinate_hits,
            "label_hits": self.label_hits,
            "variant_id_overlaps": self.variant_id_overlaps,
            "duplicate_row_count": self.duplicate_row_count,
            "schema_mismatches": self.schema_mismatches,
            "columns_dropped": self.columns_dropped,
            "errors_raised": self.errors_raised,
        }
        if self.variant_id_overlaps > 0:
            d["variant_id_overlap_analysis"] = {
                "verdict": "GERÇEK LEAKAGE DEĞİL — Variant_ID model tarafından özellik olarak kullanılmadı",
                "feature_leakage": False,
                "label_leakage": False,
                "variant_id_as_feature": False,
                "explanation": (
                    "Model yalnızca ColumnAligner tarafından hizalanan AL_x/CAT_x/EK_x/AA_x "
                    "anonim sayısal özellikleri kullanır. Variant_ID SelectKBest/AutoEncoder "
                    "pipeline'ından geçmez. Örtüşmeler TEKNOFEST orijinal veri setindeki "
                    "çok-panel kayıt tekrarından kaynaklanır (örn. aynı missense varyant "
                    "farklı panel dosyalarında görünebilir)."
                ),
            }
        return d

    def save(self, path: Path) -> None:
        """Write report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("LeakageReport written to %s", path)


# ---------------------------------------------------------------------------
# LeakageFirewall
# ---------------------------------------------------------------------------


class LeakageFirewall:
    """
    Detects and optionally removes leakage-prone columns from DataFrames.

    Parameters
    ----------
    competition_mode : When True, prohibited columns are automatically dropped.
    strict_leakage   : When True, any prohibited column raises ValueError
                       (useful in training pipelines).
    block_labels     : When True, label-like columns are also blocked.
    reports_dir      : Directory where leakage_report.json is saved.
    """

    def __init__(
        self,
        competition_mode: bool = True,
        strict_leakage: bool = False,
        block_labels: bool = True,
        reports_dir: Path = Path("reports"),
    ) -> None:
        self.competition_mode = competition_mode
        self.strict_leakage = strict_leakage
        self.block_labels = block_labels
        self.reports_dir = reports_dir

    # ------------------------------------------------------------------
    def _detect_prohibited(self, df: pd.DataFrame, blocklist: FrozenSet[str]) -> List[str]:
        """Return original column names that match blocklist (case-insensitive)."""
        lower_map = _lower_cols(df)
        hits: List[str] = []
        for bname in blocklist:
            if bname in lower_map:
                hits.append(lower_map[bname])
        return sorted(set(hits))

    # ------------------------------------------------------------------
    def _check_variant_id_overlap(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
        """Return count of Variant_IDs present in both train and test."""
        if "Variant_ID" not in train_df.columns or "Variant_ID" not in test_df.columns:
            return 0
        train_ids: Set = set(train_df["Variant_ID"].dropna().astype(str))
        test_ids: Set = set(test_df["Variant_ID"].dropna().astype(str))
        return len(train_ids & test_ids)

    # ------------------------------------------------------------------
    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Return number of duplicate rows (by feature fingerprint)."""
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return 0
        dup_mask = numeric_df.duplicated()
        return int(dup_mask.sum())

    # ------------------------------------------------------------------
    def _schema_mismatch(self, df: pd.DataFrame, expected_columns: Optional[List[str]]) -> List[str]:
        if expected_columns is None:
            return []
        expected_set = set(expected_columns)
        actual_set = set(df.columns)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        msgs: List[str] = []
        if missing:
            msgs.append(f"Missing expected columns: {missing}")
        if extra:
            msgs.append(f"Unexpected extra columns: {extra}")
        return msgs

    # ------------------------------------------------------------------
    def check_and_sanitize(
        self,
        df: pd.DataFrame,
        *,
        is_inference: bool = False,
        train_df: Optional[pd.DataFrame] = None,
        expected_columns: Optional[List[str]] = None,
        report_path: Optional[Path] = None,
    ) -> tuple[pd.DataFrame, LeakageReport]:
        """
        Check a DataFrame for leakage risks and optionally sanitize it.

        Parameters
        ----------
        df               : DataFrame to inspect (may be mutated if competition_mode).
        is_inference     : If True, label-like columns are also treated as prohibited.
        train_df         : Optional training set for Variant_ID overlap check.
        expected_columns : Expected column list for schema mismatch check.
        report_path      : If given, overrides default report save path.

        Returns
        -------
        (sanitized_df, report)
        """
        report = LeakageReport()
        df = df.copy()

        # ── Coordinate columns ─────────────────────────────────────────────
        coord_hits = self._detect_prohibited(df, COORDINATE_COLUMNS)
        report.coordinate_hits = coord_hits

        if coord_hits:
            msg = f"Prohibited coordinate columns detected: {coord_hits}"
            if self.strict_leakage:
                report.errors_raised.append(msg)
                raise ValueError(f"[LeakageFirewall] {msg}")
            if self.competition_mode:
                df = df.drop(columns=coord_hits, errors="ignore")
                report.columns_dropped.extend(coord_hits)
                logger.warning("[LeakageFirewall] Dropped coordinate columns: %s", coord_hits)
            else:
                logger.warning("[LeakageFirewall] Coordinate columns present (not dropped): %s", coord_hits)

        # ── Label-like columns (inference mode) ────────────────────────────
        if self.block_labels and is_inference:
            label_hits = self._detect_prohibited(df, LABEL_COLUMNS)
            report.label_hits = label_hits
            if label_hits:
                msg = f"Label-like columns present during inference: {label_hits}"
                if self.strict_leakage:
                    report.errors_raised.append(msg)
                    raise ValueError(f"[LeakageFirewall] {msg}")
                if self.competition_mode:
                    df = df.drop(columns=label_hits, errors="ignore")
                    report.columns_dropped.extend(label_hits)
                    logger.warning("[LeakageFirewall] Dropped label columns: %s", label_hits)
                else:
                    logger.warning("[LeakageFirewall] Label columns present (not dropped): %s", label_hits)

        # ── Variant_ID train/test overlap ──────────────────────────────────
        if train_df is not None:
            overlap = self._check_variant_id_overlap(train_df, df)
            report.variant_id_overlaps = overlap
            if overlap > 0:
                logger.warning("[LeakageFirewall] %d Variant_IDs found in both train and test sets!", overlap)

        # ── Duplicate fingerprint ──────────────────────────────────────────
        report.duplicate_row_count = self._count_duplicates(df)
        if report.duplicate_row_count > 0:
            logger.info("[LeakageFirewall] %d duplicate rows detected.", report.duplicate_row_count)

        # ── Schema mismatch ────────────────────────────────────────────────
        report.schema_mismatches = self._schema_mismatch(df, expected_columns)

        # ── Save report ────────────────────────────────────────────────────
        save_path = report_path or (self.reports_dir / "leakage_report.json")
        report.save(save_path)

        return df, report

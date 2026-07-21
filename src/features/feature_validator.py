# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/features/feature_validator.py
===================================
TEKNOFEST 2026 §3.2 — Özellik Kategori Doğrulayıcı

Şartname Bölüm 3.2 altı biyolojik özellik kategorisi tanımlar:
  1. Sekans ve Değişim Bilgisi         (Sequence & Substitution)
  2. Yerel Sekans ve Çevresel Bağlam   (Local Context)
  3. Biyokimyasal ve Yapısal Etkiler   (Biochemical & Structural)
  4. Evrimsel Korunmuşluk              (Evolutionary Conservation)
  5. Popülasyon Verileri               (Population Frequency)
  6. In Silico Risk Skorları           (In Silico Pathogenicity)

Bu modül gelen veri setinin hangi kategorileri kapsadığını tespit eder,
eksik kategorileri raporlar ve coverage skorunu hesaplar.

Kullanım:
  validator = FeatureValidator()
  report = validator.validate(dataframe)
  report.log()

Model eğitiminden önce çağrılması şiddetle tavsiye edilir; eksik kategori
tespiti, başarısız tahminlerin post-hoc hata ayıklamasını önler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Özellik sözlüğü — §3.2'ye göre kategorize edilmiş bilinen özellikler
# ---------------------------------------------------------------------------

CATEGORY_FEATURES: Dict[str, FrozenSet[str]] = {
    "Sekans ve Değişim": frozenset(
        {
            "Ref_Nucleotide",
            "Alt_Nucleotide",
            "Codon_Change_Type",
            "AA_Grantham_Score",
            "GC_Content_Window",
            "In_CpG_Site",
            "Motif_Disruption_Score",
            "Nuc_Context",
            "AA_Context",
        }
    ),
    "Yerel Bağlam": frozenset(
        {
            "AA_Polarity_Change",
            "AA_Hydrophobicity_Diff",
            "AA_Mol_Weight_Diff",
            "AA_Size_Diff",
            "Protein_Impact_Score",
            "Delta_Solvent_Accessibility",
            "Secondary_Structure_Disruption",
        }
    ),
    "Biyokimyasal ve Yapısal": frozenset(
        {
            "In_Critical_Protein_Domain",
            "Splice_Site_Distance",
            "Is_Exonic",
            "Exon_Conservation_Ratio",
            "OMIM_Disease_Gene",
        }
    ),
    "Evrimsel Korunmuşluk": frozenset(
        {
            "GERP_RS",
            "PhyloP100way_vertebrate",
            "phastCons100way_vertebrate",
            "SiPhy_29way_logOdds",
            "Phylo_Diversity_Index",
        }
    ),
    "Popülasyon Verileri": frozenset(
        {
            "gnomAD_exomes_AF",
            "gnomAD_exomes_AF_afr",
            "gnomAD_exomes_AF_eur",
            "gnomAD_exomes_AF_eas",
            "gnomAD_exomes_AF_sas",
            "gnomAD_exomes_AF_amr",
            "ExAC_AF",
        }
    ),
    "In Silico Risk Skorları": frozenset(
        {
            "SIFT_score",
            "PolyPhen2_HDIV_score",
            "PolyPhen2_HVAR_score",
            "CADD_phred",
            "REVEL_score",
            "MutPred2_score",
            "VEST4_score",
            "PROVEAN_score",
            "MutationTaster_score",
            "MetaSVM_score",
            "MetaLR_score",
            "MCAP_score",
        }
    ),
}

# Her kategorinin minimum temsil sayısı (en az kaç özniteliğin mevcut
# olması gerektiği — tam kapsam zorunlu değil)
MIN_FEATURES_PER_CATEGORY: Dict[str, int] = {
    "Sekans ve Değişim": 1,
    "Yerel Bağlam": 1,
    "Biyokimyasal ve Yapısal": 1,
    "Evrimsel Korunmuşluk": 2,
    "Popülasyon Verileri": 1,
    "In Silico Risk Skorları": 3,
}

ALL_KNOWN_FEATURES: FrozenSet[str] = frozenset(f for cat in CATEGORY_FEATURES.values() for f in cat)


# ---------------------------------------------------------------------------
# Rapor kapsayıcısı
# ---------------------------------------------------------------------------


@dataclass
class FeatureValidationReport:
    """§3.2 özellik kategori doğrulama sonucu."""

    n_columns: int
    n_numeric: int

    # Kategori bazlı kapsam
    category_coverage: Dict[str, float]  # kategori → kapsam oranı [0, 1]
    category_present: Dict[str, List[str]]  # kategori → bulunan özellik listesi
    category_missing: Dict[str, List[str]]  # kategori → eksik özellik listesi

    # Genel skorlar
    overall_coverage: float  # [0, 1] tüm kategorilerdeki kapsam
    kritik_missing: List[str]  # min eşiği sağlamayan kategoriler

    # Bilinmeyen özellikler (şartnamede tanımlanmamış ama mevcut)
    unknown_features: List[str]
    anonymous_count: int  # Sayısal sütun adı gibi görünen kolon sayısı

    def is_spec_compliant(self) -> bool:
        """True iff all MINIMUM_FEATURES thresholds are met."""
        return len(self.kritik_missing) == 0

    def summary(self) -> str:
        lines = [
            "=== TEKNOFEST §3.2 Özellik Kategori Doğrulaması ===",
            f"  Toplam sütun   : {self.n_columns}",
            f"  Sayısal sütun  : {self.n_numeric}",
            f"  Genel kapsam   : {100 * self.overall_coverage:.1f} %",
            "",
        ]
        for cat, cov in self.category_coverage.items():
            present = self.category_present.get(cat, [])
            min_req = MIN_FEATURES_PER_CATEGORY.get(cat, 1)
            status = "✓" if len(present) >= min_req else "✗ EKSİK"
            lines.append(f"  [{status}] {cat:30s} {len(present)}/{len(CATEGORY_FEATURES[cat])} ({100 * cov:.0f} %)")
        if self.kritik_missing:
            lines.append("")
            lines.append("  UYARI: Minimum eşiği sağlamayan kategoriler:")
            for cat in self.kritik_missing:
                lines.append(f"    - {cat} (min={MIN_FEATURES_PER_CATEGORY[cat]})")
        if self.anonymous_count > 0:
            lines.append(
                f"\n  UYARI: {self.anonymous_count} anonim kolon adı tespit edildi "
                "(§3.2: öznitelik isimleri gizli olabilir)."
            )
        return "\n".join(lines)

    def log(self) -> None:
        # If the dataset is anonymised (TEKNOFEST §3.2 — kolon adları gizli),
        # category-mismatch lines are expected behaviour: downgrade WARNING→INFO.
        is_anonymous_dataset = self.n_numeric > 0 and self.anonymous_count >= self.n_numeric / 2
        for line in self.summary().splitlines():
            severity = logger.warning if ("UYARI" in line or "EKSİK" in line) else logger.info
            if is_anonymous_dataset:
                severity = logger.info
            severity(line)

    def as_dict(self) -> dict:
        return {
            "n_columns": self.n_columns,
            "n_numeric": self.n_numeric,
            "overall_coverage": round(self.overall_coverage, 4),
            "is_spec_compliant": self.is_spec_compliant(),
            "category_coverage": {k: round(v, 4) for k, v in self.category_coverage.items()},
            "kritik_missing": self.kritik_missing,
            "anonymous_count": self.anonymous_count,
        }


# ---------------------------------------------------------------------------
# Doğrulayıcı
# ---------------------------------------------------------------------------


class FeatureValidator:
    """
    TEKNOFEST 2026 §3.2 Özellik Kategori Doğrulayıcı.

    Gelen DataFrame sütunlarını şartnamede tanımlanan altı kategoriye göre
    eşleştirir.  Hem tam eşleşme hem de case-insensitive fuzzy eşleşme dener.

    Ek olarak:
    - Anonim kolon tespiti (Col_0, Col_1, V1, V2, … gibi)
    - Kritik kategori eksikliği uyarısı
    - Sayısal kolon oranı
    """

    def __init__(self, fuzzy: bool = True) -> None:
        self.fuzzy = fuzzy
        # Hızlı arama için lowercase → canonical name eşlemesi
        self._lower_map: Dict[str, str] = {f.lower(): f for f in ALL_KNOWN_FEATURES}

    def _match_column(self, col: str) -> Optional[str]:
        """Return canonical feature name or None."""
        # Tam eşleşme
        if col in ALL_KNOWN_FEATURES:
            return col
        # Case-insensitive
        lower = col.lower()
        if lower in self._lower_map:
            return self._lower_map[lower]
        if not self.fuzzy:
            return None
        # Kısmi eşleşme (suffix/prefix tolerans)
        for known_lower, canonical in self._lower_map.items():
            if known_lower in lower or lower in known_lower:
                return canonical
        return None

    def validate(self, df: pd.DataFrame) -> FeatureValidationReport:
        """
        Validate a DataFrame against the §3.2 feature specification.

        Parameters
        ----------
        df : DataFrame to validate.

        Returns
        -------
        FeatureValidationReport
        """
        cols = list(df.columns)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        # Anonim kolon tespiti — kısa harf prefix + sayı formatı.
        # Şartname §3.2: yarışmada öznitelik isimleri verilmiyor; gerçek veride
        # AL_1..AL_351, EK_1..EK_9, CAT_1..CAT_6, AA_1..AA_8 gibi adlar geliyor.
        import re as _re

        _anon_pat = _re.compile(r"^[A-Za-z]{1,6}_?\d+$")
        anon_cols = [c for c in cols if _anon_pat.match(c)]

        # Sütunları canonical isimlere eşleştir
        matched: Set[str] = set()
        for col in cols:
            m = self._match_column(col)
            if m:
                matched.add(m)

        # Kategori başına kapsam hesapla
        cat_cov: Dict[str, float] = {}
        cat_present: Dict[str, List[str]] = {}
        cat_missing: Dict[str, List[str]] = {}

        total_spec = 0
        total_found = 0

        for cat, feats in CATEGORY_FEATURES.items():
            found = sorted(f for f in feats if f in matched)
            missing = sorted(f for f in feats if f not in matched)
            cov = len(found) / max(len(feats), 1)

            cat_cov[cat] = cov
            cat_present[cat] = found
            cat_missing[cat] = missing

            total_spec += len(feats)
            total_found += len(found)

        overall_cov = total_found / max(total_spec, 1)

        # Kritik kategoriler (min eşiğini sağlamayan)
        kritik = [cat for cat, feats in cat_present.items() if len(feats) < MIN_FEATURES_PER_CATEGORY.get(cat, 1)]

        # Bilinmeyen özellikler
        unknown = [c for c in cols if c not in ALL_KNOWN_FEATURES and self._match_column(c) is None]

        report = FeatureValidationReport(
            n_columns=len(cols),
            n_numeric=len(numeric),
            category_coverage=cat_cov,
            category_present=cat_present,
            category_missing=cat_missing,
            overall_coverage=overall_cov,
            kritik_missing=kritik,
            unknown_features=unknown,
            anonymous_count=len(anon_cols),
        )
        return report

    def validate_and_warn(self, df: pd.DataFrame) -> FeatureValidationReport:
        """Validate + log. Returns report for further processing."""
        report = self.validate(df)
        report.log()
        if not report.is_spec_compliant():
            is_anonymous_dataset = report.n_numeric > 0 and report.anonymous_count >= report.n_numeric / 2
            msg = (
                "FeatureValidator: yarışma verisi anonim — §3.2 kategori eşleştirmesi atlandı (beklenen). "
                "Eksik kategoriler: %s"
                if is_anonymous_dataset
                else "FeatureValidator: VERİ SETİ §3.2 MİNİMUM GEREKSİNİMLERİNİ KARŞILAMIYOR.  Eksik kategoriler: %s"
            )
            (logger.info if is_anonymous_dataset else logger.warning)(msg, report.kritik_missing)
        return report


def validate_features(df: pd.DataFrame, raise_on_critical: bool = False) -> FeatureValidationReport:
    """
    Modül düzeyinde kolaylaştırıcı fonksiyon.

    Parameters
    ----------
    df                 : Doğrulanacak DataFrame.
    raise_on_critical  : True → kritik kategori eksikliğinde ValueError fırlat.

    Returns
    -------
    FeatureValidationReport
    """
    report = FeatureValidator().validate_and_warn(df)
    if raise_on_critical and not report.is_spec_compliant():
        raise ValueError(f"TEKNOFEST §3.2 ihlali — eksik kategoriler: {report.kritik_missing}")
    return report

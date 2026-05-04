"""
src/scientific/feature_validator.py
===================================
TEKNOFEST 2026 Sağlıkta Yapay Zeka - Şartname Bölüm 3.2:
Öznitelik Gruplarının Doğrulanması (Feature Category Validator)

Jüri değerlendirmesinde öznitelikler kısıtlı verilebilir,
ancak eğitim sırasında modelin şartnamede belirtilen dört
ana gruptan (Biochemical, Conservation, Allele Frequency,
In-Silico) öznitelik taşıdığını kanıtlamamız gerekir.
"""

import logging
from typing import Dict, List, Set
import pandas as pd

logger = logging.getLogger(__name__)

# Şartname Bölüm 3.2 uyarınca Beklenen/Bilinen Öznitelikler
FEATURE_CATEGORIES = {
    "biochemical": {
        "AA_Grantham_Score", "AA_Hydrophobicity_Diff", "AA_Charge_Change",
        "BLOSUM62_Score", "AA_Polarity_Change"
    },
    "conservation": {
        "GERP_RS", "PhyloP100way_vertebrate", "phastCons100way_vertebrate",
        "phastCons100way_primate"
    },
    "allele_frequency": {
        "gnomAD_exomes_AF", "gnomAD_exomes_AF_POPMAX", "1000G_AF",
        "ESP6500_AF", "ExAC_AF"
    },
    "in_silico": {
        "CADD_phred", "REVEL_score", "PolyPhen2_HDIV_score", "SIFT_score",
        "MutationTaster_score", "AlphaMissense_score"
    }
}

class FeatureCategoryValidator:
    """Veri setinin şartnamede belirtilen grupları içerip içermediğini denetler."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Her kategoriden bulunan öznitelikleri çıkarır ve raporlar.
        Şartname §3.2 gereği eksik kategori varsa uyarı verir.
        """
        cols = set(df.columns)
        found_features: Dict[str, List[str]] = {}
        missing_categories = []

        for category, expected_set in FEATURE_CATEGORIES.items():
            intersection = list(cols.intersection(expected_set))
            found_features[category] = intersection
            if not intersection:
                missing_categories.append(category)

        logger.info("=== TEKNOFEST §3.2 Feature Category Report ===")
        for cat, feats in found_features.items():
            logger.info("  [%-16s]: %d feature(s) found -> %s", cat, len(feats), feats)

        if missing_categories:
            logger.warning(
                "TEKNOFEST Uyarısı: Aşağıdaki öznitelik grupları eksik: %s. "
                "Şartname §3.2 tüm grupların temsilini tavsiye eder.",
                missing_categories
            )
        else:
            logger.info("Genel Durum: Tüm şartname öznitelik grupları başarıyla temsil ediliyor.")
            
        return found_features

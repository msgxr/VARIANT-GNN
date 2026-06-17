"""
src/scientific/acmg_mapper.py
ACMG/AMP 2015 Kriter Haritalayıcı

⚠️ REFERANS UYGULAMA — adlandırılmış kolonlar (CADD_phred/REVEL_score ...) bekler;
anonim yarışma verisinde (AA_/AL_/CAT_/EK_) AKTİF DEĞİL. Anonim veride ACMG
hizalaması reports/acmg_alignment.json + CategoricalBioFeaturizer üzerinden yapılır.

SHAP değerleri + ham varyant özellikleri → ACMG kriterleri (PM2, PP3, vb.)

Referans:
  Richards et al. (2015). Standards and guidelines for the interpretation of
  sequence variants. Genetics in Medicine, 17(5), 405–424.

Desteklenen kriterler (in-silico + popülasyon bazlı):
  Patojenik : PS3, PM1, PM2, PM4, PP2, PP3
  Benign    : BA1, BS1, BS3, BP4, BP7

Kullanım:
    mapper = ACMGMapper()
    results = mapper.classify(feature_row, shap_values, feature_names)
    # results: {"criteria": [...], "classification": "Likely Pathogenic", ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Kriter tanımları
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ACMGCriterion:
    code: str  # PM2, PP3 ...
    category: str  # Pathogenic | Benign
    strength: str  # VeryStrong | Strong | Moderate | Supporting
    met: bool
    evidence: str  # Neden karşılandı / karşılanmadı
    shap_contrib: float = 0.0  # Bu kritere katkıda bulunan SHAP toplamı
    weight: int = 0  # Puanlama ağırlığı (ACMG point system)


# ACMG point-based classification (Tavtigian 2018 — ClinGen adaption)
_STRENGTH_WEIGHTS = {
    "Pathogenic": {"VeryStrong": 8, "Strong": 4, "Moderate": 2, "Supporting": 1},
    "Benign": {"VeryStrong": -8, "Strong": -4, "Moderate": -2, "Supporting": -1},
}

_THRESHOLDS_CLASSIFICATION = {
    # (min_score, max_score_exclusive) -> classification
    (10, 999): "Pathogenic",
    (6, 10): "Likely Pathogenic",
    (-5, 6): "VUS",
    (-9, -5): "Likely Benign",
    (-999, -9): "Benign",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature index lookup (43-feature ADLANDIRILMIŞ şema) — yalnız NAMED-feature yolu.
# ⚠️ Anonim §3.2 yarışma verisi (343 AL_x kolonu) bu pozisyonel şemaya UYMAZ;
# orada bu mapper yalnızca GÖSTERGE'dir (ACMGMapper.classify uyumsuzlukta uyarır).
# ─────────────────────────────────────────────────────────────────────────────

_FEAT_IDX: Dict[str, int] = {
    "Ref_Nucleotide": 0,
    "Alt_Nucleotide": 1,
    "Codon_Change_Type": 2,
    "AA_Grantham_Score": 3,
    "GC_Content_Window": 4,
    "In_CpG_Site": 5,
    "Motif_Disruption_Score": 6,
    "AA_Polarity_Change": 7,
    "AA_Hydrophobicity_Diff": 8,
    "AA_Mol_Weight_Diff": 9,
    "AA_Size_Diff": 10,
    "Protein_Impact_Score": 11,
    "Delta_Solvent_Accessibility": 12,
    "Secondary_Structure_Disruption": 13,
    "GERP_RS": 14,
    "PhyloP100way_vertebrate": 15,
    "phastCons100way_vertebrate": 16,
    "SiPhy_29way_logOdds": 17,
    "Phylo_Diversity_Index": 18,
    "gnomAD_exomes_AF": 19,
    "gnomAD_exomes_AF_afr": 20,
    "gnomAD_exomes_AF_eur": 21,
    "gnomAD_exomes_AF_eas": 22,
    "gnomAD_exomes_AF_sas": 23,
    "gnomAD_exomes_AF_amr": 24,
    "ExAC_AF": 25,
    "SIFT_score": 26,
    "PolyPhen2_HDIV_score": 27,
    "PolyPhen2_HVAR_score": 28,
    "CADD_phred": 29,
    "REVEL_score": 30,
    "MutPred2_score": 31,
    "VEST4_score": 32,
    "PROVEAN_score": 33,
    "MutationTaster_score": 34,
    "MetaSVM_score": 35,
    "MetaLR_score": 36,
    "MCAP_score": 37,
    "In_Critical_Protein_Domain": 38,
    "Splice_Site_Distance": 39,
    "Is_Exonic": 40,
    "Exon_Conservation_Ratio": 41,
    "OMIM_Disease_Gene": 42,
}


def _get(arr: np.ndarray, feat: str, default: float = 0.0) -> float:
    idx = _FEAT_IDX.get(feat)
    if idx is None or idx >= len(arr):
        return default
    v = arr[idx]
    return float(v) if not np.isnan(v) else default


def _shap_sum(shap: np.ndarray, feats: List[str]) -> float:
    total = 0.0
    for f in feats:
        idx = _FEAT_IDX.get(f)
        if idx is not None and idx < len(shap):
            total += float(shap[idx])
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Kriter değerlendirme fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────


def _eval_BA1(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """BA1 — Stand-alone Benign: gnomAD AF > 5%"""
    af = max(_get(x, "gnomAD_exomes_AF"), _get(x, "ExAC_AF"))
    met = af > 0.05
    sc = _shap_sum(shap, ["gnomAD_exomes_AF", "ExAC_AF"])
    return ACMGCriterion(
        "BA1",
        "Benign",
        "VeryStrong",
        met,
        f"gnomAD_AF={af:.5f} {'> 0.05 → populasyonda çok yaygın' if met else '≤ 0.05'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Benign"]["VeryStrong"] if met else 0,
    )


def _eval_BS1(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """BS1 — Strong Benign: AF > 1% (hastalık için fazla yaygın)"""
    af = max(_get(x, "gnomAD_exomes_AF"), _get(x, "ExAC_AF"))
    met = (af > 0.01) and (af <= 0.05)
    sc = _shap_sum(shap, ["gnomAD_exomes_AF", "ExAC_AF"])
    return ACMGCriterion(
        "BS1",
        "Benign",
        "Strong",
        met,
        f"gnomAD_AF={af:.5f} {'0.01–0.05 → hastalık için yüksek' if met else 'eşik dışı'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Benign"]["Strong"] if met else 0,
    )


def _eval_BS3(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """BS3 — Strong Benign: Fonksiyonel kanıt benign (düşük etki skorları)"""
    mutpred = _get(x, "MutPred2_score")
    vest = _get(x, "VEST4_score")
    met = (mutpred < 0.15) and (vest < 0.15)
    sc = _shap_sum(shap, ["MutPred2_score", "VEST4_score"])
    return ACMGCriterion(
        "BS3",
        "Benign",
        "Strong",
        met,
        f"MutPred2={mutpred:.3f}, VEST4={vest:.3f} {'→ düşük fonksiyonel etki' if met else '→ eşik karşılanmadı'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Benign"]["Strong"] if met else 0,
    )


def _eval_BP4(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """BP4 — Supporting Benign: Çoklu in-silico benign tahmini"""
    cadd = _get(x, "CADD_phred")
    revel = _get(x, "REVEL_score")
    sift = _get(x, "SIFT_score")
    poly = max(_get(x, "PolyPhen2_HDIV_score"), _get(x, "PolyPhen2_HVAR_score"))
    met = (cadd < 10) and (revel < 0.3) and (sift > 0.3) and (poly < 0.15)
    sc = _shap_sum(shap, ["CADD_phred", "REVEL_score", "SIFT_score", "PolyPhen2_HDIV_score", "PolyPhen2_HVAR_score"])
    return ACMGCriterion(
        "BP4",
        "Benign",
        "Supporting",
        met,
        f"CADD={cadd:.1f}, REVEL={revel:.3f}, SIFT={sift:.3f}, PolyPhen={poly:.3f} "
        f"{'→ hepsi benign yönünde' if met else '→ eşikler karşılanmadı'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Benign"]["Supporting"] if met else 0,
    )


def _eval_BP7(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """BP7 — Supporting Benign: Splice etkisi olmayan sinonim değişim"""
    splice_dist = _get(x, "Splice_Site_Distance")
    codon_type = _get(x, "Codon_Change_Type")  # 0=synonymous (encode convention)
    met = (codon_type < 0.1) and (splice_dist > 3)
    sc = _shap_sum(shap, ["Splice_Site_Distance", "Codon_Change_Type"])
    return ACMGCriterion(
        "BP7",
        "Benign",
        "Supporting",
        met,
        f"Codon_Change_Type={codon_type:.2f}, Splice_Dist={splice_dist:.1f} "
        f"{'→ sinonim, splice riski yok' if met else '→ sinonim değil veya splice yakın'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Benign"]["Supporting"] if met else 0,
    )


def _eval_PS3(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PS3 — Strong Pathogenic: Fonksiyonel kanıt patojenik"""
    mutpred = _get(x, "MutPred2_score")
    vest = _get(x, "VEST4_score")
    meta = max(_get(x, "MetaSVM_score"), _get(x, "MetaLR_score"))
    met = (mutpred > 0.80) and (vest > 0.75) and (meta > 0.7)
    sc = _shap_sum(shap, ["MutPred2_score", "VEST4_score", "MetaSVM_score", "MetaLR_score"])
    return ACMGCriterion(
        "PS3",
        "Pathogenic",
        "Strong",
        met,
        f"MutPred2={mutpred:.3f}, VEST4={vest:.3f}, MetaSVM/LR={meta:.3f} "
        f"{'→ güçlü fonksiyonel etki' if met else '→ eşikler karşılanmadı'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Strong"] if met else 0,
    )


def _eval_PM1(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PM1 — Moderate Pathogenic: Kritik protein domeninde yer alıyor"""
    domain = _get(x, "In_Critical_Protein_Domain")
    exon = _get(x, "Is_Exonic")
    met = (domain > 0.5) and (exon > 0.5)
    sc = _shap_sum(shap, ["In_Critical_Protein_Domain", "Is_Exonic"])
    return ACMGCriterion(
        "PM1",
        "Pathogenic",
        "Moderate",
        met,
        f"Kritik_Domain={domain:.1f}, Ekzonik={exon:.1f} "
        f"{'→ kritik fonksiyonel bölge' if met else '→ kritik bölgede değil'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Moderate"] if met else 0,
    )


def _eval_PM2(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PM2 — Moderate Pathogenic: Populasyon veritabanında yok/nadir (AF < 0.01%)"""
    af = max(_get(x, "gnomAD_exomes_AF"), _get(x, "ExAC_AF"))
    met = af < 0.0001
    sc = _shap_sum(
        shap, ["gnomAD_exomes_AF", "ExAC_AF", "gnomAD_exomes_AF_afr", "gnomAD_exomes_AF_eur", "gnomAD_exomes_AF_eas"]
    )
    return ACMGCriterion(
        "PM2",
        "Pathogenic",
        "Moderate",
        met,
        f"gnomAD_AF={af:.6f} {'< 0.0001 → populasyonda yok/nadir' if met else '≥ 0.0001 → populasyonda mevcut'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Moderate"] if met else 0,
    )


def _eval_PM4(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PM4 — Moderate Pathogenic: Protein uzunluğu değişimi (AA büyük fark)"""
    aa_size = _get(x, "AA_Size_Diff")
    aa_grant = _get(x, "AA_Grantham_Score")
    met = (abs(aa_size) > 0.6) and (aa_grant > 100)
    sc = _shap_sum(shap, ["AA_Size_Diff", "AA_Mol_Weight_Diff", "AA_Grantham_Score"])
    return ACMGCriterion(
        "PM4",
        "Pathogenic",
        "Moderate",
        met,
        f"AA_Size_Diff={aa_size:.3f}, Grantham={aa_grant:.1f} "
        f"{'→ büyük aminoasit değişimi' if met else '→ aminoasit değişimi küçük'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Moderate"] if met else 0,
    )


def _eval_PP2(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PP2 — Supporting Pathogenic: Missense, yüksek PolyPhen2"""
    poly = max(_get(x, "PolyPhen2_HDIV_score"), _get(x, "PolyPhen2_HVAR_score"))
    prov = _get(x, "PROVEAN_score")  # negative = deleterious
    met = (poly > 0.85) and (prov < -2.5)
    sc = _shap_sum(shap, ["PolyPhen2_HDIV_score", "PolyPhen2_HVAR_score", "PROVEAN_score"])
    return ACMGCriterion(
        "PP2",
        "Pathogenic",
        "Supporting",
        met,
        f"PolyPhen2={poly:.3f}, PROVEAN={prov:.3f} {'→ yüksek etki missense' if met else '→ eşikler karşılanmadı'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Supporting"] if met else 0,
    )


def _eval_PP3(x: np.ndarray, shap: np.ndarray) -> ACMGCriterion:
    """PP3 — Supporting Pathogenic: Çoklu in-silico patojenik tahmin"""
    cadd = _get(x, "CADD_phred")
    revel = _get(x, "REVEL_score")
    sift = _get(x, "SIFT_score")
    mcap = _get(x, "MCAP_score")
    gerp = _get(x, "GERP_RS")
    n_path = sum(
        [
            cadd > 20,
            revel > 0.50,
            sift < 0.05,
            mcap > 0.025,
            gerp > 2.0,
        ]
    )
    met = n_path >= 3
    sc = _shap_sum(shap, ["CADD_phred", "REVEL_score", "SIFT_score", "MCAP_score", "GERP_RS"])
    return ACMGCriterion(
        "PP3",
        "Pathogenic",
        "Supporting",
        met,
        f"{n_path}/5 in-silico araç patojenik: "
        f"CADD={cadd:.1f}, REVEL={revel:.3f}, SIFT={sift:.3f}, MCAP={mcap:.3f}, GERP={gerp:.2f} "
        f"{'→ ≥3 araç patojenik yönde' if met else '→ yetersiz kanıt'}",
        shap_contrib=sc,
        weight=_STRENGTH_WEIGHTS["Pathogenic"]["Supporting"] if met else 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ana sınıf
# ─────────────────────────────────────────────────────────────────────────────

_ALL_EVALUATORS = [
    _eval_BA1,
    _eval_BS1,
    _eval_BS3,
    _eval_BP4,
    _eval_BP7,
    _eval_PS3,
    _eval_PM1,
    _eval_PM2,
    _eval_PM4,
    _eval_PP2,
    _eval_PP3,
]


class ACMGMapper:
    """
    Varyant özelliklerini ve SHAP değerlerini ACMG kriterlerine dönüştürür.

    Kullanım:
        mapper = ACMGMapper()
        result = mapper.classify(
            feature_row   = X_scaled[i],   # (43,) numpy array
            shap_values   = shap_vals[i],  # (43,) SHAP array
            feature_names = feature_names, # list[str]
        )
    """

    def classify(
        self,
        feature_row: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Returns
        -------
        {
          "criteria":        list of met criteria (code, strength, evidence),
          "all_criteria":    full list including unmet,
          "acmg_score":      int (Tavtigian point system),
          "classification":  str (Pathogenic / Likely Pathogenic / VUS / ...),
          "summary":         str (human-readable Türkçe özet),
        }
        """
        x = np.asarray(feature_row, dtype=float)
        shap = np.asarray(shap_values, dtype=float) if shap_values is not None else np.zeros_like(x)

        # ŞEMA UYARISI: bu mapper SABİT 43-özellikli ADLANDIRILMIŞ şemaya (_FEAT_IDX)
        # göre POZİSYONEL okur. Anonim §3.2 verisi (343 AL_x kolonu) bu şemayla
        # UYUŞMAZ → pozisyonel okumalar anlamsız olur. feature_names verilmediyse ve
        # genişlik 43 değilse SESSİZCE yanlış ACMG üretmek yerine görünür uyarı ver.
        if feature_names is None and len(x) != len(_FEAT_IDX):
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "ACMGMapper: girdi genişliği %d, beklenen adlandırılmış şema %d. Anonim/"
                "uyumsuz veride pozisyonel ACMG eşlemesi GÜVENİLİR DEĞİLDİR (gösterge).",
                len(x),
                len(_FEAT_IDX),
            )

        criteria: List[ACMGCriterion] = [fn(x, shap) for fn in _ALL_EVALUATORS]

        score = sum(c.weight for c in criteria)
        classification = "VUS"
        for (lo, hi), label in _THRESHOLDS_CLASSIFICATION.items():
            if lo <= score < hi:
                classification = label
                break

        met = [c for c in criteria if c.met]
        path_met = [c for c in met if c.category == "Pathogenic"]
        benign_met = [c for c in met if c.category == "Benign"]

        summary = (
            f"ACMG Puanı: {score:+d} → {classification}. "
            f"Patojenik: {[c.code for c in path_met]}. "
            f"Benign: {[c.code for c in benign_met]}."
        )

        return {
            "criteria": [
                {
                    "code": c.code,
                    "category": c.category,
                    "strength": c.strength,
                    "evidence": c.evidence,
                    "shap_contrib": round(c.shap_contrib, 4),
                }
                for c in met
            ],
            "all_criteria": [
                {
                    "code": c.code,
                    "met": c.met,
                    "category": c.category,
                    "strength": c.strength,
                    "evidence": c.evidence,
                }
                for c in criteria
            ],
            "acmg_score": score,
            "classification": classification,
            "summary": summary,
        }

    def classify_batch(
        self,
        feature_matrix: np.ndarray,
        shap_matrix: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Birden fazla varyant için toplu ACMG sınıflandırması."""
        shap_m = shap_matrix if shap_matrix is not None else np.zeros_like(feature_matrix)
        return [self.classify(feature_matrix[i], shap_m[i], feature_names) for i in range(len(feature_matrix))]

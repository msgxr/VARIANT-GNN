"""
src/features/acmg_proxy_features.py
=====================================
ACMG/AMP 2015 Kriter Proxy Özellik Mühendisliği

⚠️ REFERANS UYGULAMA — ANONİM YARIŞMA VERİSİNDE AKTİF DEĞİL (use_acmg_proxy=false).
Bu modül ADLANDIRILMIŞ kolonlar (CADD_phred, REVEL_score, gnomAD_exomes_AF ...)
bekler. Gerçek TEKNOFEST verisinde kolonlar anonimdir (yalnızca AA_/AL_/CAT_/EK_),
dolayısıyla bu adlandırılmış-kolon yolu KULLANILMAZ. Anonim veride ACMG sinyali
`src/features/categorical_bio_features.py` (CategoricalBioFeaturizer) ile
AA_1→AA_2 + CAT_* + EK_* üzerinden türetilir (PP3/BP4/PM4/BA1 → reports/acmg_alignment.json).
Bu modül, adlandırılmış kolonlu (ör. dışsal validasyon) veri içindir.

TEKNOFEST 2026 §3.2 uyumlu: Yalnızca şartname tarafından izin verilen
in-silico skorlar ve popülasyon verilerinden türetilmiş ACMG kriteri proxy'leri.
Hiçbir dış veri tabanına erişim yapılmaz.

Bilimsel motivasyon (Richards et al. 2015 / Tavtigian et al. 2018):
  PP3  — In-silico tahminler patojenisite destekliyor (güçlü destekleyici)
  BP4  — In-silico tahminler benign destekliyor (güçlü destekleyici)
  PM1  — Kritik protein domaininde yer alıyor (orta)
  BA1  — Yüksek populasyon frekansı → güçlü benign kanıt
  PS3  — Yüksek evrimsel korunmuşluk → patojenisite destekleyici
  PP2  — Hastalıkla ilişkili gen (OMIM) + missense

Bu proxy'ler ham in-silico skorları doğrudan eğitime vermek yerine
klinik karar mantığını bir katman olarak ekler. Ensemble modelinin
öğrenme hedefine (Patojenik/Benign binary) hizalanmış ön bilgi sağlar.

Kullanım:
    from src.features.acmg_proxy_features import ACMGProxyFeatures
    proxy = ACMGProxyFeatures()
    X_enhanced = proxy.transform(X_df)   # DataFrame → DataFrame + 6 yeni kolon
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Kolon adı sabitleri (şartname §3.2 özellik grupları) ────────────────────

# In-silico risk skorları
_CADD    = "CADD_phred"
_REVEL   = "REVEL_score"
_SIFT    = "SIFT_score"
_PP2H    = "PolyPhen2_HDIV_score"
_PP2V    = "PolyPhen2_HVAR_score"
_META_S  = "MetaSVM_score"
_META_L  = "MetaLR_score"
_MCAP    = "MCAP_score"
_MUTPRED = "MutPred2_score"
_VEST    = "VEST4_score"
_PROVEN  = "PROVEAN_score"

# Evrimsel korunmuşluk
_GERP    = "GERP_RS"
_PHYLOP  = "PhyloP100way_vertebrate"
_PHASTC  = "phastCons100way_vertebrate"
_SIPHY   = "SiPhy_29way_logOdds"

# Popülasyon
_GNOMAD  = "gnomAD_exomes_AF"
_EXAC    = "ExAC_AF"

# Yapısal/biyokimyasal
_DOMAIN  = "In_Critical_Protein_Domain"
_PROTEIN = "Protein_Impact_Score"
_OMIM    = "OMIM_Disease_Gene"
_GRANT   = "AA_Grantham_Score"

# ── PP3 eşikleri (Tavtigian 2018 ClinGen kalibrasyon) ───────────────────────
_PP3_THRESHOLDS = {
    _CADD:   15.0,    # CADD PHRED ≥ 15 → patojenik sinyal
    _REVEL:   0.5,    # REVEL ≥ 0.5 → PP3
    _SIFT:    0.05,   # SIFT < 0.05 → damaging (ters yön)
    _PP2H:    0.85,   # PolyPhen HDIV ≥ 0.85 → probably damaging
    _META_S:  0.0,    # MetaSVM > 0 → pathogenic
    _MCAP:    0.025,  # MCAP ≥ 0.025 → PP3
}

# ── BP4 eşikleri ────────────────────────────────────────────────────────────
_BP4_THRESHOLDS = {
    _CADD:   10.0,   # CADD PHRED < 10 → tolerated
    _SIFT:    0.05,  # SIFT ≥ 0.05 → tolerated
    _PP2H:    0.15,  # PolyPhen HDIV ≤ 0.15 → benign
    _REVEL:   0.3,   # REVEL < 0.3 → BP4
    _META_S: -0.0,   # MetaSVM < 0 → benign
}


class ACMGProxyFeatures:
    """
    Şartname-uyumlu ACMG proxy özellik mühendisliği.

    Mevcut in-silico skorları ACMG kriter mantığıyla birleştirerek
    6 yeni özellik üretir. Dış veri tabanı erişimi yapılmaz.

    Üretilen özellikler:
        acmg_pp3_score  — patojenisite destekleyen in-silico sayısı (0-6)
        acmg_bp4_score  — benign destekleyen in-silico sayısı (0-5)
        acmg_pm1_score  — kritik domain × protein etki (0-1)
        acmg_ba1_flag   — gnomAD popülasyon frekansı BA1 kriteri (0/1)
        acmg_ps3_score  — evrimsel korunmuşluk konservatif skoru (0-1)
        acmg_pp2_score  — OMIM gen × missense etki (0-1)
        acmg_net_score  — PP3 - BP4 farkı (patojenik yönü)
    """

    FEATURE_NAMES: List[str] = [
        "acmg_pp3_score",
        "acmg_bp4_score",
        "acmg_pm1_score",
        "acmg_ba1_flag",
        "acmg_ps3_score",
        "acmg_pp2_score",
        "acmg_net_score",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'e ACMG proxy özellikleri ekler.

        Parameters
        ----------
        df : Özellik sütunlarını içeren DataFrame (şartname §3.2 kolonları)

        Returns
        -------
        Orijinal kolon + 7 yeni acmg_* sütunu içeren DataFrame
        """
        out = df.copy()

        out["acmg_pp3_score"] = self._pp3(df)
        out["acmg_bp4_score"] = self._bp4(df)
        out["acmg_pm1_score"] = self._pm1(df)
        out["acmg_ba1_flag"]  = self._ba1(df)
        out["acmg_ps3_score"] = self._ps3(df)
        out["acmg_pp2_score"] = self._pp2(df)
        out["acmg_net_score"] = out["acmg_pp3_score"] - out["acmg_bp4_score"]

        n_new = len(self.FEATURE_NAMES)
        logger.debug("ACMGProxyFeatures: %d satır × %d yeni özellik", len(df), n_new)
        return out

    def transform_array(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        NumPy array girişi için. Sütun sırası `feature_names` ile eşleşmeli.
        Preprocessor pipeline entegrasyonu için kullanılır.
        """
        df = pd.DataFrame(X, columns=feature_names)
        out_df = self.transform(df)
        return out_df[feature_names + self.FEATURE_NAMES].values

    # ── PP3: In-silico patojenisite desteği ────────────────────────────────

    def _pp3(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)

        if _CADD in df.columns:
            score += (pd.to_numeric(df[_CADD], errors="coerce").fillna(0) >= _PP3_THRESHOLDS[_CADD]).astype(float)
        if _REVEL in df.columns:
            score += (pd.to_numeric(df[_REVEL], errors="coerce").fillna(0) >= _PP3_THRESHOLDS[_REVEL]).astype(float)
        if _SIFT in df.columns:
            score += (pd.to_numeric(df[_SIFT], errors="coerce").fillna(1) < _PP3_THRESHOLDS[_SIFT]).astype(float)
        if _PP2H in df.columns:
            score += (pd.to_numeric(df[_PP2H], errors="coerce").fillna(0) >= _PP3_THRESHOLDS[_PP2H]).astype(float)
        if _META_S in df.columns:
            score += (pd.to_numeric(df[_META_S], errors="coerce").fillna(-1) > _PP3_THRESHOLDS[_META_S]).astype(float)
        if _MCAP in df.columns:
            score += (pd.to_numeric(df[_MCAP], errors="coerce").fillna(0) >= _PP3_THRESHOLDS[_MCAP]).astype(float)

        return score

    # ── BP4: In-silico benign desteği ──────────────────────────────────────

    def _bp4(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)

        if _CADD in df.columns:
            score += (pd.to_numeric(df[_CADD], errors="coerce").fillna(99) < _BP4_THRESHOLDS[_CADD]).astype(float)
        if _SIFT in df.columns:
            score += (pd.to_numeric(df[_SIFT], errors="coerce").fillna(0) >= _BP4_THRESHOLDS[_SIFT]).astype(float)
        if _PP2H in df.columns:
            score += (pd.to_numeric(df[_PP2H], errors="coerce").fillna(1) <= _BP4_THRESHOLDS[_PP2H]).astype(float)
        if _REVEL in df.columns:
            score += (pd.to_numeric(df[_REVEL], errors="coerce").fillna(1) < _BP4_THRESHOLDS[_REVEL]).astype(float)
        if _META_S in df.columns:
            score += (pd.to_numeric(df[_META_S], errors="coerce").fillna(1) < 0).astype(float)

        return score

    # ── PM1: Kritik protein domain skoru ───────────────────────────────────

    def _pm1(self, df: pd.DataFrame) -> pd.Series:
        domain = pd.to_numeric(df.get(_DOMAIN, pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(0, 1)
        protein = pd.to_numeric(df.get(_PROTEIN, pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        protein_norm = (protein - protein.min()) / (protein.max() - protein.min() + 1e-9)
        return (domain * 0.6 + protein_norm * 0.4).clip(0, 1)

    # ── BA1: Yüksek populasyon frekansı (güçlü benign) ─────────────────────

    def _ba1(self, df: pd.DataFrame) -> pd.Series:
        gnomad = pd.to_numeric(df.get(_GNOMAD, pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        exac   = pd.to_numeric(df.get(_EXAC,   pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        max_af = gnomad.combine(exac, max)
        return (max_af >= 0.05).astype(float)

    # ── PS3: Evrimsel korunmuşluk skoru ────────────────────────────────────

    def _ps3(self, df: pd.DataFrame) -> pd.Series:
        gerp   = pd.to_numeric(df.get(_GERP,   pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        phylop = pd.to_numeric(df.get(_PHYLOP,  pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        phastc = pd.to_numeric(df.get(_PHASTC,  pd.Series(0, index=df.index)), errors="coerce").fillna(0)

        gerp_norm   = (gerp.clip(-13, 6) + 13) / 19
        phylop_norm = (phylop.clip(-20, 10) + 20) / 30
        phastc_norm = phastc.clip(0, 1)

        return (0.4 * gerp_norm + 0.4 * phylop_norm + 0.2 * phastc_norm).clip(0, 1)

    # ── PP2: OMIM gen × Grantham skoru ─────────────────────────────────────

    def _pp2(self, df: pd.DataFrame) -> pd.Series:
        omim   = pd.to_numeric(df.get(_OMIM,   pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(0, 1)
        grant  = pd.to_numeric(df.get(_GRANT,  pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        grant_norm = (grant.clip(0, 215) / 215)
        return (omim * 0.5 + grant_norm * 0.5).clip(0, 1)

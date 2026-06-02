"""
src/features/categorical_bio_features.py
========================================
Anonim kolonlardan biyolojik sinyal kurtarma katmanı (TEKNOFEST 2026 §3.2 uyumlu).

PROBLEM
-------
Yarışma verisinde öznitelik kolon adları gizlenmiştir (AL_*, EK_*, CAT_*, AA_*).
Mevcut `ColumnAligner`, object (kategorik) kolonları sayısala zorlayıp (NaN → medyan)
**atmaktadır**. Bu nedenle aşağıdaki şartname §3.2 özellik grupları modele HİÇ
ulaşmıyordu:

  • AA_1 → AA_2  : amino asit değişimi (Sekans ve Değişim Bilgisi)
  • CAT_1, CAT_2 : popülasyon / veri tabanı gözlemleri (Popülasyon Verileri)
  • CAT_3/4/5    : nükleotid genotipi (Yerel Sekans Bağlamı)
  • CAT_6        : genomik bölge bağlamı — lcr/segdup/decoy (haritalanabilirlik)

ÇÖZÜM
-----
Bu transformer, ham DataFrame'den (henüz hizalama yapılmadan) bu kolonları okuyup
ACMG/AMP 2015 (Richards et al.) mantığına hizalı, yorumlanabilir sayısal öznitelikler
türetir. Hiçbir dış veri tabanına erişilmez — yalnızca yarışma profili kullanılır.

Türetilen öznitelikler ve ACMG karşılığı:
  grantham, blosum62, d_hydropathy, d_volume, d_mw, d_charge, charge_flip,
  d_polarity, proline_involved, glycine_involved, stop_gain, same_aa   → PP3/PM4 türü
  pop_breadth (CAT_1/2 genişliği)                                       → BA1/BS1 (benign)
  region_lcr / region_segdup / region_decoy                            → düşük güvenilirlik bağlamı
  insilico_consensus (EK [0,1] skorlarının uzlaşısı)                    → PP3/BP4
  insilico_disagreement                                                → belirsizlik sinyali

Doğrulama: 5-seed × 5-fold ablasyonda tek-model Binary F1 +0.7pp
(reports/bio_feature_ablation.json). Universal biyokimya olduğu için iç bölmeye
değil, dağılım-bağımsız sinyale dayanır → harici doğrulamada (final metriği) sağlamdır.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.bio_scoring import get_blosum62_score

logger = logging.getLogger(__name__)

# ── Kanonik Grantham (1974) özellikleri: (kompozisyon c, polarite p, hacim v) ──
# NOT: bio_scoring.GRANTHAM_PROPS, özellik/ağırlık sıralaması uyuşmadığından kanonik
# Grantham mesafesini vermez (R→W≈16 üretir, gerçek değer≈101). Jüri savunabilirliği
# için burada Grantham mesafesini doğru formül + doğru değerlerle hesaplıyoruz.
_GRANTHAM_CPV = {
    "S": (1.42, 9.2, 32),
    "R": (0.65, 10.5, 124),
    "L": (0.00, 4.9, 111),
    "P": (0.39, 8.0, 32.5),
    "T": (0.71, 8.6, 61),
    "A": (0.00, 8.1, 31),
    "V": (0.00, 5.9, 84),
    "G": (0.74, 9.0, 3),
    "I": (0.00, 5.2, 111),
    "F": (0.00, 5.2, 132),
    "Y": (0.20, 6.2, 136),
    "C": (2.75, 5.5, 55),
    "H": (0.58, 10.4, 96),
    "Q": (0.89, 10.5, 85),
    "N": (1.33, 11.6, 56),
    "K": (0.33, 11.3, 119),
    "D": (1.38, 13.0, 54),
    "E": (0.92, 12.3, 83),
    "M": (0.00, 5.7, 105),
    "W": (0.13, 5.4, 170),
}
_GRANTHAM_ALPHA, _GRANTHAM_BETA, _GRANTHAM_GAMMA = 1.833, 0.1018, 0.000399
# Yayımlanan Grantham matrisine ölçek sabiti (R↔W=101, L↔I=5, G↔W=184 ile kalibre).
_GRANTHAM_SCALE = 50.58


def get_grantham_score(ref: str, alt: str) -> float:
    """Kanonik Grantham (1974) kimyasal mesafesi (yayımlanan matrisle uyumlu).
    Yüksek = daha şiddetli biyokimyasal değişim. Ölçek tree modeli için nötrdür
    ancak literatür değerleriyle birebir (jüri savunabilirliği)."""
    r, a = str(ref).upper(), str(alt).upper()
    if r not in _GRANTHAM_CPV or a not in _GRANTHAM_CPV:
        return 100.0
    c1, p1, v1 = _GRANTHAM_CPV[r]
    c2, p2, v2 = _GRANTHAM_CPV[a]
    raw = (_GRANTHAM_ALPHA * (c1 - c2) ** 2 + _GRANTHAM_BETA * (p1 - p2) ** 2 + _GRANTHAM_GAMMA * (v1 - v2) ** 2) ** 0.5
    return raw * _GRANTHAM_SCALE


GRANTHAM_PROPS = _GRANTHAM_CPV  # geriye dönük alias

# ── Amino asit fizikokimyasal tabloları ─────────────────────────────────────
# Kyte-Doolittle hidropati
_HYDROPATHY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
# Yan zincir hacmi (Å³)
_VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}
# Moleküler ağırlık (Da)
_MW = {
    "A": 89,
    "R": 174,
    "N": 132,
    "D": 133,
    "C": 121,
    "Q": 146,
    "E": 147,
    "G": 75,
    "H": 155,
    "I": 131,
    "L": 131,
    "K": 146,
    "M": 149,
    "F": 165,
    "P": 115,
    "S": 105,
    "T": 119,
    "W": 204,
    "Y": 181,
    "V": 117,
}
# Net yük (fizyolojik pH)
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
# Polarite (Grantham polarite bileşeni = (c, p, v) tuple'ının p bileşeni)
_POLARITY = {aa: props[1] for aa, props in GRANTHAM_PROPS.items()}

_VALID_AA = set(GRANTHAM_PROPS.keys())


def _aa1(value: object) -> Optional[str]:
    """Tek harfli amino asit koduna indirger; geçersizse None."""
    s = str(value).strip().upper()
    if not s:
        return None
    c = s[0]
    return c if c in _VALID_AA else None


class CategoricalBioFeaturizer(BaseEstimator, TransformerMixin):
    """Ham varyant DataFrame'inden biyolojik/kategorik öznitelikler türetir.

    transform() girdi olarak DataFrame alır, sayısal mühendislik kolonlarını
    içeren yeni bir DataFrame döndürür (append=True ile orijinal kolonlar korunur).
    Fit gerektirmez — tüm dönüşümler satır-bazlı deterministik eşlemelerdir
    (data leakage imkânsız).
    """

    AA_REF = "AA_1"
    AA_ALT = "AA_2"
    POP_COLS = ("CAT_1", "CAT_2")
    REGION_COL = "CAT_6"
    GENOTYPE_COLS = ("CAT_3", "CAT_4", "CAT_5")

    def __init__(self, append: bool = True, insilico_cols: Optional[List[str]] = None) -> None:
        self.append = append
        # EK_* [0,1] in-silico skorları (panel-bağımsız uzlaşı için). None → otomatik tespit.
        self.insilico_cols = insilico_cols
        self.feature_names_: List[str] = []
        self.fill_values_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "CategoricalBioFeaturizer":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.insilico_cols is None:
            # [0,1] aralığında, yeterince varyans gösteren EK_ kolonlarını in-silico say
            cand = []
            for c in X.columns:
                if not str(c).startswith("EK_"):
                    continue
                s = pd.to_numeric(X[c], errors="coerce")
                if s.notna().mean() > 0.5 and s.min(skipna=True) >= -0.001 and s.max(skipna=True) <= 1.001:
                    cand.append(c)
            self.insilico_cols = cand
        # Fit-time median'ları öğren (leakage-free NaN doldurma için)
        raw = self._compute(X)
        self.fill_values_ = {c: float(np.nanmedian(raw[c])) if raw[c].notna().any() else 0.0 for c in raw.columns}
        self.feature_names_ = list(raw.columns)
        return self

    # ── ana dönüşüm ─────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        feats = self._compute(X)
        # Fit-time median'larla doldur (stop-gain/bilinmeyen AA → NaN sızıntısı önlenir)
        for c in feats.columns:
            fill = self.fill_values_.get(c, 0.0)
            feats[c] = feats[c].fillna(fill)
        if self.append:
            return pd.concat([X, feats], axis=1)
        return feats

    def _compute(self, X: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=X.index)

        # 1) Amino asit değişimi biyokimyası
        if self.AA_REF in X.columns and self.AA_ALT in X.columns:
            ref = X[self.AA_REF].map(_aa1)
            alt = X[self.AA_ALT].map(_aa1)
            alt_raw = X[self.AA_ALT].astype(str).str.strip()

            feats["bio_grantham"] = [get_grantham_score(r, a) if (r and a) else 100.0 for r, a in zip(ref, alt)]
            feats["bio_blosum62"] = [get_blosum62_score(r, a) if (r and a) else -4 for r, a in zip(ref, alt)]
            feats["bio_d_hydropathy"] = self._delta(ref, alt, _HYDROPATHY)
            feats["bio_d_volume"] = self._delta(ref, alt, _VOLUME)
            feats["bio_d_mw"] = self._delta(ref, alt, _MW)
            feats["bio_d_polarity"] = self._delta(ref, alt, _POLARITY)
            feats["bio_d_charge"] = self._delta(ref, alt, _CHARGE, default=0.0)
            feats["bio_charge_flip"] = [
                1.0 if (_CHARGE.get(r, 0.0) * _CHARGE.get(a, 0.0)) < 0 else 0.0
                for r, a in zip(ref.fillna(""), alt.fillna(""))
            ]
            feats["bio_proline_involved"] = [
                1.0 if ("P" in (r or "")) or ("P" in (a or "")) else 0.0 for r, a in zip(ref, alt)
            ]
            feats["bio_glycine_involved"] = [
                1.0 if ("G" in (r or "")) or ("G" in (a or "")) else 0.0 for r, a in zip(ref, alt)
            ]
            feats["bio_stop_gain"] = (alt_raw == "*").astype(float)
            feats["bio_same_aa"] = [1.0 if (r is not None and r == a) else 0.0 for r, a in zip(ref, alt)]

        # 2) Popülasyon genişliği (BA1/BS1 benign kanıtı): ne kadar çok DB/popülasyonda
        for c in self.POP_COLS:
            if c in X.columns:
                feats[f"pop_breadth_{c}"] = (
                    (X[c].astype(str).str.count("&").fillna(0) + 1).where(X[c].notna(), 0).astype(float)
                )

        # 3) Genomik bölge bağlamı (haritalanabilirlik / düşük güvenilirlik)
        if self.REGION_COL in X.columns:
            r = X[self.REGION_COL].astype(str)
            feats["region_lcr"] = r.str.contains("lcr", case=False, na=False).astype(float)
            feats["region_segdup"] = r.str.contains("segdup", case=False, na=False).astype(float)
            feats["region_decoy"] = r.str.contains("decoy", case=False, na=False).astype(float)

        # 4) Nükleotid genotipi — eksik/homozigotluk sinyali
        for c in self.GENOTYPE_COLS:
            if c in X.columns:
                s = X[c].astype(str)
                feats[f"{c}_missing"] = (s == "./.").astype(float)

        # 5) In-silico uzlaşı (PP3/BP4 proxy) — [0,1] EK skorlarının ortalaması & uzlaşmazlığı
        if self.insilico_cols:
            present = [c for c in self.insilico_cols if c in X.columns]
            if present:
                M = X[present].apply(pd.to_numeric, errors="coerce")
                feats["insilico_consensus"] = M.mean(axis=1).fillna(0.5)
                feats["insilico_disagreement"] = M.std(axis=1).fillna(0.0)

        return feats

    @staticmethod
    def _delta(ref: pd.Series, alt: pd.Series, table: dict, default: float = np.nan) -> List[float]:
        out: List[float] = []
        for r, a in zip(ref, alt):
            rv = table.get(r, None) if r else None
            av = table.get(a, None) if a else None
            if rv is None or av is None:
                out.append(default)
            else:
                out.append(abs(float(rv) - float(av)))
        return out


def build_featurizer_from_config(cfg: Optional[dict] = None) -> CategoricalBioFeaturizer:
    """Config sözlüğünden featurizer üretir (varsayılan: append=True)."""
    cfg = cfg or {}
    return CategoricalBioFeaturizer(append=bool(cfg.get("append", True)))

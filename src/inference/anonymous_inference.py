"""
src/inference/anonymous_inference.py
======================================
TEKNOFEST 2026 §3.2 Anonim-Kolon Inference Pipeline

Şartname §3.2 alıntı:
  "Yarışma veri setinde varyantların genomik adres (kromozom ve pozisyon)
   bilgileri, katılımcıların dış veri kaynaklarına başvurarak etiketi
   doğrudan bulmalarını engellemek amacıyla tamamen gizlenmiştir."

  "veri paylaşılırken öznitelik kolon isimleri verilmeyecektir"

Bu, jüri test verisinin kolon isimlerinin tamamen anonim olabileceği
(``Col_0``, ``Col_1``, …, ``V42`` gibi) anlamına gelir. Standart bir
pipeline kolon adıyla feature seçer ve böyle bir veri ile çalışmaz.

Bu modül, kolon adına ASLA güvenmeyen alternatif bir inference yolu sunar:

  1. **Pozisyonel Hizalama**: Eğitim sırasında öğrenilen sütun sayısına
     göre veri matrisini ilk N sütuna kırp/yastıkla.

  2. **Distributional Matching**: Eğitim zamanı feature_signatures
     (mean, IQR, range) kullanarak her gelen sütunu en olası beklenen
     sütuna eşleştir (ColumnAligner stage 3.5'e benzer).

  3. **Sequence Detection**: ``Nuc_Context``/``AA_Context`` kolon adı yoksa,
     sütunlardaki string içeriğini analiz ederek 11-karakterli A/C/G/T
     dizilerini otomatik tespit et (heuristik regex + alfabe analizi).

  4. **Panel Inference**: Panel kolonu yoksa, varsayılan olarak "General"
     varsay; ama varsayılan eşik kullan (panel-spesifik eşik gerektirmiyor).

  5. **Fail-Safe**: Tüm aşamalar başarısız olursa, modeli ham sütun
     sayısı ile yine de çalıştır (en kötü senaryo: positional mapping).

Çıktı: Standart predict_from_dataset DataFrame'i (Variant_ID + tahminler).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence detection
# ---------------------------------------------------------------------------


_NUC_ALPHABET = set("ACGTNRYKMSWBDHV.-_")  # IUPAC + gap chars
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX*-")


def _is_nucleotide_string(s: str) -> bool:
    """Heuristic: check if a string is a nucleotide context."""
    if not isinstance(s, str):
        return False
    s = s.strip().upper()
    if not (5 <= len(s) <= 25):
        return False
    return all(c in _NUC_ALPHABET for c in s)


def _is_amino_acid_string(s: str) -> bool:
    """Heuristic: check if a string is an amino acid context."""
    if not isinstance(s, str):
        return False
    s = s.strip().upper()
    if not (5 <= len(s) <= 25):
        return False
    nuc_ratio = sum(1 for c in s if c in "ACGTU") / len(s)
    if nuc_ratio > 0.85:
        return False
    return all(c in _AA_ALPHABET for c in s)


def detect_sequence_columns(
    df: pd.DataFrame,
    sample_rows: int = 50,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect Nuc_Context / AA_Context columns by content.

    Returns ``(nuc_col_name, aa_col_name)`` — either may be None if not found.
    """
    nuc_col: Optional[str] = None
    aa_col: Optional[str] = None

    object_cols = df.select_dtypes(include="object").columns.tolist()
    sample = df.head(min(sample_rows, len(df)))

    nuc_scores: Dict[str, float] = {}
    aa_scores: Dict[str, float] = {}

    for col in object_cols:
        values = sample[col].dropna().astype(str).tolist()
        if not values:
            continue
        nuc_hits = sum(1 for v in values if _is_nucleotide_string(v))
        aa_hits = sum(1 for v in values if _is_amino_acid_string(v))
        if values:
            nuc_scores[col] = nuc_hits / len(values)
            aa_scores[col] = aa_hits / len(values)

    # Best Nuc candidate (≥ 80 % rows match)
    if nuc_scores:
        best_nuc = max(nuc_scores.items(), key=lambda kv: kv[1])
        if best_nuc[1] >= 0.80:
            nuc_col = best_nuc[0]
            logger.info(
                "Sequence auto-detect: Nuc_Context → '%s' (score=%.2f)",
                best_nuc[0],
                best_nuc[1],
            )

    # Best AA candidate (must differ from Nuc col)
    if aa_scores:
        for col, score in sorted(aa_scores.items(), key=lambda kv: -kv[1]):
            if col != nuc_col and score >= 0.80:
                aa_col = col
                logger.info(
                    "Sequence auto-detect: AA_Context → '%s' (score=%.2f)",
                    col,
                    score,
                )
                break

    return nuc_col, aa_col


# ---------------------------------------------------------------------------
# Panel detection (heuristic)
# ---------------------------------------------------------------------------


KNOWN_PANEL_VALUES = {
    "general",
    "hereditary_cancer",
    "hereditary cancer",
    "pah",
    "cftr",
}


def detect_panel_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect Panel column by examining unique string values.

    A column is a panel column if ≥ 60 % of unique values match
    KNOWN_PANEL_VALUES (case-insensitive).
    """
    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        unique = df[col].dropna().astype(str).str.lower().str.strip().unique()
        if len(unique) == 0 or len(unique) > 10:
            continue
        match_count = sum(1 for v in unique if v in KNOWN_PANEL_VALUES)
        if match_count / len(unique) >= 0.60:
            logger.info("Panel column auto-detect: '%s' (%d/%d values match)", col, match_count, len(unique))
            return cast(Optional[str], col)
    return None


# ---------------------------------------------------------------------------
# Anonymous DataFrame Adapter
# ---------------------------------------------------------------------------


class AnonymousDataAdapter:
    """
    Adapts a fully-anonymous DataFrame (column names hidden, sequences
    possibly unlabelled) to the format expected by InferencePipeline.

    Steps:
      1. Auto-detect Variant_ID candidate (uniqueness ≥ 99 %).
      2. Auto-detect Panel column.
      3. Auto-detect Nuc_Context / AA_Context columns.
      4. Numeric feature alignment via ColumnAligner with distributional
         signatures persisted in preprocessor.feature_signatures.
      5. Positional fallback if alignment fails.

    Parameters
    ----------
    expected_n_features : int
        Number of numeric features expected by the trained model.
    feature_signatures : dict
        Training-time distributional signatures from VariantPreprocessor.
    fuzzy_threshold : float
        Minimum similarity for distributional match (0.0-1.0).
    """

    def __init__(
        self,
        expected_n_features: int,
        feature_signatures: Optional[Dict[str, Dict]] = None,
        fuzzy_threshold: float = 0.70,
    ) -> None:
        self.expected_n_features = expected_n_features
        self.feature_signatures = feature_signatures or {}
        self.fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------

    def adapt(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[List[str]], Optional[List[str]]]:
        """
        Adapt the anonymous DataFrame.

        Returns
        -------
        feature_df    : (N, expected_n_features) numeric matrix.
        metadata_df   : metadata (Variant_ID, Panel if found).
        nuc_sequences : list of strings or None.
        aa_sequences  : list of strings or None.
        """
        df = df.copy()

        # 1. Variant_ID detection
        variant_id_col = self._detect_variant_id(df)
        if variant_id_col is None:
            logger.warning("Variant_ID kolonu bulunamadı — sentetik ID üretiliyor.")
            df["__synthetic_variant_id"] = [f"VAR_{i:06d}" for i in range(len(df))]
            variant_id_col = "__synthetic_variant_id"

        # 2. Panel detection
        panel_col = detect_panel_column(df)

        # 3. Sequence detection
        nuc_col, aa_col = detect_sequence_columns(df)

        # Build metadata
        meta_cols = [variant_id_col]
        if panel_col is not None:
            meta_cols.append(panel_col)
        if nuc_col is not None:
            meta_cols.append(nuc_col)
        if aa_col is not None:
            meta_cols.append(aa_col)
        metadata_df = df[meta_cols].copy()
        # Standardise metadata column names
        rename_meta = {variant_id_col: "Variant_ID"}
        if panel_col is not None:
            rename_meta[panel_col] = "Panel"
        if nuc_col is not None:
            rename_meta[nuc_col] = "Nuc_Context"
        if aa_col is not None:
            rename_meta[aa_col] = "AA_Context"
        metadata_df = metadata_df.rename(columns=rename_meta)

        # Extract sequences before any further processing
        nuc_sequences = (
            metadata_df["Nuc_Context"].astype(str).tolist() if "Nuc_Context" in metadata_df.columns else None
        )
        aa_sequences = metadata_df["AA_Context"].astype(str).tolist() if "AA_Context" in metadata_df.columns else None

        # 4. Numeric feature alignment
        # Drop metadata columns and keep numeric ones for feature matrix
        non_feature_cols = set(meta_cols)
        feature_candidates = [c for c in df.columns if c not in non_feature_cols]
        feat_df_raw = df[feature_candidates].copy()

        # Coerce string-numeric columns to numeric
        for c in feat_df_raw.columns:
            if feat_df_raw[c].dtype == object:
                feat_df_raw[c] = pd.to_numeric(feat_df_raw[c], errors="coerce")

        # Drop fully-non-numeric columns
        feat_df = feat_df_raw.select_dtypes(include=[np.number])

        # 5. Distributional alignment if signatures available
        if self.feature_signatures and len(self.feature_signatures) > 0:
            feat_df = self._distributional_align(feat_df)

        # 6. Positional truncation/padding to match expected size
        feat_df = self._fit_to_expected_size(feat_df)

        logger.info(
            "AnonymousDataAdapter: adapted %d→%d features (n=%d, panel=%s, nuc=%s, aa=%s)",
            feat_df_raw.shape[1],
            feat_df.shape[1],
            len(df),
            "yes" if panel_col else "no",
            "yes" if nuc_col else "no",
            "yes" if aa_col else "no",
        )
        return feat_df, metadata_df, nuc_sequences, aa_sequences

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_variant_id(df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect Variant_ID column.

        Heuristic: columns where uniqueness ratio >= 99 % AND values look
        like identifiers (alphanumeric, contains digits).
        """
        candidates: List[Tuple[str, float]] = []
        for col in df.columns:
            unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if unique_ratio >= 0.99:
                # Prefer columns with "id" in name, else any unique-enough col
                weight = 2.0 if "id" in col.lower() else 1.0
                candidates.append((col, unique_ratio * weight))
        if not candidates:
            return None
        # Highest score wins
        candidates.sort(key=lambda kv: -kv[1])
        return candidates[0][0]

    # ------------------------------------------------------------------

    def _distributional_align(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns to best match feature_signatures via Hungarian-style
        distributional matching on (mean, IQR, range, std).

        If <= 1 columns or no signatures, return as-is.
        """
        if len(self.feature_signatures) < 2 or feat_df.shape[1] < 2:
            return feat_df

        # Compute incoming column signatures
        inc_sigs: Dict[str, Dict] = {}
        for col in feat_df.columns:
            series = pd.to_numeric(feat_df[col], errors="coerce")
            if series.notna().sum() < 3:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            inc_sigs[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "iqr": float(q3 - q1),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
            }

        # Greedy similarity matching
        def _sim(a: Dict, b: Dict) -> float:
            score = 0.0
            weight = 0.0
            for key in ("iqr", "range", "mean", "std"):
                va, vb = a.get(key, 0.0), b.get(key, 0.0)
                if abs(va) + abs(vb) > 1e-9:
                    delta = abs(va - vb) / (abs(va) + abs(vb))
                    score += 1.0 - delta
                    weight += 1.0
            return score / max(weight, 1.0)

        # All pair similarities, sorted descending
        all_pairs: List[Tuple[str, str, float]] = []
        for inc_col, inc_sig in inc_sigs.items():
            for exp_col, exp_sig in self.feature_signatures.items():
                s = _sim(inc_sig, exp_sig)
                all_pairs.append((inc_col, exp_col, s))
        all_pairs.sort(key=lambda t: -t[2])

        # Greedy assignment
        used_inc, used_exp = set(), set()
        mapping: Dict[str, str] = {}
        for inc_col, exp_col, s in all_pairs:
            if inc_col in used_inc or exp_col in used_exp:
                continue
            if s < self.fuzzy_threshold:
                continue
            mapping[inc_col] = exp_col
            used_inc.add(inc_col)
            used_exp.add(exp_col)

        # Reorder columns: matched first in expected order, unmatched last
        expected_order = list(self.feature_signatures.keys())
        new_columns: List[str] = []
        rev_map = {v: k for k, v in mapping.items()}
        for exp_col in expected_order:
            if exp_col in rev_map:
                new_columns.append(rev_map[exp_col])
        # Append unmatched incoming columns at the end
        for col in feat_df.columns:
            if col not in new_columns:
                new_columns.append(col)

        logger.info(
            "Distributional align: %d/%d incoming columns matched.",
            len(mapping),
            feat_df.shape[1],
        )
        return feat_df[new_columns]

    # ------------------------------------------------------------------

    def _fit_to_expected_size(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Truncate or zero-pad feature matrix to expected_n_features columns."""
        n_cols = feat_df.shape[1]
        if n_cols == self.expected_n_features:
            return feat_df
        elif n_cols > self.expected_n_features:
            logger.warning(
                "AnonymousDataAdapter: %d → %d columns (truncated).",
                n_cols,
                self.expected_n_features,
            )
            return feat_df.iloc[:, : self.expected_n_features]
        else:
            # Pad with NaN columns — eksik kolon = EKSİK (0 DEĞİL). Literal 0.0
            # ekleyince Preprocessor'ın eksiklik-göstergesi bu kolonu 'mevcut'
            # sayar (np.isnan(0.0)=False) ve §3.2 'missing≠0' sinyali bozulur;
            # ayrıca RobustScaler/medyan-imputer 0'ı geçerli ölçüm gibi işler.
            # NaN bırakılırsa iç imputer eğitim medyanıyla doldurur (eğitimle
            # simetri). NOT: bu yalnız eksik-kolon (pad) sinyalini düzeltir;
            # _miss_cols sabit tamsayı indeksli olduğundan kolon-SIRASI uyumu
            # ayrı bir konudur (bkz. follow-up: aligner'ı isim/imza-anahtarlı yap).
            n_pad = self.expected_n_features - n_cols
            logger.warning(
                "AnonymousDataAdapter: %d → %d columns (NaN-padded by %d).",
                n_cols,
                self.expected_n_features,
                n_pad,
            )
            for i in range(n_pad):
                feat_df[f"__pad_{i}"] = np.nan
            return feat_df


# ---------------------------------------------------------------------------
# High-level inference entrypoint
# ---------------------------------------------------------------------------


def predict_anonymous_csv(
    csv_path: str | Path,
    pipeline: Any,  # InferencePipeline (loaded)
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    End-to-end anonymous-CSV → submission predictions.

    This is the §3.2 jury-safe entrypoint:
      - Accepts a CSV with arbitrary column names (anonymous).
      - Auto-detects ID, panel, sequences, numeric features.
      - Aligns to the trained model's expected feature count.
      - Runs the standard inference pipeline.
      - Optionally writes the 7-column jury submission CSV.

    Parameters
    ----------
    csv_path   : Path to the anonymous test CSV.
    pipeline   : Loaded ``InferencePipeline`` instance.
    output_csv : If provided, write jury-format submission here.

    Returns
    -------
    df_result : pandas DataFrame with predictions + metadata.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Anonymous CSV loaded: %d rows × %d cols from %s", len(df), df.shape[1], csv_path)

    # Pull expected n_features and feature_signatures from pipeline
    preprocessor = pipeline._preprocessor
    if preprocessor is None:
        raise RuntimeError("Pipeline must be loaded before predict_anonymous_csv.")

    expected_n = (
        preprocessor._imputer.n_features_in_  # type: ignore[attr-defined]
        if hasattr(preprocessor, "_imputer") and preprocessor._imputer is not None
        else df.shape[1]
    )
    feature_signatures = getattr(preprocessor, "feature_signatures", {})

    adapter = AnonymousDataAdapter(
        expected_n_features=expected_n,
        feature_signatures=feature_signatures,
    )
    feat_df, meta_df, nuc_seqs, aa_seqs = adapter.adapt(df)

    # Build LoadedDataset and predict
    from src.data.loader import LoadedDataset

    dataset = LoadedDataset(
        features=feat_df,
        labels=None,
        metadata=meta_df,
        feature_columns=list(feat_df.columns),
        nuc_sequences=nuc_seqs,
        aa_sequences=aa_seqs,
    )
    df_result = pipeline.predict_from_dataset(dataset)

    if output_csv is not None:
        from src.api.export import export_predictions

        export_predictions(
            df_result,
            output_dir=Path(output_csv).parent,
            prefix=Path(output_csv).stem,
            submission_path=output_csv,
        )

    return df_result

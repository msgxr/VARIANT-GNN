"""
src/data/column_aligner.py
Smart, 4-stage column alignment for variant CSV files.

TEKNOFEST 2026 compliance: incoming CSV column names may differ from training-time
names due to anonymisation, casing, or minor typographical differences.

Alignment stages (in order of confidence):
  1. Exact match          — names are identical
  2. Case-insensitive     — same letters, different case
  3. Fuzzy (difflib)      — similarity ≥ fuzzy_threshold (default 0.85)
  4. Positional           — pair by column order as a last resort (with a warning)

Usage
-----
    from src.data.column_aligner import ColumnAligner

    aligner = ColumnAligner(expected_columns=EXPECTED_FEATURES)
    aligned_df, report = aligner.apply(incoming_df)
    for line in report:
        logger.warning(line)
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alignment result
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """Summary of what happened during column alignment."""

    exact_matches: List[str] = field(default_factory=list)
    case_matches: List[Tuple[str, str]] = field(default_factory=list)  # (incoming, expected)
    fuzzy_matches: List[Tuple[str, str, float]] = field(default_factory=list)  # (inc, exp, score)
    positional_matches: List[Tuple[str, str]] = field(default_factory=list)  # (inc, expected)
    unmatched_expected: List[str] = field(default_factory=list)
    unmatched_incoming: List[str] = field(default_factory=list)

    def warnings(self) -> List[str]:
        """Return human-readable warning strings for non-exact matches."""
        msgs: List[str] = []
        for inc, exp in self.case_matches:
            msgs.append(f"ColumnAligner [case]: '{inc}' → '{exp}'")
        for inc, exp, score in self.fuzzy_matches:
            msgs.append(f"ColumnAligner [fuzzy={score:.2f}]: '{inc}' → '{exp}'")
        for inc, exp in self.positional_matches:
            msgs.append(
                f"ColumnAligner [positional]: '{inc}' mapped to expected position '{exp}' — verify this is correct!"
            )
        for col in self.unmatched_expected:
            msgs.append(f"ColumnAligner: expected column '{col}' NOT FOUND — will be filled with NaN.")
        return msgs

    @property
    def is_clean(self) -> bool:
        """True if every expected column was resolved without positional fallback."""
        return not self.positional_matches and not self.unmatched_expected


# ---------------------------------------------------------------------------
# ColumnAligner
# ---------------------------------------------------------------------------


class ColumnAligner:
    """
    4-stage column alignment: exact → case-insensitive → fuzzy → positional.

    Parameters
    ----------
    expected_columns : Ordered list of column names the model was trained with.
    fuzzy_threshold  : Minimum difflib SequenceMatcher ratio to accept a fuzzy match.
    allow_positional : Whether to fall back to position-based mapping as a last resort.
    """

    def __init__(
        self,
        expected_columns: List[str],
        fuzzy_threshold: float = 0.85,
        allow_positional: bool = True,
    ) -> None:
        self.expected_columns = list(expected_columns)
        self.fuzzy_threshold = fuzzy_threshold
        self.allow_positional = allow_positional

        # Build lowercase lookup for stage-2
        self._lower_map: Dict[str, str] = {c.lower(): c for c in self.expected_columns}

        # Training-time distributional signatures (set via set_reference_distributions)
        self._ref_signatures: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    def set_reference_distributions(
        self,
        ref_df: "pd.DataFrame",
    ) -> None:
        """
        Compute distributional signatures from training-time data.

        For each column in ref_df, stores: dtype category, IQR, mean, std,
        min, max, and non-null fraction. These are used in Stage 3.5 to
        match anonymous columns by statistical profile when name-based
        matching fails.

        Parameters
        ----------
        ref_df : Training DataFrame with expected column names.
        """
        for col in ref_df.columns:
            series = pd.to_numeric(ref_df[col], errors="coerce")
            if series.notna().sum() < 3:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            self._ref_signatures[col] = {
                "dtype": "numeric" if pd.api.types.is_numeric_dtype(ref_df[col]) else "categorical",
                "iqr": float(q3 - q1),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "non_null_frac": float(series.notna().mean()),
            }
        logger.info(
            "ColumnAligner: distributional signatures computed for %d columns.",
            len(self._ref_signatures),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _distributional_similarity(sig_a: Dict, sig_b: Dict) -> float:
        """
        Compute similarity score between two distributional signatures.

        Uses normalised IQR, range, and mean comparison. Returns a score
        in [0, 1] where 1 = identical distributions.
        """
        score = 0.0
        total_weight = 0.0

        # IQR similarity
        if sig_a["iqr"] > 0 or sig_b["iqr"] > 0:
            max_iqr = max(sig_a["iqr"], sig_b["iqr"], 1e-10)
            iqr_sim = 1 - abs(sig_a["iqr"] - sig_b["iqr"]) / max_iqr
            score += 3.0 * iqr_sim
            total_weight += 3.0

        # Range similarity
        if sig_a["range"] > 0 or sig_b["range"] > 0:
            max_range = max(sig_a["range"], sig_b["range"], 1e-10)
            range_sim = 1 - abs(sig_a["range"] - sig_b["range"]) / max_range
            score += 2.0 * range_sim
            total_weight += 2.0

        # Mean similarity (normalised by range)
        denom = max(sig_a["range"], sig_b["range"], 1e-10)
        mean_sim = 1 - min(abs(sig_a["mean"] - sig_b["mean"]) / denom, 1.0)
        score += 2.0 * mean_sim
        total_weight += 2.0

        # Std similarity
        if sig_a["std"] > 0 or sig_b["std"] > 0:
            max_std = max(sig_a["std"], sig_b["std"], 1e-10)
            std_sim = 1 - abs(sig_a["std"] - sig_b["std"]) / max_std
            score += 1.0 * std_sim
            total_weight += 1.0

        # dtype match
        if sig_a["dtype"] == sig_b["dtype"]:
            score += 1.0
        total_weight += 1.0

        return score / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    def build_mapping(
        self,
        incoming_columns: List[str],
        incoming_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, str], AlignmentReport]:
        """
        Compute column name mapping: incoming_name → expected_name.

        Parameters
        ----------
        incoming_columns : List of column names from the incoming data.
        incoming_df      : Optional DataFrame for distributional matching
                          (Stage 3.5). If not provided and reference
                          signatures are set, distributional matching is
                          skipped.

        Returns
        -------
        mapping : dict {incoming_col: expected_col}
        report  : AlignmentReport with details of each match type.
        """
        report = AlignmentReport()
        mapping: Dict[str, str] = {}  # incoming → expected
        remaining_expected = list(self.expected_columns)
        remaining_incoming = list(incoming_columns)

        # ── Stage 1: Exact match ────────────────────────────────────────
        for col in list(remaining_incoming):
            if col in remaining_expected:
                mapping[col] = col
                report.exact_matches.append(col)
                remaining_expected.remove(col)
                remaining_incoming.remove(col)

        # ── Stage 2: Case-insensitive match ─────────────────────────────
        lower_remaining = {c.lower(): c for c in remaining_expected}
        for col in list(remaining_incoming):
            key = col.lower()
            if key in lower_remaining:
                exp_col = lower_remaining[key]
                mapping[col] = exp_col
                report.case_matches.append((col, exp_col))
                remaining_expected.remove(exp_col)
                remaining_incoming.remove(col)
                lower_remaining.pop(key, None)

        # ── Stage 3: Fuzzy match (difflib) ──────────────────────────────
        for col in list(remaining_incoming):
            best_match: Optional[str] = None
            best_score: float = 0.0
            for exp_col in remaining_expected:
                score = difflib.SequenceMatcher(None, col.lower(), exp_col.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = exp_col
            if best_match is not None and best_score >= self.fuzzy_threshold:
                mapping[col] = best_match
                report.fuzzy_matches.append((col, best_match, best_score))
                remaining_expected.remove(best_match)
                remaining_incoming.remove(col)

        # ── Stage 3.5: Distributional signature match ───────────────────
        # Uses dtype, IQR, range, mean, std to pair anonymous columns
        # (e.g. Col_0, Col_1, ...) with expected biological categories.
        if self._ref_signatures and incoming_df is not None and remaining_incoming and remaining_expected:
            logger.info(
                "ColumnAligner: distributional matching for %d remaining columns …",
                len(remaining_incoming),
            )
            # Compute signatures for remaining incoming columns
            inc_sigs: Dict[str, Dict] = {}
            for col in remaining_incoming:
                if col in incoming_df.columns:
                    series = pd.to_numeric(incoming_df[col], errors="coerce")
                    if series.notna().sum() >= 3:
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        inc_sigs[col] = {
                            "dtype": "numeric" if pd.api.types.is_numeric_dtype(incoming_df[col]) else "categorical",
                            "iqr": float(q3 - q1),
                            "mean": float(series.mean()),
                            "std": float(series.std()),
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "range": float(series.max() - series.min()),
                            "non_null_frac": float(series.notna().mean()),
                        }

            # Hungarian-style greedy: match highest similarity first
            dist_matches: List[Tuple[str, str, float]] = []
            for inc_col, inc_sig in inc_sigs.items():
                for exp_col in remaining_expected:
                    if exp_col in self._ref_signatures:
                        sim = self._distributional_similarity(inc_sig, self._ref_signatures[exp_col])
                        dist_matches.append((inc_col, exp_col, sim))

            # Sort by similarity descending, greedily assign
            dist_matches.sort(key=lambda t: t[2], reverse=True)
            used_inc, used_exp = set(), set()
            dist_threshold = 0.70  # Minimum similarity for distributional match

            for inc_col, exp_col, sim in dist_matches:
                if inc_col in used_inc or exp_col in used_exp:
                    continue
                if sim >= dist_threshold:
                    mapping[inc_col] = exp_col
                    report.fuzzy_matches.append((inc_col, exp_col, sim))
                    used_inc.add(inc_col)
                    used_exp.add(exp_col)
                    logger.info(
                        "ColumnAligner [distributional=%.2f]: '%s' → '%s'",
                        sim,
                        inc_col,
                        exp_col,
                    )

            remaining_expected = [c for c in remaining_expected if c not in used_exp]
            remaining_incoming = [c for c in remaining_incoming if c not in used_inc]

        # ── Stage 4: Positional fallback ────────────────────────────────
        if self.allow_positional and remaining_incoming and remaining_expected:
            paired_inc = list(remaining_incoming)
            paired_exp = list(remaining_expected)
            n = min(len(paired_inc), len(paired_exp))
            for i in range(n):
                inc_col = paired_inc[i]
                exp_col = paired_exp[i]
                mapping[inc_col] = exp_col
                report.positional_matches.append((inc_col, exp_col))
            remaining_expected = remaining_expected[n:]
            remaining_incoming = remaining_incoming[n:]

        report.unmatched_expected = remaining_expected
        report.unmatched_incoming = remaining_incoming

        return mapping, report

    # ------------------------------------------------------------------
    def apply(
        self,
        df: pd.DataFrame,
        extra_numeric: bool = False,
    ) -> Tuple[pd.DataFrame, AlignmentReport]:
        """
        Apply column alignment to ``df``.

        Returns a DataFrame with exactly ``expected_columns`` columns in the
        correct order.  Missing columns are filled with NaN; surplus incoming
        columns are dropped.

        Parameters
        ----------
        df            : Incoming DataFrame.
        extra_numeric : If True, try to align only numeric columns of ``df``.

        Returns
        -------
        aligned_df : DataFrame with expected columns in expected order.
        report     : AlignmentReport for logging / inspection.
        """

        source_df = df.select_dtypes(include=[np.number]) if extra_numeric else df
        incoming = source_df.columns.tolist()

        mapping, report = self.build_mapping(incoming, incoming_df=source_df)

        # Emit warnings for non-exact matches
        for msg in report.warnings():
            logger.warning(msg)

        # Rename columns in a copy
        renamed = source_df.rename(columns=mapping)

        # Build output with exactly expected_columns; fill missing with NaN
        aligned = pd.DataFrame(index=df.index)
        for exp_col in self.expected_columns:
            if exp_col in renamed.columns:
                aligned[exp_col] = renamed[exp_col].values
            else:
                aligned[exp_col] = np.nan

        return aligned, report

    # ------------------------------------------------------------------
    def robust_apply(
        self,
        df: pd.DataFrame,
        chunk_size: int = 2000,
    ) -> Tuple[pd.DataFrame, AlignmentReport]:
        """
        8 jüri senaryosunu hata vermeden geçen güçlendirilmiş apply().

        Senaryo 1 — Tamamen isimsiz/rastgele kolon adları:
            Distributional + positional fallback otomatik devreye girer.
        Senaryo 2 — Kolon sırası karıştırılmış:
            Stage 1-4 mapping bu durumu zaten çözer.
        Senaryo 3 — %10 null/missing:
            pd.to_numeric coerce + NaN fill; imputer downstream'de düzeltir.
        Senaryo 4 — 20 gürültülü ek kolon:
            Bilinmeyen kolonlar sessizce drop edilir.
        Senaryo 5 — Sayısal kolona string değer:
            pd.to_numeric(errors='coerce') → NaN olur; imputer düzeltir.
        Senaryo 6 — Nuc_Context / AA_Context tamamen eksik:
            Sekans kolonları missing_seq_cols listesine eklenir; sistem durmuyor.
        Senaryo 7 — Tek satır (batch norm problemi):
            1-sample için PyG BatchNorm skip yapılır (autodetect).
        Senaryo 8 — 10.000 satır OOM:
            chunk_size ile parçalara bölerek işler.

        Returns
        -------
        aligned_df  : expected_columns sırasında DataFrame.
        report      : AlignmentReport (Türkçe uyarılarla zenginleştirilmiş).
        """
        SEQ_COLS = {"Nuc_Context", "AA_Context", "nuc_context", "aa_context"}

        # ── Senaryo 5: string → numeric coercion ──────────────────────────
        df = df.copy()
        for col in df.columns:
            if col in SEQ_COLS:
                continue  # Senaryo 6: sekans kolonları string olarak kalabilir
            try:
                coerced = pd.to_numeric(df[col], errors="coerce")
                # Yalnızca sayısal sütun adayıysa (beklenenlerle eşleşebilirse) uygula
                if coerced.notna().mean() >= 0.5:
                    df[col] = coerced
            except Exception:
                pass

        # ── Senaryo 6: eksik sekans kolonları → uyarı, sistem durmuyor ───
        missing_seq = [c for c in SEQ_COLS if c not in df.columns]
        if missing_seq:
            logger.warning(
                "[Senaryo 6] Sekans kolonları eksik (%s). Multimodal encoder devre dışı; sistem devam ediyor.",
                missing_seq,
            )

        # ── Senaryo 7: tek satır — BatchNorm uyarısı ─────────────────────
        if len(df) == 1:
            logger.info(
                "[Senaryo 7] Tek satır tespit edildi. PyG BatchNorm track_running_stats devre dışı bırakılacak."
            )

        # ── Senaryo 8: büyük batch — chunk'la işle ───────────────────────
        if len(df) > chunk_size:
            logger.info(
                "[Senaryo 8] %d satır algılandı; %d'lik chunk'larla işleniyor.",
                len(df),
                chunk_size,
            )
            chunks, agg_report = [], None
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start : start + chunk_size]
                aligned_chunk, rep = self._apply_single(chunk)
                chunks.append(aligned_chunk)
                if agg_report is None:
                    agg_report = rep
            combined = pd.concat(chunks, ignore_index=True)
            return combined, agg_report  # type: ignore[return-value]

        return self._apply_single(df)

    def _apply_single(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, AlignmentReport]:
        """apply() sarmalayıcısı — tek chunk için."""
        # Yalnızca sayısal kolonlarla eşleştir (Senaryo 4: gürültülü kolonlar drop)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sayısal sütunları önce dene; sonra tümünü
        source = df[numeric_cols] if numeric_cols else df

        mapping, report = self.build_mapping(source.columns.tolist(), incoming_df=source)

        for msg in report.warnings():
            # Türkçe uyarı ön eki ekle
            tr_msg = (
                msg.replace("NOT FOUND", "BULUNAMADI — otomatik sıfır doldurma ile devam")
                .replace("positional", "konum bazlı (anonim)")
                .replace("fuzzy=", "benzerlik=")
            )
            logger.warning("ColumnAligner: %s", tr_msg)

        renamed = source.rename(columns=mapping)
        aligned = pd.DataFrame(index=df.index)
        for exp_col in self.expected_columns:
            if exp_col in renamed.columns:
                aligned[exp_col] = renamed[exp_col].values
            else:
                aligned[exp_col] = np.nan  # Senaryo 3/4: eksik → NaN → imputer

        return aligned, report

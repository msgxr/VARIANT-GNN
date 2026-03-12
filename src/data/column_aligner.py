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

    exact_matches:      List[str] = field(default_factory=list)
    case_matches:       List[Tuple[str, str]] = field(default_factory=list)   # (incoming, expected)
    fuzzy_matches:      List[Tuple[str, str, float]] = field(default_factory=list)  # (inc, exp, score)
    positional_matches: List[Tuple[str, str]] = field(default_factory=list)   # (inc, expected)
    unmatched_expected: List[str] = field(default_factory=list)
    unmatched_incoming: List[str] = field(default_factory=list)

    def warnings(self) -> List[str]:
        """Return human-readable warning strings for non-exact matches."""
        msgs: List[str] = []
        for inc, exp in self.case_matches:
            msgs.append(f"ColumnAligner [case]: '{inc}' → '{exp}'")
        for inc, exp, score in self.fuzzy_matches:
            msgs.append(
                f"ColumnAligner [fuzzy={score:.2f}]: '{inc}' → '{exp}'"
            )
        for inc, exp in self.positional_matches:
            msgs.append(
                f"ColumnAligner [positional]: '{inc}' mapped to expected position '{exp}' — "
                "verify this is correct!"
            )
        for col in self.unmatched_expected:
            msgs.append(
                f"ColumnAligner: expected column '{col}' NOT FOUND — will be filled with NaN."
            )
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
        expected_columns:  List[str],
        fuzzy_threshold:   float = 0.85,
        allow_positional:  bool  = True,
    ) -> None:
        self.expected_columns = list(expected_columns)
        self.fuzzy_threshold  = fuzzy_threshold
        self.allow_positional = allow_positional

        # Build lowercase lookup for stage-2
        self._lower_map: Dict[str, str] = {c.lower(): c for c in self.expected_columns}

    # ------------------------------------------------------------------
    def build_mapping(
        self, incoming_columns: List[str]
    ) -> Tuple[Dict[str, str], AlignmentReport]:
        """
        Compute column name mapping: incoming_name → expected_name.

        Returns
        -------
        mapping : dict {incoming_col: expected_col}
        report  : AlignmentReport with details of each match type.
        """
        report   = AlignmentReport()
        mapping:  Dict[str, str] = {}   # incoming → expected
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
            best_score: float         = 0.0
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
        df:            pd.DataFrame,
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
        import numpy as np

        source_df = df.select_dtypes(include=[np.number]) if extra_numeric else df
        incoming  = source_df.columns.tolist()

        mapping, report = self.build_mapping(incoming)

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

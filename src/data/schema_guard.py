"""
src/data/schema_guard.py
Runtime schema enforcement: validates that a DataFrame matches the expected
feature schema before model inference.  Raises on hard violations; emits
warnings on soft violations (unexpected extra columns, minor type mismatches).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SchemaViolation:
    """A single schema constraint violation."""

    column: str
    violation_type: str  # "missing" | "type_mismatch" | "unexpected"
    detail: str


@dataclass
class SchemaGuardResult:
    """Aggregated result from SchemaGuard.validate()."""

    hard_violations: List[SchemaViolation] = field(default_factory=list)
    soft_violations: List[SchemaViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.hard_violations

    def summary(self) -> str:
        parts = [f"SchemaGuard: {len(self.hard_violations)} hard, {len(self.soft_violations)} soft violations"]
        for v in self.hard_violations:
            parts.append(f"  HARD [{v.violation_type}] {v.column}: {v.detail}")
        for v in self.soft_violations:
            parts.append(f"  SOFT [{v.violation_type}] {v.column}: {v.detail}")
        return "\n".join(parts)


class SchemaGuard:
    """
    Validates incoming DataFrames against a declared feature schema.

    Parameters
    ----------
    required_columns   : Columns that must be present.
    numeric_columns    : Subset of required_columns that must be numeric dtype.
    optional_columns   : Columns allowed but not required.
    allow_extra        : If False, extra columns are soft-violations (default True).
    """

    def __init__(
        self,
        required_columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None,
        allow_extra: bool = True,
    ) -> None:
        self.required_columns = list(required_columns)
        self.numeric_columns: List[str] = list(numeric_columns or required_columns)
        self.optional_columns: List[str] = list(optional_columns or [])
        self.allow_extra = allow_extra

    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> SchemaGuardResult:
        """
        Validate ``df`` against the declared schema.

        Returns
        -------
        SchemaGuardResult with hard and soft violations listed.
        """
        result = SchemaGuardResult()
        known: set[str] = set(self.required_columns) | set(self.optional_columns)
        present = set(df.columns)

        # ── Missing required columns ──────────────────────────────────────
        for col in self.required_columns:
            if col not in present:
                result.hard_violations.append(
                    SchemaViolation(
                        column=col,
                        violation_type="missing",
                        detail=f"Required column '{col}' is absent from DataFrame.",
                    )
                )

        # ── Type check: numeric columns must be numeric ───────────────────
        for col in self.numeric_columns:
            if col in present and not pd.api.types.is_numeric_dtype(df[col]):
                result.hard_violations.append(
                    SchemaViolation(
                        column=col,
                        violation_type="type_mismatch",
                        detail=f"Column '{col}' must be numeric; got dtype '{df[col].dtype}'.",
                    )
                )

        # ── Unexpected extra columns ──────────────────────────────────────
        if not self.allow_extra:
            for col in sorted(present - known):
                result.soft_violations.append(
                    SchemaViolation(
                        column=col,
                        violation_type="unexpected",
                        detail=f"Column '{col}' is not in the declared schema.",
                    )
                )

        summary = result.summary()
        if result.hard_violations:
            logger.error(summary)
        elif result.soft_violations:
            logger.warning(summary)
        else:
            logger.debug("SchemaGuard: DataFrame passed all checks. shape=%s", df.shape)

        return result

    # ------------------------------------------------------------------
    def validate_or_raise(self, df: pd.DataFrame) -> None:
        """Validate and raise ValueError if hard violations exist."""
        result = self.validate(df)
        if not result.is_valid:
            raise ValueError(result.summary())

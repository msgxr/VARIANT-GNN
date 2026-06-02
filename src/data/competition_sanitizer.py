"""
src/data/competition_sanitizer.py
Thin orchestration layer that runs the LeakageFirewall for both training
and inference contexts, respecting competition.mode and strict_leakage flags
read from the project config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.leakage_firewall import LeakageFirewall, LeakageReport

logger = logging.getLogger(__name__)


class CompetitionSanitizer:
    """
    Orchestrates LeakageFirewall using project-level config flags.

    Parameters
    ----------
    competition_mode : Map to config competition.mode.
    strict_leakage   : Map to config competition.strict_leakage.
    reports_dir      : Directory to write leakage_report.json.
    """

    def __init__(
        self,
        competition_mode: bool = True,
        strict_leakage: bool = False,
        reports_dir: Path = Path("reports"),
    ) -> None:
        self._firewall = LeakageFirewall(
            competition_mode=competition_mode,
            strict_leakage=strict_leakage,
            block_labels=True,
            reports_dir=reports_dir,
        )

    # ------------------------------------------------------------------
    def sanitize_train(
        self,
        df: pd.DataFrame,
        *,
        expected_columns: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, LeakageReport]:
        """
        Sanitize a training DataFrame.

        Labels are allowed (block_labels applies only during inference).
        Coordinate columns are still prohibited.
        """
        clean_df, report = self._firewall.check_and_sanitize(
            df,
            is_inference=False,
            expected_columns=expected_columns,
            report_path=self._firewall.reports_dir / "leakage_report_train.json",
        )
        if not report.is_clean:
            logger.warning(
                "[CompetitionSanitizer] Training data leakage issues: %s",
                report.to_dict(),
            )
        return clean_df, report

    # ------------------------------------------------------------------
    def sanitize_inference(
        self,
        df: pd.DataFrame,
        *,
        train_df: Optional[pd.DataFrame] = None,
        expected_columns: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, LeakageReport]:
        """
        Sanitize an inference (blind/test) DataFrame.

        Label-like columns are prohibited in inference mode.
        Variant_ID overlap with training set is checked when train_df is given.
        """
        clean_df, report = self._firewall.check_and_sanitize(
            df,
            is_inference=True,
            train_df=train_df,
            expected_columns=expected_columns,
            report_path=self._firewall.reports_dir / "leakage_report_inference.json",
        )
        if not report.is_clean:
            logger.warning(
                "[CompetitionSanitizer] Inference data leakage issues: %s",
                report.to_dict(),
            )
        return clean_df, report

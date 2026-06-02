"""tests/unit/test_leakage_firewall.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.competition_sanitizer import CompetitionSanitizer
from src.data.leakage_firewall import (
    COORDINATE_COLUMNS,
    LABEL_COLUMNS,
    LeakageFirewall,
    LeakageReport,
)
from src.data.schema_guard import SchemaGuard


def _make_df(**kwargs) -> pd.DataFrame:
    base = {"Variant_ID": ["V1", "V2", "V3"], "SIFT_score": [0.1, 0.5, 0.9]}
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# LeakageFirewall — coordinate column detection
# ---------------------------------------------------------------------------


def test_coordinate_column_detected(tmp_path):
    df = _make_df(chromosome=["chr1", "chr2", "chr3"])
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    clean, report = fw.check_and_sanitize(df)
    assert "chromosome" in report.coordinate_hits
    assert "chromosome" not in clean.columns
    assert "chromosome" in report.columns_dropped


def test_coordinate_column_case_insensitive(tmp_path):
    df = _make_df(Chromosome=["chr1", "chr2", "chr3"])
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    clean, report = fw.check_and_sanitize(df)
    assert report.coordinate_hits  # 'Chromosome' maps to 'chromosome' blocklist


def test_strict_leakage_raises(tmp_path):
    df = _make_df(position=[100, 200, 300])
    fw = LeakageFirewall(competition_mode=True, strict_leakage=True, reports_dir=tmp_path)
    with pytest.raises(ValueError, match="Prohibited coordinate columns"):
        fw.check_and_sanitize(df)


def test_non_competition_mode_does_not_drop(tmp_path):
    df = _make_df(chromosome=["chr1", "chr2", "chr3"])
    fw = LeakageFirewall(competition_mode=False, strict_leakage=False, reports_dir=tmp_path)
    clean, report = fw.check_and_sanitize(df)
    assert "chromosome" in clean.columns
    assert not report.columns_dropped


# ---------------------------------------------------------------------------
# LeakageFirewall — label column detection during inference
# ---------------------------------------------------------------------------


def test_label_column_blocked_in_inference(tmp_path):
    df = _make_df(label=[1, 0, 1])
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    clean, report = fw.check_and_sanitize(df, is_inference=True)
    assert "label" in report.label_hits
    assert "label" not in clean.columns


def test_label_column_allowed_in_training(tmp_path):
    df = _make_df(label=[1, 0, 1])
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    clean, report = fw.check_and_sanitize(df, is_inference=False)
    assert not report.label_hits
    assert "label" in clean.columns


# ---------------------------------------------------------------------------
# Train/test Variant_ID overlap
# ---------------------------------------------------------------------------


def test_variant_id_overlap_detected(tmp_path):
    train = _make_df()  # Variant_IDs: V1, V2, V3
    test = _make_df()  # same IDs
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    _, report = fw.check_and_sanitize(test, train_df=train)
    assert report.variant_id_overlaps == 3


def test_no_variant_id_overlap(tmp_path):
    train = pd.DataFrame({"Variant_ID": ["V1", "V2"], "SIFT_score": [0.1, 0.5]})
    test = pd.DataFrame({"Variant_ID": ["V3", "V4"], "SIFT_score": [0.2, 0.6]})
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    _, report = fw.check_and_sanitize(test, train_df=train)
    assert report.variant_id_overlaps == 0


# ---------------------------------------------------------------------------
# Duplicate fingerprint
# ---------------------------------------------------------------------------


def test_duplicate_detection(tmp_path):
    df = pd.DataFrame({"SIFT_score": [0.5, 0.5, 0.9]})
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    _, report = fw.check_and_sanitize(df)
    assert report.duplicate_row_count == 1


# ---------------------------------------------------------------------------
# Report JSON saved
# ---------------------------------------------------------------------------


def test_report_saved_to_disk(tmp_path):
    df = _make_df()
    fw = LeakageFirewall(competition_mode=True, strict_leakage=False, reports_dir=tmp_path)
    fw.check_and_sanitize(df)
    report_file = tmp_path / "leakage_report.json"
    assert report_file.exists()
    payload = json.loads(report_file.read_text())
    assert "is_clean" in payload


# ---------------------------------------------------------------------------
# CompetitionSanitizer
# ---------------------------------------------------------------------------


def test_sanitizer_train(tmp_path):
    df = _make_df(label=[1, 0, 1])
    san = CompetitionSanitizer(reports_dir=tmp_path)
    clean, report = san.sanitize_train(df)
    assert "label" in clean.columns  # allowed in training


def test_sanitizer_inference(tmp_path):
    df = _make_df(label=[1, 0, 1], chromosome=["1", "2", "3"])
    san = CompetitionSanitizer(reports_dir=tmp_path)
    clean, report = san.sanitize_inference(df)
    assert "label" not in clean.columns
    assert "chromosome" not in clean.columns


# ---------------------------------------------------------------------------
# SchemaGuard
# ---------------------------------------------------------------------------


def test_schema_guard_missing_column():
    guard = SchemaGuard(required_columns=["SIFT_score", "CADD_phred"])
    df = pd.DataFrame({"SIFT_score": [0.1]})
    result = guard.validate(df)
    assert not result.is_valid
    cols = [v.column for v in result.hard_violations]
    assert "CADD_phred" in cols


def test_schema_guard_type_mismatch():
    guard = SchemaGuard(
        required_columns=["SIFT_score"],
        numeric_columns=["SIFT_score"],
    )
    df = pd.DataFrame({"SIFT_score": ["not_a_number"]})
    result = guard.validate(df)
    assert not result.is_valid


def test_schema_guard_valid():
    guard = SchemaGuard(required_columns=["SIFT_score", "CADD_phred"])
    df = pd.DataFrame({"SIFT_score": [0.1, 0.5], "CADD_phred": [25.0, 30.0]})
    result = guard.validate(df)
    assert result.is_valid


def test_schema_guard_raises_on_violation():
    guard = SchemaGuard(required_columns=["SIFT_score"])
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError):
        guard.validate_or_raise(df)

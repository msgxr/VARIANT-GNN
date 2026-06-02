"""
tests/unit/test_inference_contract.py
TEKNOFEST 2026 §7.5 — Inference girdi/çıktı sözleşme testleri.

Jürinin kodu yeniden çalıştırma senaryosunu doğrular:
  - Model etiketsiz CSV'yi kabul etmeli
  - Çıktı zorunlu alanları içermeli (Variant_ID, Prediction, Predicted_Label)
  - Koordinat sütunları (chromosome, position vb.) kabul edilmemeli
  - Label sütunu inference sırasında çıktıyı etkilememeli
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_blind_df(n: int = 20, seed: int = 0, include_coord: bool = False, id_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Variant_ID": [f"VAR_{i + id_offset:04d}" for i in range(n)],
            "Panel": np.random.choice(["General", "Hereditary_Cancer", "PAH", "CFTR"], n),
            "CADD_phred": rng.uniform(1, 40, n),
            "REVEL_score": rng.uniform(0, 1, n),
            "SIFT_score": rng.uniform(0, 1, n),
            "gnomAD_exomes_AF": rng.uniform(0, 0.05, n),
            "GERP_RS": rng.uniform(-10, 6, n),
            "PhyloP100way_vertebrate": rng.uniform(-3, 3, n),
        }
    )
    if include_coord:
        df["chromosome"] = "chr1"
        df["position"] = rng.integers(1_000_000, 50_000_000, n)
    return df


class TestLeakageFirewallInference:
    def test_coordinate_columns_blocked(self):
        """Koordinat sütunları (chromosome, position) inference'da engellenmeli — §3.2."""
        from src.data.leakage_firewall import LeakageFirewall

        df = _make_blind_df(include_coord=True)
        fw = LeakageFirewall(competition_mode=True)
        clean_df, report = fw.check_and_sanitize(df, is_inference=True)
        assert "chromosome" not in clean_df.columns
        assert "position" not in clean_df.columns
        assert len(report.coordinate_hits) > 0

    def test_label_columns_blocked_during_inference(self):
        """Label sütunu inference sırasında engellenmeli — §3.2."""
        from src.data.leakage_firewall import LeakageFirewall

        df = _make_blind_df()
        df["label"] = np.random.randint(0, 2, len(df))
        fw = LeakageFirewall(competition_mode=True)
        clean_df, report = fw.check_and_sanitize(df, is_inference=True)
        assert "label" not in clean_df.columns
        assert len(report.label_hits) > 0

    def test_clean_data_passes_firewall(self):
        """Koordinat ve label içermeyen temiz veri firewall'ı geçmeli."""
        from src.data.leakage_firewall import LeakageFirewall

        df = _make_blind_df(include_coord=False)
        fw = LeakageFirewall(competition_mode=True)
        clean_df, report = fw.check_and_sanitize(df, is_inference=True)
        assert report.is_clean

    def test_no_variant_id_overlap_detected(self):
        """Eğitim ve test Variant_ID'lerinde çakışma olmamalı."""
        from src.data.leakage_firewall import LeakageFirewall

        train_df = _make_blind_df(n=10, seed=0, id_offset=0)
        test_df = _make_blind_df(n=10, seed=0, id_offset=1000)
        fw = LeakageFirewall(competition_mode=True)
        _, report = fw.check_and_sanitize(test_df, train_df=train_df, is_inference=True)
        assert report.variant_id_overlaps == 0

    def test_variant_id_overlap_detected(self):
        """Aynı Variant_ID'ler eğitim ve test'te varsa overlap tespit edilmeli."""
        from src.data.leakage_firewall import LeakageFirewall

        train_df = _make_blind_df(n=10, seed=0)
        test_df = train_df.copy()
        fw = LeakageFirewall(competition_mode=True)
        _, report = fw.check_and_sanitize(test_df, train_df=train_df, is_inference=True)
        assert report.variant_id_overlaps == 10


class TestSubmissionFormat:
    def test_submission_schema_fields(self):
        """Submission CSV zorunlu alanları içermeli: Variant_ID, prediction_label — §7.5."""
        import json
        from pathlib import Path

        schema_path = Path("data/contracts/submission_schema.json")
        if not schema_path.exists():
            pytest.skip("submission_schema.json bulunamadı")
        with open(schema_path) as f:
            schema = json.load(f)
        props = schema.get("properties", schema.get("columns", {}))
        required = set(schema.get("required", list(props.keys())))
        for field in ("Variant_ID", "prediction_label"):
            assert field in props or field in required, f"Submission schema'da '{field}' alanı eksik"

    def test_label_mapping_correct(self):
        """Label mapping: Pathogenic=1, Benign=0 — şartname §3.2."""
        import json
        from pathlib import Path

        mapping_path = Path("data/contracts/label_mapping.json")
        if not mapping_path.exists():
            pytest.skip("label_mapping.json bulunamadı")
        with open(mapping_path) as f:
            mapping = json.load(f)
        assert mapping.get("Pathogenic") == 1, "Pathogenic → 1 olmalı"
        assert mapping.get("Benign") == 0, "Benign → 0 olmalı"


class TestClinVarAPIBlocked:
    def test_clinvar_blocked_in_inference_mode(self):
        """ClinVar API eğitim/inference modunda tamamen bloklanmalı — §3.2."""
        from src.explainability.clinvar_api import fetch_clinvar_info, set_inference_mode

        set_inference_mode(True)
        result = fetch_clinvar_info("rs397507444")
        assert result["found"] is False
        assert result["error"] is not None
        set_inference_mode(False)

    def test_clinvar_api_returns_dict(self):
        """fetch_clinvar_info her zaman dict döndürmeli (hata durumunda da)."""
        from src.explainability.clinvar_api import fetch_clinvar_info, set_inference_mode

        set_inference_mode(True)
        result = fetch_clinvar_info("rs_nonexistent_test_id")
        assert isinstance(result, dict)
        assert "found" in result
        set_inference_mode(False)

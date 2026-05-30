"""
tests/unit/test_bio_scoring.py
Unit tests for biological scoring (src/features/bio_scoring.py).
"""
from __future__ import annotations

import pytest


class TestBLOSUM62:
    def test_known_substitution(self):
        from src.features.bio_scoring import get_blosum62_score
        # A→A diagonal should be 4
        assert get_blosum62_score("A", "A") == 4
        # W→W diagonal should be 11 (highest)
        assert get_blosum62_score("W", "W") == 11

    def test_symmetric(self):
        from src.features.bio_scoring import get_blosum62_score
        assert get_blosum62_score("A", "R") == get_blosum62_score("R", "A")
        assert get_blosum62_score("C", "W") == get_blosum62_score("W", "C")

    def test_case_insensitive(self):
        from src.features.bio_scoring import get_blosum62_score
        assert get_blosum62_score("a", "r") == get_blosum62_score("A", "R")

    def test_unknown_returns_default(self):
        from src.features.bio_scoring import get_blosum62_score
        assert get_blosum62_score("X", "A") == -4  # default fallback

    def test_negative_for_dissimilar(self):
        from src.features.bio_scoring import get_blosum62_score
        # W and D are very dissimilar
        assert get_blosum62_score("W", "D") < 0

    def test_all_20_amino_acids_covered(self):
        from src.features.bio_scoring import BLOSUM62
        aas = "ARNDCQEGHILKMFPSTWYV"
        for a1 in aas:
            for a2 in aas:
                assert (a1, a2) in BLOSUM62, f"Missing ({a1}, {a2})"


class TestGranthamScore:
    def test_identical_aa_zero(self):
        from src.features.bio_scoring import get_grantham_score
        assert get_grantham_score("A", "A") == 0.0

    def test_known_pair(self):
        from src.features.bio_scoring import get_grantham_score
        # A→W Grantham distance is 44.5 — meaningfully high
        score = get_grantham_score("A", "W")
        assert score > 40.0

    def test_symmetric(self):
        from src.features.bio_scoring import get_grantham_score
        assert abs(get_grantham_score("A", "V") - get_grantham_score("V", "A")) < 1e-6

    def test_case_insensitive(self):
        from src.features.bio_scoring import get_grantham_score
        assert abs(get_grantham_score("a", "v") - get_grantham_score("A", "V")) < 1e-6

    def test_unknown_returns_default(self):
        from src.features.bio_scoring import get_grantham_score
        assert get_grantham_score("X", "A") == 100.0

    def test_range(self):
        from src.features.bio_scoring import get_grantham_score
        aas = "ARNDCQEGHILKMFPSTWYV"
        for a1 in aas:
            for a2 in aas:
                score = get_grantham_score(a1, a2)
                assert score >= 0.0, f"Negative score for ({a1}, {a2})"

    def test_conservative_vs_radical(self):
        from src.features.bio_scoring import get_grantham_score
        # I→L conservative (similar hydrophobicity)
        # I→W radical (very different)
        assert get_grantham_score("I", "L") < get_grantham_score("I", "W")

# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/unit/test_innovations_v2.py
===================================
İnovasyon modülü testleri:
  1. ACMGProxyFeatures  — ACMG kriter proxy özellik mühendisliği
  2. TTAPredictor       — Test-Time Augmentation
  3. RankAggregator     — Borda Count / Rank aggregation ensemble
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 1. ACMGProxyFeatures
# ─────────────────────────────────────────────────────────────────────────────


class TestACMGProxyFeatures:
    def _make_df(self, n: int = 50, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "CADD_phred": rng.uniform(0, 60, n),
                "REVEL_score": rng.uniform(0, 1, n),
                "SIFT_score": rng.uniform(0, 1, n),
                "PolyPhen2_HDIV_score": rng.uniform(0, 1, n),
                "MetaSVM_score": rng.uniform(-1, 1, n),
                "MCAP_score": rng.uniform(0, 1, n),
                "GERP_RS": rng.uniform(-13, 6, n),
                "PhyloP100way_vertebrate": rng.uniform(-20, 10, n),
                "phastCons100way_vertebrate": rng.uniform(0, 1, n),
                "gnomAD_exomes_AF": rng.uniform(0, 0.1, n),
                "ExAC_AF": rng.uniform(0, 0.1, n),
                "In_Critical_Protein_Domain": rng.integers(0, 2, n).astype(float),
                "Protein_Impact_Score": rng.uniform(0, 1, n),
                "OMIM_Disease_Gene": rng.integers(0, 2, n).astype(float),
                "AA_Grantham_Score": rng.uniform(0, 215, n),
            }
        )

    def test_output_columns_present(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df()
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        for col in ACMGProxyFeatures.FEATURE_NAMES:
            assert col in out.columns, f"Kolon eksik: {col}"

    def test_output_shape(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=100)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out.shape == (100, len(df.columns) + len(ACMGProxyFeatures.FEATURE_NAMES))

    def test_pp3_range(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=200)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out["acmg_pp3_score"].between(0, 6).all()

    def test_bp4_range(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=200)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out["acmg_bp4_score"].between(0, 5).all()

    def test_pm1_range(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=200)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out["acmg_pm1_score"].between(0, 1).all()

    def test_ba1_flag_binary(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=100)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert set(out["acmg_ba1_flag"].unique()).issubset({0.0, 1.0})

    def test_ba1_flag_high_af(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = pd.DataFrame(
            {
                "gnomAD_exomes_AF": [0.06, 0.01],
                "ExAC_AF": [0.0, 0.0],
            }
        )
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out["acmg_ba1_flag"].iloc[0] == 1.0
        assert out["acmg_ba1_flag"].iloc[1] == 0.0

    def test_net_score_equals_pp3_minus_bp4(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=50)
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        np.testing.assert_allclose(
            out["acmg_net_score"].values,
            (out["acmg_pp3_score"] - out["acmg_bp4_score"]).values,
            rtol=1e-6,
        )

    def test_missing_columns_handled_gracefully(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = pd.DataFrame({"CADD_phred": [20.0, 5.0, 30.0]})
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert "acmg_pp3_score" in out.columns
        assert len(out) == 3

    def test_pathogenic_scores_higher_for_damaging_variants(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = pd.DataFrame(
            {
                "CADD_phred": [40.0, 2.0],
                "REVEL_score": [0.9, 0.1],
                "SIFT_score": [0.001, 0.9],
                "PolyPhen2_HDIV_score": [0.99, 0.01],
                "MetaSVM_score": [0.8, -0.8],
                "MCAP_score": [0.8, 0.005],
            }
        )
        proxy = ACMGProxyFeatures()
        out = proxy.transform(df)
        assert out["acmg_pp3_score"].iloc[0] > out["acmg_pp3_score"].iloc[1]

    def test_transform_array(self):
        from src.features.acmg_proxy_features import ACMGProxyFeatures

        df = self._make_df(n=30)
        proxy = ACMGProxyFeatures()
        feature_names = list(df.columns)
        X = df.values
        out = proxy.transform_array(X, feature_names)
        expected_cols = len(feature_names) + len(ACMGProxyFeatures.FEATURE_NAMES)
        assert out.shape == (30, expected_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TTAPredictor
# ─────────────────────────────────────────────────────────────────────────────


class _MockEnsemble:
    """Test için basit mock ensemble."""

    def predict(self, X):
        N = X.shape[0]
        proba = np.column_stack([np.full(N, 0.4), np.full(N, 0.6)])
        preds = np.ones(N, dtype=int)
        return preds, proba


class _MockPreprocessor:
    """Test için basit mock preprocessor."""

    def transform(self, X):
        return X.astype(float)


class TestTTAPredictor:
    def test_basic_predict(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(0).normal(size=(20, 5))
        tta = TTAPredictor(n_augmentations=5, seed=42)
        result = tta.predict(ens, pre, X)
        assert result.proba_mean.shape == (20, 2)
        assert result.proba_std.shape == (20,)
        assert result.predictions.shape == (20,)
        assert result.uncertainty_tta.shape == (20,)

    def test_proba_sums_to_one(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(1).normal(size=(15, 5))
        tta = TTAPredictor(n_augmentations=5, seed=0)
        result = tta.predict(ens, pre, X)
        np.testing.assert_allclose(result.proba_mean.sum(axis=1), np.ones(15), atol=1e-6)

    def test_uncertainty_range(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(2).normal(size=(30, 5))
        tta = TTAPredictor(n_augmentations=8, seed=0)
        result = tta.predict(ens, pre, X)
        assert (result.uncertainty_tta >= 0).all()
        assert (result.uncertainty_tta <= 1).all()

    def test_n_augmentations_recorded(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(3).normal(size=(10, 5))
        tta = TTAPredictor(n_augmentations=7, seed=0)
        result = tta.predict(ens, pre, X)
        assert result.n_augmentations == 7

    def test_zero_noise_deterministic(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(4).normal(size=(10, 5))
        tta = TTAPredictor(n_augmentations=5, noise_scale=0.0, seed=0)
        result = tta.predict(ens, pre, X)
        np.testing.assert_allclose(result.proba_std, 0.0, atol=1e-10)

    def test_predict_with_threshold(self):
        from src.inference.tta import TTAPredictor

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        X = np.random.default_rng(5).normal(size=(20, 5))
        tta = TTAPredictor(n_augmentations=3, seed=0)
        preds, prob, unc = tta.predict_with_threshold(ens, pre, X, threshold=0.5)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    def test_dataframe_output(self):
        from src.inference.tta import tta_predict_dataframe

        ens = _MockEnsemble()
        pre = _MockPreprocessor()
        feat_cols = [f"f{i}" for i in range(5)]
        df = pd.DataFrame(
            np.random.default_rng(6).normal(size=(15, 5)),
            columns=feat_cols,
        )
        out = tta_predict_dataframe(ens, pre, df, feat_cols, n_augmentations=3)
        assert "TTA_Probability" in out.columns
        assert "TTA_Uncertainty" in out.columns
        assert "TTA_Prediction" in out.columns
        assert "TTA_Flag" in out.columns
        assert set(out["TTA_Prediction"].unique()).issubset({"Pathogenic", "Benign"})


# ─────────────────────────────────────────────────────────────────────────────
# 3. RankAggregator
# ─────────────────────────────────────────────────────────────────────────────


class TestRankAggregator:
    def _make_probas(self, n: int = 50, k: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        return [rng.uniform(0, 1, (n, 2)) for _ in range(k)]

    def test_borda_output_shape(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=40, k=3)
        agg = RankAggregator(method="borda")
        out = agg.aggregate(probas)
        assert out.shape == (40,)

    def test_borda_range(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=100, k=4)
        agg = RankAggregator(method="borda")
        out = agg.aggregate(probas)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_reciprocal_rank_range(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=60, k=3)
        agg = RankAggregator(method="reciprocal_rank")
        out = agg.aggregate(probas)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_geometric_mean_range(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=50, k=4)
        agg = RankAggregator(method="geometric_mean")
        out = agg.aggregate(probas)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_accepts_1d_probas(self):
        from src.ensemble.rank_aggregation import RankAggregator

        rng = np.random.default_rng(0)
        probas = [rng.uniform(0, 1, 30) for _ in range(3)]
        agg = RankAggregator(method="borda")
        out = agg.aggregate(probas)
        assert out.shape == (30,)

    def test_all_same_model_returns_same_order(self):
        from src.ensemble.rank_aggregation import RankAggregator

        p = np.linspace(0, 1, 50)
        probas = [p.copy() for _ in range(4)]
        agg = RankAggregator(method="borda")
        out = agg.aggregate(probas)
        diffs = np.diff(out)
        assert (diffs >= 0).all(), "Aynı model → monoton sıra korunmalı"

    def test_weighted_aggregation(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=30, k=4, seed=99)
        agg_eq = RankAggregator(method="borda")
        agg_w = RankAggregator(method="borda", weights=[2.0, 1.0, 1.0, 0.5])
        out_eq = agg_eq.aggregate(probas)
        out_w = agg_w.aggregate(probas)
        assert not np.allclose(out_eq, out_w), "Ağırlıklar farklı sonuç vermeli"

    def test_compare_with_weighted_avg(self):
        from src.ensemble.rank_aggregation import RankAggregator

        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 80)
        probas = self._make_probas(n=80, k=3, seed=7)
        agg = RankAggregator(method="borda")
        result = agg.compare_with_weighted_avg(probas, y)
        assert "rank_aggregation_f1" in result
        assert "weighted_avg_f1" in result
        assert "delta_f1" in result
        assert 0.0 <= result["rank_aggregation_f1"] <= 1.0

    def test_unknown_method_raises(self):
        from src.ensemble.rank_aggregation import RankAggregator

        probas = self._make_probas(n=10, k=2)
        agg = RankAggregator(method="unknown_method")  # type: ignore
        with pytest.raises(ValueError, match="Bilinmeyen method"):
            agg.aggregate(probas)

# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""Backward-compatible re-export. The canonical implementation lives in src.api.pipeline."""

try:
    from src.api.pipeline import InferencePipeline
except ImportError as _e:
    raise ImportError(
        f"InferencePipeline yuklenemedi (src.api.pipeline): {_e}\nKontrol et: src/api/pipeline.py ve bagimliliklari."
    ) from _e

__all__ = ["InferencePipeline"]

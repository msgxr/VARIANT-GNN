"""
src/api/rest_api.py
VARIANT-GNN — FastAPI REST Endpoint

Hastane Bilgi Yönetim Sistemi (HBYS) entegrasyonu için tasarlanmıştır.
Herhangi bir HTTP istemcisinden (curl, Postman, Python requests) çağrılabilir.

Endpoints:
  GET  /health          → Sistem sağlık kontrolü
  GET  /info            → Model ve versiyon bilgisi
  POST /predict         → CSV veya JSON varyant verisi → tahmin
  POST /predict/json    → JSON body ile tek/çoklu varyant tahmini
  GET  /predict/sample  → Örnek tahmin (demo, model yüklü olmalı)

Kurulum:
  pip install fastapi uvicorn python-multipart

Çalıştırma:
  uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --reload

Docker:
  docker-compose up variant-gnn-api
"""
from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── FastAPI imports ──────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse  # noqa: F401  (availability check)
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Stub — FastAPI kurulu değilken modül import hatası vermesin
    class FastAPI:           # type: ignore
        def __init__(self, *a, **kw): pass
        def get(self, *a, **kw):  return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def add_middleware(self, *a, **kw): pass
    class BaseModel: pass    # type: ignore
    def Field(*a, **kw): return None  # type: ignore


# ── Application ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VARIANT-GNN API",
    description=(
        "Genetik varyant patojenite tahmini için REST API. "
        "TEKNOFEST 2026 | Takım XYRA3"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

if _FASTAPI_AVAILABLE:
    import os
    _cors_origins = os.environ.get(
        "CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],          # Sadece gerekli metodlar
        allow_headers=["Content-Type", "Accept", "Authorization"],  # Sadece gerekli headerlar
        allow_credentials=False,
    )

# Dosya yükleme limiti: 10 MB
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# ── Pipeline (lazy-loaded) ────────────────────────────────────────────────────
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.api.pipeline import InferencePipeline
        _pipeline = InferencePipeline()
        _pipeline.load()
        logger.info("InferencePipeline yüklendi.")
    return _pipeline


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class VariantRecord(BaseModel):
    """Tek bir varyant kaydı — tüm sayısal feature'lar + opsiyonel metadata."""
    Variant_ID: Optional[str] = Field(None, description="Varyant kimliği")
    Panel: Optional[str]      = Field(None, description="Panel adı: General | Hereditary_Cancer | PAH | CFTR")

    class Config:
        extra = "allow"           # Bilinmeyen feature sütunlarını kabul et


class PredictJsonRequest(BaseModel):
    variants: List[Dict[str, Any]] = Field(
        ...,
        description="Her biri bir varyantı temsil eden dict listesi",
        min_length=1,
    )


class PredictionResult(BaseModel):
    Variant_ID:      Optional[str]
    Prediction:      str           # "Pathogenic" | "Benign"
    Probability:     float
    Calibrated_Risk: float
    Confidence:      float
    High_Risk:       bool
    Clinical_Flag:   str           # ✅ / 🔶 / ⚠️


class PredictResponse(BaseModel):
    status:      str = "success"
    model:       str = "VARIANT-GNN v2.0"
    timestamp:   str
    n_variants:  int
    latency_ms:  float
    results:     List[PredictionResult]
    summary: Dict[str, Any]


# ── Health & Info ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Sistem sağlık kontrolü — yük dengeleyici ve Kubernetes probe için."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "variant-gnn-api",
        "version": "2.0.0",
    }


@app.get("/info", tags=["System"])
def info():
    """Model versiyonu ve konfigürasyon bilgisi."""
    try:
        from src.config import get_settings
        cfg = get_settings()
        return {
            "model": "VARIANT-GNN",
            "version": "2.0.0",
            "team": "XYRA3",
            "competition": "TEKNOFEST 2026",
            "architecture": "XGBoost + LightGBM + VariantGATv2GNN + DNN",
            "ensemble_weights": cfg.ensemble.weights,
            "threshold": cfg.thresholds.classification,
            "high_risk_threshold": cfg.thresholds.high_risk,
            "panels": ["General", "Hereditary_Cancer", "PAH", "CFTR"],
            "human_in_the_loop": {
                "enabled": True,
                "uncertainty_flag_threshold": 0.30,
                "high_confidence_threshold": 0.15,
                "description": (
                    "MC-Dropout belirsizliği > 0.30 olan varyantlar "
                    "⚠️ Uzman Değerlendirmesi Gerekli olarak işaretlenir."
                ),
            },
        }
    except Exception:
        # İç hata detayları dışarıya sızdırılmaz
        logger.exception("/info endpoint config hatası")
        return {"model": "VARIANT-GNN", "version": "2.0.0", "status": "config_unavailable"}


# ── CSV Upload Predict ────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_csv(file: UploadFile = File(..., description="CSV varyant dosyası")):
    """
    CSV dosyası yükle → tahmin al.

    Örnek kullanım:
        curl -X POST http://localhost:8000/predict \\
             -F "file=@data/test_variants.csv"
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Sadece CSV dosyaları kabul edilir.",
        )

    t0 = time.perf_counter()
    try:
        contents = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Dosya boyutu limiti aşıldı (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB).",
            )
        df_raw = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception:
        logger.exception("CSV parse hatasi")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV dosyası okunamadı. Geçerli bir CSV formatı kullanın.",
        )

    return _run_prediction(df_raw, t0)


# ── JSON Body Predict ─────────────────────────────────────────────────────────
@app.post("/predict/json", response_model=PredictResponse, tags=["Prediction"])
def predict_json(request: PredictJsonRequest):
    """
    JSON body ile tahmin.

    Örnek kullanım:
        curl -X POST http://localhost:8000/predict/json \\
             -H "Content-Type: application/json" \\
             -d '{"variants": [{"Variant_ID": "V001", "Panel": "General", ...}]}'
    """
    t0 = time.perf_counter()
    df_raw = pd.DataFrame(request.variants)
    return _run_prediction(df_raw, t0)


# ── Sample Predict (demo) ─────────────────────────────────────────────────────
@app.get("/predict/sample", tags=["Prediction"])
def predict_sample():
    """
    Rastgele oluşturulmuş örnek veri ile demo tahmini.
    Model yüklü olmalıdır.
    """
    t0  = time.perf_counter()
    rng = np.random.default_rng(42)
    df_raw = pd.DataFrame({
        "Variant_ID": [f"DEMO_{i:03d}" for i in range(5)],
        "Panel": ["General"] * 5,
        **{f"feature_{i}": rng.uniform(0, 1, 5).tolist() for i in range(43)},
    })
    return _run_prediction(df_raw, t0)


# ── Core prediction logic ─────────────────────────────────────────────────────
def _run_prediction(df_raw: pd.DataFrame, t0: float) -> dict:
    # Bos DataFrame kontrolu
    if df_raw.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Girdi bos — hic satir yok. Gecerli bir CSV veya JSON girin.",
        )

    try:
        pipeline = _get_pipeline()
    except Exception:
        logger.exception("Pipeline yukleme hatasi")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model servisi kullanılamıyor. Sistem yöneticisiyle iletişime geçin.",
        )

    try:
        df_result = pipeline.predict_from_dataframe(df_raw)
    except Exception:
        logger.exception("Tahmin hatasi")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tahmin sırasında hata oluştu. Veri formatını kontrol edin.",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Build results list ────────────────────────────────────────────────────
    results = []
    for _, row in df_result.iterrows():
        results.append(PredictionResult(
            Variant_ID      = str(row.get("Variant_ID", ""))    or None,
            Prediction      = str(row.get("Prediction",  "?")),
            Probability     = float(row.get("Probability",     0.0)),
            Calibrated_Risk = float(row.get("Calibrated_Risk", 0.0)),
            Confidence      = float(row.get("Confidence",      0.0)),
            High_Risk       = bool(row.get("High_Risk",        False)),
            Clinical_Flag   = str(row.get("Clinical_Flag",     "🔶 Orta Güven")),
        ))

    n        = len(results)
    path_n   = sum(1 for r in results if r.Prediction == "Pathogenic")
    expert_n = sum(1 for r in results if "Uzman" in r.Clinical_Flag)
    hc_n     = sum(1 for r in results if "Yüksek" in r.Clinical_Flag)

    return PredictResponse(
        timestamp  = datetime.utcnow().isoformat(),
        n_variants = n,
        latency_ms = round(latency_ms, 2),
        results    = results,
        summary    = {
            "total":                  n,
            "pathogenic":             path_n,
            "benign":                 n - path_n,
            "pathogenic_rate_pct":    round(100 * path_n  / max(n, 1), 1),
            "expert_review_required": expert_n,
            "high_confidence":        hc_n,
            "latency_per_variant_ms": round(latency_ms / max(n, 1), 3),
            "human_in_the_loop": (
                f"{expert_n}/{n} varyant uzman değerlendirmesine yönlendirildi "
                f"(MC-Dropout belirsizliği > 0.30)"
            ),
        },
    ).model_dump()

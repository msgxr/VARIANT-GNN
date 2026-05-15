"""
src/config/schema.py
=====================
TEKNOFEST 2026 — Pydantic v2 YAML Yapılandırma Doğrulayıcı (TD-011)

`configs/*.yaml` dosyalarında geçersiz tip, eksik alan veya geçersiz değer
geldiğinde **fail-fast** davranır. Her hata net Türkçe mesajla raporlanır
ve programın yanlış konfigürasyonla sessizce yanlış sonuç üretmesini
engeller.

Kullanım:

    from src.config.schema import validate_config_dict, validate_config_file

    raw = yaml.safe_load(open("configs/default.yaml"))
    validated = validate_config_dict(raw)        # tipler/sınırlar OK ise döner
    # veya doğrudan:
    validated = validate_config_file("configs/default.yaml")

Bu modül `src/config/settings.py` ile **birlikte** çalışır:
  - `settings.py` — runtime için zengin dataclass arayüzü.
  - `schema.py`   — diskteki YAML'i doğrulayan giriş kapısı.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section schemas
# ---------------------------------------------------------------------------


class _SectionBase(BaseModel):
    """
    Tüm alt-bölüm modelleri için ortak temel.

    `extra="ignore"`: Tanımlanmamış alanlar sessizce göz ardı edilir.
    Bu, mevcut `configs/*.yaml` dosyalarının (panel display_name,
    eski experiment flag'leri vb.) kırılmadan doğrulanmasını sağlar.

    Asıl koruma: TİP HATALARI ve SINIR İHLALLERİ yakalanır
    (örn. negatif epoch, > 1.0 dropout, geçersiz string Literal).
    """
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class PathsSchema(_SectionBase):
    data_dir:     str = "data"
    models_dir:   str = "models"
    reports_dir:  str = "reports"
    output_dir:   Optional[str] = None


class GNNSchema(_SectionBase):
    hidden_dim:                int   = Field(128, gt=0, le=2048)
    epochs:                    int   = Field(50,  ge=1,  le=10_000)
    lr:                        float = Field(0.001, gt=0, le=1.0)
    weight_decay:              float = Field(1e-4,  ge=0, le=1.0)
    use_gat:                   bool  = True
    knn_k:                     int   = Field(10, ge=2, le=200)
    knn_threshold:             float = Field(0.30, ge=0, le=1.0)
    early_stopping_patience:   int   = Field(20, ge=0, le=1_000)
    use_multimodal:            bool  = True
    seq_enc_dim:               int   = Field(32, gt=0, le=512)
    model_type:                str   = "gatv2"


class DNNSchema(_SectionBase):
    hidden_dim:    int   = Field(128, gt=0, le=2048)
    epochs:        int   = Field(20,  ge=1, le=1_000)
    lr:            float = Field(0.001, gt=0, le=1.0)
    weight_decay:  float = Field(1e-4,  ge=0, le=1.0)


class XGBSchema(_SectionBase):
    objective:        str   = "binary:logistic"
    eval_metric:      str   = "logloss"
    max_depth:        int   = Field(6,   ge=1, le=64)
    learning_rate:    float = Field(0.05, gt=0, le=1.0)
    subsample:        float = Field(0.8, gt=0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0, le=1.0)
    n_estimators:     int   = Field(200, ge=1, le=10_000)
    n_jobs:           int   = Field(-1)
    min_child_weight: int   = Field(3,   ge=0)
    reg_alpha:        float = Field(0.05, ge=0)
    reg_lambda:       float = Field(1.0,  ge=0)


class LGBMSchema(_SectionBase):
    num_leaves:        int   = Field(63,  ge=2)
    max_depth:         int   = Field(-1)             # -1 = sınırsız
    learning_rate:     float = Field(0.05, gt=0, le=1.0)
    n_estimators:      int   = Field(300, ge=1, le=10_000)
    subsample:         float = Field(0.8, gt=0, le=1.0)
    colsample_bytree:  float = Field(0.8, gt=0, le=1.0)
    reg_alpha:         float = Field(0.1, ge=0)
    reg_lambda:        float = Field(1.0, ge=0)
    min_child_samples: int   = Field(10,  ge=1)
    n_jobs:            int   = -1
    verbose:           int   = -1


class EnsembleSchema(_SectionBase):
    weights:           List[float] = Field(default_factory=lambda: [0.30, 0.30, 0.25, 0.15])
    optimize_weights:  bool = True

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError(f"ensemble.weights tam olarak 4 eleman olmalı (XGB,LGB,GNN,DNN); {len(v)} verildi.")
        if any(w < 0 for w in v):
            raise ValueError(f"ensemble.weights tüm değerleri ≥ 0 olmalı; {v}")
        if sum(v) <= 0:
            raise ValueError("ensemble.weights toplam ağırlık > 0 olmalı (auto-normalize için).")
        return v


class PreprocessingSchema(_SectionBase):
    corr_threshold:           float = Field(0.25, ge=0, le=1.0)
    use_autoencoder:          bool  = True
    autoencoder_encoding_dim: int   = Field(16, gt=0, le=2048)
    autoencoder_epochs:       int   = Field(10, ge=1, le=1_000)
    use_feature_selection:    bool  = True
    k_best_features:          int   = Field(35, ge=1, le=10_000)
    smote_enabled:            bool  = False
    use_bio_scoring:          bool  = False
    use_acmg_proxy:           bool  = False


class CalibrationSchema(_SectionBase):
    enabled: bool = True
    method:  Literal["isotonic", "sigmoid"] = "isotonic"


class TrainingSchema(_SectionBase):
    test_size:      float = Field(0.20, gt=0, lt=1.0)
    cv_folds:       int   = Field(5,   ge=2, le=20)
    batch_size:     int   = Field(32,  ge=1, le=8192)
    loss_function:  Literal["weighted_bce", "focal"] = "weighted_bce"
    focal_gamma:    float = Field(2.0, ge=0)
    # Opsiyonel ek alanlar (PSR/PDR config'lerinde görülen)
    calibration_size:    Optional[float] = Field(None, gt=0, lt=1.0)
    optimize_weights:    Optional[bool]  = None
    save_artifacts:      Optional[bool]  = None
    report_panel_metrics: Optional[bool] = None


class ThresholdSchema(_SectionBase):
    classification: float = Field(0.50, ge=0, le=1.0)
    high_risk:      float = Field(0.70, ge=0, le=1.0)


class SchemaSection(_SectionBase):
    target_column: str       = "Label"
    id_columns:    List[str] = Field(default_factory=lambda: ["Variant_ID"])
    non_feature_columns: List[str] = Field(default_factory=list)
    label_mapping: Dict[str, int] = Field(default_factory=dict)


class PanelEntrySchema(_SectionBase):
    """Panel başına meta — sayım alanları opsiyonel (panels.yaml uyumu için)."""
    train_pathogenic: Optional[int] = Field(default=None, ge=0)
    train_benign:     Optional[int] = Field(default=None, ge=0)
    test_pathogenic:  Optional[int] = Field(default=None, ge=0)
    test_benign:      Optional[int] = Field(default=None, ge=0)


class ExternalValidationSchema(_SectionBase):
    enabled:            bool      = True
    metrics:            List[str] = Field(default_factory=list)
    export_predictions: bool      = True


class ReproducibilitySchema(_SectionBase):
    seed:               int  = Field(42, ge=0)
    deterministic_torch: bool = True


# ---------------------------------------------------------------------------
# Top-level config schema
# ---------------------------------------------------------------------------


class ConfigSchema(BaseModel):
    """
    Tüm `configs/*.yaml` dosyaları için Pydantic v2 üst-seviye şema.

    Eksik alan = varsayılan değer kullanılır; bilinmeyen alan = ValidationError.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    seed:    int = Field(42, ge=0)
    device:  Literal["auto", "cpu", "cuda"] = "auto"
    mode:    Optional[str] = None      # "train" | "psr" | "pdr" — bilgi amaçlı

    paths:               PathsSchema               = Field(default_factory=PathsSchema)
    gnn:                 GNNSchema                 = Field(default_factory=GNNSchema)
    dnn:                 DNNSchema                 = Field(default_factory=DNNSchema)
    xgb:                 XGBSchema                 = Field(default_factory=XGBSchema)
    lgbm:                LGBMSchema                = Field(default_factory=LGBMSchema)
    ensemble:            EnsembleSchema            = Field(default_factory=EnsembleSchema)
    preprocessing:       PreprocessingSchema       = Field(default_factory=PreprocessingSchema)
    calibration:         CalibrationSchema         = Field(default_factory=CalibrationSchema)
    training:            TrainingSchema            = Field(default_factory=TrainingSchema)
    thresholds:          ThresholdSchema           = Field(default_factory=ThresholdSchema)
    schema_:             SchemaSection             = Field(default_factory=SchemaSection, alias="schema")
    panels:              Dict[str, PanelEntrySchema] = Field(default_factory=dict)
    external_validation: ExternalValidationSchema  = Field(default_factory=ExternalValidationSchema)
    reproducibility:     ReproducibilitySchema     = Field(default_factory=ReproducibilitySchema)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Tüm config doğrulama hataları için tek noktalı exception."""


def _format_pydantic_error(exc: ValidationError) -> str:
    """Pydantic v2 hatalarını insan-okur Türkçe mesaja çevir."""
    lines = ["Yapılandırma doğrulaması başarısız:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        typ = err.get("type", "")
        msg = err.get("msg", "")
        inp = err.get("input")
        lines.append(f"  • [{loc}] tip={typ}  →  {msg}  (girdi={inp!r})")
    return "\n".join(lines)


def validate_config_dict(raw: Dict[str, Any]) -> ConfigSchema:
    """
    Bir dict'i ConfigSchema'ya doğrula (fail-fast).

    Parameters
    ----------
    raw : YAML'dan elde edilen ham dict.

    Returns
    -------
    ConfigSchema  : Tip-güvenli, normalleştirilmiş yapılandırma.

    Raises
    ------
    ConfigValidationError  : Geçersiz tip / eksik alan / sınır dışı değer
                              durumlarında — ayrıntılı hata mesajıyla.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Beklenen dict; gelen tip={type(raw).__name__} (içerik: {raw!r})"
        )

    try:
        return ConfigSchema.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(_format_pydantic_error(exc)) from exc


def validate_config_file(path: str | Path) -> ConfigSchema:
    """
    Bir YAML dosyasını okuyup doğrula.

    Parameters
    ----------
    path : YAML dosya yolu.

    Returns
    -------
    ConfigSchema

    Raises
    ------
    FileNotFoundError      : Dosya yoksa.
    ConfigValidationError  : Yapılandırma geçersizse.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    try:
        validated = validate_config_dict(raw)
    except ConfigValidationError as exc:
        # Hata mesajına dosya adını ekle
        raise ConfigValidationError(f"{path}\n{exc}") from exc

    logger.info("Config doğrulandı: %s (mode=%s)", path, validated.mode or "default")
    return validated

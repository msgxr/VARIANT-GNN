"""
src/config/settings.py
Centralised configuration loader — reads configs/default.yaml and
exposes a frozen Settings dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _BASE_DIR / "configs" / "default.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class PathSettings:
    data_dir: Path
    models_dir: Path
    reports_dir: Path

    # derived model paths
    xgb_model_path: Path = field(init=False)
    gnn_model_path: Path = field(init=False)
    dnn_model_path: Path = field(init=False)
    autoencoder_path: Path = field(init=False)
    preprocessor_path: Path = field(init=False)
    calibrator_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.xgb_model_path   = self.models_dir / "xgb_model.json"
        self.gnn_model_path   = self.models_dir / "gnn_model.pth"
        self.dnn_model_path   = self.models_dir / "dnn_model.pth"
        self.autoencoder_path = self.models_dir / "autoencoder.pth"
        self.preprocessor_path = self.models_dir / "preprocessor.pkl"
        self.calibrator_path  = self.models_dir / "calibrator.pkl"

    def create_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class GNNSettings:
    hidden_dim: int = 64
    epochs: int = 20
    lr: float = 0.005
    weight_decay: float = 1e-4
    use_gat: bool = False
    knn_k: int = 5                    # k for cosine k-NN sample graph
    early_stopping_patience: int = 5  # 0 = disabled


@dataclass
class DNNSettings:
    hidden_dim: int = 128
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 1e-4


@dataclass
class XGBSettings:
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_estimators: int = 150
    n_jobs: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_estimators": self.n_estimators,
            "n_jobs": self.n_jobs,
            "random_state": 42,
        }


@dataclass
class EnsembleSettings:
    weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    optimize_weights: bool = False


@dataclass
class PreprocessingSettings:
    corr_threshold: float = 0.25
    use_autoencoder: bool = True
    autoencoder_encoding_dim: int = 16
    autoencoder_epochs: int = 10
    use_feature_selection: bool = False
    k_best_features: int = 30
    smote_enabled: bool = True


@dataclass
class CalibrationSettings:
    enabled: bool = True
    method: str = "isotonic"  # isotonic | sigmoid


@dataclass
class TrainingSettings:
    test_size: float = 0.2
    cv_folds: int = 5
    batch_size: int = 32


@dataclass
class ThresholdSettings:
    classification: float = 0.5
    high_risk: float = 0.7


@dataclass
class SchemaSettings:
    target_column: str = "Label"
    id_columns: List[str] = field(default_factory=lambda: ["Variant_ID"])
    non_feature_columns: List[str] = field(
        default_factory=lambda: ["Panel", "Nuc_Context", "AA_Context"]
    )
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "pathogenic": 1,
            "likely pathogenic": 1,
            "benign": 0,
            "likely benign": 0,
            "1": 1,
            "1.0": 1,
            "0": 0,
            "0.0": 0,
        }
    )


@dataclass
class PanelSettings:
    """Per-panel variant counts (TEKNOFEST 2026 Şartname Bölüm 3.2)."""
    train_pathogenic: int = 0
    train_benign: int = 0
    test_pathogenic: int = 0
    test_benign: int = 0


@dataclass
class ExternalValidationSettings:
    enabled: bool = True
    metrics: List[str] = field(
        default_factory=lambda: ["f1", "roc_auc", "brier_score", "precision", "recall"]
    )
    export_predictions: bool = True


@dataclass
class Settings:
    seed: int
    device: str
    paths: PathSettings
    gnn: GNNSettings
    dnn: DNNSettings
    xgb: XGBSettings
    ensemble: EnsembleSettings
    preprocessing: PreprocessingSettings
    calibration: CalibrationSettings
    training: TrainingSettings
    thresholds: ThresholdSettings
    schema: SchemaSettings
    panels: Dict[str, PanelSettings] = field(default_factory=dict)
    external_validation: ExternalValidationSettings = field(
        default_factory=ExternalValidationSettings
    )


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load and validate settings from a YAML configuration file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    raw = _load_yaml(path)

    base_dir = _BASE_DIR

    raw_paths = raw.get("paths", {})
    paths = PathSettings(
        data_dir   = base_dir / raw_paths.get("data_dir", "data"),
        models_dir = base_dir / raw_paths.get("models_dir", "models"),
        reports_dir= base_dir / raw_paths.get("reports_dir", "reports"),
    )

    raw_gnn = raw.get("gnn", {})
    gnn = GNNSettings(
        hidden_dim               = raw_gnn.get("hidden_dim", 64),
        epochs                   = raw_gnn.get("epochs", 20),
        lr                       = raw_gnn.get("lr", 0.005),
        weight_decay             = raw_gnn.get("weight_decay", 1e-4),
        use_gat                  = raw_gnn.get("use_gat", False),
        knn_k                    = raw_gnn.get("knn_k", 5),
        early_stopping_patience  = raw_gnn.get("early_stopping_patience", 5),
    )

    raw_dnn = raw.get("dnn", {})
    dnn = DNNSettings(
        hidden_dim   = raw_dnn.get("hidden_dim", 128),
        epochs       = raw_dnn.get("epochs", 20),
        lr           = raw_dnn.get("lr", 0.001),
        weight_decay = raw_dnn.get("weight_decay", 1e-4),
    )

    raw_xgb = raw.get("xgb", {})
    xgb = XGBSettings(
        objective       = raw_xgb.get("objective", "binary:logistic"),
        eval_metric     = raw_xgb.get("eval_metric", "logloss"),
        max_depth       = raw_xgb.get("max_depth", 6),
        learning_rate   = raw_xgb.get("learning_rate", 0.05),
        subsample       = raw_xgb.get("subsample", 0.8),
        colsample_bytree= raw_xgb.get("colsample_bytree", 0.8),
        n_estimators    = raw_xgb.get("n_estimators", 150),
        n_jobs          = raw_xgb.get("n_jobs", -1),
    )

    raw_ens = raw.get("ensemble", {})
    ensemble = EnsembleSettings(
        weights          = raw_ens.get("weights", [0.4, 0.4, 0.2]),
        optimize_weights = raw_ens.get("optimize_weights", False),
    )

    raw_pre = raw.get("preprocessing", {})
    preprocessing = PreprocessingSettings(
        corr_threshold          = raw_pre.get("corr_threshold", 0.25),
        use_autoencoder         = raw_pre.get("use_autoencoder", True),
        autoencoder_encoding_dim= raw_pre.get("autoencoder_encoding_dim", 16),
        autoencoder_epochs      = raw_pre.get("autoencoder_epochs", 10),
        use_feature_selection   = raw_pre.get("use_feature_selection", False),
        k_best_features         = raw_pre.get("k_best_features", 30),
        smote_enabled           = raw_pre.get("smote_enabled", True),
    )

    raw_cal = raw.get("calibration", {})
    calibration = CalibrationSettings(
        enabled = raw_cal.get("enabled", True),
        method  = raw_cal.get("method", "isotonic"),
    )

    raw_tr = raw.get("training", {})
    training = TrainingSettings(
        test_size  = raw_tr.get("test_size", 0.2),
        cv_folds   = raw_tr.get("cv_folds", 5),
        batch_size = raw_tr.get("batch_size", 32),
    )

    raw_thr = raw.get("thresholds", {})
    thresholds = ThresholdSettings(
        classification = raw_thr.get("classification", 0.5),
        high_risk      = raw_thr.get("high_risk", 0.7),
    )

    raw_sch = raw.get("schema", {})
    label_map_raw = raw_sch.get(
        "label_mapping",
        {"pathogenic": 1, "likely pathogenic": 1, "benign": 0, "likely benign": 0},
    )
    schema = SchemaSettings(
        target_column = raw_sch.get("target_column", "Label"),
        id_columns    = raw_sch.get("id_columns", ["Variant_ID"]),
        non_feature_columns = raw_sch.get("non_feature_columns", ["Panel", "Nuc_Context", "AA_Context"]),
        label_mapping = {str(k): int(v) for k, v in label_map_raw.items()},
    )

    # ── Panel tanımları ───────────────────────────────────────────
    raw_panels = raw.get("panels", {})
    panels: Dict[str, PanelSettings] = {}
    for panel_name, panel_data in raw_panels.items():
        panels[panel_name] = PanelSettings(
            train_pathogenic = panel_data.get("train_pathogenic", 0),
            train_benign     = panel_data.get("train_benign", 0),
            test_pathogenic  = panel_data.get("test_pathogenic", 0),
            test_benign      = panel_data.get("test_benign", 0),
        )

    # ── External validation ───────────────────────────────────────
    raw_ev = raw.get("external_validation", {})
    external_validation = ExternalValidationSettings(
        enabled           = raw_ev.get("enabled", True),
        metrics           = raw_ev.get("metrics", ["f1", "roc_auc", "brier_score"]),
        export_predictions = raw_ev.get("export_predictions", True),
    )

    return Settings(
        seed                = raw.get("seed", 42),
        device              = raw.get("device", "auto"),
        paths               = paths,
        gnn                 = gnn,
        dnn                 = dnn,
        xgb                 = xgb,
        ensemble            = ensemble,
        preprocessing       = preprocessing,
        calibration         = calibration,
        training            = training,
        thresholds          = thresholds,
        schema              = schema,
        panels              = panels,
        external_validation = external_validation,
    )


# Module-level singleton
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Return cached Settings, loading on first call."""
    global _settings
    if _settings is None:
        _settings = load_settings(config_path)
    return _settings


def reset_settings() -> None:
    """Force reload on next get_settings() call (useful in tests)."""
    global _settings
    _settings = None

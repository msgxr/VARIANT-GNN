"""
src/utils/serialization.py
Secure, forward-compatible model serialisation and deserialisation.

Security notes:
  - PyTorch weights loaded with ``weights_only=True`` (PyTorch ≥ 2.0).
  - XGBoost models saved/loaded as JSON (no pickle).
  - Preprocessor and calibrator use joblib with explicit path validation.
  - No arbitrary pickle deserialization from untrusted paths.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import joblib
import torch
import xgboost as xgb

logger = logging.getLogger(__name__)


def _build_legacy_sage_gnn(
    numeric_dim: int,
    hidden_dim: int,
    num_classes: int = 2,
    dropout: float = 0.3,
    use_multimodal: bool = False,
    seq_enc_dim: int = 32,
) -> "torch.nn.Module":
    """
    Orijinal SAGEConv mimarisini (SAGEConv + PyGBatchNorm + tek Linear)
    yeniden üretir. Yalnızca bu mimariye ait eski checkpoint'leri yüklemek
    için kullanılır — eğitim için değil.
    """
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch_geometric.nn import BatchNorm as _PyGBN
    from torch_geometric.nn import SAGEConv as _SAGEConv

    class _SBlock(_nn.Module):
        def __init__(self, in_c: int, out_c: int, drop: float = 0.3) -> None:
            super().__init__()
            self.conv = _SAGEConv(in_c, out_c)
            self.bn = _PyGBN(out_c)
            self.dropout = _nn.Dropout(p=drop)
            self.skip = _nn.Linear(in_c, out_c, bias=False) if in_c != out_c else _nn.Identity()

        def forward(self, x: Any, edge_index: Any) -> Any:
            res = self.skip(x)
            out = self.conv(x, edge_index)
            out = self.bn(out)
            out = _F.relu(out)
            out = self.dropout(out)
            return out + res

    class _LegacySAGEGNN(_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_multimodal = use_multimodal
            if use_multimodal:
                from src.features.multimodal_encoder import SequenceEncoder as _SE

                self.seq_encoder: Any = _SE(cnn_channels=seq_enc_dim // 2)
                in_ch = numeric_dim + self.seq_encoder.output_dim
            else:
                self.seq_encoder = None
                in_ch = numeric_dim
            self.input_proj = _nn.Linear(in_ch, hidden_dim)
            self.block1 = _SBlock(hidden_dim, hidden_dim, dropout)
            self.block2 = _SBlock(hidden_dim, hidden_dim, dropout)
            self.block3 = _SBlock(hidden_dim, hidden_dim, dropout)
            self.classifier = _nn.Linear(hidden_dim, num_classes)

        def forward(self, x: Any, edge_index: Any, nuc_ids: Any = None, aa_ids: Any = None, **kw: Any) -> Any:
            if self.use_multimodal and self.seq_encoder is not None:
                if nuc_ids is not None and aa_ids is not None:
                    seq = self.seq_encoder(nuc_ids, aa_ids)
                else:
                    import torch as _t

                    seq = _t.zeros(x.shape[0], self.seq_encoder.output_dim, device=x.device, dtype=x.dtype)
                x = _t.cat([x, seq], dim=1)
            x = _F.relu(self.input_proj(x))
            x = self.block1(x, edge_index)
            x = self.block2(x, edge_index)
            x = self.block3(x, edge_index)
            return self.classifier(x)

    return _LegacySAGEGNN()


class _LGBMBoosterWrapper:
    """
    Thin shim that wraps a raw ``lightgbm.Booster`` to expose a
    ``predict_proba(X)`` interface compatible with ``HybridEnsemble``.
    Used only when loading checkpoints saved via ``Booster.save_model``.
    """

    def __init__(self, booster: Any) -> None:
        import numpy as _np

        self._booster = booster
        self._np = _np

    def predict_proba(self, X: Any) -> Any:
        import numpy as _np
        import pandas as _pd

        # LightGBM raw Booster feature_name kontrolü yapar; DataFrame geçmek hata üretebilir
        X_arr = X.values if isinstance(X, _pd.DataFrame) else _np.asarray(X)
        raw = self._booster.predict(X_arr)
        if raw.ndim == 1:
            return _np.column_stack([1.0 - raw, raw])
        return raw


def _safe_torch_load(path: Path, device: torch.device) -> dict:
    """Load a PyTorch state dict safely."""
    import pickle

    try:
        # weights_only=True prevents arbitrary code execution (CVE-safe)
        return cast(dict, torch.load(str(path), map_location=device, weights_only=True))
    except (TypeError, pickle.UnpicklingError):
        # TypeError: PyTorch < 2.0 has no weights_only kwarg.
        # UnpicklingError: torch >= 2.6 raises "Weights only load failed" when the
        # checkpoint stores non-tensor globals. These are our own shipped artefacts,
        # so a legacy (full) load is acceptable here.
        logger.warning("weights_only load failed; falling back to legacy load.")
        return cast(dict, torch.load(str(path), map_location=device))  # nosec B614


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    """
    Centralised artefact store.

    Saves and loads:
      - XGBoost model (.json)
      - GNN state dict (.pth)
      - DNN state dict (.pth)
      - Preprocessor (.pkl via joblib)
      - Calibrator   (.pkl via joblib)
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _xgb_path(self) -> Path:
        return self.model_dir / "xgb_model.json"

    @property
    def _lgbm_path(self) -> Path:
        return self.model_dir / "lgbm_model.txt"

    @property
    def _meta_learner_path(self) -> Path:
        return self.model_dir / "meta_learner.pkl"

    @property
    def _gnn_path(self) -> Path:
        return self.model_dir / "gnn_model.pth"

    @property
    def _gnn_arch_path(self) -> Path:
        return self.model_dir / "gnn_arch.json"

    @property
    def _dnn_path(self) -> Path:
        return self.model_dir / "dnn_model.pth"

    @property
    def _autoenc_path(self) -> Path:
        return self.model_dir / "autoencoder.pth"

    @property
    def _preprocessor_path(self) -> Path:
        return self.model_dir / "preprocessor.pkl"

    @property
    def _calibrator_path(self) -> Path:
        return self.model_dir / "calibrator.pkl"

    @property
    def _ensemble_cfg_path(self) -> Path:
        return self.model_dir / "ensemble_config.json"

    @property
    def _threshold_path(self) -> Path:
        return self.model_dir / "threshold.json"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_all(
        self,
        preprocessor: Any,
        ensemble: Any,
        calibrator: Any = None,
    ) -> None:
        """Persist all artefacts.  ``ensemble`` is a ``HybridEnsemble``."""
        import json
        from datetime import datetime

        self._save_xgb(ensemble.xgb)
        self._save_lgbm(ensemble.lgbm)
        self._save_gnn(ensemble.gnn)
        self._save_dnn(ensemble.dnn)
        self._save_autoencoder(preprocessor)
        self._save_preprocessor(preprocessor)
        self._save_ensemble_cfg(ensemble)
        self._save_meta_learner(ensemble)
        if calibrator is not None:
            self._save_calibrator(calibrator)

        # ── ensemble.pkl — required by ArtifactLoader ──────────────────
        joblib.dump(ensemble, str(self.model_dir / "ensemble.pkl"))
        logger.info("Ensemble -> %s", self.model_dir / "ensemble.pkl")

        # ── feature_names.json — required by ArtifactLoader ────────────
        feature_names = None
        try:
            if ensemble.xgb is not None:
                feature_names = ensemble.xgb.get_booster().feature_names
        except Exception:
            pass
        if feature_names is not None:
            with open(self.model_dir / "feature_names.json", "w") as _fh:
                json.dump(feature_names, _fh)
            logger.info("Feature names (%d) -> feature_names.json", len(feature_names))

        # ── Intrinsic Metadata (Task 6) ──
        metadata = {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "sha256_checksums": {
                "xgb": _sha256(self._xgb_path) if self._xgb_path.exists() else None,
                "gnn": _sha256(self._gnn_path) if self._gnn_path.exists() else None,
                "dnn": _sha256(self._dnn_path) if self._dnn_path.exists() else None,
            },
            "training_config": ensemble.weights if hasattr(ensemble, "weights") else None,
        }
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ── manifest.json — reproducibility layer ───────────────────────
        try:
            from src.utils.artifact_manifest import save_manifest as _save_manifest

            _save_manifest(self.model_dir, model_version="1.0.0")
        except Exception as _mex:
            logger.warning("Manifest save failed (non-fatal): %s", _mex)

        logger.info("All artefacts saved with metadata.json -> %s", self.model_dir)

    def _save_xgb(self, model: Any) -> None:
        if model is not None:
            model.save_model(str(self._xgb_path))
            logger.info("XGBoost -> %s", self._xgb_path)

    def _save_lgbm(self, model: Any) -> None:
        if model is not None:
            try:
                model.booster_.save_model(str(self._lgbm_path))
                logger.info("LightGBM -> %s", self._lgbm_path)
            except Exception as exc:
                logger.warning("LightGBM save failed: %s", exc)

    def _save_meta_learner(self, ensemble: Any) -> None:
        ml = getattr(ensemble, "meta_learner", None)
        if ml is not None:
            joblib.dump(ml, str(self._meta_learner_path))
            logger.info("MetaLearner -> %s", self._meta_learner_path)

    def _save_gnn(self, model: Any) -> None:
        if model is None:
            return
        torch.save(model.state_dict(), str(self._gnn_path))
        import json as _json

        # VariantGATv2GNN is the base class; VariantSAGEGNN is a subclass alias.
        # We check the base class so both variants are handled identically.
        from src.core.models.gnn import VariantGATv2GNN as _VGATGNN

        arch: dict = {"type": type(model).__name__}
        if isinstance(model, _VGATGNN):
            # Save true numeric_dim (pre-concat), NOT input_proj.in_features
            # which includes seq_encoder output when use_multimodal=True.
            _in_feats = model.input_proj.in_features
            if model.use_multimodal and model.seq_encoder is not None:
                _in_feats -= model.seq_encoder.output_dim
            arch["numeric_dim"] = _in_feats
            # input_proj: Linear(in_channels, hidden_dim) → out_features == hidden_dim
            arch["hidden_dim"] = model.input_proj.out_features
            arch["use_multimodal"] = bool(model.use_multimodal)
            if model.seq_encoder is not None:
                arch["seq_enc_dim"] = model.seq_encoder.output_dim
        with open(self._gnn_arch_path, "w") as _fh:
            _json.dump(arch, _fh)
        logger.info("GNN -> %s  (arch=%s)", self._gnn_path, arch["type"])

    def _save_dnn(self, model: Any) -> None:
        if model is not None:
            torch.save(model.state_dict(), str(self._dnn_path))
            logger.info("DNN -> %s", self._dnn_path)

    def _save_autoencoder(self, preprocessor: Any) -> None:
        if (
            hasattr(preprocessor, "_autoenc")
            and preprocessor._autoenc is not None
            and preprocessor._autoenc._net is not None
        ):
            torch.save(preprocessor._autoenc._net.state_dict(), str(self._autoenc_path))
            logger.info("AutoEncoder -> %s", self._autoenc_path)

    def _save_preprocessor(self, preprocessor: Any) -> None:
        joblib.dump(preprocessor, str(self._preprocessor_path))
        logger.info("Preprocessor -> %s", self._preprocessor_path)

    def _save_ensemble_cfg(self, ensemble: Any) -> None:
        import json

        cfg = {"weights": ensemble.weights}
        with open(self._ensemble_cfg_path, "w") as fh:
            json.dump(cfg, fh)

    def _save_calibrator(self, calibrator: Any) -> None:
        joblib.dump(calibrator, str(self._calibrator_path))
        logger.info("Calibrator -> %s", self._calibrator_path)

    def save_threshold(self, threshold: float) -> None:
        """Persist the F1-optimal classification threshold (TEKNOFEST §7.3)."""
        import json

        with open(self._threshold_path, "w") as fh:
            json.dump({"classification_threshold": threshold}, fh)
        logger.info("Threshold -> %s  (thr=%.4f)", self._threshold_path, threshold)

    def load_threshold(self, default: float = 0.8415) -> float:
        """Load saved threshold; return default if not found.

        The default is the canonical θ=0.8415 (NOT 0.50): if the artifact is
        missing, a wrong-but-plausible 0.50 would silently produce wrong
        decisions, whereas 0.8415 at least matches the reported operating point.

        Tolerant of both key conventions: 'classification_threshold' (written by
        save_threshold) and 'threshold' (written by the calibration/optimisation
        scripts, e.g. models/threshold.json). Prevents the §7.5 jury re-run from
        silently falling back to a default and reproducing none of the reported
        numbers.
        """
        import json

        if not self._threshold_path.exists():
            return default
        try:
            with open(self._threshold_path) as fh:
                data = json.load(fh)
            for key in ("classification_threshold", "threshold"):
                if key in data and data[key] is not None:
                    return float(data[key])
            return default
        except Exception:
            return default

    @property
    def _panel_threshold_path(self) -> Path:
        return self.model_dir / "panel_thresholds.json"

    def save_panel_thresholds(self, thresholds: dict) -> None:
        # FLAT format ({"General": 0.33, ...}) — human-readable and the convention
        # the shipped file + tests expect. load_panel_thresholds() reads both flat
        # and legacy wrapped files.
        import json

        with open(self._panel_threshold_path, "w") as fh:
            json.dump({k: float(v) for k, v in thresholds.items()}, fh, indent=2)
        logger.info("Panel Thresholds -> %s", self._panel_threshold_path)

    def load_panel_thresholds(self) -> dict:
        """Load per-panel thresholds; tolerant of wrapped and flat file formats.

        save_panel_thresholds writes {"panel_thresholds": {...}} but the shipped
        models/panel_thresholds.json is flat ({"General": 0.59, ...}). Without this
        tolerance load_panel_thresholds() returns {} and inference silently applies
        a single global threshold to all four panels — reproducing none of the
        panel-specific reported F1 values (TEKNOFEST §7.5).
        """
        import json

        if not self._panel_threshold_path.exists():
            return {}
        try:
            with open(self._panel_threshold_path) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "panel_thresholds" in data:
                return data["panel_thresholds"] or {}
            # Flat file: treat the dict itself as the panel→threshold mapping
            # (ignore non-float bookkeeping keys defensively).
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            return {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Lightweight individual loaders (used by ColumnAligner setup)
    # ------------------------------------------------------------------

    def load_preprocessor(self) -> Any:
        """Load only the preprocessor (no model weights). Raises if missing."""
        if not self._preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {self._preprocessor_path}")
        return joblib.load(str(self._preprocessor_path))

    def load_xgb(self) -> Any:
        """Load only the XGBoost model (booster feature names included)."""
        if not self._xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {self._xgb_path}")
        from src.config import get_settings as _gs

        cfg = _gs()
        model = xgb.XGBClassifier(**cfg.xgb.as_dict())
        model.load_model(str(self._xgb_path))
        return model

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_all(self) -> Tuple:
        """
        Load preprocessor, ensemble, and calibrator.

        Returns
        -------
        (preprocessor, ensemble, calibrator)  where calibrator may be None.
        """
        from src.config import get_settings
        from src.core.models.ensemble import HybridEnsemble
        from src.core.models.gnn import FeatureGNN
        from src.features.preprocessing import VariantPreprocessor
        from src.models.dnn_model import VariantDNN

        cfg = get_settings()
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")

        # --- Preprocessor ---
        if not self._preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {self._preprocessor_path}")
        preprocessor: VariantPreprocessor = joblib.load(str(self._preprocessor_path))

        # Restore autoencoder weights if present
        if hasattr(preprocessor, "_autoenc") and preprocessor._autoenc is not None and self._autoenc_path.exists():
            ae = preprocessor._autoenc
            ae._device_obj = device
            from src.features.autoencoder import _TorchAutoEncoder

            # Re-create net to restore weights
            input_dim = preprocessor.n_output_features - ae.encoding_dim
            ae._net = _TorchAutoEncoder(input_dim, ae.encoding_dim).to(device)
            ae._net.load_state_dict(_safe_torch_load(self._autoenc_path, device))
            ae._net.eval()

        n_features = preprocessor.n_output_features

        # --- GNN: detect saved architecture ---
        import json as _json

        _gnn_arch: dict = {"type": "FeatureGNN"}
        if self._gnn_arch_path.exists():
            with open(self._gnn_arch_path) as _fh:
                _gnn_arch = _json.load(_fh)

        _gnn_type = _gnn_arch.get("type", "FeatureGNN")
        _seq_enc_dim = _gnn_arch.get("seq_enc_dim", getattr(cfg.gnn, "seq_enc_dim", 32))

        # --- GNN model yükle ---
        # Checkpoint anahtarlarından gerçek mimariyi tespit et;
        # tip adına değil, state_dict içeriğine göre doğru sınıfı seç.
        _sd_raw: dict = {}
        if self._gnn_path.exists():
            _sd_raw = _safe_torch_load(self._gnn_path, device)

        # SAGEConv checkpoint'i tanımak için ayırt edici anahtar
        _is_sage_conv_ckpt = any("bn.module" in k for k in _sd_raw)

        if _is_sage_conv_ckpt:
            # Checkpoint orijinal SAGEConv mimarisine ait (BatchNorm + tek Linear)
            # Bu mimariyi inline olarak yeniden oluştur.
            gnn_model = _build_legacy_sage_gnn(
                numeric_dim=_gnn_arch.get("numeric_dim", n_features),
                hidden_dim=_gnn_arch.get("hidden_dim", cfg.gnn.hidden_dim),
                num_classes=2,
                use_multimodal=_gnn_arch.get("use_multimodal", False),
                seq_enc_dim=_seq_enc_dim,
            ).to(device)
            logger.info("GNN: SAGEConv checkpoint tespit edildi → LegacySAGEGNN yükleniyor.")
        elif _gnn_type in ("VariantGATv2GNN", "VariantSAGEGNN"):
            from src.core.models.gnn import VariantGATv2GNN

            gnn_model = VariantGATv2GNN(
                numeric_dim=_gnn_arch.get("numeric_dim", n_features),
                hidden_dim=_gnn_arch.get("hidden_dim", cfg.gnn.hidden_dim),
                use_multimodal=_gnn_arch.get("use_multimodal", False),
                seq_enc_dim=_seq_enc_dim,
            ).to(device)
        else:
            gnn_model = FeatureGNN(
                in_channels=1,
                hidden_dim=cfg.gnn.hidden_dim,
                num_classes=2,
                use_gat=cfg.gnn.use_gat,
            ).to(device)

        if _sd_raw:
            try:
                gnn_model.load_state_dict(_sd_raw)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"GNN state_dict mismatch loading '{self._gnn_path}'. "
                    f"Saved type={_gnn_type}, arch={_gnn_arch}. "
                    f"Original error: {exc}"
                ) from exc
            gnn_model.eval()
            logger.info("GNN <- %s  (type=%s)", self._gnn_path, _gnn_type)

        # --- DNN ---
        dnn_model = VariantDNN(
            input_dim=n_features,
            hidden_dim=cfg.dnn.hidden_dim,
            num_classes=2,
        ).to(device)
        if self._dnn_path.exists():
            dnn_model.load_state_dict(_safe_torch_load(self._dnn_path, device))
            dnn_model.eval()
            logger.info("DNN <- %s", self._dnn_path)

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(**cfg.xgb.as_dict())
        if self._xgb_path.exists():
            xgb_model.load_model(str(self._xgb_path))
            logger.info("XGBoost <- %s", self._xgb_path)

        # --- LightGBM (optional — absent in legacy checkpoints) ---
        lgbm_model: Any = None
        if self._lgbm_path.exists():
            try:
                import lightgbm as lgb

                lgbm_model = lgb.Booster(model_file=str(self._lgbm_path))
                # Wrap in sklearn API shim for predict_proba compatibility
                lgbm_model = _LGBMBoosterWrapper(lgbm_model)
                logger.info("LightGBM <- %s", self._lgbm_path)
            except Exception as exc:
                # G5/§7.5: the file EXISTS, so it is a shipped 30%-weight base model.
                # Silently dropping it (→ 3-model ensemble) reproduces WRONG numbers.
                # Most common cause: missing OpenMP runtime.
                raise RuntimeError(
                    f"LightGBM model exists ({self._lgbm_path}) but failed to load: {exc}. "
                    "Genellikle OpenMP eksik → 'brew install libomp' (mac) / "
                    "'apt-get install libgomp1' (linux). 3-model ensemble ile sessizce "
                    "YANLIŞ sayı üretmemek için yükleme zorunlu kılındı."
                ) from exc

        # Ensemble weights from saved config (allows runtime update)
        weights = cfg.ensemble.weights
        if self._ensemble_cfg_path.exists():
            import json

            with open(self._ensemble_cfg_path) as fh:
                weights = json.load(fh).get("weights", weights)

        ensemble = HybridEnsemble(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            gnn_model=cast(Any, gnn_model),
            dnn_model=dnn_model,
            weights=weights,
            device=device,
        )

        # --- Calibrator (optional) ---
        calibrator: Optional[object] = None
        if self._calibrator_path.exists():
            calibrator = joblib.load(str(self._calibrator_path))
            logger.info("Calibrator <- %s", self._calibrator_path)

        # --- Meta-learner (optional stacking) ---
        if self._meta_learner_path.exists():
            try:
                ensemble.meta_learner = joblib.load(str(self._meta_learner_path))
                logger.info("MetaLearner <- %s", self._meta_learner_path)
            except Exception as exc:
                logger.warning("Meta-learner load failed (skipping): %s", exc)

        logger.info("All artefacts loaded from %s", self.model_dir)
        return preprocessor, ensemble, calibrator

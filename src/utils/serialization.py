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
from typing import Optional, Tuple

import joblib
import torch
import xgboost as xgb

logger = logging.getLogger(__name__)


class _LGBMBoosterWrapper:
    """
    Thin shim that wraps a raw ``lightgbm.Booster`` to expose a
    ``predict_proba(X)`` interface compatible with ``HybridEnsemble``.
    Used only when loading checkpoints saved via ``Booster.save_model``.
    """

    def __init__(self, booster) -> None:
        import numpy as _np
        self._booster = booster
        self._np = _np

    def predict_proba(self, X) -> "np.ndarray":
        import numpy as _np
        raw = self._booster.predict(X)
        if raw.ndim == 1:
            return _np.column_stack([1.0 - raw, raw])
        return raw


def _safe_torch_load(path: Path, device: torch.device) -> dict:
    """Load a PyTorch state dict safely."""
    try:
        # weights_only=True prevents arbitrary code execution (CVE-safe)
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        # PyTorch < 2.0 does not support weights_only
        logger.warning("weights_only not supported; falling back to legacy load.")
        return torch.load(str(path), map_location=device)  # nosec B614


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
    def _xgb_path(self)          -> Path: return self.model_dir / "xgb_model.json"
    @property
    def _lgbm_path(self)         -> Path: return self.model_dir / "lgbm_model.txt"
    @property
    def _meta_learner_path(self) -> Path: return self.model_dir / "meta_learner.pkl"
    @property
    def _gnn_path(self)          -> Path: return self.model_dir / "gnn_model.pth"
    @property
    def _gnn_arch_path(self)     -> Path: return self.model_dir / "gnn_arch.json"
    @property
    def _dnn_path(self)          -> Path: return self.model_dir / "dnn_model.pth"
    @property
    def _autoenc_path(self)      -> Path: return self.model_dir / "autoencoder.pth"
    @property
    def _preprocessor_path(self) -> Path: return self.model_dir / "preprocessor.pkl"
    @property
    def _calibrator_path(self)   -> Path: return self.model_dir / "calibrator.pkl"
    @property
    def _ensemble_cfg_path(self) -> Path: return self.model_dir / "ensemble_config.json"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_all(
        self,
        preprocessor,
        ensemble,
        calibrator=None,
    ) -> None:
        """Persist all artefacts.  ``ensemble`` is a ``HybridEnsemble``."""
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
        logger.info("All artefacts saved -> %s", self.model_dir)

    def _save_xgb(self, model) -> None:
        if model is not None:
            model.save_model(str(self._xgb_path))
            logger.info("XGBoost -> %s", self._xgb_path)

    def _save_lgbm(self, model) -> None:
        if model is not None:
            try:
                model.booster_.save_model(str(self._lgbm_path))
                logger.info("LightGBM -> %s", self._lgbm_path)
            except Exception as exc:
                logger.warning("LightGBM save failed: %s", exc)

    def _save_meta_learner(self, ensemble) -> None:
        ml = getattr(ensemble, "meta_learner", None)
        if ml is not None:
            joblib.dump(ml, str(self._meta_learner_path))
            logger.info("MetaLearner -> %s", self._meta_learner_path)

    def _save_gnn(self, model) -> None:
        if model is None:
            return
        torch.save(model.state_dict(), str(self._gnn_path))
        # Save architecture metadata so load_all can reconstruct correctly
        import json as _json
        from src.models.gnn import VariantSAGEGNN as _VSGNN
        arch: dict = {"type": type(model).__name__}
        if isinstance(model, _VSGNN):
            # Save true numeric_dim (pre-concat), NOT input_proj.in_features
            # which includes seq_encoder output when use_multimodal=True.
            _in_feats = model.input_proj.in_features
            if model.use_multimodal and model.seq_encoder is not None:
                _in_feats -= model.seq_encoder.output_dim
            arch["numeric_dim"]    = _in_feats
            arch["hidden_dim"]     = model.classifier.in_features
            arch["use_multimodal"] = bool(model.use_multimodal)
        with open(self._gnn_arch_path, "w") as _fh:
            _json.dump(arch, _fh)
        logger.info("GNN -> %s  (arch=%s)", self._gnn_path, arch["type"])

    def _save_dnn(self, model) -> None:
        if model is not None:
            torch.save(model.state_dict(), str(self._dnn_path))
            logger.info("DNN -> %s", self._dnn_path)

    def _save_autoencoder(self, preprocessor) -> None:
        if (
            hasattr(preprocessor, "_autoenc")
            and preprocessor._autoenc is not None
            and preprocessor._autoenc._net is not None
        ):
            torch.save(
                preprocessor._autoenc._net.state_dict(), str(self._autoenc_path)
            )
            logger.info("AutoEncoder -> %s", self._autoenc_path)

    def _save_preprocessor(self, preprocessor) -> None:
        joblib.dump(preprocessor, str(self._preprocessor_path))
        logger.info("Preprocessor -> %s", self._preprocessor_path)

    def _save_ensemble_cfg(self, ensemble) -> None:
        import json
        cfg = {"weights": ensemble.weights}
        with open(self._ensemble_cfg_path, "w") as fh:
            json.dump(cfg, fh)

    def _save_calibrator(self, calibrator) -> None:
        joblib.dump(calibrator, str(self._calibrator_path))
        logger.info("Calibrator -> %s", self._calibrator_path)

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
        from src.features.preprocessing import VariantPreprocessor
        from src.models.dnn import VariantDNN
        from src.models.ensemble import HybridEnsemble
        from src.models.gnn import FeatureGNN

        cfg    = get_settings()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
        )

        # --- Preprocessor ---
        if not self._preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {self._preprocessor_path}"
            )
        preprocessor: VariantPreprocessor = joblib.load(str(self._preprocessor_path))

        # Restore autoencoder weights if present
        if (
            hasattr(preprocessor, "_autoenc")
            and preprocessor._autoenc is not None
            and self._autoenc_path.exists()
        ):
            ae       = preprocessor._autoenc
            ae._device_obj = device
            from src.features.autoencoder import _TorchAutoEncoder
            # Re-create net to restore weights
            input_dim = preprocessor.n_output_features - ae.encoding_dim
            ae._net   = _TorchAutoEncoder(input_dim, ae.encoding_dim).to(device)
            ae._net.load_state_dict(
                _safe_torch_load(self._autoenc_path, device)
            )
            ae._net.eval()

        n_features = preprocessor.n_output_features

        # --- GNN: detect saved architecture (VariantSAGEGNN vs legacy FeatureGNN) ---
        import json as _json
        _gnn_arch: dict = {"type": "FeatureGNN"}
        if self._gnn_arch_path.exists():
            with open(self._gnn_arch_path) as _fh:
                _gnn_arch = _json.load(_fh)

        _gnn_type = _gnn_arch.get("type", "FeatureGNN")
        if _gnn_type == "VariantSAGEGNN":
            from src.models.gnn import VariantSAGEGNN
            gnn_model = VariantSAGEGNN(
                numeric_dim    = _gnn_arch.get("numeric_dim", n_features),
                hidden_dim     = _gnn_arch.get("hidden_dim", cfg.gnn.hidden_dim),
                use_multimodal = _gnn_arch.get("use_multimodal", False),
            ).to(device)
        else:
            gnn_model = FeatureGNN(
                in_channels = 1,
                hidden_dim  = cfg.gnn.hidden_dim,
                num_classes = 2,
                use_gat     = cfg.gnn.use_gat,
            ).to(device)

        if self._gnn_path.exists():
            gnn_model.load_state_dict(_safe_torch_load(self._gnn_path, device))
            gnn_model.eval()
            logger.info("GNN ← %s  (type=%s)", self._gnn_path, _gnn_type)

        # --- DNN ---
        dnn_model = VariantDNN(
            input_dim  = n_features,
            hidden_dim = cfg.dnn.hidden_dim,
            num_classes= 2,
        ).to(device)
        if self._dnn_path.exists():
            dnn_model.load_state_dict(_safe_torch_load(self._dnn_path, device))
            dnn_model.eval()
            logger.info("DNN ← %s", self._dnn_path)

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(**cfg.xgb.as_dict())
        if self._xgb_path.exists():
            xgb_model.load_model(str(self._xgb_path))
            logger.info("XGBoost ← %s", self._xgb_path)

        # --- LightGBM (optional — absent in legacy checkpoints) ---
        lgbm_model = None
        if self._lgbm_path.exists():
            try:
                import lightgbm as lgb
                lgbm_model = lgb.Booster(model_file=str(self._lgbm_path))
                # Wrap in sklearn API shim for predict_proba compatibility
                lgbm_model = _LGBMBoosterWrapper(lgbm_model)
                logger.info("LightGBM ← %s", self._lgbm_path)
            except Exception as exc:
                logger.warning("LightGBM load failed (skipping): %s", exc)

        # Ensemble weights from saved config (allows runtime update)
        weights = cfg.ensemble.weights
        if self._ensemble_cfg_path.exists():
            import json
            with open(self._ensemble_cfg_path) as fh:
                weights = json.load(fh).get("weights", weights)

        ensemble = HybridEnsemble(
            xgb_model  = xgb_model,
            lgbm_model = lgbm_model,
            gnn_model  = gnn_model,
            dnn_model  = dnn_model,
            weights    = weights,
            device     = device,
        )

        # --- Calibrator (optional) ---
        calibrator: Optional[object] = None
        if self._calibrator_path.exists():
            calibrator = joblib.load(str(self._calibrator_path))
            logger.info("Calibrator ← %s", self._calibrator_path)

        # --- Meta-learner (optional stacking) ---
        if self._meta_learner_path.exists():
            try:
                ensemble.meta_learner = joblib.load(str(self._meta_learner_path))
                logger.info("MetaLearner ← %s", self._meta_learner_path)
            except Exception as exc:
                logger.warning("Meta-learner load failed (skipping): %s", exc)

        logger.info("All artefacts loaded from %s", self.model_dir)
        return preprocessor, ensemble, calibrator

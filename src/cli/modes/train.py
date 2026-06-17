"""src/cli/modes/train.py — train and train_panels modes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import LoadedDataset, load_csv
from src.scientific.calibration.calibrator import EnsembleCalibrator
from src.scientific.metrics.metrics import evaluate, evaluate_per_panel, find_best_threshold
from src.scientific.metrics.plots import save_all_plots
from src.training.trainer import VariantTrainer
from src.utils.seeds import set_global_seed
from src.utils.serialization import ModelStore


def _get_labelled_data(data_file: Any, cfg: Any) -> LoadedDataset:
    """Load a labelled dataset; search default paths if data_file is None."""
    candidates = []
    if data_file:
        candidates.append(Path(data_file))
    candidates += [
        cfg.paths.data_dir / "train_variants.csv",
        Path("data/train_variants.csv"),
    ]
    for path in candidates:
        if path.exists():
            logging.info("Loading dataset: %s", path)
            ds = load_csv(path)
            if ds.labels is None:
                logging.error("No labels found in %s.", path)
                sys.exit(1)
            return ds
    logging.error("No labelled dataset found.")
    sys.exit(1)


def mode_train(args: argparse.Namespace, cfg: Any) -> None:
    """Leakage-free training + 5-fold CV + calibration + test evaluation."""
    ds = _get_labelled_data(args.data_file, cfg)
    assert ds.labels is not None  # _get_labelled_data sys.exit()s when labels are None

    from src.features.feature_validator import FeatureValidator

    fv_report = FeatureValidator().validate_and_warn(ds.features)
    logging.info("Feature coverage: %.1f %%", 100 * fv_report.overall_coverage)

    panel = getattr(args, "panel", None)
    if panel and "Panel" in ds.metadata.columns:
        mask = ds.metadata["Panel"] == panel
        valid_positions = list(np.where(mask.values)[0])
        from src.data.loader import LoadedDataset

        ds = LoadedDataset(
            features=ds.features[mask].reset_index(drop=True),
            labels=ds.labels[mask.values],
            metadata=ds.metadata[mask].reset_index(drop=True),
            feature_columns=ds.feature_columns,
            nuc_sequences=([ds.nuc_sequences[i] for i in valid_positions] if ds.nuc_sequences else None),
            aa_sequences=([ds.aa_sequences[i] for i in valid_positions] if ds.aa_sequences else None),
        )
        assert ds.labels is not None  # set above from ds.labels[mask.values]
        logging.info("Panel filter: %s (%d variants)", panel, len(ds.labels))

    X = ds.features.values
    y = ds.labels
    assert y is not None  # labels guaranteed non-None (see assert after load)
    set_global_seed(cfg.seed)
    cfg.paths.create_dirs()

    # ── Group-aware leakage guard (TEKNOFEST §7.5) ──────────────────────────
    # The same variant appears across panels (panel overlap) and as augmented
    # near-twins; splitting by row leaks it into both train and test, inflating
    # internal metrics. Group by base Variant_ID (strip any _aug suffix) so a
    # variant never straddles the split.
    groups = None
    if "Variant_ID" in ds.metadata.columns and len(ds.metadata) == len(X):
        groups = ds.metadata["Variant_ID"].astype(str).str.replace(r"_aug\d*$", "", regex=True).values
        n_groups = len(np.unique(groups))
        logging.info("Group-aware splitting ON: %d rows → %d unique variants", len(X), n_groups)

    # Panel labels for optional Domain-Adversarial (DANN) DNN training.
    panels_arr = None
    if "Panel" in ds.metadata.columns and len(ds.metadata) == len(X):
        panels_arr = ds.metadata["Panel"].astype(str).values

    trainer = VariantTrainer()
    result = trainer.train(X, y, nuc_seqs=ds.nuc_sequences, aa_seqs=ds.aa_sequences, groups=groups, panels=panels_arr)
    logging.info("CV summary — Binary F1 (§7.3): %.4f ± %.4f", result.mean_cv_f1, result.std_cv_f1)

    # Stacking meta-öğrenicisini GERÇEK group-aware OOF üzerinde yeniden eğit
    # (Wolpert 1992 doğru stacking). Trainer içindeki inner-val fit'i suboptimaldir;
    # OOF fit nested-CV'de +0.59pp / held-out +0.85pp (reports/stacking_improvement.json).
    try:
        _oof_path = cfg.paths.reports_dir / "oof_per_model.npz"
        if _oof_path.exists() and hasattr(result.ensemble, "fit_meta_learner_from_oof"):
            _oofz = np.load(_oof_path)
            result.ensemble.fit_meta_learner_from_oof(_oofz["oof"], _oofz["labels"])
            logging.info("Meta-learner GENUINE OOF üzerinde yeniden eğitildi (stacking iyileştirmesi).")
    except Exception as _oof_exc:
        logging.warning("OOF meta-learner override atlandı: %s", _oof_exc)

    preprocessor = result.preprocessor
    ensemble = result.ensemble

    # Reuse the SAME held-out split the trainer used (group-aware, consistent).
    all_indices = np.arange(len(X))
    test_indices = result.test_indices
    if test_indices is None:  # backward-compat fallback
        _, test_indices = train_test_split(
            all_indices, test_size=cfg.training.test_size, stratify=y, random_state=cfg.seed
        )
    train_indices = np.setdiff1d(all_indices, test_indices)
    X_tr, y_tr = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Adversarial leakage guard (§7.5): no variant may appear in both splits.
    if groups is not None:
        _overlap = np.intersect1d(np.unique(groups[train_indices]), np.unique(groups[test_indices]))
        if len(_overlap) > 0:
            raise RuntimeError(
                f"LEAKAGE: {len(_overlap)} Variant_ID(s) straddle train/test "
                f"(e.g. {_overlap[:3].tolist()}). Group-aware split failed."
            )
        logging.info(
            "Leakage guard PASSED: 0 variants straddle train/test (test n=%d, %d unique variants).",
            len(test_indices),
            len(np.unique(groups[test_indices])),
        )

    # Calibration split from the train portion — group-aware so calibration/
    # threshold are never fit on a variant also in train. cal_indices are in
    # ORIGINAL X space so panel labels can be recovered consistently below.
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        _g_tr = groups[train_indices]
        _pos_tr, _pos_cal = next(
            GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg.seed + 99).split(
                np.arange(len(train_indices)), y_tr, _g_tr
            )
        )
    else:
        _pos_tr, _pos_cal = train_test_split(
            np.arange(len(train_indices)), test_size=0.15, stratify=y_tr, random_state=cfg.seed + 99
        )
    cal_indices = train_indices[_pos_cal]
    X_cal, y_cal = X[cal_indices], y[cal_indices]
    X_cal_proc = preprocessor.transform(X_cal)
    _, raw_cal_proba = ensemble.predict(X_cal_proc)
    calibrator = EnsembleCalibrator(method=cfg.calibration.method)
    calibrator.fit(raw_cal_proba, y_cal)

    X_test_proc = preprocessor.transform(X_test)
    _, raw_tst_proba = ensemble.predict(X_test_proc)
    cal_test_proba = calibrator.transform(raw_tst_proba)

    cal_cal_proba = calibrator.transform(raw_cal_proba)
    # KARAR EŞİĞİ — TEKNOFEST RESMİ Q&A (2026-06-02): gerçek test seti ~%20 PATOJENİK /
    # %80 benign (eğitimin TERSİ; 50/50 dengeli ESKİ şartnameydi, GEÇERSİZ). F1 patojenik-
    # odaklı (pos_label=1) ve patojenik artık AZINLIK sınıf. Eşik %20-patojenik prior'da
    # türetilir; %74-poz/50-50'de türetmek bu sette F1 KAYBETTİRİR (ölçüldü: θ=0.6831→0.55
    # vs θ≈0.84→0.60). Tam group-aware OOF'ta %20-patojenik resample, 25x medyan F1-optimal.
    _TEST_POS = 0.20
    best_thr = None
    _thr_source = "oof_20pct_resample_median25"  # birincil yol (aşağıdaki try bloğu)
    try:
        _oofz2 = np.load(cfg.paths.reports_dir / "oof_per_model.npz")
        _meta = getattr(ensemble, "meta_learner", None)
        if _meta is not None:
            # HAM full-stack OOF proba — inference (ExternalValidationRunner) etiketi
            # HAM olasılıkta verir (kalibrasyon yalnız 'calibrated_risk' GÖSTERİMİ için),
            # bu yüzden eşik de HAM'da türetilir → derivation==inference tutarlılığı.
            _oof_pr = _meta.predict_proba(_oofz2["oof"])[:, 1]
            _yoof = _oofz2["labels"].astype(int)
            _po = np.where(_yoof == 1)[0]
            _ne = np.where(_yoof == 0)[0]
            # %20-patojenik: pozitif azınlık → pos = neg * (0.20/0.80) = neg/4
            _npos = max(1, int(round(len(_ne) * _TEST_POS / (1 - _TEST_POS))))
            _ths = []
            for _s in range(25):
                _rr = np.random.RandomState(cfg.seed + _s)
                _sub = _rr.choice(_po, min(_npos, len(_po)), replace=False)
                _bi = np.concatenate([_sub, _ne])
                _t, _ = find_best_threshold(_yoof[_bi], _oof_pr[_bi], metric="f1")
                _ths.append(_t)
            best_thr = float(np.median(_ths))
            logging.info(
                "Threshold from %d%%-PATOJENİK OOF resample (25x medyan, TEKNOFEST resmi test prior): %.4f",
                int(_TEST_POS * 100),
                best_thr,
            )
    except Exception as _thr_exc:
        logging.warning("%20-patojenik OOF eşik atlandı, cal-set'e düşülüyor: %s", _thr_exc)
    if best_thr is None:
        best_thr, _ = find_best_threshold(y_cal, cal_cal_proba[:, 1], metric="f1")
        _thr_source = "calibration_set_fallback"
        logging.info("Threshold from calibration set (fallback, n=%d): %.4f", len(y_cal), best_thr)

    report = evaluate(y_test, raw_tst_proba, threshold=best_thr)  # inference HAM'da karar verir

    # GERÇEK yarışma metriği — %20-patojenik (TEKNOFEST resmi test prior) F1 (held-out
    # resample, 300x). Headline'daki hold-out F1 %75-poz iç settedir, jüri skoru DEĞİL.
    # reports/competition_jury_f1.json (§III.9 dürüstlük).
    try:
        _ptst = raw_tst_proba[:, 1]  # inference ile ayni: HAM olasilik
        _pp = np.where(y_test == 1)[0]
        _nn = np.where(y_test == 0)[0]
        _npos_t = max(1, int(round(len(_nn) * _TEST_POS / (1 - _TEST_POS))))
        _bf = []
        for _s in range(300):
            _rr = np.random.RandomState(_s)
            _sub = _rr.choice(_pp, min(_npos_t, len(_pp)), replace=False)
            _bi = np.concatenate([_sub, _nn])
            _yy = y_test[_bi]
            _qq = _ptst[_bi]
            _yp = (_qq >= best_thr).astype(int)
            _tp = int(((_yp == 1) & (_yy == 1)).sum())
            _fp = int(((_yp == 1) & (_yy == 0)).sum())
            _fn = int(((_yp == 0) & (_yy == 1)).sum())
            _bf.append(2 * _tp / (2 * _tp + _fp + _fn) if (2 * _tp + _fp + _fn) else 0.0)
        _cj = float(np.mean(_bf))
        logging.info(
            "%d%%-PATOJENİK jüri-prior F1 (GERÇEK yarışma beklentisi) = %.4f ± %.4f @ θ=%.4f",
            int(_TEST_POS * 100),
            _cj,
            float(np.std(_bf)),
            best_thr,
        )
        json.dump(
            {
                "competition_jury_f1": round(_cj, 4),
                "competition_jury_f1_std": round(float(np.std(_bf)), 4),
                "test_pos_rate": _TEST_POS,
                "threshold": round(best_thr, 4),
                "holdout_f1_75pct_pos": round(report.binary_f1, 4),
                "_note": "competition_jury_f1 = TEKNOFEST RESMİ test prior'i (%20 patojenik/%80 benign) "
                "held-out resample (300x) F1 — GERCEK beklenen yarisma skoru. holdout_f1 = "
                "%75-poz ic hold-out (ayrim gucu, juri skoru DEGIL).",
            },
            open(cfg.paths.reports_dir / "competition_jury_f1.json", "w"),
            indent=2,
            ensure_ascii=False,
        )
    except Exception as _bexc:
        logging.warning("Competition-jury F1 raporu atlandı: %s", _bexc)
    report.log(prefix="TEST")

    store = ModelStore(cfg.paths.models_dir)
    store.save_all(preprocessor, ensemble, calibrator)
    store.save_threshold(best_thr)

    try:
        import joblib as _jl

        from src.scientific.ood_detector import OODDetector

        X_train_proc = preprocessor.transform(X_tr)
        _ood_det = OODDetector(z_threshold=3.5, ood_frac_thresh=0.25)
        _ood_det.fit(X_train_proc)
        _ood_path = cfg.paths.models_dir / "ood_detector.pkl"
        _jl.dump(_ood_det, str(_ood_path))
        logging.info("OOD detector saved → %s", _ood_path)
    except Exception as exc:
        logging.warning("OOD detector training failed (non-fatal): %s", exc)

    save_all_plots(report, y_test, cal_test_proba, cfg.paths.reports_dir)

    panel_reports_dict: dict = {}
    panel_thresholds_dict: dict = {}

    if "Panel" in ds.metadata.columns:
        test_panels = ds.metadata["Panel"].values[test_indices]
        panel_reports = evaluate_per_panel(y_test, raw_tst_proba, test_panels, threshold=best_thr)
        for pname, prep in panel_reports.items():
            prep.log(prefix=f"PANEL_{pname}")
            panel_reports_dict[pname] = prep.as_dict()

        # RESMİ SKORLAMA (TEKNOFEST Q&A): her panel kendi %20-patojenik test setinde
        # ayrı F1, sonra 4-panel ORTALAMASI = nihai yarışma skoru. Genel %20-F1'den farklı.
        try:
            _off: dict[str, Optional[float]] = {}
            for _pn in np.unique(test_panels):
                _m = test_panels == _pn
                _yp = y_test[_m]
                _pr = raw_tst_proba[_m, 1]
                _po = np.where(_yp == 1)[0]
                _ne = np.where(_yp == 0)[0]
                if len(_po) == 0 or len(_ne) == 0:
                    _off[str(_pn)] = None
                    continue
                _np20 = max(1, int(round(len(_ne) * 0.20 / 0.80)))
                _fs = []
                for _s in range(300):
                    _rr = np.random.RandomState(_s)
                    _sub = _rr.choice(_po, min(_np20, len(_po)), replace=False)
                    _bi = np.concatenate([_sub, _ne])
                    _yy = _yp[_bi]
                    _yhat = (_pr[_bi] >= best_thr).astype(int)
                    _tp = int(((_yhat == 1) & (_yy == 1)).sum())
                    _fp = int(((_yhat == 1) & (_yy == 0)).sum())
                    _fn = int(((_yhat == 0) & (_yy == 1)).sum())
                    _fs.append(2 * _tp / (2 * _tp + _fp + _fn) if (2 * _tp + _fp + _fn) else 0.0)
                _off[str(_pn)] = round(float(np.mean(_fs)), 4)
            _valid = [v for v in _off.values() if v is not None]
            _official = round(float(np.mean(_valid)), 4) if _valid else None
            logging.info("RESMİ SKOR (4-panel %%20-patojenik F1 ortalaması) = %s | paneller=%s", _official, _off)
            _cjp = cfg.paths.reports_dir / "competition_jury_f1.json"
            if _cjp.exists():
                _cj = json.load(open(_cjp, encoding="utf-8"))
                _cj["official_per_panel_f1_20pct"] = _off
                _cj["official_competition_score"] = _official
                _cj["_official_note"] = (
                    "TEKNOFEST RESMİ skorlama: her panel %20-patojenik test'inde "
                    "ayrı F1, sonra 4-panel ORTALAMASI. CFTR küçük-n → gürültülü."
                )
                json.dump(_cj, open(_cjp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as _offexc:
            logging.warning("Resmi per-panel skor atlandı: %s", _offexc)

        # PANEL-BAZLI EŞİK denemesi: resmi skor panel-ORTALAMA olduğundan her panel kendi
        # %20-θ'sını CAL'de türetip kendi test'inde uygular → zayıf paneli (PAH) kurtarıp
        # ortalamayı global-tek-θ'dan yükseltebilir mi? (rakiplere karşı 2. edge adayı)
        try:
            _calp = ds.metadata["Panel"].values[cal_indices] if len(ds.metadata) == len(X) else np.array([])
            if len(_calp) == len(y_cal):
                _rc = raw_cal_proba[:, 1]
                _ppth: dict[str, Optional[float]] = {}
                _ppf1: dict[str, Optional[float]] = {}
                for _pn in np.unique(test_panels):
                    _cm = _calp == _pn
                    _ycal_p = y_cal[_cm]
                    _rcal_p = _rc[_cm]
                    _cpo = np.where(_ycal_p == 1)[0]
                    _cne = np.where(_ycal_p == 0)[0]
                    if len(_cpo) < 3 or len(_cne) < 3:
                        _ppth[str(_pn)] = None
                        _ppf1[str(_pn)] = None
                        continue
                    _cnp = max(1, int(round(len(_cne) * 0.25)))
                    _tl = []
                    for _s in range(25):
                        _r = np.random.RandomState(cfg.seed + _s)
                        _ix = np.concatenate([_r.choice(_cpo, min(_cnp, len(_cpo)), replace=False), _cne])
                        _t, _ = find_best_threshold(_ycal_p[_ix], _rcal_p[_ix], metric="f1")
                        _tl.append(_t)
                    _pth = float(np.median(_tl))
                    _ppth[str(_pn)] = round(_pth, 4)
                    _tm = test_panels == _pn
                    _yp = y_test[_tm]
                    _pr = raw_tst_proba[_tm, 1]
                    _po = np.where(_yp == 1)[0]
                    _ne = np.where(_yp == 0)[0]
                    if len(_po) == 0 or len(_ne) == 0:
                        _ppf1[str(_pn)] = None
                        continue
                    _np20 = max(1, int(round(len(_ne) * 0.25)))
                    _fs = []
                    for _s in range(300):
                        _r = np.random.RandomState(_s)
                        _ix = np.concatenate([_r.choice(_po, min(_np20, len(_po)), replace=False), _ne])
                        _yy = _yp[_ix]
                        _yh = (_pr[_ix] >= _pth).astype(int)
                        _tp = int(((_yh == 1) & (_yy == 1)).sum())
                        _fp = int(((_yh == 1) & (_yy == 0)).sum())
                        _fn = int(((_yh == 0) & (_yy == 1)).sum())
                        _fs.append(2 * _tp / (2 * _tp + _fp + _fn) if (2 * _tp + _fp + _fn) else 0.0)
                    _ppf1[str(_pn)] = round(float(np.mean(_fs)), 4)
                _vv = [v for v in _ppf1.values() if v is not None]
                _pta = round(float(np.mean(_vv)), 4) if _vv else None
                logging.info(
                    "PANEL-BAZLI θ resmi skor = %s (global-θ: %s) | θ=%s | F1=%s", _pta, _official, _ppth, _ppf1
                )
                if _cjp.exists():
                    _cj = json.load(open(_cjp, encoding="utf-8"))
                    _cj["per_panel_threshold_score"] = _pta
                    _cj["per_panel_thresholds_20pct"] = _ppth
                    _cj["per_panel_threshold_f1"] = _ppf1
                    json.dump(_cj, open(_cjp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as _ppexc:
            logging.warning("Panel-bazlı θ denemesi atlandı: %s", _ppexc)

        cal_panels = np.array([])
        if len(ds.metadata) == len(X):
            # Reuse the exact group-aware calibration indices computed above.
            cal_panels = ds.metadata["Panel"].values[cal_indices]

        if len(cal_panels) == len(y_cal):
            panel_thresholds_dict = ensemble.optimise_panel_thresholds(
                X_cal_proc, y_cal, cast(Sequence[str], cal_panels)
            )
            store.save_panel_thresholds(panel_thresholds_dict)
        else:
            logging.warning(
                "Panel threshold optimization skipped: cal_panels size (%d) != y_cal (%d).",
                len(cal_panels),
                len(y_cal),
            )

    feat_val_path = cfg.paths.reports_dir / "feature_validation.json"
    with open(feat_val_path, "w") as fh:
        json.dump(fv_report.as_dict(), fh, indent=2, ensure_ascii=False)

    report_path = cfg.paths.reports_dir / "cv_report.json"
    with open(report_path, "w") as fh:
        json.dump(
            {
                "competition_metric": "Binary F1 — 2*TP/(2*TP+FP+FN), pos_label=1 (TEKNOFEST §7.3)",
                "metric_note": "Primary: binary_f1 (Pathogenic). macro_f1 is auxiliary.",
                "swa_enabled": True,
                "mean_cv_binary_f1": result.mean_cv_f1,
                "std_cv_binary_f1": result.std_cv_f1,
                "test_binary_f1": report.binary_f1,
                "test_macro_f1": report.macro_f1,
                "best_threshold": best_thr,
                "threshold_source": _thr_source,
                "feature_coverage": round(fv_report.overall_coverage, 4),
                "anonymous_columns": fv_report.anonymous_count,
                "folds": [vars(r) for r in result.fold_results],
                "test_metrics": report.as_dict(),
                "panel_metrics": panel_reports_dict,
                "panel_thresholds": panel_thresholds_dict,
            },
            fh,
            indent=2,
        )
    logging.info("CV report saved → %s", report_path)
    logging.info("Training complete.")


def mode_train_panels(args: argparse.Namespace, cfg: Any) -> None:
    """DEPRECATED — mode_train'e yönlendirilir.

    Eski uygulama; (a) trainer.train'i ``groups=`` OLMADAN çağırıyor, (b) kalibrasyon
    ve per-panel test bölmelerini SATIR-BAZLI ``train_test_split`` ile yapıyor (aynı
    Variant_ID train/test'i çaprazlayıp withdrawn-leaky sayıları geri sokuyordu), ve
    (c) eşiği KALİBRE olasılıkta türetiyordu (kanonik mode_train HAM'da türetir →
    derivation==inference). Kanonik ``mode_train`` zaten group-aware eğitim + per-panel
    değerlendirme (evaluate_per_panel, panel eşikleri) yapar; bu mod ona delege edilir.
    """
    logging.warning(
        "mode_train_panels DEPRECATED (satır-bazlı split + calibrated-threshold sızıntısı). "
        "Kanonik group-aware mode_train kullanılıyor; per-panel değerlendirme zaten dahil."
    )
    return mode_train(args, cfg)

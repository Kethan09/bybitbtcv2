from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from data import load_or_download
from features import feature_columns, make_features, make_target


def fit_platt(p_raw: np.ndarray, y: np.ndarray):
    # Logistic calibration on a chronological validation slice.
    model = LogisticRegression(solver="lbfgs")
    z = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / np.clip(1 - p_raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model.fit(z, y.astype(int))
    return model


def calibrated_probability(base, calibrator, x: pd.DataFrame) -> np.ndarray:
    raw = base.predict_proba(x)[:, 1]
    z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return calibrator.predict_proba(z)[:, 1]


def train_horizon(raw: pd.DataFrame, horizon: int, cfg: dict):
    features = make_features(raw)
    y = make_target(raw, horizon)
    cols = feature_columns(features)
    ds = features[cols].copy()
    ds["y"] = y
    ds = ds.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ds) < 20_000:
        raise RuntimeError(f"Not enough clean 1m rows for {horizon}m: {len(ds):,}")

    n = len(ds)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    tr, va, te = ds.iloc[:i1], ds.iloc[i1:i2], ds.iloc[i2:]

    params = dict(cfg["xgb"])
    base = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", **params)
    base.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], verbose=False)

    raw_val = base.predict_proba(va[cols])[:, 1]
    calibrator = fit_platt(raw_val, va["y"].to_numpy(dtype=int))
    p_val = calibrated_probability(base, calibrator, va[cols])
    p_test = calibrated_probability(base, calibrator, te[cols])

    threshold = float(cfg["min_confidence"])
    conf_test = np.maximum(p_test, 1 - p_test)
    mask = conf_test >= threshold
    pred = (p_test >= 0.5).astype(int)

    metrics = {
        "horizon_minutes": horizon,
        "rows": len(ds),
        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
        "test_auc": float(roc_auc_score(te["y"], p_test)),
        "test_accuracy_all": float(accuracy_score(te["y"], pred)),
        "test_brier": float(brier_score_loss(te["y"], p_test)),
        "high_conf_threshold": threshold,
        "test_high_conf_coverage": float(mask.mean()),
        "test_high_conf_accuracy": float(accuracy_score(te["y"].to_numpy(dtype=int)[mask], pred[mask])) if mask.any() else None,
        "test_high_conf_count": int(mask.sum()),
        "class_balance_test_up": float(te["y"].mean()),
        "last_test_timestamp": str(te.index[-1]),
        "feature_count": len(cols),
    }

    out = Path(cfg["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "base_model": base,
        "calibrator": calibrator,
        "features": cols,
        "horizon": horizon,
        "threshold": threshold,
        "version": "2.0-bybit-index",
    }
    joblib.dump(artifact, out / f"model_{horizon}m.joblib")
    (out / f"metrics_{horizon}m.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    cache = Path(cfg["data_dir"]) / f"{cfg['symbol']}_{cfg['history_days']}d_index_1m.parquet"
    raw = load_or_download(cfg["symbol"], int(cfg["history_days"]), str(cache))
    print(f"Loaded {len(raw):,} 1m Bybit index candles: {raw.index[0]} -> {raw.index[-1]}")
    for h in (5, 15):
        print(f"\n=== TRAIN {h} MINUTE ===")
        metrics = train_horizon(raw, h, cfg)
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

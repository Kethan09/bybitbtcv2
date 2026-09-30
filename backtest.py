from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

from data import load_or_download
from features import make_features, make_target


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    cache = Path(cfg["data_dir"]) / f"{cfg['symbol']}_{cfg['history_days']}d_index_1m.parquet"
    raw = load_or_download(cfg["symbol"], int(cfg["history_days"]), str(cache))
    features = make_features(raw)

    for h in (5, 15):
        art = joblib.load(Path(cfg["model_dir"]) / f"model_{h}m.joblib")
        # Evaluate only the final 15% so this is aligned with training's unseen test partition.
        y = make_target(raw, h)
        ds = features[art["features"]].copy()
        ds["y"] = y
        ds = ds.replace([np.inf, -np.inf], np.nan).dropna()
        start = int(len(ds) * 0.85)
        test = ds.iloc[start:]
        raw_p = art["base_model"].predict_proba(test[art["features"]])[:, 1]
        z = np.log(np.clip(raw_p, 1e-6, 1 - 1e-6) / np.clip(1 - raw_p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        p = art["calibrator"].predict_proba(z)[:, 1]
        conf = np.maximum(p, 1 - p)
        mask = conf >= float(cfg["min_confidence"])
        pred = (p >= 0.5).astype(int)
        print(json.dumps({
            "horizon": h,
            "test_rows": len(test),
            "all_accuracy": round(float(accuracy_score(test["y"], pred)), 4),
            "high_conf_threshold": float(cfg["min_confidence"]),
            "high_conf_count": int(mask.sum()),
            "high_conf_coverage": round(float(mask.mean()), 4),
            "high_conf_accuracy": None if not mask.any() else round(float(accuracy_score(test["y"].to_numpy(dtype=int)[mask], pred[mask])), 4),
        }, indent=2))


if __name__ == "__main__":
    main()

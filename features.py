from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    ad = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = au / (ad + EPS)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["open", "high", "low", "close", "volume"]].copy().sort_index()
    c, o, h, l, v = x["close"], x["open"], x["high"], x["low"], x["volume"]

    for n in (1, 2, 3, 5, 10, 15, 20, 30, 60, 120):
        x[f"ret_{n}"] = c.pct_change(n)

    for n in (5, 9, 21, 50, 100, 200):
        ema = c.ewm(span=n, adjust=False, min_periods=n).mean()
        x[f"ema_{n}_dist"] = c / (ema + EPS) - 1

    x["rsi7"] = _rsi(c, 7)
    x["rsi14"] = _rsi(c, 14)
    x["rsi28"] = _rsi(c, 28)

    atr14 = _atr(x, 14)
    x["atr14_pct"] = atr14 / (c + EPS)
    x["atr28_pct"] = _atr(x, 28) / (c + EPS)
    x["adx14"] = _adx(x, 14)

    macd_fast = c.ewm(span=12, adjust=False).mean()
    macd_slow = c.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    sig = macd.ewm(span=9, adjust=False).mean()
    x["macd_pct"] = macd / (c + EPS)
    x["macd_hist_pct"] = (macd - sig) / (c + EPS)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["bb_z"] = (c - mid) / (sd + EPS)
    x["bb_width"] = (2 * sd) / (mid + EPS)

    rng = (h - l).clip(lower=EPS)
    body = c - o
    x["body_pct"] = body / (o + EPS)
    x["range_pct"] = rng / (c + EPS)
    x["close_location"] = (c - l) / rng
    x["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    x["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng

    for n in (10, 20, 60, 120):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        x[f"dist_hh_{n}"] = c / (hh + EPS) - 1
        x[f"dist_ll_{n}"] = c / (ll + EPS) - 1

    vm, vs = v.rolling(20).mean(), v.rolling(20).std()
    x["rel_volume"] = v / (vm + EPS)
    x["volume_z"] = (v - vm) / (vs + EPS)

    typical = (h + l + c) / 3
    x["vwap_dist_60"] = c / (((typical * v).rolling(60).sum() / (v.rolling(60).sum() + EPS)) + EPS) - 1

    for rule, prefix in (("5min", "m5"), ("15min", "m15")):
        b = _resample(x[["open", "high", "low", "close", "volume"]], rule)
        bc = b["close"]
        ema20 = bc.ewm(span=20, adjust=False, min_periods=20).mean()
        ema50 = bc.ewm(span=50, adjust=False, min_periods=50).mean()
        x[f"{prefix}_ret"] = bc.pct_change().reindex(x.index, method="ffill")
        x[f"{prefix}_ema20"] = (bc / (ema20 + EPS) - 1).reindex(x.index, method="ffill")
        x[f"{prefix}_ema50"] = (bc / (ema50 + EPS) - 1).reindex(x.index, method="ffill")
        x[f"{prefix}_rsi"] = _rsi(bc, 14).reindex(x.index, method="ffill")
        x[f"{prefix}_atr"] = (_atr(b, 14) / (bc + EPS)).reindex(x.index, method="ffill")
        x[f"{prefix}_adx"] = _adx(b, 14).reindex(x.index, method="ffill")
        x[f"{prefix}_body"] = ((b["close"] - b["open"]) / (b["open"] + EPS)).reindex(x.index, method="ffill")

    # Continuous time-of-day seasonality.
    mod = x.index.hour * 60 + x.index.minute
    x["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * mod / 1440)

    # Current 1m candle must be closed before inference.
    return x.replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    # Bybit Odds Up/Down: future index price strictly above/below matched entry.
    future_close = df["close"].shift(-horizon)
    y = (future_close > df["close"]).astype("float")
    y[future_close.isna()] = np.nan
    return y


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "volume"}
    return [c for c in frame.columns if c not in excluded]

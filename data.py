from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BYBIT_BASE = "https://api.bybit.com"


def _get(session: requests.Session, path: str, params: dict) -> dict:
    for attempt in range(7):
        try:
            r = session.get(BYBIT_BASE + path, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error {payload.get('retCode')}: {payload.get('retMsg')}")
            return payload
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("unreachable")


def _download_kline(path: str, category: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download paginated klines from Bybit.

    Bybit returns each page newest -> oldest.  We therefore page backwards
    using the `end` parameter so a multi-day 1m request is not truncated to
    the newest 1,000 candles.
    """
    rows: list[list[str]] = []
    page_end = end_ms
    interval_ms = 60_000 if interval == "1" else int(interval) * 60_000
    session = requests.Session()
    session.headers.update({"User-Agent": "BTCUSDT-Bybit-Odds-Probability-Bot/1.1"})

    # Safety guard: 90 days of 1m data needs ~130 pages per endpoint.
    max_pages = 1000
    pages = 0

    while page_end > start_ms and pages < max_pages:
        payload = _get(
            session,
            path,
            {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": page_end,
                "limit": 1000,
            },
        )
        batch = payload["result"]["list"]
        if not batch:
            break

        # API response is reverse chronological: newest first.
        rows.extend(batch)
        oldest_start = min(int(r[0]) for r in batch)
        next_end = oldest_start - 1
        pages += 1

        if next_end >= page_end:
            break
        page_end = next_end
        time.sleep(0.12)

        if pages % 25 == 0:
            print(f"  downloaded {len(rows):,} {interval}m candles ...")

        # Once the page reaches the requested start, we are finished.
        if oldest_start <= start_ms:
            break

    if not rows:
        raise RuntimeError(f"Bybit returned no data for {symbol} {interval}")

    cols = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    width = len(rows[0])
    frame = pd.DataFrame(rows, columns=cols[:width])
    for c in [c for c in frame.columns if c != "open_time"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True)
    frame = frame.set_index("open_time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.loc[(frame.index >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
                      (frame.index <= pd.to_datetime(end_ms, unit="ms", utc=True))]
    return frame

def download_1m(symbol: str = "BTCUSDT", days: int = 90, cache_file: str | None = None) -> pd.DataFrame:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    # Index price is the settlement reference for Bybit Odds Up/Down.
    index_df = _download_kline(
        "/v5/market/index-price-kline", "linear", symbol, "1", start_ms, end_ms
    )

    # Linear market kline supplies traded volume features. It is optional for training.
    market_df = _download_kline(
        "/v5/market/kline", "linear", symbol, "1", start_ms, end_ms
    )
    market_df = market_df.rename(columns={
        "open": "m_open", "high": "m_high", "low": "m_low", "close": "m_close",
        "volume": "volume", "turnover": "turnover"
    })

    df = index_df.join(market_df[[c for c in ["volume", "turnover"] if c in market_df.columns]], how="left")
    df["volume"] = df.get("volume", pd.Series(index=df.index, dtype=float)).fillna(0.0)
    df["turnover"] = df.get("turnover", pd.Series(index=df.index, dtype=float)).fillna(0.0)
    df = df.loc[df.index <= pd.Timestamp.now(tz="UTC").floor("min")]

    if cache_file:
        p = Path(cache_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
    return df


def load_or_download(symbol: str, days: int, cache_file: str) -> pd.DataFrame:
    p = Path(cache_file)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            expected = max(1, int(days * 1440 * 0.90))
            if len(df) >= expected:
                return df
            print(f"Cached data has only {len(df):,} candles; expected about {expected:,}. Re-downloading full history...")
        except Exception as exc:
            print(f"Could not use cache {p}: {exc}. Re-downloading...")
    return download_1m(symbol, days, cache_file)

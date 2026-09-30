from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import websocket
import yaml

from features import make_features


class BybitBTCProbabilityBot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbol = cfg["symbol"].upper()
        self.threshold = float(cfg["min_confidence"])
        self.webhook_url = str(cfg.get("webhook_url", "")).strip()
        tg = cfg.get("telegram", {}) or {}
        self.telegram_enabled = bool(tg.get("enabled", False))
        self.telegram_bot_token = str(tg.get("bot_token", "")).strip()
        self.telegram_chat_id = str(tg.get("chat_id", "")).strip()
        self.telegram_send_no_trade = bool(tg.get("send_no_trade", False))
        self.models = {
            5: joblib.load(Path(cfg["model_dir"]) / "model_5m.joblib"),
            15: joblib.load(Path(cfg["model_dir"]) / "model_15m.joblib"),
        }
        self.buffer = self._load_history()
        self.last_emitted = {5: None, 15: None}
        self.last_ticker = {}
        self.signal_log = Path(cfg.get("paper_log", "data/paper_signals.csv"))
        self.signal_log.parent.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> pd.DataFrame:
        p = Path(self.cfg["data_dir"]) / f"{self.cfg['symbol']}_{self.cfg['history_days']}d_index_1m.parquet"
        if not p.exists():
            raise FileNotFoundError("Run train.py first to download Bybit index 1m data and train models.")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].sort_index().tail(3000)

    def on_closed_kline(self, data: dict):
        start = pd.to_datetime(int(data["start"]), unit="ms", utc=True)
        row = pd.DataFrame([{ 
            "open": float(data["open"]), "high": float(data["high"]),
            "low": float(data["low"]), "close": float(data["close"]), "volume": float(data.get("volume", 0.0))
        }], index=[start])
        self.buffer = pd.concat([self.buffer.drop(index=start, errors="ignore"), row]).sort_index().tail(3000)

        # Every closed minute is a potential Odds entry point. The 5m/15m horizon starts NOW.
        self.emit_prediction(5)
        self.emit_prediction(15)

    def _predict(self, horizon: int):
        frame = make_features(self.buffer)
        art = self.models[horizon]
        x = frame[art["features"]].iloc[[-1]]
        if x.isna().any(axis=None):
            return None
        raw = float(art["base_model"].predict_proba(x)[0, 1])
        z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - raw)).reshape(-1, 1)
        p_up = float(art["calibrator"].predict_proba(z)[0, 1])
        return p_up

    def _filters(self, frame: pd.DataFrame, direction: str) -> tuple[bool, list[str]]:
        s = self.cfg.get("signal", {})
        reasons = []
        if bool(s.get("require_trend_filter", True)):
            trend_up = frame["ema_21_dist"].iloc[-1] > 0 and frame["ema_50_dist"].iloc[-1] > 0
            trend_down = frame["ema_21_dist"].iloc[-1] < 0 and frame["ema_50_dist"].iloc[-1] < 0
            if direction == "UP" and not trend_up:
                reasons.append("5m/1m trend not bullish")
            if direction == "DOWN" and not trend_down:
                reasons.append("5m/1m trend not bearish")

        adx = float(frame["adx14"].iloc[-1])
        atr = float(frame["atr14_pct"].iloc[-1])
        if adx < float(s.get("min_adx", 16)):
            reasons.append(f"ADX too low ({adx:.1f})")
        if not (float(s.get("min_atr_pct", 0.00015)) <= atr <= float(s.get("max_atr_pct", 0.0045))):
            reasons.append(f"ATR regime filtered ({atr:.5f})")
        return len(reasons) == 0, reasons

    def _telegram_message(self, payload: dict) -> str:
        action = payload["action"]
        emoji = "🟢" if action == "UP" else "🔴" if action == "DOWN" else "⚪"
        h = int(payload["horizon_minutes"])
        other = payload.get("confirmation_probability_up_pct")
        if other is not None:
            other_direction = "UP" if float(other) >= 50 else "DOWN"
            other_line = f"<b>{15 if h == 5 else 5}M confirmation:</b> {other_direction} {float(other):.2f}% UP / {100-float(other):.2f}% DOWN"
        else:
            other_line = ""
        return (
            f"{emoji} <b>BTCUSDT {action} SIGNAL</b>\n\n"
            f"<b>Expiry:</b> {h} minutes\n"
            f"<b>Index price:</b> ${payload['index_price']:,.2f}\n"
            f"<b>UP:</b> {payload['probability_up_pct']:.2f}%\n"
            f"<b>DOWN:</b> {payload['probability_down_pct']:.2f}%\n"
            f"<b>Confidence:</b> {payload['confidence_pct']:.2f}%\n"
            + (other_line + "\n" if other_line else "") +
            f"<b>Time:</b> {payload['signal_time_local']}\n"
            f"<b>Mode:</b> Prediction only\n\n"
            f"<i>{payload['reason']}</i>"
        )

    def _send_telegram(self, payload: dict):
        if not self.telegram_enabled:
            return
        if not self.telegram_bot_token or not self.telegram_chat_id:
            print("Telegram is enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID is missing.")
            return
        if payload["action"] == "NO TRADE" and not self.telegram_send_no_trade:
            return
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        body = {
            "chat_id": self.telegram_chat_id,
            "text": self._telegram_message(payload),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=body, timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"Telegram error: {exc}")

    def _emit(self, payload: dict):
        print(json.dumps(payload, ensure_ascii=False))
        exists = self.signal_log.exists()
        with self.signal_log.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=payload.keys())
            if not exists:
                w.writeheader()
            w.writerow(payload)
        if self.webhook_url and payload["action"] != "NO TRADE":
            try:
                requests.post(self.webhook_url, json=payload, timeout=5)
            except requests.RequestException as exc:
                print(f"Webhook error: {exc}")
        self._send_telegram(payload)

    def emit_prediction(self, horizon: int):
        frame = make_features(self.buffer)
        p = self._predict(horizon)
        if p is None:
            return
        direction = "UP" if p >= 0.5 else "DOWN"
        confidence = max(p, 1 - p)
        ok_filter, reasons = self._filters(frame, direction)

        other_h = 15 if horizon == 5 else 5
        other_p = self._predict(other_h)
        agreement = other_p is None or ("UP" if other_p >= 0.5 else "DOWN") == direction
        require_agreement = bool(self.cfg.get("signal", {}).get("require_model_agreement", True))

        if confidence < self.threshold:
            action = "NO TRADE"
            reasons.append(f"confidence {confidence:.3f} < {self.threshold:.2f}")
        elif require_agreement and not agreement:
            action = "NO TRADE"
            reasons.append("5m/15m model disagreement")
        elif not ok_filter:
            action = "NO TRADE"
        else:
            action = direction
            reasons.append("high-confidence setup")

        ts = frame.index[-1]
        if self.last_emitted[horizon] == ts:
            return
        self.last_emitted[horizon] = ts

        local = ts.tz_convert(self.cfg.get("timezone", "Asia/Kolkata"))
        payload = {
            "symbol": self.symbol,
            "signal_time_utc": ts.isoformat(),
            "signal_time_local": local.isoformat(),
            "horizon_minutes": horizon,
            "entry_reference": "Bybit index price at signal time",
            "index_price": round(float(self.buffer["close"].iloc[-1]), 2),
            "direction": direction,
            "probability_up_pct": round(p * 100, 2),
            "probability_down_pct": round((1 - p) * 100, 2),
            "confidence_pct": round(confidence * 100, 2),
            "confirmation_probability_up_pct": None if other_p is None else round(other_p * 100, 2),
            "action": action,
            "reason": "; ".join(reasons),
            "paper_mode": True,
        }
        self._emit(payload)


def run():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    bot = BybitBTCProbabilityBot(cfg)
    # Bybit public linear kline stream. We use the closed 1m candles only.
    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
            if msg.get("topic") != f"kline.1.{bot.symbol}":
                return
            for k in msg.get("data", []):
                if bool(k.get("confirm")):
                    bot.on_closed_kline(k)
        except Exception as exc:
            print(f"message error: {exc}")

    def on_error(ws, error):
        print(f"websocket error: {error}")

    def on_close(ws, code, msg):
        print(f"websocket closed: {code} {msg}")

    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "args": [f"kline.1.{bot.symbol}"]}))
        print(f"Connected to Bybit public linear stream: {bot.symbol} 1m")
        print("PAPER MODE: predictions only; no Bybit Odds order is submitted.")

    url = "wss://stream.bybit.com/v5/public/linear"
    while True:
        try:
            ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            print(f"connection failure: {exc}")
        time.sleep(3)


if __name__ == "__main__":
    run()

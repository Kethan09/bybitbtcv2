# BTCUSDT Bybit Odds Probability Bot (Python)

This is a **prediction + alert bot**, not an automatic Bybit Odds execution bot.

It reads Bybit BTCUSDT 1-minute market data, evaluates the market after each **closed 1-minute candle**, and predicts the probability that the **Bybit index price** will be above or below the current reference price after:

- 5 minutes
- 15 minutes

This target is closer to the way Bybit Odds Up/Down works than predicting the color of a fixed chart candle, because Bybit says Up/Down settlement compares the index price at expiry with the matched entry price and the duration is 5 or 15 minutes. citeturn315477search1turn315477search2

## Important limitation

Bybit states that Bybit Odds does not support API/bot/copy-trading integration at launch. This project therefore **does not submit Odds orders**. It generates high-confidence alerts that you can compare with the Odds screen manually. citeturn315477search13

## Data design

The trainer uses Bybit's official Index Price Kline endpoint for the price target and Bybit linear market kline data for volume context. Bybit documents the Index Price Kline endpoint at `/v5/market/index-price-kline`, with 1-minute intervals available. citeturn245103search0turn245103search2

The live process uses Bybit's public linear 1-minute kline websocket and only processes `confirm=true` candles, meaning the 1-minute candle is closed. citeturn315477search0

## Install on Windows

```powershell
cd C:\path\to\btc_usdt_prediction_bot
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

Edit `config.yaml` first. Start with 90 days. For serious evaluation, later train with 180+ days and keep the final test segment untouched.

```powershell
python train.py
```

Outputs:

```text
models/model_5m.joblib
models/model_15m.joblib
models/metrics_5m.json
models/metrics_15m.json
data/BTCUSDT_90d_index_1m.parquet
```

The training split is chronological: 70% train, 15% calibration/validation, 15% final unseen test. The probability is calibrated on the validation slice, not on the final test slice.

## Run live

```powershell
python live_bot.py
```

Example:

```json
{
  "symbol": "BTCUSDT",
  "horizon_minutes": 5,
  "direction": "UP",
  "probability_up_pct": 88.4,
  "probability_down_pct": 11.6,
  "confidence_pct": 88.4,
  "action": "UP"
}
```

The default filter is deliberately strict:

- minimum confidence: 85%
- optional 5m/15m agreement
- trend filter
- ADX regime filter
- ATR volatility filter
- no signal below the threshold

A high model probability is **not a guarantee** of an 85% real-world win rate. The only number that matters is the bot's audited out-of-sample/live result after enough signals.

## Paper log

Every prediction is saved to:

```text
`data/paper_signals.csv`
```

Use this to record the subsequent 5m/15m outcome and measure real-time accuracy.

## Bybit Odds timing

Bybit says the Up/Down clock begins when the order is matched, rather than being tied to a universal 5-minute chart candle boundary. Therefore the live predictor evaluates **from the current closed 1-minute reference forward by 5 or 15 minutes**. citeturn315477search3turn315477search6

## Do not use a fake 95% guarantee

A model can say 92%, but that does not make 92% true. Always judge the model using:

- final unseen test accuracy at the selected threshold
- Brier score / calibration
- high-confidence coverage
- maximum consecutive losses
- performance across trending/ranging/high-volatility regimes
- live paper-trading results

Only after the live paper results are stable should you consider using real-money Odds contracts.

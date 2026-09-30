# Telegram signal setup

The bot can send only high-confidence `UP` / `DOWN` signals to Telegram. `NO TRADE` is not sent by default.

## 1. Create your Telegram bot

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot`.
3. Give the bot a name and username.
4. BotFather will give you a **bot token**. Keep it private.

## 2. Start your bot

Open the new bot in Telegram and send:

```text
/start
```

## 3. Get your chat ID

Open this URL in a browser after sending `/start`:

```text
https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
```

Replace `YOUR_BOT_TOKEN` with your real token.

In the JSON response, find:

```text
"chat":{"id":123456789
```

That number is your `chat_id`.

## 4. Put the values into `config.yaml`

```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "123456789"
  send_no_trade: false
```

For a private Telegram chat, the ID is normally a positive integer. For a group/channel it can have a different format.

## 5. Run the bot

```powershell
python train.py
python live_bot.py
```

When a qualifying signal appears, Telegram will receive a message like:

```text
🟢 BTCUSDT UP SIGNAL

Expiry: 5 minutes
Index price: $108,250.00
UP: 91.20%
DOWN: 8.80%
Confidence: 91.20%
15M confirmation: UP 86.70% UP / 13.30% DOWN
Time: 2026-09-30T17:01:00+05:30
Mode: Prediction only
```

## Security

Do not post your Telegram bot token publicly or commit it to GitHub. For a production deployment, use environment variables or a secret manager instead of storing the token directly in `config.yaml`.

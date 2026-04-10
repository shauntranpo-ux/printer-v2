# printer-v2

Kalshi BTC prediction market trading bot. Runs a 4-model AI ensemble (Claude, GPT-4o, Gemini, DeepSeek) in parallel every 15 minutes to score active markets, apply 5-gate risk checks, size via half-Kelly, and place limit orders.

## Architecture

```
coinbase_feed.py  ──▶  ensemble.py  ──▶  risk_gates.py  ──▶  strategy.py  ──▶  kalshi_client.py
    (BTC price)       (4 AI models)      (5 risk gates)    (Kelly sizing)       (place order)
                                               │
                                          database.py ──▶ dashboard.py
                                               │
                                        telegram_alerts.py
```

`runner.py` (TradingBot) orchestrates everything on 15-minute clock-aligned cycles.

---

## 1. Setup

### Clone and install

```bash
git clone https://github.com/shauntranpo-ux/printer-v2
cd printer-v2
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.template .env
# Open .env and fill in every required value
```

---

## 2. API Keys

| Service | Where to get it |
|---------|----------------|
| **Kalshi** API key + RSA key pair | [kalshi.com](https://kalshi.com) → Account → API → Create key (download the `.pem` file) |
| **Anthropic** (Claude) | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| **OpenAI** (GPT-4o) | [platform.openai.com](https://platform.openai.com) → API Keys |
| **Google** (Gemini) | [aistudio.google.com](https://aistudio.google.com) → Get API key |
| **DeepSeek** | [platform.deepseek.com](https://platform.deepseek.com) → API Keys |
| **Telegram** bot token | Message [@BotFather](https://t.me/BotFather) → `/newbot` |
| **Telegram** chat ID | Message [@userinfobot](https://t.me/userinfobot) to get your chat ID |

### Kalshi RSA key

After creating an API key on Kalshi, place the downloaded `.pem` file in the project root and set in `.env`:

```
KALSHI_PRIVATE_KEY=./kalshi_private_key.pem
```

Or paste the raw PEM content directly as the value:

```
KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n...
```

---

## 3. Run locally

```bash
# Terminal 1 — trading bot
python runner.py

# Terminal 2 — dashboard
python dashboard.py
```

Dashboard at `http://localhost:8080`. Auto-refreshes every 30 seconds.

---

## 4. Deploy to Railway

### Initial deploy

1. Create a new project at [railway.app](https://railway.app)
2. Connect your GitHub repo
3. Railway auto-detects the `Procfile` and starts two services:
   - **web** — Flask dashboard (Railway assigns `$PORT` automatically)
   - **worker** — trading bot loop

### Set environment variables

In the Railway dashboard → Variables, add every key from `.env.template`. For the Kalshi private key, paste the raw PEM string as `KALSHI_PRIVATE_KEY`.

### Persistent database

Attach a Railway Volume mounted at `/app` so `printer_v2.db` survives redeploys:

Railway dashboard → your service → Volumes → Add volume → mount at `/app`

Then set `DB_PATH=/app/printer_v2.db` in Railway variables.

### Demo / paper trade first

Set `KALSHI_DEMO=true` to run on Kalshi's demo environment with no real money.

---

## 5. Kill switch

To stop the bot without killing the process (useful in production):

```bash
# In the project directory (or Railway shell)
touch STOP
```

The bot checks for a `STOP` file at the top of every cycle. When found it:
1. Sends a Telegram kill-switch alert
2. Logs the shutdown event
3. Exits the loop cleanly

Remove the file to allow restart:

```bash
rm STOP
```

---

## 6. Dashboard

Navigate to the Railway web service URL (or `http://localhost:8080`).

| Section | What it shows |
|---------|--------------|
| **Status dot** | Green = bot running, Red = STOP file detected |
| **BTC Price** | Live price from Coinbase public API (30s cache) |
| **Balance** | Kalshi available cash (60s cache) |
| **Today P&L** | Realised P&L in dollars and percentage of wagered |
| **Daily Loss Used** | Progress bar — red fills toward $100 limit |
| **Win Rate** | Wins / total closed trades today |
| **Sharpe** | Trade-level Sharpe ratio (mean P&L / std P&L) |
| **Open Positions** | Live bid price, P&L%, and time since entry |
| **Last 20 Trades** | Entry/exit prices, P&L, and exit reason |
| **Last Ensemble** | Per-model probability, consensus, and TRADE/SKIP/WAIT action |

API endpoints (JSON):

```
GET /api/stats       — all dashboard data
GET /api/positions   — open positions with live prices
GET /api/trades      — last 20 trades
GET /health          — {"status": "ok"} for Railway healthcheck
```

---

## Files

| File | Purpose |
|------|---------|
| `runner.py` | TradingBot — 15-minute orchestration loop |
| `ensemble.py` | 4 AI models in parallel, consensus aggregation |
| `risk_gates.py` | 5-gate async pre-trade filter |
| `strategy.py` | Kelly sizing, enter_trade, check_exits |
| `kalshi_client.py` | Kalshi REST API v2 (RSA auth, rate limiter) |
| `coinbase_feed.py` | Coinbase WebSocket BTC feed + candle builder |
| `telegram_alerts.py` | Trade entry/exit/error alerts via Telegram Bot API |
| `database.py` | SQLite via aiosqlite — trades, stats, ensemble log |
| `dashboard.py` | Flask single-page dashboard + JSON API |
| `config.py` | pydantic-settings — all env vars, validation |

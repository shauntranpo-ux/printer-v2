# printer-v2

Kalshi BTC options trading bot. 15-minute loop, 4-model ensemble, Kelly sizing, 5-gate risk checks.

## Architecture

```
coinbase_feed.py  →  ensemble.py  →  risk_gates.py  →  strategy.py  →  kalshi_client.py
      ↓                                                                        ↓
  (BTC price)                                                           (place order)
                                         ↓
                                    database.py  →  dashboard.py
                                         ↓
                                  telegram_alerts.py
```

All components are orchestrated by `runner.py` on a 15-minute clock-aligned cadence.

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd printer-v2
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env and fill in all required values
```

Required keys:
- `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` — from your Kalshi account
- `COINBASE_API_KEY` and `COINBASE_API_SECRET` — from Coinbase Advanced Trade

### 3. Add your Kalshi private key

Place your RSA private key at the path specified in `KALSHI_PRIVATE_KEY_PATH` (default: `./kalshi_private_key.pem`).

### 4. Run locally

```bash
# Start the trading bot
python runner.py

# Start the dashboard (separate terminal)
python dashboard.py
```

Dashboard available at `http://localhost:5000` (password from `DASHBOARD_PASSWORD`).

## Deploy to Railway

### One-time setup

1. Install the Railway CLI: `npm install -g @railway/cli`
2. `railway login`
3. `railway init` in the project directory
4. Add all env vars from `.env.template` in the Railway dashboard

### Deploy

```bash
railway up
```

Railway will start two processes from the `Procfile`:
- **web** — Flask dashboard (auto-assigned port)
- **worker** — Trading bot loop

### Persistent volume (recommended)

Attach a Railway volume at `/app` so `printer_v2.db` survives redeploys.

## Risk Gates

All 5 gates must pass before any order is placed:

| Gate | Check | Default Threshold |
|------|-------|-------------------|
| 1 | Daily drawdown | < 5% of balance |
| 2 | Open exposure | < 20% of balance |
| 3 | Ensemble confidence | ≥ 0.60 |
| 4 | BTC realized volatility | < 4% (1h) |
| 5 | Trade frequency | ≥ 15 min since last trade |

## Ensemble Models

| Model | Strategy |
|-------|----------|
| Trend | EMA crossover + ADX |
| Mean Reversion | Bollinger Band + RSI |
| Momentum | Rate of change + MACD |
| Volatility Regime | ATR + VIX proxy |

Weights are recalibrated dynamically based on each model's rolling accuracy.

## Files

| File | Purpose |
|------|---------|
| `runner.py` | Main 15-minute trading loop |
| `kalshi_client.py` | Kalshi REST API wrapper |
| `coinbase_feed.py` | Coinbase WebSocket BTC feed |
| `ensemble.py` | 4-model AI ensemble engine |
| `risk_gates.py` | 5-gate pre-trade risk checks |
| `strategy.py` | Kelly sizing + entry/exit logic |
| `dashboard.py` | Flask web dashboard |
| `telegram_alerts.py` | Telegram notification system |
| `database.py` | SQLite trade logging |
| `config.py` | Settings loaded from .env |

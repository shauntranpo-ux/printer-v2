"""
config.py — All settings loaded from .env

Responsibilities:
- Load all environment variables at import time via python-dotenv
- Expose a single `cfg` object with typed, validated fields
- Fail fast with a clear error if any required variable is missing
- Provide sensible defaults for optional parameters
"""

import os
from dotenv import load_dotenv

load_dotenv()

# TODO: implement Config dataclass or namespace with all settings
# TODO: validate required keys on startup and raise ConfigError if missing

# --- Kalshi ---
# KALSHI_API_KEY_ID
# KALSHI_PRIVATE_KEY_PATH
# KALSHI_BASE_URL  (default: https://trading-api.kalshi.com/trade-api/v2)

# --- Coinbase ---
# COINBASE_API_KEY
# COINBASE_API_SECRET

# --- Ensemble ---
# ENSEMBLE_CONFIDENCE_MIN  (default: 0.60)
# MODEL_WEIGHTS_TREND, MODEL_WEIGHTS_MEAN_REV, MODEL_WEIGHTS_MOMENTUM, MODEL_WEIGHTS_VOL

# --- Risk ---
# MAX_DAILY_DRAWDOWN_PCT   (default: 0.05)
# MAX_POSITION_EXPOSURE    (default: 0.20)
# MIN_MINUTES_BETWEEN_TRADES (default: 15)
# MAX_BTC_VOL_THRESHOLD    (default: 0.04)

# --- Kelly ---
# KELLY_FRACTION           (default: 0.5)
# MAX_POSITION_PCT         (default: 0.10)

# --- Telegram ---
# TELEGRAM_TOKEN
# TELEGRAM_CHAT_ID

# --- Dashboard ---
# DASHBOARD_PASSWORD
# PORT                     (default: 5000)

# --- Database ---
# DATABASE_PATH            (default: printer_v2.db)

# TODO: expose cfg = Config(...) with all fields parsed and validated

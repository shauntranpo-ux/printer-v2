"""
config.py — All settings loaded from .env
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    pass


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise ConfigError(f"Required env var missing or empty: {key}")
    return val


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _float(key: str, default: float) -> float:
    raw = os.getenv(key, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a float, got: {raw!r}")


def _int(key: str, default: int) -> int:
    raw = os.getenv(key, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an int, got: {raw!r}")


@dataclass
class KalshiConfig:
    api_key_id: str
    private_key_path: Path
    base_url: str


@dataclass
class CoinbaseConfig:
    api_key: str
    api_secret: str
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"


@dataclass
class EnsembleConfig:
    confidence_min: float       # minimum confidence to trade
    weight_trend: float
    weight_mean_rev: float
    weight_momentum: float
    weight_vol: float

    def __post_init__(self):
        total = self.weight_trend + self.weight_mean_rev + self.weight_momentum + self.weight_vol
        if abs(total - 1.0) > 0.01:
            raise ConfigError(
                f"Model weights must sum to 1.0, got {total:.3f}. "
                "Check MODEL_WEIGHT_* env vars."
            )


@dataclass
class RiskConfig:
    max_daily_drawdown_pct: float   # e.g. 0.05 = 5%
    max_position_exposure: float    # e.g. 0.20 = 20% of balance
    min_minutes_between_trades: int
    max_btc_vol_threshold: float    # e.g. 0.04 = 4% realized vol


@dataclass
class KellyConfig:
    fraction: float         # 0.5 = half-Kelly
    max_position_pct: float # hard cap per trade


@dataclass
class TelegramConfig:
    token: str      # empty string means disabled
    chat_id: str


@dataclass
class DashboardConfig:
    password: str
    port: int


@dataclass
class Config:
    kalshi: KalshiConfig
    coinbase: CoinbaseConfig
    ensemble: EnsembleConfig
    risk: RiskConfig
    kelly: KellyConfig
    telegram: TelegramConfig
    dashboard: DashboardConfig
    database_path: Path
    env: str  # "live" | "demo"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram.token and self.telegram.chat_id)


def _load() -> Config:
    try:
        kalshi = KalshiConfig(
            api_key_id=_require("KALSHI_API_KEY_ID"),
            private_key_path=Path(_optional("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")),
            base_url=_optional(
                "KALSHI_BASE_URL",
                "https://trading-api.kalshi.com/trade-api/v2",
            ),
        )
        if not kalshi.private_key_path.exists():
            raise ConfigError(
                f"Kalshi private key not found at: {kalshi.private_key_path}. "
                "Set KALSHI_PRIVATE_KEY_PATH or place the file there."
            )

        coinbase = CoinbaseConfig(
            api_key=_require("COINBASE_API_KEY"),
            api_secret=_require("COINBASE_API_SECRET"),
        )

        # Normalize weights before building EnsembleConfig so they always sum to 1.0
        raw_weights = [
            _float("MODEL_WEIGHT_TREND", 0.30),
            _float("MODEL_WEIGHT_MEAN_REV", 0.25),
            _float("MODEL_WEIGHT_MOMENTUM", 0.25),
            _float("MODEL_WEIGHT_VOL", 0.20),
        ]
        total_w = sum(raw_weights)
        w_trend, w_mr, w_mom, w_vol = [w / total_w for w in raw_weights]

        ensemble = EnsembleConfig(
            confidence_min=_float("ENSEMBLE_CONFIDENCE_MIN", 0.60),
            weight_trend=w_trend,
            weight_mean_rev=w_mr,
            weight_momentum=w_mom,
            weight_vol=w_vol,
        )

        risk = RiskConfig(
            max_daily_drawdown_pct=_float("MAX_DAILY_DRAWDOWN_PCT", 0.05),
            max_position_exposure=_float("MAX_POSITION_EXPOSURE", 0.20),
            min_minutes_between_trades=_int("MIN_MINUTES_BETWEEN_TRADES", 15),
            max_btc_vol_threshold=_float("MAX_BTC_VOL_THRESHOLD", 0.04),
        )

        kelly = KellyConfig(
            fraction=_float("KELLY_FRACTION", 0.5),
            max_position_pct=_float("MAX_POSITION_PCT", 0.10),
        )

        telegram = TelegramConfig(
            token=_optional("TELEGRAM_TOKEN", ""),
            chat_id=_optional("TELEGRAM_CHAT_ID", ""),
        )

        dashboard = DashboardConfig(
            password=_optional("DASHBOARD_PASSWORD", "changeme"),
            port=_int("PORT", 5000),
        )

        env = _optional("KALSHI_ENV", "live")
        if env not in ("live", "demo"):
            raise ConfigError(f"KALSHI_ENV must be 'live' or 'demo', got: {env!r}")

        return Config(
            kalshi=kalshi,
            coinbase=coinbase,
            ensemble=ensemble,
            risk=risk,
            kelly=kelly,
            telegram=telegram,
            dashboard=dashboard,
            database_path=Path(_optional("DATABASE_PATH", "printer_v2.db")),
            env=env,
        )

    except ConfigError as exc:
        print(f"[config] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


cfg = _load()

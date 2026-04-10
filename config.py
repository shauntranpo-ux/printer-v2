"""
config.py — All settings loaded from environment variables via pydantic-settings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------
    MAX_BET_SIZE: float = Field(default=25.0, description="Max dollars per trade")
    DAILY_LOSS_LIMIT: float = Field(default=100.0, description="Halt trading after losing this much today (dollars)")
    KELLY_FRACTION: float = Field(default=0.5, description="Fractional Kelly multiplier (0.5 = half-Kelly)")
    MIN_EDGE: float = Field(default=0.02, description="Minimum implied edge (2%) required to place a trade")
    MIN_CONFIDENCE: float = Field(default=0.14, description="Minimum ensemble confidence score to trade")
    MAX_MODEL_SPREAD: float = Field(default=0.50, description="Max allowed disagreement between models (abort if exceeded)")
    MAX_OPEN_POSITIONS: int = Field(default=6, description="Maximum concurrent open positions")
    STOP_LOSS_PCT: float = Field(default=0.67, description="Close position when it loses this fraction of cost (67% — e.g. entry 60¢ exits at ~20¢)")
    CONFIDENCE_DECAY_EXIT: float = Field(default=0.20, description="Exit open position when market bid drops below this (20¢ — market is pricing <20% win probability)")
    TAKE_PROFIT_PCT: float = Field(default=0.55, description="Close position at +55% profit (or let expire if market bid ≥ 75¢)")
    TRAILING_STOP_LOCK_PCT: float = Field(default=0.30, description="Activate trailing stop once peak P&L reaches +30%")
    TRAILING_STOP_EXIT_PCT: float = Field(default=0.20, description="Exit trailing stop if P&L drops below +20% from peak")

    # ------------------------------------------------------------------
    # Kalshi
    # ------------------------------------------------------------------
    KALSHI_API_KEY: str = Field(default="", description="Kalshi API key ID")
    KALSHI_PRIVATE_KEY: str = Field(default="", description="RSA private key — raw PEM string (paste full key including headers)")
    KALSHI_BASE_URL: str = Field(
        default="https://api.elections.kalshi.com/trade-api/v2",
        description="Kalshi REST API base URL",
    )
    KALSHI_DEMO: bool = Field(default=False, description="Use Kalshi demo environment")

    # ------------------------------------------------------------------
    # Coinbase
    # ------------------------------------------------------------------
    COINBASE_WS_URL: str = Field(
        default="wss://advanced-trade-ws.coinbase.com",
        description="Coinbase Advanced Trade WebSocket URL",
    )
    BTC_PRODUCT_ID: str = Field(default="BTC-USD", description="Coinbase product ID for BTC")
    PRICE_STALENESS_SECONDS: int = Field(default=30, description="Max seconds before price is considered stale")

    # ------------------------------------------------------------------
    # Multi-asset trading
    # ------------------------------------------------------------------
    SUPPORTED_ASSETS: str = Field(
        default="BTC,ETH,SOL,XRP,DOGE,HYPE,BNB",
        description="Comma-separated crypto assets to trade 15m Kalshi markets for",
    )

    @property
    def supported_assets_list(self) -> list[str]:
        return [a.strip().upper() for a in self.SUPPORTED_ASSETS.split(",") if a.strip()]

    # ------------------------------------------------------------------
    # AI Models
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic Claude API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API key")

    CLAUDE_MODEL: str = Field(default="claude-sonnet-4-5", description="Claude model ID")
    GPT_MODEL: str = Field(default="gpt-4o", description="OpenAI model ID")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model ID")
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat", description="DeepSeek model ID (deepseek-chat for JSON, deepseek-reasoner for CoT)")

    # ------------------------------------------------------------------
    # Ensemble weights (auto-normalized if they don't sum to 1.0)
    # ------------------------------------------------------------------
    CLAUDE_WEIGHT: float = Field(default=0.30, description="Claude vote weight")
    GPT_WEIGHT: float = Field(default=0.25, description="GPT vote weight")
    GEMINI_WEIGHT: float = Field(default=0.25, description="Gemini vote weight")
    DEEPSEEK_WEIGHT: float = Field(default=0.20, description="DeepSeek vote weight")

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram bot token from @BotFather")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Telegram chat/group ID for alerts")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DB_PATH: str = Field(default="printer_v2.db", description="SQLite database file path")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    DASHBOARD_PORT: int = Field(default=8080, description="Flask dashboard port")
    DASHBOARD_HOST: str = Field(default="0.0.0.0", description="Flask dashboard bind host")

    # ------------------------------------------------------------------
    # Market filters
    # ------------------------------------------------------------------
    RESPECT_TIME_FILTERS: bool = Field(
        default=True,
        description="Apply UTC time-based bet sizing (high/medium/low activity windows)",
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("KELLY_FRACTION")
    @classmethod
    def kelly_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"KELLY_FRACTION must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("MIN_CONFIDENCE", "CONFIDENCE_DECAY_EXIT")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Confidence values must be in (0.0, 1.0), got {v}")
        return v

    @field_validator("STOP_LOSS_PCT", "MIN_EDGE", "MAX_MODEL_SPREAD")
    @classmethod
    def pct_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Percentage values must be in (0.0, 1.0), got {v}")
        return v

    @field_validator("MAX_BET_SIZE", "DAILY_LOSS_LIMIT")
    @classmethod
    def dollar_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Dollar limits must be positive, got {v}")
        return v

    @field_validator("MAX_OPEN_POSITIONS")
    @classmethod
    def positions_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"MAX_OPEN_POSITIONS must be ≥ 1, got {v}")
        return v

    # ------------------------------------------------------------------
    # Model-level validator: normalize ensemble weights
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def normalize_weights(self) -> "Settings":
        total = self.CLAUDE_WEIGHT + self.GPT_WEIGHT + self.GEMINI_WEIGHT + self.DEEPSEEK_WEIGHT
        if abs(total - 1.0) > 0.001:
            self.CLAUDE_WEIGHT   /= total
            self.GPT_WEIGHT      /= total
            self.GEMINI_WEIGHT   /= total
            self.DEEPSEEK_WEIGHT /= total
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def env(self) -> str:
        return "demo" if self.KALSHI_DEMO else "live"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)

    @property
    def ensemble_weights(self) -> dict[str, float]:
        return {
            "claude":   self.CLAUDE_WEIGHT,
            "gpt":      self.GPT_WEIGHT,
            "gemini":   self.GEMINI_WEIGHT,
            "deepseek": self.DEEPSEEK_WEIGHT,
        }

    # ------------------------------------------------------------------
    # validate() — explicit startup check with aggregated errors
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Call once at startup. Raises RuntimeError listing ALL missing/invalid
        settings so the operator can fix everything in one shot.
        """
        errors: list[str] = []

        # Required API keys
        required_keys = {
            "KALSHI_API_KEY":    self.KALSHI_API_KEY,
            "KALSHI_PRIVATE_KEY": self.KALSHI_PRIVATE_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "OPENAI_API_KEY":    self.OPENAI_API_KEY,
            "GEMINI_API_KEY":    self.GEMINI_API_KEY,
            "DEEPSEEK_API_KEY":  self.DEEPSEEK_API_KEY,
        }
        for name, val in required_keys.items():
            if not val.strip():
                errors.append(f"  • {name} is required but not set")

        # Optional but warn if Telegram is half-configured
        if bool(self.TELEGRAM_BOT_TOKEN) != bool(self.TELEGRAM_CHAT_ID):
            errors.append(
                "  • TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set "
                "(or both left empty)"
            )

        # Kalshi private key — must look like a PEM block
        if self.KALSHI_PRIVATE_KEY.strip() and not self.KALSHI_PRIVATE_KEY.strip().startswith("-----BEGIN"):
            errors.append(
                "  • KALSHI_PRIVATE_KEY must be a raw PEM string starting with '-----BEGIN'"
            )

        # Stop-loss vs bet size sanity
        if self.STOP_LOSS_PCT >= 1.0:
            errors.append(f"  • STOP_LOSS_PCT ({self.STOP_LOSS_PCT}) must be < 1.0")

        if self.DAILY_LOSS_LIMIT < self.MAX_BET_SIZE:
            errors.append(
                f"  • DAILY_LOSS_LIMIT (${self.DAILY_LOSS_LIMIT}) is less than "
                f"MAX_BET_SIZE (${self.MAX_BET_SIZE}) — you'd halt after one trade"
            )

        if errors:
            msg = "Configuration errors — fix these before starting the bot:\n" + "\n".join(errors)
            raise RuntimeError(msg)

        # All good
        _print_summary(self)


def _print_summary(s: Settings) -> None:
    """Log a concise config summary at startup (no secrets)."""
    print(
        f"[config] printer-v2  env={s.env}  "
        f"bet=${s.MAX_BET_SIZE}  loss_limit=${s.DAILY_LOSS_LIMIT}  "
        f"kelly={s.KELLY_FRACTION}  min_conf={s.MIN_CONFIDENCE}  "
        f"max_positions={s.MAX_OPEN_POSITIONS}\n"
        f"[config] weights  claude={s.CLAUDE_WEIGHT:.2f}  gpt={s.GPT_WEIGHT:.2f}  "
        f"gemini={s.GEMINI_WEIGHT:.2f}  deepseek={s.DEEPSEEK_WEIGHT:.2f}\n"
        f"[config] models   claude={s.CLAUDE_MODEL}  gpt={s.GPT_MODEL}  "
        f"gemini={s.GEMINI_MODEL}  deepseek={s.DEEPSEEK_MODEL}\n"
        f"[config] db={s.DB_PATH}  dashboard={s.DASHBOARD_HOST}:{s.DASHBOARD_PORT}  "
        f"telegram={'enabled' if s.telegram_enabled else 'disabled'}"
    )


# Single instance — import this everywhere
try:
    settings = Settings()
except Exception as exc:
    print(f"[config] FATAL: failed to load settings — {exc}", file=sys.stderr)
    sys.exit(1)

"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket BTC price feed
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from config import settings

log = logging.getLogger(__name__)

CANDLE_SECONDS = 15 * 60   # 900 s per candle


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StaleDataError(Exception):
    """Raised when BTC price data is older than PRICE_STALENESS_SECONDS."""


# ---------------------------------------------------------------------------
# Candle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float          # approximate incremental vol from 24h delta
    timestamp: datetime       # UTC start of the 15m interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_for(dt: datetime) -> datetime:
    """Snap a UTC datetime down to the nearest 15-minute boundary."""
    snapped_minute = (dt.minute // 15) * 15
    return dt.replace(minute=snapped_minute, second=0, microsecond=0)


def _empty_candle(price: float, bucket: datetime) -> dict:
    return {"open": price, "high": price, "low": price,
            "close": price, "volume": 0.0, "timestamp": bucket}


# ---------------------------------------------------------------------------
# CoinbaseFeed
# ---------------------------------------------------------------------------

class CoinbaseFeed:
    def __init__(self) -> None:
        # ---- Public state (per spec) ----
        self.current_price: float = 0.0
        self.last_update:   datetime | None = None
        self.candles:       deque[Candle] = deque(maxlen=25)
        self.current_candle: dict = {}       # open/high/low/close/volume/timestamp

        # ---- Private bookkeeping ----
        self._volume_24h:     float = 0.0
        self._candle_start_vol: float = 0.0  # 24h vol snapshot at candle open
        self._current_bucket: datetime | None = None

        self._price_event = asyncio.Event()
        self._running:  bool = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_current_price(self) -> float:
        """Return current BTC price. Raises StaleDataError if data is old."""
        if self.is_stale():
            age = "never"
            if self.last_update:
                secs = (datetime.now(timezone.utc) - self.last_update).total_seconds()
                age = f"{secs:.0f}s ago"
            raise StaleDataError(
                f"BTC price data is stale (last update: {age}). "
                f"Threshold: {settings.PRICE_STALENESS_SECONDS}s"
            )
        return self.current_price

    def get_momentum(self) -> float:
        """
        Calculate BTC momentum from the last 4 completed 15m candles.

        Three components (all normalised to [-1, 1] via tanh):
          • Direction  (40%) — recency-weighted candle body direction
          • Velocity   (40%) — overall price change across the 4-candle window
          • Volume     (20%) — whether recent volume is above average
                              (amplifies direction, does not reverse it)

        Returns a float in [-1.0, 1.0]:
          positive → bullish  |  negative → bearish  |  0 → neutral
        """
        candles = list(self.candles)[-4:]
        n = len(candles)
        if n < 2:
            return 0.0

        # --- Direction: recency-weighted body slope per candle ---
        all_weights = [0.1, 0.2, 0.3, 0.4]    # oldest → newest
        weights = all_weights[-n:]
        w_sum = sum(weights)

        direction_sum = 0.0
        for c, w in zip(candles, weights):
            if c.open > 0:
                pct = (c.close - c.open) / c.open
                # tanh(pct * 100): a 1% candle move → tanh(1) ≈ 0.76
                direction_sum += math.tanh(pct * 100) * w
        direction_score = direction_sum / w_sum

        # --- Velocity: price change across the full window ---
        first_close = candles[0].close
        last_close  = candles[-1].close
        velocity_score = 0.0
        if first_close > 0:
            velocity_score = math.tanh((last_close - first_close) / first_close * 100)

        # --- Volume: is recent candle volume above average? ---
        vols = [c.volume for c in candles]
        mean_vol = sum(vols) / n
        vol_factor = 0.0
        if mean_vol > 0:
            # ratio > 0 means recent vol is above average
            ratio = vols[-1] / mean_vol - 1.0
            # Volume confirms direction; it never flips the sign
            vol_sign = 1.0 if direction_score >= 0 else -1.0
            vol_factor = math.tanh(ratio) * vol_sign

        momentum = 0.40 * direction_score + 0.40 * velocity_score + 0.20 * vol_factor
        return max(-1.0, min(1.0, momentum))

    def get_ohlcv(self, n: int = 4) -> list[Candle]:
        """Return the last n completed 15m candles."""
        return list(self.candles)[-n:]

    def is_stale(self) -> bool:
        """True if no tick received, or last tick is older than PRICE_STALENESS_SECONDS."""
        if self.last_update is None:
            return True
        age = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return age > settings.PRICE_STALENESS_SECONDS

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch connect() as a background task."""
        self._task = asyncio.create_task(self.connect(), name="coinbase-feed")
        log.info("CoinbaseFeed started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("CoinbaseFeed stopped")

    async def wait_for_price(self, timeout: float = 30.0) -> float:
        """Block until the first tick is received. Returns current price."""
        await asyncio.wait_for(self._price_event.wait(), timeout=timeout)
        return self.current_price

    # ------------------------------------------------------------------
    # Main connect loop (exponential backoff on disconnect)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Long-running reconnect loop. Connects, subscribes, and handles
        messages until stopped. Backs off exponentially on failures.
        If a connection was alive for > 30s it is considered successful
        and backoff resets to 1s.
        """
        self._running = True
        delay   = 1.0
        attempt = 0

        while self._running:
            attempt += 1
            connect_ts = asyncio.get_event_loop().time()

            try:
                log.info(
                    "Connecting to Coinbase WS (attempt %d): %s",
                    attempt, settings.COINBASE_WS_URL,
                )
                await self._connect_once()

            except asyncio.CancelledError:
                raise

            except (ConnectionClosed, WebSocketException) as exc:
                log.warning("WS connection closed (attempt %d): %s", attempt, exc)

            except OSError as exc:
                log.warning("Network error (attempt %d): %s", attempt, exc)

            except Exception as exc:
                log.error("Unexpected feed error (attempt %d): %s", attempt, exc)

            if not self._running:
                break

            # How long did the connection actually live?
            lived = asyncio.get_event_loop().time() - connect_ts
            if lived > 30.0:
                # Treat as a healthy connection that dropped — reset backoff
                log.info(
                    "Feed reconnecting (was alive for %.0fs) — resetting backoff",
                    lived,
                )
                delay   = 1.0
                attempt = 0
                await asyncio.sleep(1.0)
            else:
                log.warning(
                    "Feed reconnecting in %.0fs (attempt %d)...", delay, attempt
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)

    async def _connect_once(self) -> None:
        """Single WebSocket session: connect → subscribe → message loop."""
        async with websockets.connect(
            settings.COINBASE_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
            max_size=2 ** 20,   # 1 MB
        ) as ws:
            await ws.send(json.dumps({
                "type":        "subscribe",
                "product_ids": [settings.BTC_PRODUCT_ID],
                "channel":     "ticker",
            }))
            log.info("Subscribed to %s ticker channel", settings.BTC_PRODUCT_ID)

            async for raw in ws:
                if not self._running:
                    return
                await self._handle_message(raw)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.debug("Unparseable WS message: %r", raw[:120])
            return

        channel = msg.get("channel", "")

        if channel == "ticker":
            for event in msg.get("events", []):
                for tick in event.get("tickers", []):
                    self._process_tick(tick)

        elif channel == "subscriptions":
            log.info("Subscription confirmed: %s", msg)

        elif channel == "heartbeats":
            pass    # routine keepalive — ignore

        elif msg.get("type") == "error":
            log.error(
                "Coinbase WS error — %s: %s",
                msg.get("error", "?"),
                msg.get("message", ""),
            )

        # Silently drop any other message types (e.g., sequence gaps)

    def _process_tick(self, tick: dict) -> None:
        price_raw = tick.get("price") or tick.get("last_trade_price", "")
        vol_raw   = tick.get("volume_24_h") or tick.get("volume_24h", "0")

        try:
            price     = float(price_raw)
            vol_24h   = float(vol_raw)
        except (ValueError, TypeError):
            return

        if price <= 0:
            return

        now    = datetime.now(timezone.utc)
        bucket = _bucket_for(now)

        # Update public state
        self.current_price = price
        self.last_update   = now
        self._volume_24h   = vol_24h

        # ---- Candle management ----
        if not self.current_candle:
            # Very first tick — open the initial candle
            self._current_bucket    = bucket
            self._candle_start_vol  = vol_24h
            self.current_candle     = _empty_candle(price, bucket)

        elif bucket != self._current_bucket:
            # 15-minute boundary crossed — finalise and store completed candle
            incremental_vol = max(vol_24h - self._candle_start_vol, 0.0)
            finished = Candle(
                open      = self.current_candle["open"],
                high      = self.current_candle["high"],
                low       = self.current_candle["low"],
                close     = price,          # price at the boundary is the close
                volume    = incremental_vol,
                timestamp = self._current_bucket,   # type: ignore[arg-type]
            )
            self.candles.append(finished)
            log.debug(
                "Candle closed [%s] O=%.2f H=%.2f L=%.2f C=%.2f V=%.4f",
                self._current_bucket.strftime("%H:%M"),  # type: ignore[union-attr]
                finished.open, finished.high,
                finished.low, finished.close, finished.volume,
            )

            # Open the new candle
            self._current_bucket   = bucket
            self._candle_start_vol = vol_24h
            self.current_candle    = _empty_candle(price, bucket)

        else:
            # Mid-candle tick — update OHLCV
            c = self.current_candle
            if price > c["high"]:
                c["high"] = price
            if price < c["low"]:
                c["low"] = price
            c["close"]  = price
            c["volume"] = max(vol_24h - self._candle_start_vol, 0.0)

        self._price_event.set()

    # ------------------------------------------------------------------
    # Backward-compat aliases (used by ensemble.py and runner.py)
    # ------------------------------------------------------------------

    def get_price(self) -> float:
        """Alias for get_current_price() — does not raise on stale data."""
        return self.current_price

    def get_candles_15m(self, n: int | None = None) -> list[Candle]:
        candles = list(self.candles)
        return candles if n is None else candles[-n:]

    def get_candles_1h(self, n: int | None = None) -> list[Candle]:
        """Synthesise 1h candles by merging groups of four 15m candles."""
        source = list(self.candles)
        hourly: list[Candle] = []
        for i in range(0, len(source) - 3, 4):
            group = source[i : i + 4]
            if len(group) == 4:
                hourly.append(Candle(
                    open      = group[0].open,
                    high      = max(c.high for c in group),
                    low       = min(c.low  for c in group),
                    close     = group[-1].close,
                    volume    = sum(c.volume for c in group),
                    timestamp = group[0].timestamp,
                ))
        return hourly if n is None else hourly[-n:]

"""
coinbase_feed.py — Coinbase WebSocket BTC price feed
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_ID = "BTC-USD"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float    # unix seconds, start of candle
    interval_sec: int


@dataclass
class Tick:
    price: float
    volume_24h: float
    timestamp: float


# ---------------------------------------------------------------------------
# Candle aggregator
# ---------------------------------------------------------------------------

class CandleAggregator:
    """Aggregates raw ticks into fixed-interval OHLCV candles."""

    def __init__(self, interval_sec: int, maxlen: int = 200):
        self.interval_sec = interval_sec
        self._candles: deque[Candle] = deque(maxlen=maxlen)
        self._open: float | None = None
        self._high: float = 0.0
        self._low: float = float("inf")
        self._volume: float = 0.0
        self._bucket: int = 0   # unix timestamp of current bucket start

    def _bucket_for(self, ts: float) -> int:
        return int(ts // self.interval_sec) * self.interval_sec

    def update(self, price: float, volume: float, ts: float) -> Candle | None:
        """Feed a tick. Returns a completed candle when a new interval starts."""
        bucket = self._bucket_for(ts)
        completed: Candle | None = None

        if self._open is None:
            # First tick ever
            self._bucket = bucket
            self._open = price

        elif bucket != self._bucket:
            # New interval — emit completed candle
            completed = Candle(
                open=self._open,
                high=self._high,
                low=self._low,
                close=price,        # close of old = first of new
                volume=self._volume,
                timestamp=self._bucket,
                interval_sec=self.interval_sec,
            )
            self._candles.append(completed)
            log.debug("Candle closed [%ds]: O=%.2f H=%.2f L=%.2f C=%.2f V=%.4f",
                      self.interval_sec, completed.open, completed.high,
                      completed.low, completed.close, completed.volume)

            # Reset for new bucket
            self._bucket = bucket
            self._open = price
            self._high = price
            self._low = price
            self._volume = volume
            return completed

        # Update current bucket
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._volume += volume
        return None

    def get_candles(self, n: int | None = None) -> list[Candle]:
        candles = list(self._candles)
        if n is not None:
            candles = candles[-n:]
        return candles

    @property
    def current_close(self) -> float | None:
        """Best estimate of current price (close of in-progress candle)."""
        return None  # updated by feed directly


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

class CoinbaseFeed:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        reconnect_delay: float = 3.0,
        max_reconnect_delay: float = 60.0,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay

        self._price: float = 0.0
        self._volume_24h: float = 0.0
        self._last_tick_ts: float = 0.0

        self._agg_15m = CandleAggregator(interval_sec=15 * 60)
        self._agg_1h  = CandleAggregator(interval_sec=60 * 60)

        self._price_event = asyncio.Event()
        self._running = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_price(self) -> float:
        return self._price

    def get_volume_24h(self) -> float:
        return self._volume_24h

    def get_candles_15m(self, n: int | None = None) -> list[Candle]:
        return self._agg_15m.get_candles(n)

    def get_candles_1h(self, n: int | None = None) -> list[Candle]:
        return self._agg_1h.get_candles(n)

    async def wait_for_price(self, timeout: float = 30.0) -> float:
        """Block until at least one tick has been received."""
        await asyncio.wait_for(self._price_event.wait(), timeout=timeout)
        return self._price

    def is_stale(self, max_age_sec: float = 10.0) -> bool:
        if self._last_tick_ts == 0.0:
            return True
        return (time.time() - self._last_tick_ts) > max_age_sec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="coinbase-feed")
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

    # ------------------------------------------------------------------
    # WebSocket loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        delay = self._reconnect_delay
        while self._running:
            try:
                await self._connect()
                delay = self._reconnect_delay   # reset on clean run
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("Feed disconnected: %s — reconnecting in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _connect(self) -> None:
        log.info("Connecting to Coinbase WebSocket: %s", WS_URL)
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
            await self._subscribe(ws)
            log.info("Subscribed to %s ticker", PRODUCT_ID)
            async for raw in ws:
                if not self._running:
                    return
                self._handle_message(raw)

    async def _subscribe(self, ws) -> None:
        msg = {
            "type": "subscribe",
            "product_ids": [PRODUCT_ID],
            "channel": "ticker",
        }
        # Coinbase Advanced Trade WS uses JWT auth for private channels;
        # the public ticker channel works without credentials.
        await ws.send(json.dumps(msg))

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type") or msg.get("channel")

        # Coinbase Advanced Trade wraps events in {channel, events:[...]}
        if msg.get("channel") == "ticker":
            for event in msg.get("events", []):
                for tick in event.get("tickers", []):
                    self._process_tick(tick)
        elif msg_type == "ticker":
            # Legacy format fallback
            self._process_tick(msg)

    def _process_tick(self, tick: dict) -> None:
        price_str = tick.get("price") or tick.get("last_trade_price", "0")
        vol_str   = tick.get("volume_24_h") or tick.get("volume_24h", "0")

        try:
            price = float(price_str)
            volume = float(vol_str)
        except (ValueError, TypeError):
            return

        if price <= 0:
            return

        ts = time.time()
        self._price = price
        self._volume_24h = volume
        self._last_tick_ts = ts

        # Feed candle aggregators (volume per tick not available from ticker,
        # use 0 — we track price candles; volume_24h is a snapshot not incremental)
        self._agg_15m.update(price, 0.0, ts)
        self._agg_1h.update(price, 0.0, ts)

        self._price_event.set()

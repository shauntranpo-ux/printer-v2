"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket price feed (multi-asset)

Subscribes to ALL supported assets in a single WebSocket connection.
Maintains per-asset candles, momentum, and staleness state.

Backward-compatible: all existing methods (get_current_price, get_momentum,
is_stale, etc.) still work and default to BTC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from config import settings

log = logging.getLogger(__name__)

CANDLE_SECONDS = 15 * 60   # 900 s per candle

# Assets available on Coinbase Advanced Trade
COINBASE_PRODUCT_MAP: dict[str, str] = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
}

# Assets not on Coinbase — fetched via Binance public REST
BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StaleDataError(Exception):
    """Raised when price data is older than PRICE_STALENESS_SECONDS."""


# ---------------------------------------------------------------------------
# Candle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Per-asset state (extracted so CoinbaseFeed can hold N of them)
# ---------------------------------------------------------------------------

@dataclass
class _AssetState:
    current_price:    float = 0.0
    last_update:      Optional[datetime] = None
    candles:          deque = field(default_factory=lambda: deque(maxlen=25))
    current_candle:   dict  = field(default_factory=dict)
    volume_24h:       float = 0.0
    candle_start_vol: float = 0.0
    current_bucket:   Optional[datetime] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_for(dt: datetime) -> datetime:
    snapped_minute = (dt.minute // 15) * 15
    return dt.replace(minute=snapped_minute, second=0, microsecond=0)


def _empty_candle(price: float, bucket: datetime) -> dict:
    return {"open": price, "high": price, "low": price,
            "close": price, "volume": 0.0, "timestamp": bucket}


def _compute_momentum(state: _AssetState) -> float:
    """Compute momentum score [-1, +1] from last 4 candles."""
    candles = list(state.candles)[-4:]
    n = len(candles)
    if n < 2:
        return 0.0

    all_weights = [0.1, 0.2, 0.3, 0.4]
    weights = all_weights[-n:]
    w_sum   = sum(weights)

    direction_sum = 0.0
    for c, w in zip(candles, weights):
        if c.open > 0:
            pct = (c.close - c.open) / c.open
            direction_sum += math.tanh(pct * 100) * w
    direction_score = direction_sum / w_sum

    first_close = candles[0].close
    last_close  = candles[-1].close
    velocity_score = 0.0
    if first_close > 0:
        velocity_score = math.tanh((last_close - first_close) / first_close * 100)

    vols     = [c.volume for c in candles]
    mean_vol = sum(vols) / n
    vol_factor = 0.0
    if mean_vol > 0:
        ratio    = vols[-1] / mean_vol - 1.0
        vol_sign = 1.0 if direction_score >= 0 else -1.0
        vol_factor = math.tanh(ratio) * vol_sign

    momentum = 0.40 * direction_score + 0.40 * velocity_score + 0.20 * vol_factor
    return max(-1.0, min(1.0, momentum))


def _process_tick_into(state: _AssetState, price: float, vol_24h: float) -> None:
    """Update _AssetState from a single price tick."""
    now    = datetime.now(timezone.utc)
    bucket = _bucket_for(now)

    state.current_price = price
    state.last_update   = now
    state.volume_24h    = vol_24h

    if not state.current_candle:
        state.current_bucket    = bucket
        state.candle_start_vol  = vol_24h
        state.current_candle    = _empty_candle(price, bucket)

    elif bucket != state.current_bucket:
        incremental_vol = max(vol_24h - state.candle_start_vol, 0.0)
        finished = Candle(
            open      = state.current_candle["open"],
            high      = state.current_candle["high"],
            low       = state.current_candle["low"],
            close     = price,
            volume    = incremental_vol,
            timestamp = state.current_bucket,   # type: ignore[arg-type]
        )
        state.candles.append(finished)
        state.current_bucket    = bucket
        state.candle_start_vol  = vol_24h
        state.current_candle    = _empty_candle(price, bucket)

    else:
        c = state.current_candle
        if price > c["high"]: c["high"] = price
        if price < c["low"]:  c["low"]  = price
        c["close"]  = price
        c["volume"] = max(vol_24h - state.candle_start_vol, 0.0)


# ---------------------------------------------------------------------------
# CoinbaseFeed — subscribes to all Coinbase assets in one WS connection
# ---------------------------------------------------------------------------

class CoinbaseFeed:
    """
    Multi-asset price feed via Coinbase Advanced Trade WebSocket.

    All supported Coinbase assets are subscribed in a single connection.
    Assets not on Coinbase (BNB, HYPE) are polled via Binance REST.

    Backward-compatible: un-keyed methods (get_current_price, get_momentum,
    is_stale, get_ohlcv) always refer to BTC.
    Use the *_for(asset) variants for multi-asset access.
    """

    def __init__(self) -> None:
        # Build per-asset state for ALL assets in config
        all_assets: list[str] = settings.supported_assets_list or ["BTC"]

        # Coinbase assets
        self._cb_products: dict[str, str] = {}   # asset → product_id
        # Reverse map: product_id → asset (for routing incoming ticks)
        self._product_to_asset: dict[str, str] = {}
        for asset in all_assets:
            if asset in COINBASE_PRODUCT_MAP:
                pid = COINBASE_PRODUCT_MAP[asset]
                self._cb_products[asset]           = pid
                self._product_to_asset[pid]        = asset

        # Binance REST assets
        self._binance_assets: dict[str, str] = {
            a: s for a, s in BINANCE_SYMBOL_MAP.items() if a in all_assets
        }

        # Per-asset state
        self._state: dict[str, _AssetState] = {
            a: _AssetState() for a in all_assets
        }

        self._primary = "BTC"   # backward-compat default

        # WS machinery
        self._price_event = asyncio.Event()
        self._running:  bool = False
        self._ws_task:  asyncio.Task | None = None
        self._rest_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Multi-asset public interface
    # ------------------------------------------------------------------

    def get_price_for(self, asset: str) -> float:
        state = self._state.get(asset)
        return state.current_price if state else 0.0

    def get_momentum_for(self, asset: str) -> float:
        state = self._state.get(asset)
        return _compute_momentum(state) if state else 0.0

    def is_stale_for(self, asset: str) -> bool:
        state = self._state.get(asset)
        if state is None or state.last_update is None:
            return True
        age = (datetime.now(timezone.utc) - state.last_update).total_seconds()
        return age > settings.PRICE_STALENESS_SECONDS

    def get_ohlcv_for(self, asset: str, n: int = 4) -> list[Candle]:
        state = self._state.get(asset)
        return list(state.candles)[-n:] if state else []

    def assets(self) -> list[str]:
        return list(self._state.keys())

    # ------------------------------------------------------------------
    # Backward-compatible BTC interface
    # ------------------------------------------------------------------

    @property
    def current_price(self) -> float:
        return self._state[self._primary].current_price

    @property
    def last_update(self) -> Optional[datetime]:
        return self._state[self._primary].last_update

    @property
    def candles(self) -> deque:
        return self._state[self._primary].candles

    def get_current_price(self) -> float:
        if self.is_stale():
            state = self._state[self._primary]
            age   = "never"
            if state.last_update:
                secs = (datetime.now(timezone.utc) - state.last_update).total_seconds()
                age = f"{secs:.0f}s ago"
            raise StaleDataError(
                f"BTC price data is stale (last update: {age}). "
                f"Threshold: {settings.PRICE_STALENESS_SECONDS}s"
            )
        return self.current_price

    def get_momentum(self) -> float:
        return self.get_momentum_for(self._primary)

    def is_stale(self) -> bool:
        return self.is_stale_for(self._primary)

    def get_ohlcv(self, n: int = 4) -> list[Candle]:
        return self.get_ohlcv_for(self._primary, n)

    def get_price(self) -> float:
        return self.current_price

    def get_candles_15m(self, n: int | None = None) -> list[Candle]:
        candles = list(self._state[self._primary].candles)
        return candles if n is None else candles[-n:]

    def get_candles_1h(self, n: int | None = None) -> list[Candle]:
        source = list(self._state[self._primary].candles)
        hourly: list[Candle] = []
        for i in range(0, len(source) - 3, 4):
            group = source[i: i + 4]
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._ws_task   = asyncio.create_task(self._ws_loop(),   name="coinbase-feed-ws")
        self._rest_task = asyncio.create_task(self._rest_loop(), name="coinbase-feed-rest")
        log.info(
            "CoinbaseFeed started — Coinbase: %s  REST: %s",
            list(self._cb_products.keys()),
            list(self._binance_assets.keys()),
        )

    async def stop(self) -> None:
        self._running = False
        for task in (self._ws_task, self._rest_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        log.info("CoinbaseFeed stopped")

    async def wait_for_price(self, timeout: float = 30.0) -> float:
        await asyncio.wait_for(self._price_event.wait(), timeout=timeout)
        return self.current_price

    # ------------------------------------------------------------------
    # WebSocket loop (Coinbase assets)
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        self._running = True
        delay   = 1.0
        attempt = 0

        while self._running:
            attempt += 1
            connect_ts = asyncio.get_event_loop().time()
            try:
                await self._connect_once()
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, WebSocketException) as exc:
                log.warning("WS closed (attempt %d): %s", attempt, exc)
            except OSError as exc:
                log.warning("Network error (attempt %d): %s", attempt, exc)
            except Exception as exc:
                log.error("Feed error (attempt %d): %s", attempt, exc)

            if not self._running:
                break

            lived = asyncio.get_event_loop().time() - connect_ts
            if lived > 30.0:
                delay = 1.0; attempt = 0
                await asyncio.sleep(1.0)
            else:
                log.warning("Feed reconnecting in %.0fs...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)

    async def _connect_once(self) -> None:
        if not self._cb_products:
            return   # no Coinbase assets configured

        product_ids = list(self._cb_products.values())
        async with websockets.connect(
            settings.COINBASE_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
            max_size=2 ** 20,
        ) as ws:
            await ws.send(json.dumps({
                "type":        "subscribe",
                "product_ids": product_ids,
                "channel":     "ticker",
            }))
            log.info("Subscribed to %s", product_ids)

            async for raw in ws:
                if not self._running:
                    return
                await self._handle_message(raw)

    async def _handle_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        channel = msg.get("channel", "")
        if channel == "ticker":
            for event in msg.get("events", []):
                for tick in event.get("tickers", []):
                    self._route_tick(tick)
        elif msg.get("type") == "error":
            log.error("Coinbase WS error: %s %s",
                      msg.get("error"), msg.get("message"))

    def _route_tick(self, tick: dict) -> None:
        product_id = tick.get("product_id", "")
        asset      = self._product_to_asset.get(product_id)
        if not asset:
            return

        price_raw = tick.get("price") or tick.get("last_trade_price", "")
        vol_raw   = tick.get("volume_24_h") or tick.get("volume_24h", "0")
        try:
            price   = float(price_raw)
            vol_24h = float(vol_raw)
        except (ValueError, TypeError):
            return
        if price <= 0:
            return

        _process_tick_into(self._state[asset], price, vol_24h)
        log.debug("Tick [%s] $%.4f", asset, price)
        self._price_event.set()

    # ------------------------------------------------------------------
    # REST loop (Binance public API for assets not on Coinbase)
    # ------------------------------------------------------------------

    async def _rest_loop(self) -> None:
        """Poll Binance public price endpoint every 10s for BNB/HYPE."""
        if not self._binance_assets:
            return

        import aiohttp   # optional import — only needed if REST assets configured

        self._running = True
        while self._running:
            for asset, symbol in self._binance_assets.items():
                try:
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            data  = await resp.json()
                            price = float(data.get("price", 0))
                            if price > 0:
                                _process_tick_into(self._state[asset], price, 0.0)
                                log.debug("REST tick [%s] $%.4f", asset, price)
                                self._price_event.set()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("Binance REST fetch failed for %s: %s", asset, exc)

            await asyncio.sleep(10)

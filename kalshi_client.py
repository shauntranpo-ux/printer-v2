"""
kalshi_client.py — Kalshi REST API v2 client (httpx async, RSA auth)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from urllib.parse import urlparse
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from config import settings

log = logging.getLogger(__name__)

_RATE_LIMIT_RPS = 10      # max requests per second
_TIMEOUT        = 10.0    # seconds per request


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class KalshiError(Exception):
    """Base class for all Kalshi API errors."""
    def __init__(self, message: str, status_code: int = 0, raw: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw or {}


class KalshiAuthError(KalshiError):
    """401 / 403 — bad key, expired token, or signature mismatch."""


class KalshiRateLimitError(KalshiError):
    """429 — too many requests. Check retry_after attribute."""
    def __init__(self, message: str, retry_after: float = 1.0):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class KalshiMarketClosedError(KalshiError):
    """Market is no longer accepting orders (halted, closed, or settled)."""


class KalshiInsufficientFundsError(KalshiError):
    """Insufficient cash balance to place the requested order."""


# ---------------------------------------------------------------------------
# Rate limiter — token bucket, 10 req/s
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple fixed-rate limiter: at most `rate` calls per second."""

    def __init__(self, rate: float = _RATE_LIMIT_RPS):
        self._interval = 1.0 / rate
        self._last_ts  = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now  = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _load_key() -> RSAPrivateKey:
    private_key_str = settings.KALSHI_PRIVATE_KEY.strip()
    if not private_key_str:
        raise KalshiAuthError("KALSHI_PRIVATE_KEY is not set")
    # Railway (and many env-var stores) serialize newlines as the two-character
    # sequence backslash-n.  Expand them so PEM parsing succeeds.
    private_key_str = private_key_str.replace("\\n", "\n")
    key = serialization.load_pem_private_key(
        private_key_str.encode(), password=None
    )
    if not isinstance(key, RSAPrivateKey):
        raise KalshiAuthError("KALSHI_PRIVATE_KEY is not an RSA private key")
    return key


def _rsa_sign(key: RSAPrivateKey, message: str) -> str:
    """RSA-PSS-SHA256 signature as required by the Kalshi v2 API."""
    raw = key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# Error response parser
# ---------------------------------------------------------------------------

def _raise_for_response(resp: httpx.Response) -> None:
    """Map HTTP error responses to typed KalshiError subclasses."""
    if resp.is_success:
        return

    # Try to parse the Kalshi error envelope.
    # New API wraps errors: {"error": {"code": ..., "message": ..., "details": ...}}
    # Old API used flat:    {"code": ..., "message": ...}
    try:
        body: dict = resp.json()
    except Exception:
        body = {}

    err_obj = body.get("error", body)   # unwrap nested envelope if present
    code    = err_obj.get("code", "") or err_obj.get("details", "")
    message = err_obj.get("message", "") or resp.text[:200]
    status  = resp.status_code

    if status in (401, 403):
        raise KalshiAuthError(
            f"Auth failed ({status}): {message}", status_code=status, raw=body
        )

    if status == 429:
        retry_after = float(resp.headers.get("Retry-After", "1"))
        raise KalshiRateLimitError(
            f"Rate limited — retry after {retry_after}s", retry_after=retry_after
        )

    if status in (400, 403):
        lc = (code + message).lower()
        if "insufficient" in lc or "funds" in lc or "balance" in lc:
            raise KalshiInsufficientFundsError(
                f"Insufficient funds: {message}", status_code=status, raw=body
            )
        if "closed" in lc or "halted" in lc or "settled" in lc or "inactive" in lc:
            raise KalshiMarketClosedError(
                f"Market not active: {message}", status_code=status, raw=body
            )

    raise KalshiError(f"Kalshi API error {status}: {message}", status_code=status, raw=body)


# ---------------------------------------------------------------------------
# KalshiClient
# ---------------------------------------------------------------------------

class KalshiClient:
    def __init__(self) -> None:
        self._api_key   = settings.KALSHI_API_KEY
        self._key       = _load_key()
        self._base_url  = settings.KALSHI_BASE_URL.rstrip("/")
        self._limiter   = _RateLimiter(_RATE_LIMIT_RPS)
        self._http      = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def auth_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Generate RSA-PSS-SHA256 signed auth headers for a Kalshi v2 request.

        Signature covers:  timestamp_ms + METHOD_UPPER + full_path_without_query
        Full path includes the base URL path, e.g. /trade-api/v2/portfolio/balance
        """
        ts  = str(int(time.time() * 1000))
        bare_path  = path.split("?")[0]
        # Sign with the full URL path (base path prefix + endpoint)
        base_path  = urlparse(self._base_url).path.rstrip("/")
        full_path  = base_path + bare_path
        msg = ts + method.upper() + full_path
        sig = _rsa_sign(self._key, msg)
        return {
            "KALSHI-ACCESS-KEY":       self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    # ------------------------------------------------------------------
    # HTTP primitives
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> Any:
        await self._limiter.acquire()
        headers = await self.auth_headers("GET", path)
        resp = await self._http.get(path, headers=headers, params=params)
        _raise_for_response(resp)
        return resp.json()

    async def _post(self, path: str, body: dict) -> Any:
        await self._limiter.acquire()
        headers = await self.auth_headers("POST", path)
        resp = await self._http.post(path, headers=headers, json=body)
        _raise_for_response(resp)
        return resp.json()

    async def _delete(self, path: str) -> Any:
        await self._limiter.acquire()
        headers = await self.auth_headers("DELETE", path)
        resp = await self._http.delete(path, headers=headers)
        _raise_for_response(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def get_btc_15m_markets(self) -> list[dict]:
        """
        Return open BTC markets expiring within the next 20 minutes.

        Tries KXBTC series first; falls back to a BTC keyword search if
        the primary series returns nothing.

        Each element contains:
          ticker, yes_ask, yes_bid, no_ask, no_bid,
          volume, open_interest, close_time, strike_price
        """
        now        = int(time.time())
        window_end = now + 30 * 60     # 30 minutes out

        def _fetch_params(series: str) -> dict:
            return {
                "series_ticker": series,
                "status":        "open",
                "min_close_ts":  now,
                "max_close_ts":  window_end,
                "limit":         20,
            }

        # Try primary series ticker (15-minute BTC markets)
        data = await self._get("/markets", _fetch_params("KXBTC15M"))
        markets: list[dict] = data.get("markets", [])

        # Fallback: try KXBTC series
        if not markets:
            log.debug("No KXBTC15M markets — trying KXBTC fallback")
            data = await self._get("/markets", _fetch_params("KXBTC"))
            markets = data.get("markets", [])

        if not markets:
            log.warning("No open BTC 15m markets found")
            return []

        # Sort by soonest close first
        markets.sort(key=lambda m: m.get("close_time", ""))

        result = []
        for m in markets:
            result.append({
                "ticker":        m.get("ticker", ""),
                "yes_ask":       m.get("yes_ask", 0),
                "yes_bid":       m.get("yes_bid", 0),
                "no_ask":        m.get("no_ask", 0),
                "no_bid":        m.get("no_bid", 0),
                "volume":        m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
                "close_time":    m.get("close_time", ""),
                "strike_price":  m.get("floor_strike") or m.get("cap_strike") or 0,
            })

        log.debug("Found %d open BTC markets", len(result))
        return result

    async def get_market(self, ticker: str) -> dict:
        """Return the full market object for a given ticker."""
        data = await self._get(f"/markets/{ticker}")
        return data.get("market", data)

    async def get_order_book(self, ticker: str, depth: int = 10) -> dict:
        """
        Return the order book for a market.

        Returns:
          {
            "yes_bids": [{"price": int, "size": int}, ...],   # sorted best→worst
            "yes_asks": [{"price": int, "size": int}, ...],
            "no_bids":  [...],
            "no_asks":  [...],
          }
        """
        data = await self._get(
            f"/markets/{ticker}/orderbook", {"depth": depth}
        )
        book = data.get("orderbook", {})

        def _parse(raw: list | None) -> list[dict]:
            out = []
            for entry in (raw or []):
                # API returns either [price, size] lists or {"price":…,"size":…} dicts
                if isinstance(entry, list) and len(entry) >= 2:
                    out.append({"price": int(entry[0]), "size": int(entry[1])})
                elif isinstance(entry, dict):
                    out.append({"price": int(entry.get("price", 0)),
                                "size":  int(entry.get("size", 0))})
            return out

        return {
            "yes_bids": _parse(book.get("yes")),
            "yes_asks": _parse(book.get("yes_asks")),
            "no_bids":  _parse(book.get("no")),
            "no_asks":  _parse(book.get("no_asks")),
        }

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Return available cash balance in dollars."""
        data = await self._get("/portfolio/balance")
        # Kalshi balance is in cents
        cents = data.get("balance", 0)
        return round(cents / 100, 2)

    async def get_open_positions(self) -> list[dict]:
        """
        Return open BTC positions only.

        Each element:
          ticker, side ("YES"/"NO"), quantity, avg_entry_price (cents)
        """
        data = await self._get("/portfolio/positions")
        out  = []
        for p in data.get("market_positions", []):
            ticker = p.get("ticker", "")
            net    = p.get("position", 0)     # positive = YES, negative = NO
            if net == 0:
                continue
            # Filter to BTC markets only
            ticker_up = ticker.upper()
            if "BTC" not in ticker_up and "KXBTC" not in ticker_up:
                continue
            out.append({
                "ticker":          ticker,
                "side":            "YES" if net > 0 else "NO",
                "quantity":        abs(net),
                "avg_entry_price": p.get("market_exposure", 0) // max(abs(net), 1),
            })
        return out

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        ticker: str,
        side:   str,    # "yes" | "no"
        price:  int,    # cents (1–99)
        count:  int,    # number of contracts
        action: str = "buy",   # "buy" | "sell"
    ) -> dict:
        """
        Place a limit order.

        Returns:
          {"order_id": str, "status": str, "filled_price": int | None}
        """
        side_lc   = side.lower()
        action_lc = action.lower()
        yes_price = price if side_lc == "yes" else (100 - price)
        no_price  = price if side_lc == "no"  else (100 - price)

        body: dict[str, Any] = {
            "ticker":    ticker,
            "action":    action_lc,
            "side":      side_lc,
            "type":      "limit",
            "count":     count,
            "yes_price": yes_price,
            "no_price":  no_price,
        }

        try:
            data = await self._post("/portfolio/orders", body)
        except KalshiError:
            raise
        except httpx.HTTPStatusError as exc:
            _raise_for_response(exc.response)
            raise

        order = data.get("order", data)
        log.info(
            "Order placed: %s  %s %s %s x%d @ %dc  status=%s",
            order.get("order_id", "?"), action_lc.upper(), side_lc.upper(), ticker,
            count, price, order.get("status", "?"),
        )
        return {
            "order_id":    order.get("order_id", ""),
            "status":      order.get("status", "unknown"),
            "filled_price": order.get("yes_price") if order.get("quantity_filled") else None,
        }

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order. Returns True on success.
        Returns False (without raising) if already filled or cancelled.
        """
        try:
            await self._delete(f"/portfolio/orders/{order_id}")
            log.info("Order cancelled: %s", order_id)
            return True
        except KalshiError as exc:
            if exc.status_code == 404:
                log.warning("cancel_order: %s not found (already settled?)", order_id)
                return False
            # Already filled / market closed are acceptable failures
            msg = str(exc).lower()
            if "filled" in msg or "closed" in msg or "cancelled" in msg:
                log.warning("cancel_order: %s — %s", order_id, exc)
                return False
            raise

    async def get_order(self, order_id: str) -> dict:
        """Fetch the current state of a single order."""
        data  = await self._get(f"/portfolio/orders/{order_id}")
        order = data.get("order", data)
        return {
            "order_id":         order.get("order_id", ""),
            "ticker":           order.get("ticker", ""),
            "side":             order.get("side", ""),
            "status":           order.get("status", "unknown"),
            "count":            order.get("count", 0),
            "quantity_filled":  order.get("quantity_filled", 0),
            "yes_price":        order.get("yes_price", 0),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._http.aclose()

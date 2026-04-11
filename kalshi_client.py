"""
kalshi_client.py — Kalshi REST API v2 client (httpx async, RSA auth)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
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

    # Kalshi series ticker prefixes per asset  (tries 15M variant first, then base)
    _SERIES_MAP: dict[str, list[str]] = {
        "BTC":  ["KXBTC15M", "KXBTC"],
        "ETH":  ["KXETH15M", "KXETH"],
        "SOL":  ["KXSOL15M", "KXSOL"],
        "XRP":  ["KXXRP15M", "KXXRP"],
        "DOGE": ["KXDOGE15M", "KXDOGE"],
        "HYPE": ["KXHYPE15M", "KXHYPE"],
        "BNB":  ["KXBNB15M",  "KXBNB"],
    }

    async def get_crypto_15m_markets(self, asset: str = "BTC") -> list[dict]:
        """
        Return open 15-minute markets for any supported crypto asset.
        Mirrors get_btc_15m_markets() but is asset-agnostic.
        """
        asset    = asset.upper()
        series_list = self._SERIES_MAP.get(asset, [f"KX{asset}15M", f"KX{asset}"])

        now        = int(time.time())
        window_end = now + 30 * 60

        def _params(series: str) -> dict:
            return {
                "series_ticker": series,
                "status":        "open",
                "min_close_ts":  now,
                "max_close_ts":  window_end,
                "limit":         20,
            }

        markets: list[dict] = []
        for series in series_list:
            data    = await self._get("/markets", _params(series))
            markets = data.get("markets", [])
            if markets:
                break

        if not markets:
            log.debug("No open %s 15m markets found", asset)
            return []

        markets.sort(key=lambda m: m.get("close_time", ""))
        result = []
        for m in markets:
            yes_ask = m.get("yes_ask") or 0
            no_ask  = m.get("no_ask")  or 0
            yes_bid = m.get("yes_bid") or 0
            no_bid  = m.get("no_bid")  or 0
            # Derive missing ask from opposite bid if the API returned 0
            if yes_ask == 0 and no_bid > 0:
                yes_ask = 100 - no_bid
            if no_ask == 0 and yes_bid > 0:
                no_ask = 100 - yes_bid
            # Last resort: use last_price as a mid-market estimate
            last = m.get("last_price") or 0
            if yes_ask == 0 and last > 0:
                yes_ask = last
            if no_ask == 0 and yes_ask > 0:
                no_ask = 100 - yes_ask
            volume = m.get("volume") or m.get("volume_24h") or 0
            result.append({
                "ticker":        m.get("ticker", ""),
                "title":         m.get("title", ""),
                "yes_ask":       yes_ask,
                "yes_bid":       yes_bid,
                "no_ask":        no_ask,
                "no_bid":        no_bid,
                "volume":        volume,
                "open_interest": m.get("open_interest", 0),
                "close_time":    m.get("close_time", ""),
                "strike_price":  m.get("floor_strike") or m.get("cap_strike") or 0,
                "asset":         asset,
            })

        log.debug("Found %d open %s markets", len(result), asset)
        return result

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
                try:
                    if isinstance(entry, list):
                        if len(entry) >= 2:
                            out.append({"price": int(entry[0]), "size": int(entry[1])})
                    elif isinstance(entry, dict):
                        out.append({"price": int(entry.get("price", 0)),
                                    "size":  int(entry.get("size", 0))})
                except (ValueError, TypeError):
                    continue
            return out

        yes_bids = _parse(book.get("yes") or book.get("yes_bids") or [])
        no_bids  = _parse(book.get("no")  or book.get("no_bids")  or [])

        # Kalshi order books only expose bids for each side.
        # Asks are derived from the opposite side's best bid:
        #   YES ask = 100 - best NO bid  (cheapest YES you can buy)
        #   NO  ask = 100 - best YES bid (cheapest NO  you can buy)
        yes_asks = [{"price": 100 - e["price"], "size": e["size"]} for e in no_bids]
        no_asks  = [{"price": 100 - e["price"], "size": e["size"]} for e in yes_bids]

        return {
            "yes_bids": yes_bids,
            "yes_asks": yes_asks,
            "no_bids":  no_bids,
            "no_asks":  no_asks,
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
        ticker:     str,
        side:       str,               # "yes" | "no"
        count:      int,               # number of contracts
        action:     str = "buy",       # "buy" | "sell"
        price:      int | None = None, # cents (1–99) — required for limit, omit for market
        order_type: str = "limit",     # "limit" | "market"
    ) -> dict:
        """
        Place a limit or market order.

        Market orders: omit price, pass order_type="market". Fills immediately
        at best available price; returns status="resting" if no counter-party.
        Limit  orders: provide price in cents (1–99).

        Returns:
          {"order_id": str, "status": str, "filled_price": int | None}
        """
        side_lc   = side.lower()
        action_lc = action.lower()

        body: dict[str, Any] = {
            "ticker":          ticker,
            "client_order_id": str(uuid.uuid4()),   # required by Kalshi v2
            "action":          action_lc,
            "side":            side_lc,
            "type":            order_type,
            "count":           count,
        }

        if order_type == "limit":
            if price is None:
                raise ValueError("price is required for limit orders")
            yes_price = price if side_lc == "yes" else (100 - price)
            no_price  = price if side_lc == "no"  else (100 - price)
            body["yes_price"] = yes_price
            body["no_price"]  = no_price
        elif order_type == "market":
            # Kalshi market orders still need a worst-acceptable yes_price.
            # Buy:  sweep aggressively (pay up to 99¢ YES / 99¢ NO side)
            # Sell: accept any price (down to 1¢)
            if action_lc == "buy":
                body["yes_price"] = 99 if side_lc == "yes" else 1
            else:
                body["yes_price"] = 1 if side_lc == "yes" else 99

        try:
            data = await self._post("/portfolio/orders", body)
        except KalshiError:
            raise
        except httpx.HTTPStatusError as exc:
            _raise_for_response(exc.response)
            raise

        order     = data.get("order", data)
        status    = order.get("status", "?")
        price_str = f" @ {price}¢" if price is not None else " (market)"
        log.info(
            "Order placed: %s  %s %s %s x%d%s  status=%s",
            order.get("order_id", "?"), action_lc.upper(), side_lc.upper(),
            ticker, count, price_str, status,
        )

        qty_filled = order.get("quantity_filled", 0)
        fp = None
        if qty_filled:
            # Actual fill price lives in the fills array (the sweep yes_price we sent
            # is NOT the execution price for market orders).
            fills = order.get("fills", [])
            if fills:
                fp = (fills[0].get("yes_price") if side_lc == "yes"
                      else fills[0].get("no_price"))
            # Fallback: use order price field if fills absent (limit orders)
            if fp is None:
                fp = order.get("yes_price") if side_lc == "yes" else order.get("no_price")

        return {
            "order_id":     order.get("order_id", ""),
            "status":       status,
            "filled_price": fp,
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

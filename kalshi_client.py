"""
kalshi_client.py — Kalshi REST API wrapper
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OrderBook:
    ticker: str
    yes_bids: list[tuple[int, int]]   # (price_cents, quantity)
    yes_asks: list[tuple[int, int]]
    no_bids: list[tuple[int, int]]
    no_asks: list[tuple[int, int]]

    def best_yes_ask(self) -> int | None:
        return self.yes_asks[0][0] if self.yes_asks else None

    def best_no_ask(self) -> int | None:
        return self.no_asks[0][0] if self.no_asks else None


@dataclass
class Position:
    ticker: str
    market_title: str
    yes_contracts: int
    no_contracts: int


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str           # "yes" | "no"
    contracts: int
    price: int          # cents
    status: str         # "resting" | "filled" | "canceled" | "partially_filled"
    filled: int
    remaining: int


@dataclass
class Balance:
    available_balance: int   # cents
    portfolio_value: int


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _load_private_key(path: Path) -> RSAPrivateKey:
    pem = path.read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, RSAPrivateKey):
        raise ValueError("Key at %s is not an RSA private key" % path)
    return key


def _sign(key: RSAPrivateKey, message: str) -> str:
    signature = key.sign(
        message.encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KalshiClient:
    def __init__(self, api_key_id: str, private_key_path: Path, base_url: str):
        self._api_key_id = api_key_id
        self._key = _load_private_key(private_key_path)
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=10.0,
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = ts + method.upper() + path
        sig = _sign(self._key, msg)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    async def _get(self, path: str, params: dict | None = None) -> Any:
        headers = self._auth_headers("GET", path)
        resp = await self._http.get(path, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: dict) -> Any:
        headers = self._auth_headers("POST", path)
        resp = await self._http.post(path, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()

    async def _delete(self, path: str) -> Any:
        headers = self._auth_headers("DELETE", path)
        resp = await self._http.delete(path, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        data = await self._get("/portfolio/balance")
        return Balance(
            available_balance=data["balance"],
            portfolio_value=data.get("portfolio_value", 0),
        )

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def get_market(self, ticker: str) -> dict:
        data = await self._get(f"/markets/{ticker}")
        return data["market"]

    async def get_orderbook(self, ticker: str, depth: int = 5) -> OrderBook:
        data = await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        book = data["orderbook"]

        def _parse(side: list) -> list[tuple[int, int]]:
            return [(entry[0], entry[1]) for entry in (side or [])]

        return OrderBook(
            ticker=ticker,
            yes_bids=_parse(book.get("yes", [])),
            yes_asks=_parse(book.get("yes_asks", [])),
            no_bids=_parse(book.get("no", [])),
            no_asks=_parse(book.get("no_asks", [])),
        )

    async def find_btc_market(self, series_ticker: str = "KXBTC") -> dict | None:
        """Return the nearest-expiry open BTC market."""
        data = await self._get(
            "/markets",
            {"series_ticker": series_ticker, "status": "open", "limit": 10},
        )
        markets = data.get("markets", [])
        if not markets:
            return None
        # Sort by close_time ascending to get nearest expiry
        markets.sort(key=lambda m: m.get("close_time", ""))
        return markets[0]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        ticker: str,
        side: str,          # "yes" | "no"
        contracts: int,
        price: int,         # cents (1-99)
        order_type: str = "limit",
        client_order_id: str | None = None,
    ) -> Order:
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "count": contracts,
            "type": order_type,
        }
        if order_type == "limit":
            body["yes_price"] = price if side == "yes" else (100 - price)
            body["no_price"] = price if side == "no" else (100 - price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        data = await self._post("/portfolio/orders", body)
        return _parse_order(data["order"])

    async def cancel_order(self, order_id: str) -> Order:
        data = await self._delete(f"/portfolio/orders/{order_id}")
        return _parse_order(data["order"])

    async def get_order(self, order_id: str) -> Order:
        data = await self._get(f"/portfolio/orders/{order_id}")
        return _parse_order(data["order"])

    async def get_open_orders(self, ticker: str | None = None) -> list[Order]:
        params: dict[str, Any] = {"status": "resting"}
        if ticker:
            params["ticker"] = ticker
        data = await self._get("/portfolio/orders", params)
        return [_parse_order(o) for o in data.get("orders", [])]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        data = await self._get("/portfolio/positions")
        out = []
        for p in data.get("market_positions", []):
            if p.get("position", 0) != 0:
                out.append(Position(
                    ticker=p["ticker"],
                    market_title=p.get("market_title", ""),
                    yes_contracts=max(p["position"], 0),
                    no_contracts=max(-p["position"], 0),
                ))
        return out

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_order(o: dict) -> Order:
    return Order(
        order_id=o["order_id"],
        ticker=o["ticker"],
        side=o["side"],
        contracts=o.get("count", 0),
        price=o.get("yes_price", 0),
        status=o.get("status", "unknown"),
        filled=o.get("quantity_filled", 0),
        remaining=o.get("quantity_remaining", 0),
    )

"""
database.py — SQLite trade logging
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kalshi_order_id TEXT    UNIQUE,
    market_ticker   TEXT    NOT NULL,
    direction       TEXT    NOT NULL CHECK(direction IN ('yes', 'no')),
    contracts       INTEGER NOT NULL,
    entry_price     REAL    NOT NULL,       -- cents per contract
    close_price     REAL,
    dollar_size     REAL    NOT NULL,
    pnl             REAL,
    status          TEXT    NOT NULL DEFAULT 'open'
                            CHECK(status IN ('open', 'closed', 'cancelled', 'error')),
    signal_id       INTEGER REFERENCES signals(id),
    opened_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    direction       TEXT    NOT NULL CHECK(direction IN ('yes', 'no', 'flat')),
    confidence      REAL    NOT NULL,
    weight_trend    REAL    NOT NULL,
    weight_mean_rev REAL    NOT NULL,
    weight_momentum REAL    NOT NULL,
    weight_vol      REAL    NOT NULL,
    btc_price       REAL    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS gate_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gate_num    INTEGER NOT NULL,
    gate_name   TEXT    NOT NULL,
    passed      INTEGER NOT NULL CHECK(passed IN (0, 1)),
    reason      TEXT,
    signal_id   INTEGER REFERENCES signals(id),
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS pnl_daily (
    day             TEXT PRIMARY KEY,   -- ISO date YYYY-MM-DD
    trades_count    INTEGER NOT NULL DEFAULT 0,
    winning_trades  INTEGER NOT NULL DEFAULT 0,
    gross_pnl       REAL    NOT NULL DEFAULT 0.0,
    fees            REAL    NOT NULL DEFAULT 0.0,
    net_pnl         REAL    NOT NULL DEFAULT 0.0,
    ending_balance  REAL,
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_signals_created  ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_gate_signal      ON gate_events(signal_id);
"""


@dataclass
class TradeRow:
    id: int
    kalshi_order_id: str | None
    market_ticker: str
    direction: str
    contracts: int
    entry_price: float
    close_price: float | None
    dollar_size: float
    pnl: float | None
    status: str
    signal_id: int | None
    opened_at: str
    closed_at: str | None


@dataclass
class SignalRow:
    id: int
    direction: str
    confidence: float
    weight_trend: float
    weight_mean_rev: float
    weight_momentum: float
    weight_vol: float
    btc_price: float
    created_at: str


@dataclass
class DailyPnL:
    day: str
    trades_count: int
    winning_trades: int
    gross_pnl: float
    fees: float
    net_pnl: float
    ending_balance: float | None


class Database:
    def __init__(self, path: Path):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        log.info("Database ready at %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._db is None:
            raise RuntimeError("Database.connect() has not been called")
        yield self._db

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def log_signal(
        self,
        direction: str,
        confidence: float,
        weights: dict[str, float],
        btc_price: float,
    ) -> int:
        async with self._conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO signals
                    (direction, confidence, weight_trend, weight_mean_rev,
                     weight_momentum, weight_vol, btc_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    direction,
                    confidence,
                    weights.get("trend", 0.0),
                    weights.get("mean_rev", 0.0),
                    weights.get("momentum", 0.0),
                    weights.get("vol", 0.0),
                    btc_price,
                ),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def log_trade(
        self,
        kalshi_order_id: str | None,
        market_ticker: str,
        direction: str,
        contracts: int,
        entry_price: float,
        dollar_size: float,
        signal_id: int | None = None,
    ) -> int:
        async with self._conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO trades
                    (kalshi_order_id, market_ticker, direction, contracts,
                     entry_price, dollar_size, signal_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (kalshi_order_id, market_ticker, direction, contracts,
                 entry_price, dollar_size, signal_id),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def update_trade(
        self,
        trade_id: int,
        *,
        close_price: float,
        pnl: float,
        status: str,
    ) -> None:
        async with self._conn() as db:
            await db.execute(
                """
                UPDATE trades
                SET close_price = ?,
                    pnl         = ?,
                    status      = ?,
                    closed_at   = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                (close_price, pnl, status, trade_id),
            )
            await db.commit()

    async def get_open_trades(self) -> list[TradeRow]:
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at"
            )
            rows = await cursor.fetchall()
            return [TradeRow(**dict(r)) for r in rows]

    async def get_recent_trades(self, n: int = 20) -> list[TradeRow]:
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (n,)
            )
            rows = await cursor.fetchall()
            return [TradeRow(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Gate events
    # ------------------------------------------------------------------

    async def log_gate_event(
        self,
        gate_num: int,
        gate_name: str,
        passed: bool,
        reason: str = "",
        signal_id: int | None = None,
    ) -> None:
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO gate_events (gate_num, gate_name, passed, reason, signal_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (gate_num, gate_name, int(passed), reason, signal_id),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    async def upsert_daily_pnl(
        self,
        day: date | str | None = None,
        *,
        ending_balance: float | None = None,
    ) -> DailyPnL:
        """Recompute today's P&L from the trades table and upsert into pnl_daily."""
        day_str = str(day) if day else date.today().isoformat()
        async with self._conn() as db:
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*)                                AS trades_count,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS winning_trades,
                    COALESCE(SUM(pnl), 0.0)                AS gross_pnl
                FROM trades
                WHERE DATE(opened_at) = ? AND status = 'closed'
                """,
                (day_str,),
            )
            row = await cursor.fetchone()
            trades_count = row["trades_count"] or 0
            winning_trades = row["winning_trades"] or 0
            gross_pnl = row["gross_pnl"] or 0.0
            net_pnl = gross_pnl  # TODO: subtract fees when fee tracking is added

            await db.execute(
                """
                INSERT INTO pnl_daily
                    (day, trades_count, winning_trades, gross_pnl, fees, net_pnl,
                     ending_balance, updated_at)
                VALUES (?, ?, ?, ?, 0.0, ?, ?,
                        strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                ON CONFLICT(day) DO UPDATE SET
                    trades_count   = excluded.trades_count,
                    winning_trades = excluded.winning_trades,
                    gross_pnl      = excluded.gross_pnl,
                    net_pnl        = excluded.net_pnl,
                    ending_balance = COALESCE(excluded.ending_balance, pnl_daily.ending_balance),
                    updated_at     = excluded.updated_at
                """,
                (day_str, trades_count, winning_trades, gross_pnl, net_pnl, ending_balance),
            )
            await db.commit()

        return DailyPnL(
            day=day_str,
            trades_count=trades_count,
            winning_trades=winning_trades,
            gross_pnl=gross_pnl,
            fees=0.0,
            net_pnl=net_pnl,
            ending_balance=ending_balance,
        )

    async def get_daily_pnl(self, n_days: int = 7) -> list[DailyPnL]:
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM pnl_daily ORDER BY day DESC LIMIT ?", (n_days,)
            )
            rows = await cursor.fetchall()
            return [DailyPnL(**dict(r)) for r in rows]

    async def get_today_pnl(self) -> DailyPnL:
        rows = await self.get_daily_pnl(n_days=1)
        if rows and rows[0].day == date.today().isoformat():
            return rows[0]
        return DailyPnL(
            day=date.today().isoformat(),
            trades_count=0,
            winning_trades=0,
            gross_pnl=0.0,
            fees=0.0,
            net_pnl=0.0,
            ending_balance=None,
        )

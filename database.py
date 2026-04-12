"""
database.py — SQLite trade logging (aiosqlite async)
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    market_ticker           TEXT    NOT NULL,
    direction               TEXT    NOT NULL CHECK(direction IN ('YES', 'NO')),
    entry_price             REAL    NOT NULL,
    size_dollars            REAL    NOT NULL,
    contracts               INTEGER NOT NULL,
    kelly_fraction          REAL,
    edge                    REAL,
    ensemble_confidence     REAL,
    model_spread            REAL,
    btc_price_at_entry      REAL,
    btc_momentum            REAL,
    status                  TEXT    NOT NULL DEFAULT 'open'
                                    CHECK(status IN ('open', 'closed', 'expired')),
    exit_price              REAL,
    exit_reason             TEXT,
    pnl_dollars             REAL,
    closed_at               TEXT,
    peak_pnl_pct            REAL,
    claude_prob             REAL,
    gpt_prob                REAL,
    gemini_prob             REAL,
    deepseek_prob           REAL
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date                TEXT PRIMARY KEY,
    total_trades        INTEGER NOT NULL DEFAULT 0,
    winning_trades      INTEGER NOT NULL DEFAULT 0,
    total_pnl           REAL    NOT NULL DEFAULT 0.0,
    total_wagered       REAL    NOT NULL DEFAULT 0.0,
    daily_loss_used     REAL    NOT NULL DEFAULT 0.0,
    sharpe_ratio        REAL,
    win_rate            REAL,
    max_drawdown        REAL
);

CREATE TABLE IF NOT EXISTS ensemble_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    market_ticker   TEXT    NOT NULL,
    claude_prob     REAL,
    gpt_prob        REAL,
    gemini_prob     REAL,
    deepseek_prob   REAL,
    consensus_prob  REAL,
    model_spread    REAL,
    confidence      REAL,
    action          TEXT    CHECK(action IN ('TRADE', 'SKIP', 'WAIT')),
    skip_reason     TEXT
);

CREATE TABLE IF NOT EXISTS bot_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    event_type  TEXT    CHECK(event_type IN
                        ('startup', 'shutdown', 'error', 'crash', 'restart')),
    message     TEXT
);

CREATE TABLE IF NOT EXISTS bot_kv (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    updated TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_ensemble_ts      ON ensemble_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type      ON bot_events(event_type);
"""


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeRow:
    id: int
    timestamp: str
    market_ticker: str
    direction: str
    entry_price: float
    size_dollars: float
    contracts: int
    kelly_fraction: float | None
    edge: float | None
    ensemble_confidence: float | None
    model_spread: float | None
    btc_price_at_entry: float | None
    btc_momentum: float | None
    status: str
    exit_price: float | None
    exit_reason: str | None
    pnl_dollars: float | None
    closed_at: str | None
    peak_pnl_pct: float | None = None
    claude_prob: float | None = None
    gpt_prob: float | None = None
    gemini_prob: float | None = None
    deepseek_prob: float | None = None
    asset_symbol: str | None = None
    tp_order_id: str | None = None
    sl_order_id: str | None = None


@dataclass
class DailyStats:
    date: str
    total_trades: int
    winning_trades: int
    total_pnl: float
    total_wagered: float
    daily_loss_used: float
    sharpe_ratio: float | None
    win_rate: float | None
    max_drawdown: float | None


@dataclass
class EnsembleLogRow:
    id: int
    timestamp: str
    market_ticker: str
    claude_prob: float | None
    gpt_prob: float | None
    gemini_prob: float | None
    deepseek_prob: float | None
    consensus_prob: float | None
    model_spread: float | None
    confidence: float | None
    action: str | None
    skip_reason: str | None


@dataclass
class BotEventRow:
    id: int
    timestamp: str
    event_type: str
    message: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return date.today().isoformat()


def _sharpe(pnls: list[float]) -> float | None:
    """Trade-level Sharpe: mean / std of individual trade P&Ls."""
    if len(pnls) < 2:
        return None
    n = len(pnls)
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance)
    return (mean / std) if std > 0 else None


def _max_drawdown(pnls: list[float]) -> float | None:
    """Peak-to-trough max drawdown of cumulative P&L series."""
    if not pnls:
        return None
    peak = 0.0
    cumul = 0.0
    max_dd = 0.0
    for p in pnls:
        cumul += p
        if cumul > peak:
            peak = cumul
        dd = peak - cumul
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and create tables. Call once at startup."""
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self.init_db()
        log.info("Database ready: %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            log.info("Database closed")

    async def init_db(self) -> None:
        """Create all tables and indexes. Safe to call on existing DB."""
        if self._db is None:
            raise RuntimeError("Call connect() before init_db()")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()

    async def _migrate(self) -> None:
        """Forward-only schema migrations — safe to run on every startup."""
        async with self._conn() as db:
            # Add new columns to trades (no-op if they already exist)
            for col, coltype in [
                ("peak_pnl_pct",  "REAL"),
                ("claude_prob",   "REAL"),
                ("gpt_prob",      "REAL"),
                ("gemini_prob",   "REAL"),
                ("deepseek_prob", "REAL"),
                ("asset_symbol",  "TEXT"),   # multi-asset: BTC/ETH/SOL/XRP/DOGE etc.
                ("tp_order_id",   "TEXT"),   # Kalshi order ID for resting TP sell
                ("sl_order_id",   "TEXT"),   # Kalshi order ID for resting SL sell (future)
            ]:
                try:
                    await db.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
                    await db.commit()
                    log.info("Migration: added column trades.%s", col)
                except Exception:
                    pass  # column already exists

            # Remove the old restrictive CHECK on exit_reason so new exit types
            # (take_profit, trailing_stop) can be written without constraint errors.
            cur = await db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'"
            )
            row = await cur.fetchone()
            if row and "CHECK(exit_reason IN" in (row[0] or ""):
                log.info("Migration: relaxing exit_reason CHECK constraint")
                await db.execute("""
                    CREATE TABLE trades_migrated (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp           TEXT    NOT NULL,
                        market_ticker       TEXT    NOT NULL,
                        direction           TEXT    NOT NULL CHECK(direction IN ('YES','NO')),
                        entry_price         REAL    NOT NULL,
                        size_dollars        REAL    NOT NULL,
                        contracts           INTEGER NOT NULL,
                        kelly_fraction      REAL,
                        edge                REAL,
                        ensemble_confidence REAL,
                        model_spread        REAL,
                        btc_price_at_entry  REAL,
                        btc_momentum        REAL,
                        status              TEXT NOT NULL DEFAULT 'open'
                                            CHECK(status IN ('open','closed','expired')),
                        exit_price          REAL,
                        exit_reason         TEXT,
                        pnl_dollars         REAL,
                        closed_at           TEXT,
                        peak_pnl_pct        REAL,
                        claude_prob         REAL,
                        gpt_prob            REAL,
                        gemini_prob         REAL,
                        deepseek_prob       REAL,
                        asset_symbol        TEXT,
                        tp_order_id         TEXT,
                        sl_order_id         TEXT
                    )
                """)
                # Use COALESCE for newer columns that may not exist in old rows
                await db.execute("""
                    INSERT INTO trades_migrated
                        SELECT id, timestamp, market_ticker, direction, entry_price,
                               size_dollars, contracts, kelly_fraction, edge,
                               ensemble_confidence, model_spread, btc_price_at_entry,
                               btc_momentum, status, exit_price, exit_reason,
                               pnl_dollars, closed_at,
                               peak_pnl_pct, claude_prob, gpt_prob, gemini_prob, deepseek_prob,
                               asset_symbol, tp_order_id, sl_order_id
                        FROM trades
                """)
                await db.execute("DROP TABLE trades")
                await db.execute("ALTER TABLE trades_migrated RENAME TO trades")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
                )
                await db.commit()
                log.info("Migration: trades table updated successfully")

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._db is None:
            raise RuntimeError("Database not connected — call connect() first")
        yield self._db

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------

    async def log_trade(
        self,
        market_ticker: str,
        direction: str,
        entry_price: float,
        size_dollars: float,
        contracts: int,
        *,
        kelly_fraction: float | None = None,
        edge: float | None = None,
        ensemble_confidence: float | None = None,
        model_spread: float | None = None,
        btc_price_at_entry: float | None = None,
        btc_momentum: float | None = None,
        asset_symbol: str | None = None,
        claude_prob: float | None = None,
        gpt_prob: float | None = None,
        gemini_prob: float | None = None,
        deepseek_prob: float | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Insert a new trade. Returns the new row id."""
        ts = timestamp or _now_utc()
        direction = direction.upper()
        async with self._conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO trades (
                    timestamp, market_ticker, direction, entry_price,
                    size_dollars, contracts, kelly_fraction, edge,
                    ensemble_confidence, model_spread,
                    btc_price_at_entry, btc_momentum, asset_symbol,
                    claude_prob, gpt_prob, gemini_prob, deepseek_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, market_ticker, direction, entry_price,
                    size_dollars, contracts, kelly_fraction, edge,
                    ensemble_confidence, model_spread,
                    btc_price_at_entry, btc_momentum, asset_symbol,
                    claude_prob, gpt_prob, gemini_prob, deepseek_prob,
                ),
            )
            await db.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("log_trade: INSERT returned no row ID")
            row_id: int = cursor.lastrowid
            log.info(
                "Trade logged: id=%d %s %s x%d @ %.2f  edge=%.3f  conf=%.3f",
                row_id, direction, market_ticker, contracts, entry_price,
                edge or 0, ensemble_confidence or 0,
            )
            return row_id

    async def update_trade(
        self,
        trade_id: int,
        *,
        status: str,
        exit_price: float,
        exit_reason: str,
        pnl_dollars: float,
        closed_at: str | None = None,
    ) -> None:
        """Update exit fields when a trade closes."""
        async with self._conn() as db:
            await db.execute(
                """
                UPDATE trades
                SET status      = ?,
                    exit_price  = ?,
                    exit_reason = ?,
                    pnl_dollars = ?,
                    closed_at   = ?
                WHERE id = ?
                """,
                (
                    status, exit_price, exit_reason,
                    pnl_dollars, closed_at or _now_utc(),
                    trade_id,
                ),
            )
            await db.commit()
            log.info(
                "Trade updated: id=%d  status=%s  exit=%.4f  pnl=$%.2f  reason=%s",
                trade_id, status, exit_price, pnl_dollars, exit_reason,
            )

    async def update_bracket_orders(
        self,
        trade_id: int,
        *,
        tp_order_id: str | None = None,
        sl_order_id: str | None = None,
    ) -> None:
        """Store the Kalshi order IDs for resting TP/SL limit orders."""
        async with self._conn() as db:
            await db.execute(
                """
                UPDATE trades
                SET tp_order_id = COALESCE(?, tp_order_id),
                    sl_order_id = COALESCE(?, sl_order_id)
                WHERE id = ?
                """,
                (tp_order_id, sl_order_id, trade_id),
            )
            await db.commit()

    async def get_trade(self, trade_id: int) -> TradeRow | None:
        """Fetch a single trade by id. Returns None if not found."""
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            return TradeRow(**dict(row)) if row else None

    async def get_open_trades(self) -> list[TradeRow]:
        """All trades with status='open', ordered oldest first."""
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp ASC"
            )
            rows = await cursor.fetchall()
            return [TradeRow(**dict(r)) for r in rows]

    async def get_recent_trades(self, limit: int = 20) -> list[TradeRow]:
        """Most recent trades regardless of status — for the dashboard."""
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [TradeRow(**dict(r)) for r in rows]

    async def get_all_closed_trades(self) -> list[dict]:
        """All closed/expired trades with full column set — for backtesting."""
        async with self._conn() as db:
            cursor = await db.execute(
                """
                SELECT * FROM trades
                WHERE  status IN ('closed', 'expired')
                  AND  pnl_dollars IS NOT NULL
                ORDER  BY timestamp ASC
                """
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_all_ensemble_log(self) -> list[dict]:
        """All ensemble_log rows — for model attribution backtesting."""
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM ensemble_log ORDER BY timestamp ASC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # daily_stats
    # ------------------------------------------------------------------

    async def get_daily_stats(self, day: str | None = None) -> DailyStats:
        """
        Return today's row from daily_stats.
        Returns a zeroed-out DailyStats if no row exists yet.
        """
        target = day or _today()
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT * FROM daily_stats WHERE date = ?", (target,)
            )
            row = await cursor.fetchone()
            if row:
                return DailyStats(**dict(row))
            return DailyStats(
                date=target,
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
                total_wagered=0.0,
                daily_loss_used=0.0,
                sharpe_ratio=None,
                win_rate=None,
                max_drawdown=None,
            )

    async def update_daily_stats(self, day: str | None = None) -> DailyStats:
        """
        Recompute all stats from the trades table for `day` and upsert into
        daily_stats. Call this after every trade closes.
        """
        target = day or _today()

        async with self._conn() as db:
            cursor = await db.execute(
                """
                SELECT pnl_dollars, size_dollars
                FROM   trades
                WHERE  DATE(timestamp) = ?
                  AND  status IN ('closed', 'expired')
                ORDER BY closed_at ASC
                """,
                (target,),
            )
            closed_rows = await cursor.fetchall()

        pnls:   list[float] = [r["pnl_dollars"]  for r in closed_rows
                               if r["pnl_dollars"]  is not None]
        sizes:  list[float] = [r["size_dollars"] for r in closed_rows
                               if r["size_dollars"] is not None]

        total_trades    = len(closed_rows)
        winning_trades  = sum(1 for p in pnls if p > 0)
        total_pnl       = sum(pnls)
        total_wagered   = sum(sizes)
        daily_loss_used = abs(sum(p for p in pnls if p < 0))
        win_rate        = winning_trades / total_trades if total_trades else None
        sharpe          = _sharpe(pnls)
        max_dd          = _max_drawdown(pnls)

        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO daily_stats (
                    date, total_trades, winning_trades, total_pnl,
                    total_wagered, daily_loss_used,
                    sharpe_ratio, win_rate, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_trades    = excluded.total_trades,
                    winning_trades  = excluded.winning_trades,
                    total_pnl       = excluded.total_pnl,
                    total_wagered   = excluded.total_wagered,
                    daily_loss_used = excluded.daily_loss_used,
                    sharpe_ratio    = excluded.sharpe_ratio,
                    win_rate        = excluded.win_rate,
                    max_drawdown    = excluded.max_drawdown
                """,
                (
                    target, total_trades, winning_trades, total_pnl,
                    total_wagered, daily_loss_used,
                    sharpe, win_rate, max_dd,
                ),
            )
            await db.commit()

        stats = DailyStats(
            date=target,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            total_wagered=total_wagered,
            daily_loss_used=daily_loss_used,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            max_drawdown=max_dd,
        )
        log.info(
            "Daily stats updated [%s]: trades=%d  pnl=$%.2f  win=%.0f%%  sharpe=%s",
            target, total_trades, total_pnl,
            (win_rate or 0) * 100,
            f"{sharpe:.3f}" if sharpe is not None else "n/a",
        )
        return stats

    # ------------------------------------------------------------------
    # ensemble_log
    # ------------------------------------------------------------------

    async def log_ensemble(
        self,
        market_ticker: str,
        *,
        claude_prob: float | None = None,
        gpt_prob: float | None = None,
        gemini_prob: float | None = None,
        deepseek_prob: float | None = None,
        consensus_prob: float | None = None,
        model_spread: float | None = None,
        confidence: float | None = None,
        action: str | None = None,
        skip_reason: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Log one ensemble decision cycle. Returns the new row id."""
        ts = timestamp or _now_utc()
        async with self._conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO ensemble_log (
                    timestamp, market_ticker,
                    claude_prob, gpt_prob, gemini_prob, deepseek_prob,
                    consensus_prob, model_spread, confidence,
                    action, skip_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, market_ticker,
                    claude_prob, gpt_prob, gemini_prob, deepseek_prob,
                    consensus_prob, model_spread, confidence,
                    action, skip_reason,
                ),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # bot_events
    # ------------------------------------------------------------------

    async def log_event(
        self,
        event_type: str,
        message: str = "",
        *,
        timestamp: str | None = None,
    ) -> None:
        """Persist a lifecycle event (startup, shutdown, error, crash, restart)."""
        ts = timestamp or _now_utc()
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO bot_events (timestamp, event_type, message) VALUES (?, ?, ?)",
                (ts, event_type, message),
            )
            await db.commit()
        log.info("Event logged: [%s] %s", event_type, message[:120] if message else "")

    # ------------------------------------------------------------------
    # bot_kv — simple key/value store for live state
    # ------------------------------------------------------------------

    async def get_balance(self) -> float | None:
        """Return the last balance written by the runner, or None."""
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT value FROM bot_kv WHERE key = 'balance'"
            )
            row = await cur.fetchone()
            return float(row[0]) if row else None

    async def set_balance(self, balance: float) -> None:
        """Persist the current Kalshi cash balance (dollars)."""
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO bot_kv (key, value, updated) VALUES ('balance', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value   = excluded.value,
                    updated = excluded.updated
                """,
                (str(balance), _now_utc()),
            )
            await db.commit()

    async def get_bot_enabled(self) -> bool:
        """Return True if the bot has been started via the dashboard."""
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT value FROM bot_kv WHERE key = 'bot_enabled'"
            )
            row = await cur.fetchone()
            return row[0] == "1" if row else False

    async def set_bot_enabled(self, enabled: bool) -> None:
        """Persist bot enabled/disabled state (survives container restarts)."""
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO bot_kv (key, value, updated) VALUES ('bot_enabled', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value   = excluded.value,
                    updated = excluded.updated
                """,
                ("1" if enabled else "0", _now_utc()),
            )
            await db.commit()

    async def get_market_watch(self) -> dict | None:
        """Return the latest cycle watch data written by the runner, or None."""
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT value FROM bot_kv WHERE key = 'market_watch'"
            )
            row = await cur.fetchone()
            return json.loads(row[0]) if row else None

    async def set_market_watch(self, data: dict) -> None:
        """Persist current cycle market scan data for the dashboard."""
        log.info("[DB] set_market_watch: cycle_ts=%s signals=%d markets=%d",
                 data.get("cycle_ts", "?"),
                 len(data.get("signals", [])),
                 len(data.get("markets", [])))
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO bot_kv (key, value, updated) VALUES ('market_watch', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value   = excluded.value,
                    updated = excluded.updated
                """,
                (json.dumps(data), _now_utc()),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Advanced analytics
    # ------------------------------------------------------------------

    async def update_peak_pnl(self, trade_id: int, peak_pnl_pct: float) -> None:
        """Update peak_pnl_pct for a trade (called each exit-check cycle)."""
        async with self._conn() as db:
            await db.execute(
                "UPDATE trades SET peak_pnl_pct = ? WHERE id = ?",
                (peak_pnl_pct, trade_id),
            )
            await db.commit()

    async def get_win_rate_by_direction(self) -> dict:
        """Return win rate separately for YES and NO trades (all-time)."""
        async with self._conn() as db:
            cur = await db.execute(
                """
                SELECT direction,
                       COUNT(*) AS total,
                       SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) AS wins
                FROM   trades
                WHERE  status IN ('closed', 'expired')
                GROUP  BY direction
                """
            )
            rows = await cur.fetchall()

        result: dict = {
            "YES": {"total": 0, "wins": 0, "win_rate": None},
            "NO":  {"total": 0, "wins": 0, "win_rate": None},
        }
        for row in rows:
            d     = row["direction"]
            total = row["total"]
            wins  = row["wins"]
            if d in result:
                result[d] = {
                    "total":    total,
                    "wins":     wins,
                    "win_rate": wins / total if total else None,
                }
        return result

    async def get_model_performance(self) -> dict:
        """
        Return accuracy per model based on whether the model's probability
        aligned with the winning trade direction.
        """
        async with self._conn() as db:
            cur = await db.execute(
                """
                SELECT claude_prob, gpt_prob, gemini_prob, deepseek_prob,
                       direction, pnl_dollars
                FROM   trades
                WHERE  status IN ('closed', 'expired')
                  AND  pnl_dollars IS NOT NULL
                """
            )
            rows = await cur.fetchall()

        model_cols = {
            "claude":   "claude_prob",
            "gpt":      "gpt_prob",
            "gemini":   "gemini_prob",
            "deepseek": "deepseek_prob",
        }
        stats: dict = {m: {"correct": 0, "total": 0} for m in model_cols}

        for row in rows:
            trade_was_yes = row["direction"] == "YES"
            trade_won     = (row["pnl_dollars"] or 0) > 0
            for model, col in model_cols.items():
                prob = row[col]
                if prob is None:
                    continue
                stats[model]["total"] += 1
                model_said_yes = prob > 0.5
                # Correct when model direction aligned with entry AND trade won,
                # or misaligned AND trade lost.
                if (model_said_yes == trade_was_yes) == trade_won:
                    stats[model]["correct"] += 1

        return {
            m: {
                "total":    v["total"],
                "correct":  v["correct"],
                "accuracy": v["correct"] / v["total"] if v["total"] else None,
            }
            for m, v in stats.items()
        }

    async def get_last_7_days_stats(self) -> list[dict]:
        """Return daily P&L for the last 7 calendar days (chronological order)."""
        async with self._conn() as db:
            cur = await db.execute(
                """
                SELECT date, total_pnl, total_trades, win_rate
                FROM   daily_stats
                ORDER  BY date DESC
                LIMIT  7
                """
            )
            rows = await cur.fetchall()
        return [
            {
                "date":         row["date"],
                "total_pnl":    row["total_pnl"],
                "total_trades": row["total_trades"],
                "win_rate":     row["win_rate"],
            }
            for row in reversed(rows)
        ]

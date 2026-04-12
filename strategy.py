"""
strategy.py — Kelly sizing, trade entry, and exit management

Owns the full lifecycle of a trade: sizing, entry, stop-loss/decay/expiry
exits. All database writes and Telegram alerts happen here so runner.py
only needs to call enter_trade() and check_exits().
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import settings
from database import TradeRow
from kalshi_client import KalshiMarketClosedError

log = logging.getLogger(__name__)

_MIN_BET_DOLLARS = 0.50     # never risk less than $0.50
_ROUNDING        = 0.50     # round to nearest $0.50

# ---------------------------------------------------------------------------
# Fill-rate model constants
# ---------------------------------------------------------------------------

# Lookup table: (spread_low_cents, spread_high_cents, fill_probability)
# Empirical estimates — calibrate with fill_rate_log.csv over time.
_FILL_PROB_TABLE: list[tuple[float, float, float]] = [
    (0.0,  5.0,  1.00),   # 0–5¢ spread:  near-certain fill (tight market)
    (5.0,  10.0, 0.90),   # 5–10¢ spread: 90% expected fill rate
    (10.0, 15.0, 0.75),   # 10–15¢ spread: 75% expected fill rate
    (15.0, 20.0, 0.60),   # 15–20¢ spread: 60% — post-gate safety net
]
_FILL_PROB_DEFAULT      = 0.50   # fallback if spread exceeds all table entries
_FILL_IMPROVEMENT_CENTS = 1      # improve limit price by 1¢ per attempt
_FILL_MAX_IMPROVEMENTS  = 2      # max price improvements (2¢ total worst case)
_FILL_WAIT_SECONDS      = 10     # seconds to wait before checking fill status
_FILL_TOTAL_BUDGET      = 30     # total seconds budget across all fill attempts
_FILL_LOG_PATH          = Path("fill_rate_log.csv")
_FILL_LOG_HEADER        = [
    "timestamp", "asset", "side", "spread_at_entry",
    "attempted_contracts", "filled_contracts", "fill_time_seconds",
    "price_improvements_used", "final_fill_price",
]


def _estimate_fill_prob(spread_cents: float) -> float:
    """
    Estimate the probability of a limit order filling based on bid-ask spread.

    Uses a lookup table of empirical estimates.  Calibrate the table entries
    with real fill data from fill_rate_log.csv as it accumulates.
    """
    for lo, hi, prob in _FILL_PROB_TABLE:
        if lo <= spread_cents < hi:
            return prob
    return _FILL_PROB_DEFAULT


def _log_fill_rate(
    asset:                   str,
    side:                    str,
    spread_at_entry:         float,
    attempted_contracts:     int,
    filled_contracts:        int,
    fill_time_seconds:       float,
    price_improvements_used: int,
    final_fill_price:        int,
) -> None:
    """
    Append a fill event to fill_rate_log.csv for future calibration of
    fill probability estimates and spread-filter thresholds.

    Columns: timestamp, asset, side, spread_at_entry, attempted_contracts,
             filled_contracts, fill_time_seconds, price_improvements_used,
             final_fill_price
    """
    write_header = not _FILL_LOG_PATH.exists() or _FILL_LOG_PATH.stat().st_size == 0
    try:
        with _FILL_LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(_FILL_LOG_HEADER)
            writer.writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                asset,
                side,
                round(spread_at_entry, 2),
                attempted_contracts,
                filled_contracts,
                round(fill_time_seconds, 2),
                price_improvements_used,
                final_fill_price,
            ])
    except Exception as exc:
        log.warning("fill_rate_log write failed: %s", exc)


def _round_half(value: float) -> float:
    """Round value down to the nearest $0.50 increment."""
    return math.floor(value / _ROUNDING) * _ROUNDING


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    def __init__(self, kalshi_client, database, telegram) -> None:
        self._kalshi   = kalshi_client
        self._db       = database
        self._telegram = telegram
        # In-memory set of trade IDs currently being closed.
        # Prevents concurrent calls (main cycle + exit monitor) from both
        # entering _close_trade for the same trade before either commits to DB.
        self._closing_trades: set[int] = set()
        # Tickers currently mid-entry (order placed, DB not yet committed).
        # Prevents a concurrent cycle from entering the same ticker during
        # the 20s rate-limit backoff sleep inside kalshi_client._post().
        self._entering_tickers: set[str] = set()

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    async def calculate_bet_size(
        self,
        edge:         float,    # consensus_prob - market_ask (decimal)
        market_price: float,    # ask price in decimal (0.01 – 0.99)
        direction:    str,      # "yes" | "no"  (for logging only)
    ) -> float:
        """
        Half-Kelly position sizing:

          odds      = (1 - market_price) / market_price
          kelly_pct = edge / odds
          kelly_pct *= KELLY_FRACTION          (0.5 → half-Kelly)
          raw_size  = kelly_pct * bankroll
          size      = min(raw_size, MAX_BET_SIZE)
          size      = round down to nearest $0.50
          minimum   = $1.00 (returns 0.0 if below minimum)
        """
        if market_price <= 0.0 or market_price >= 1.0 or edge <= 0.0:
            return 0.0

        odds      = (1.0 - market_price) / market_price
        kelly_pct = (edge / odds) * settings.KELLY_FRACTION

        if kelly_pct <= 0.0:
            return 0.0

        bankroll = await self._kalshi.get_balance()    # dollars
        raw_size = kelly_pct * bankroll
        size     = min(raw_size, settings.MAX_BET_SIZE)
        size     = _round_half(size)

        log.debug(
            "Kelly [%s]: edge=%.3f price=%.2f odds=%.3f kelly=%.3f "
            "bankroll=$%.2f raw=$%.2f → size=$%.2f",
            direction, edge, market_price, odds,
            kelly_pct, bankroll, raw_size, size,
        )

        return size if size >= _MIN_BET_DOLLARS else 0.0

    # ------------------------------------------------------------------
    # Position limit check
    # ------------------------------------------------------------------

    async def can_open_position(self) -> bool:
        """True if fewer than MAX_OPEN_POSITIONS trades are currently open."""
        open_trades = await self._db.get_open_trades()
        below_limit = len(open_trades) < settings.MAX_OPEN_POSITIONS
        if not below_limit:
            log.info(
                "Position limit: %d/%d open — skipping new entry",
                len(open_trades), settings.MAX_OPEN_POSITIONS,
            )
        return below_limit

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def enter_trade(
        self,
        market:          dict,
        ensemble_result: Any,
        gate_result:     Any,
        *,
        btc_price:       float | None = None,   # asset price at entry (kept name for compat)
        btc_momentum:    float | None = None,
        asset_symbol:    str          = "BTC",  # which asset this market is for
        size_multiplier: float = 1.0,
    ) -> TradeRow | None:
        """
        Full entry flow:
          1. Calculate edge + bet size via Kelly
          2. Estimate ask price for contract sizing (market order fills at best available)
          3. Derive contract count (floor)
          4. Place market order with up to 4 retries (1 s gap)
          5. Log trade to database using actual fill price
          6. Send Telegram entry alert
          7. Return TradeRow (or None on any failure)
        """
        direction = ensemble_result.direction   # "yes" | "no" | "flat"
        if direction not in ("yes", "no"):
            return None

        # Guard: only one open trade per market ticker
        ticker = market.get("ticker", "")
        open_trades = await self._db.get_open_trades()
        if any(t.market_ticker == ticker for t in open_trades):
            log.info("Ticker %s already has an open trade — skipping entry", ticker)
            return None

        # Guard: prevent a concurrent cycle from entering the same ticker while
        # a rate-limit backoff sleep (20s) holds an in-progress order open.
        # asyncio is single-threaded, so this check + add is atomic (no await).
        if ticker in self._entering_tickers:
            log.info(
                "Ticker %s entry already in progress (concurrent cycle guard) — skipping",
                ticker,
            )
            return None
        self._entering_tickers.add(ticker)

        # Step 1 — edge check + flat bet sizing
        ask_cents = market.get("yes_ask" if direction == "yes" else "no_ask") or 0
        if ask_cents == 0:
            log.info("%s: no real ask price — refusing to size a trade with zero price", ticker)
            self._entering_tickers.discard(ticker)
            return None
        market_price = ask_cents / 100.0

        # Price cap: refuse to buy a contract priced ≥ 77¢ — too expensive, minimal upside
        if ask_cents >= 77:
            log.info(
                "Skipping %s — %s ask is %d¢ (≥77¢ price cap)",
                ticker, direction.upper(), ask_cents,
            )
            self._entering_tickers.discard(ticker)
            return None

        # P(win): YES trades use P(YES), NO trades use 1-P(YES)
        p_win = (ensemble_result.consensus_prob
                 if direction == "yes"
                 else 1.0 - ensemble_result.consensus_prob)
        edge  = p_win - market_price

        if edge <= 0.0:
            log.info(
                "No edge on %s (p_win=%.3f price=%.2f edge=%.3f) — skipping entry",
                ticker, p_win, market_price, edge,
            )
            self._entering_tickers.discard(ticker)
            return None

        # Flat bet: MAX_BET_SIZE scaled by time/streak multiplier.
        # Kelly is too aggressive/conservative at small bankrolls with narrow AI edges —
        # it consistently rounds to $0 and blocks valid TRADE signals.
        bet_size = math.floor(settings.MAX_BET_SIZE * size_multiplier / _ROUNDING) * _ROUNDING
        bet_size = max(bet_size, _MIN_BET_DOLLARS)

        # Step 2/3 — contracts (floor division; each contract costs ask_cents¢)
        contracts = int(bet_size / market_price)    # both in $
        if contracts < 1:
            log.info("Contract count rounds to 0 — skipping entry")
            self._entering_tickers.discard(ticker)
            return None

        # Step 4 — fill-rate-aware limit order with price improvement.
        # try/finally ensures _entering_tickers is always cleaned up even if an
        # unexpected exception propagates out of the fill loop or DB write below.
        try:
            return await self._do_fill_and_log(
                ticker=ticker,
                direction=direction,
                ask_cents=ask_cents,
                market_price=market_price,
                contracts=contracts,
                bet_size=bet_size,
                p_win=p_win,
                edge=edge,
                market=market,
                ensemble_result=ensemble_result,
                btc_price=btc_price,
                btc_momentum=btc_momentum,
                asset_symbol=asset_symbol,
                size_multiplier=size_multiplier,
            )
        finally:
            self._entering_tickers.discard(ticker)

    async def _do_fill_and_log(
        self,
        ticker,
        direction,
        ask_cents,
        market_price,
        contracts,
        bet_size,
        p_win,
        edge,
        market,
        ensemble_result,
        btc_price,
        btc_momentum,
        asset_symbol,
        size_multiplier,
    ):
        """Inner fill + DB log logic for enter_trade (called inside try/finally)."""
        # (continued from enter_trade — _entering_tickers guard already set by caller)
        #
        # Algorithm:
        #   1. Estimate fill probability from bid-ask spread width.
        #   2. Place limit order at ask price.
        #   3. Wait FILL_WAIT_SECONDS, poll fill status via get_order().
        #   4. If filled:          proceed.
        #   5. If partial ≥ MIN_FILL_CONTRACTS: keep partial, cancel remainder.
        #   6. If partial < MIN_FILL_CONTRACTS: note as micro-fill, let expire.
        #   7. If unfilled:        improve price by FILL_IMPROVEMENT_CENTS,
        #                          cancel old order, resubmit (max FILL_MAX_IMPROVEMENTS).
        #   8. After all improvements, still unfilled: cancel, return None.
        #   Total time budget: FILL_TOTAL_BUDGET seconds.
        #   All fill events are appended to fill_rate_log.csv for calibration.

        # Spread at order time (best-effort; 0 if bid unavailable)
        _bid_entry      = market.get("yes_bid" if direction == "yes" else "no_bid") or 0
        spread_at_entry = float(ask_cents - _bid_entry) if _bid_entry > 0 else 0.0
        fill_prob       = _estimate_fill_prob(spread_at_entry)

        log.info(
            "[ORDER ATTEMPT] ticker=%s dir=%s contracts=%d limit=%d\u00a2 size=$%.2f"
            " | spread=%.0f\u00a2 fill_prob=%.0f%%",
            ticker, direction.upper(), contracts, ask_cents, bet_size,
            spread_at_entry, fill_prob * 100,
        )

        t_start:           float     = time.monotonic()
        current_price:     int       = ask_cents
        price_improvements: int      = 0
        filled_contracts:  int       = 0
        final_fill_price:  int       = ask_cents
        active_order_id:   str | None = None

        for attempt in range(_FILL_MAX_IMPROVEMENTS + 1):

            # Enforce total time budget before each attempt
            elapsed = time.monotonic() - t_start
            if elapsed >= _FILL_TOTAL_BUDGET:
                log.info(
                    "[ORDER] Time budget exhausted (%.0fs / %ds) — cancelling %s",
                    elapsed, _FILL_TOTAL_BUDGET, ticker,
                )
                if active_order_id:
                    try:
                        await self._kalshi.cancel_order(active_order_id)
                    except Exception:
                        pass
                break

            # Subsequent attempts: cancel previous resting order, improve price
            if attempt > 0:
                if active_order_id:
                    _cancelled = False
                    try:
                        _cancelled = await self._kalshi.cancel_order(active_order_id)
                    except Exception as exc:
                        log.warning("[ORDER] cancel prev order failed: %s", exc)
                        _cancelled = True  # assume cancelled; poll below will re-verify

                    if not _cancelled:
                        # cancel_order returns False when the order was already filled
                        # (Kalshi 400/404 response). Fetch the actual fill to avoid
                        # placing a duplicate order and creating an orphaned position.
                        log.info(
                            "[ORDER] Cancel returned False for %s — order already filled, "
                            "fetching fill details", active_order_id[:8],
                        )
                        try:
                            _fill = await self._kalshi.get_order(active_order_id)
                            _qty  = _fill.get("quantity_filled", 0)
                            if _qty > 0:
                                filled_contracts = _qty
                                final_fill_price = (
                                    _fill.get("yes_price" if direction == "yes" else "no_price")
                                    or current_price
                                )
                                log.info(
                                    "[ORDER] Filled during cancel window \u00d7%d @ %d\u00a2 for %s",
                                    filled_contracts, final_fill_price, ticker,
                                )
                                active_order_id = None
                                break
                        except Exception as exc:
                            log.warning("[ORDER] get_order after cancel-fail: %s", exc)

                    active_order_id = None
                current_price   += _FILL_IMPROVEMENT_CENTS
                price_improvements += 1
                log.info(
                    "[ORDER] Price improvement %d/%d \u2192 %d\u00a2 for %s",
                    price_improvements, _FILL_MAX_IMPROVEMENTS, current_price, ticker,
                )

            # Place limit order
            try:
                _order = await self._kalshi.place_order(
                    ticker=ticker,
                    side=direction,
                    count=contracts,
                    price=current_price,
                    order_type="limit",
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                if any(kw in exc_str for kw in (
                    "insufficient", "balance", "funds", "not enough",
                    "invalid order", "invalid_order", "bad request",
                )):
                    log.error(
                        "[ORDER] FAILED (permanent) for %s: %s — not retrying",
                        ticker, exc,
                    )
                    break   # exits loop with filled_contracts == 0
                log.error("[ORDER] FAILED attempt %d for %s: %s", attempt + 1, ticker, exc)
                continue    # retry same price on transient API error

            active_order_id = _order.get("order_id") or ""
            init_status     = _order.get("status", "unknown")

            # Immediate full fill (Kalshi uses "executed" for fully filled)
            if init_status in ("filled", "executed"):
                filled_contracts = contracts
                final_fill_price = _order.get("filled_price") or current_price
                log.info(
                    "[ORDER] Immediate fill \u00d7%d @ %d\u00a2 for %s",
                    filled_contracts, final_fill_price, ticker,
                )
                active_order_id = None
                break

            # Order is resting — wait, then poll fill state
            time_left   = _FILL_TOTAL_BUDGET - (time.monotonic() - t_start)
            actual_wait = min(float(_FILL_WAIT_SECONDS), max(0.0, time_left - 1.0))
            if actual_wait > 0:
                log.debug("[ORDER] waiting %.0fs for fill on %s", actual_wait, ticker)
                await asyncio.sleep(actual_wait)

            poll: dict = {}
            if active_order_id:
                try:
                    poll = await self._kalshi.get_order(active_order_id)
                except Exception as exc:
                    log.warning("[ORDER] get_order failed (%s): %s", active_order_id, exc)

            poll_status  = poll.get("status", init_status)
            qty_filled   = poll.get("quantity_filled", 0)
            poll_price   = (
                poll.get("yes_price" if direction == "yes" else "no_price")
                or current_price
            )

            if poll_status in ("filled", "executed") or qty_filled >= contracts:
                # Full fill confirmed by poll
                filled_contracts = contracts
                final_fill_price = poll_price
                log.info(
                    "[ORDER] Full fill \u00d7%d @ %d\u00a2 for %s (%.0fs)",
                    filled_contracts, final_fill_price, ticker,
                    time.monotonic() - t_start,
                )
                active_order_id = None
                break

            if qty_filled > 0:
                # Partial fill — cancel unfilled remainder
                filled_contracts = qty_filled
                final_fill_price = poll_price
                if active_order_id:
                    try:
                        await self._kalshi.cancel_order(active_order_id)
                    except Exception:
                        pass
                    active_order_id = None

                if filled_contracts >= settings.MIN_FILL_CONTRACTS:
                    log.info(
                        "[ORDER] Partial fill \u00d7%d/%d (\u2265 MIN_FILL=%d) — "
                        "proceeding, remainder cancelled for %s",
                        filled_contracts, contracts,
                        settings.MIN_FILL_CONTRACTS, ticker,
                    )
                else:
                    log.info(
                        "[ORDER] Micro-fill \u00d7%d/%d (< MIN_FILL=%d) — "
                        "letting ride to expiry without active management for %s",
                        filled_contracts, contracts,
                        settings.MIN_FILL_CONTRACTS, ticker,
                    )
                break

            # Still completely unfilled
            if attempt < _FILL_MAX_IMPROVEMENTS:
                log.info(
                    "[ORDER] Unfilled after %.0fs (attempt %d/%d) — improving price for %s",
                    time.monotonic() - t_start,
                    attempt + 1, _FILL_MAX_IMPROVEMENTS + 1,
                    ticker,
                )
            else:
                # All improvements exhausted and still unfilled — cancel and stop
                if active_order_id:
                    try:
                        await self._kalshi.cancel_order(active_order_id)
                    except Exception:
                        pass
                    active_order_id = None
                log.info(
                    "[ORDER] No fill after %d improvement(s) — cancelling %s",
                    _FILL_MAX_IMPROVEMENTS, ticker,
                )

        # ── Fill summary ─────────────────────────────────────────────────────
        fill_time_seconds = time.monotonic() - t_start

        log.info(
            "[ORDER RESULT] ticker=%s dir=%s attempted=%d filled=%d "
            "improvements=%d fill_time=%.1fs price=%d\u00a2",
            ticker, direction.upper(), contracts, filled_contracts,
            price_improvements, fill_time_seconds, final_fill_price,
        )

        _log_fill_rate(
            asset                   = asset_symbol,
            side                    = direction.upper(),
            spread_at_entry         = spread_at_entry,
            attempted_contracts     = contracts,
            filled_contracts        = filled_contracts,
            fill_time_seconds       = fill_time_seconds,
            price_improvements_used = price_improvements,
            final_fill_price        = final_fill_price,
        )

        if filled_contracts == 0:
            await self._telegram.send_error(
                f"No fill for {ticker} ({direction.upper()} \u00d7{contracts}) "
                f"after {price_improvements} price improvement(s) — order cancelled.",
                "no_fill",
            )
            return None

        # Update to actual filled quantity and price for steps 5+
        contracts    = filled_contracts
        filled_cents = final_fill_price
        actual_cost  = contracts * (filled_cents / 100.0)

        # Step 5 — log to database
        spread        = ensemble_result.spread if ensemble_result.models else None
        ts            = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        claude_prob   = ensemble_result.claude.probability   if ensemble_result.claude   else None
        gpt_prob      = ensemble_result.gpt.probability      if ensemble_result.gpt      else None
        gemini_prob   = ensemble_result.gemini.probability   if ensemble_result.gemini   else None
        deepseek_prob = ensemble_result.deepseek.probability if ensemble_result.deepseek else None
        trade_id = await self._db.log_trade(
            market_ticker       = ticker,
            direction           = direction.upper(),
            entry_price         = float(filled_cents),
            size_dollars        = actual_cost,
            contracts           = contracts,
            edge                = edge,
            ensemble_confidence = ensemble_result.confidence,
            model_spread        = spread,
            btc_price_at_entry  = btc_price,
            btc_momentum        = btc_momentum,
            asset_symbol        = asset_symbol,
            claude_prob         = claude_prob,
            gpt_prob            = gpt_prob,
            gemini_prob         = gemini_prob,
            deepseek_prob       = deepseek_prob,
            timestamp           = ts,
        )

        trade = TradeRow(
            id                  = trade_id,
            timestamp           = ts,
            market_ticker       = ticker,
            direction           = direction.upper(),
            entry_price         = float(filled_cents),
            size_dollars        = actual_cost,
            contracts           = contracts,
            kelly_fraction      = None,
            edge                = edge,
            ensemble_confidence = ensemble_result.confidence,
            model_spread        = spread,
            btc_price_at_entry  = btc_price,
            btc_momentum        = btc_momentum,
            asset_symbol        = asset_symbol,
            status              = "open",
            exit_price          = None,
            exit_reason         = None,
            pnl_dollars         = None,
            closed_at           = None,
            claude_prob         = claude_prob,
            gpt_prob            = gpt_prob,
            gemini_prob         = gemini_prob,
            deepseek_prob       = deepseek_prob,
        )

        # Step 6 — Telegram alert
        await self._telegram.send_trade_entry(trade)

        log.info(
            "Trade entered: id=%d [%s] %s %s ×%d @ %d¢ (limit) | "
            "p_win=%.3f edge=%.3f conf=%.3f cost=$%.2f",
            trade_id, asset_symbol, direction.upper(), ticker, contracts,
            filled_cents, p_win, edge, ensemble_result.confidence, actual_cost,
        )

        return trade

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    async def check_exits(self, open_trades: list[TradeRow]) -> list[TradeRow]:
        """
        Evaluate every open trade for exit conditions.

        Triggers checked per trade (first match wins):
          1. Market resolved → expired
          2. Close time < now + 2min → let expire naturally (skip)
          3. pnl_pct >= TAKE_PROFIT_PCT → take_profit
          4. Trailing stop activates at peak >= TRAILING_STOP_LOCK_PCT
          5. pnl_pct <= -STOP_LOSS_PCT → stop_loss
          6. current bid < CONFIDENCE_DECAY_EXIT × 100¢ → decay

        Returns the list of trades closed this cycle.
        """
        closed: list[TradeRow] = []
        for trade in open_trades:
            result = await self._evaluate_exit(trade)
            if result is not None:
                closed.append(result)
        return closed

    async def _evaluate_exit(self, trade: TradeRow) -> TradeRow | None:
        ticker = trade.market_ticker

        # --- Trigger 1: market already resolved ---
        try:
            ob = await self._kalshi.get_order_book(ticker)
        except Exception as exc:
            if isinstance(exc, KalshiMarketClosedError):
                log.info(
                    "Trade %d: %s resolved — fetching settlement result", trade.id, ticker
                )
                try:
                    mkt    = await self._kalshi.get_market(ticker)
                    result = (mkt.get("result") or "").lower()
                    if result == "yes":
                        resolved_price = 100 if trade.direction == "YES" else 0
                    elif result == "no":
                        resolved_price = 0 if trade.direction == "YES" else 100
                    else:
                        log.warning(
                            "Trade %d: %s has no settlement result yet (result=%r) — using entry price",
                            trade.id, ticker, result,
                        )
                        resolved_price = int(trade.entry_price)
                except Exception as fetch_exc:
                    log.warning(
                        "Trade %d: failed to fetch settlement for %s: %s — using entry price",
                        trade.id, ticker, fetch_exc,
                    )
                    resolved_price = int(trade.entry_price)
                return await self._close_trade(trade, resolved_price, "expired")
            log.warning(
                "Trade %d: orderbook fetch failed (%s) — skipping exit check",
                trade.id, exc,
            )
            return None

        # Current best bid for our side
        # If bids are empty the contract is effectively worthless on that side —
        # use 0 so stop-loss and decay exits fire correctly instead of masking the loss
        if trade.direction == "YES":
            bids        = ob.get("yes_bids", [])
            current_bid = bids[0]["price"] if bids else 0
        else:
            bids        = ob.get("no_bids", [])
            current_bid = bids[0]["price"] if bids else 0

        # --- Compute P&L fraction and update peak ---
        entry   = trade.entry_price  # cents (stored as float)
        pnl_pct = (current_bid - entry) / entry if entry else 0.0

        peak = trade.peak_pnl_pct or 0.0
        if pnl_pct > peak:
            peak = pnl_pct
            await self._db.update_peak_pnl(trade.id, peak)

        # --- Trigger 2: less than 2 minutes to close → let expire ---
        try:
            mkt = await self._kalshi.get_market(ticker)
            close_str = mkt.get("close_time", "")
            if close_str:
                close_dt = datetime.fromisoformat(
                    close_str.replace("Z", "+00:00")
                )
                if close_dt - datetime.now(timezone.utc) < timedelta(minutes=2):
                    log.debug(
                        "Trade %d: %s closes in <2min — letting expire naturally",
                        trade.id, ticker,
                    )
                    return None
        except Exception:
            pass    # don't block other checks if market fetch fails

        # --- Trigger 3: take profit ---
        if pnl_pct >= settings.TAKE_PROFIT_PCT:
            log.info(
                "Trade %d: take profit — pnl_pct=%.1f%% (bid=%d¢ entry=%.0f¢)",
                trade.id, pnl_pct * 100, current_bid, entry,
            )
            return await self._close_trade(trade, current_bid, "take_profit")

        # --- Trigger 4: trailing stop ---
        # Activates once peak reaches +TRAILING_STOP_LOCK_PCT;
        # exits if current drops below +TRAILING_STOP_EXIT_PCT
        if (peak >= settings.TRAILING_STOP_LOCK_PCT
                and pnl_pct < settings.TRAILING_STOP_EXIT_PCT):
            log.info(
                "Trade %d: trailing stop — peak=%.1f%% current=%.1f%% (bid=%d¢ entry=%.0f¢)",
                trade.id, peak * 100, pnl_pct * 100, current_bid, entry,
            )
            return await self._close_trade(trade, current_bid, "trailing_stop")

        # --- Trigger 5: stop loss ---
        if pnl_pct <= -settings.STOP_LOSS_PCT:
            log.info(
                "Trade %d: stop loss — pnl_pct=%.1f%% (bid=%d¢ entry=%.0f¢)",
                trade.id, pnl_pct * 100, current_bid, entry,
            )
            return await self._close_trade(trade, current_bid, "stop_loss")

        # --- Trigger 6: confidence decay (market price collapsed) ---
        decay_threshold_cents = int(settings.CONFIDENCE_DECAY_EXIT * 100)
        if current_bid < decay_threshold_cents:
            log.info(
                "Trade %d: decay — bid %d¢ < threshold %d¢",
                trade.id, current_bid, decay_threshold_cents,
            )
            return await self._close_trade(trade, current_bid, "decay")

        return None

    async def _close_trade(
        self,
        trade:       TradeRow,
        exit_price:  int,       # cents — current best bid, used as P&L fallback
        exit_reason: str,       # "stop_loss" | "decay" | "take_profit" |
                                # "trailing_stop" | "expired" | "manual"
    ) -> TradeRow:
        """
        Close a trade: place market sell (unless expired), update DB,
        send Telegram alert.

        Uses actual fill price for P&L; falls back to current bid estimate.
        Sell failures are logged but never block the database update —
        the position resolves naturally at market expiry.
        """
        # ── Concurrency guard (in-memory) ──────────────────────────────────
        # Python/asyncio is single-threaded: the check + add below are atomic
        # (no await between them). This prevents the main cycle and exit monitor
        # from both entering here for the same trade before either commits to DB.
        if trade.id in self._closing_trades:
            log.warning(
                "Trade %d: close already in progress — skipping (concurrency guard)",
                trade.id,
            )
            return trade
        self._closing_trades.add(trade.id)

        try:
            return await self._do_close(trade, exit_price, exit_reason)
        finally:
            self._closing_trades.discard(trade.id)

    async def _do_close(
        self,
        trade:       TradeRow,
        exit_price:  int,
        exit_reason: str,
    ) -> TradeRow:
        final_exit_price = exit_price

        # ── DB race guard ───────────────────────────────────────────────────
        # Belt-and-suspenders: verify the trade is still open in DB before
        # placing any sell order (e.g. a prior cycle already closed it).
        fresh = await self._db.get_trade(trade.id)
        if fresh is None or fresh.status != "open":
            log.warning(
                "Trade %d: already closed/expired in DB (status=%s) — aborting (DB race guard)",
                trade.id, fresh.status if fresh else "not found",
            )
            return trade

        # ── Limit sell ──────────────────────────────────────────────────────
        if exit_reason != "expired":
            if exit_price <= 0:
                # No buyers in the book — placing any sell order is pointless.
                # Record the loss at 0¢ and let the position expire naturally.
                log.info(
                    "Trade %d: bid is 0 for %s — no buyers, recording at 0¢ and letting expire",
                    trade.id, trade.market_ticker,
                )
            else:
                # Limit sell at the current best bid — fills immediately if buyers exist,
                # rests otherwise. We cancel resting orders and record at estimated price.
                _MAX_ATTEMPTS = 5
                for attempt in range(_MAX_ATTEMPTS):
                    if attempt > 0:
                        await asyncio.sleep(1.0)
                        log.info(
                            "Limit sell retry %d/%d — trade %d (%s)",
                            attempt, _MAX_ATTEMPTS - 1, trade.id, trade.market_ticker,
                        )

                    try:
                        sell_result = await self._kalshi.place_order(
                            ticker=trade.market_ticker,
                            side=trade.direction.lower(),
                            count=trade.contracts,
                            action="sell",
                            price=exit_price,
                            order_type="limit",
                        )
                    except Exception as exc:
                        log.warning(
                            "Trade %d: limit sell attempt %d/%d failed (%s)",
                            trade.id, attempt + 1, _MAX_ATTEMPTS, exc,
                        )
                        if attempt == _MAX_ATTEMPTS - 1:
                            await self._telegram.send_error(
                                f"Limit sell failed for trade {trade.id} "
                                f"({trade.market_ticker}): {exc}\n"
                                f"Position may still be open on Kalshi — check manually.",
                                "sell_order_failed",
                            )
                        continue

                    sell_status = sell_result.get("status", "unknown")
                    if sell_status in ("filled", "partially_filled", "executed"):
                        fp = sell_result.get("filled_price")
                        if fp:
                            final_exit_price = fp
                        log.info(
                            "Trade %d: limit sell filled — attempt %d/%d  fill=%d¢",
                            trade.id, attempt + 1, _MAX_ATTEMPTS, final_exit_price,
                        )
                        break

                    # Order placed but resting (no counterparty) — cancel and stop
                    order_id = sell_result.get("order_id", "")
                    if order_id:
                        try:
                            await self._kalshi.cancel_order(order_id)
                        except Exception as cancel_exc:
                            log.warning("Failed to cancel sell order %s: %s", order_id, cancel_exc)
                    log.warning(
                        "Trade %d: limit sell for %s resting unfilled (status: %s) "
                        "— no counter-party, recording at estimated price",
                        trade.id, trade.market_ticker, sell_status,
                    )
                    await self._telegram.send_error(
                        f"Limit sell for trade {trade.id} ({trade.market_ticker}) "
                        f"resting unfilled (status: {sell_status}) — no counter-party. "
                        f"Position may still be open on Kalshi. Check manually.",
                        "sell_order_failed",
                    )
                    break  # record at estimated price and close in DB regardless

        # ── Persist result ──────────────────────────────────────────────────
        pnl       = (final_exit_price - trade.entry_price) * trade.contracts / 100.0
        closed_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        db_status = "expired" if exit_reason == "expired" else "closed"

        await self._db.update_trade(
            trade.id,
            status      = db_status,
            exit_price  = float(final_exit_price),
            exit_reason = exit_reason,
            pnl_dollars = pnl,
            closed_at   = closed_ts,
        )
        await self._db.update_daily_stats()

        closed_trade = TradeRow(
            **{**trade.__dict__,
               "status":      db_status,
               "exit_price":  float(final_exit_price),
               "exit_reason": exit_reason,
               "pnl_dollars": pnl,
               "closed_at":   closed_ts},
        )
        await self._telegram.send_trade_exit(closed_trade, exit_reason)
        log.info(
            "Trade %d closed  ticker=%s  reason=%s  exit=%d¢  P&L=$%.2f",
            trade.id, trade.market_ticker, exit_reason, final_exit_price, pnl,
        )
        return closed_trade

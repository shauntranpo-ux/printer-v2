"""
strategy.py — Kelly sizing, trade entry, and exit management

Owns the full lifecycle of a trade: sizing, entry, stop-loss/decay/expiry
exits. All database writes and Telegram alerts happen here so runner.py
only needs to call enter_trade() and check_exits().
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings
from database import TradeRow

log = logging.getLogger(__name__)

_MIN_BET_DOLLARS = 0.50     # never risk less than $0.50
_ROUNDING        = 0.50     # round to nearest $0.50


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

        # Step 1 — edge check + flat bet sizing
        ask_cents = market.get("yes_ask" if direction == "yes" else "no_ask") or 0
        order_book_empty = ask_cents == 0
        if order_book_empty:
            # Order book empty — no market makers yet. Use 50¢ as fair-value assumption
            # for edge/sizing. Market order sends directly to Kalshi and fills at actual
            # price (or cancels if no counterparty). Edge = p_win - 0.50.
            ask_cents = 50
            log.info(
                "%s: order book empty — using 50¢ fair value for sizing, sending market order",
                ticker,
            )
        market_price = ask_cents / 100.0

        # Price cap: refuse to buy a contract priced ≥ 77¢ — too expensive, minimal upside
        if ask_cents >= 77:
            log.info(
                "Skipping %s — %s ask is %d¢ (≥77¢ price cap)",
                ticker, direction.upper(), ask_cents,
            )
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
            return None

        # Flat bet: MAX_BET_SIZE scaled by time/streak multiplier.
        # Kelly is too aggressive/conservative at small bankrolls with narrow AI edges —
        # it consistently rounds to $0 and blocks valid TRADE signals.
        bet_size = math.floor(settings.MAX_BET_SIZE * size_multiplier / _ROUNDING) * _ROUNDING
        bet_size = max(bet_size, _MIN_BET_DOLLARS)

        # Step 2/3 — contracts (floor division; each contract costs ask_cents¢)
        # When the order book is empty we don't know the real fill price.
        # Size conservatively using worst-case 99¢ to avoid spending more than MAX_BET_SIZE.
        if order_book_empty:
            contracts = max(1, int(bet_size / 0.99))
            log.info(
                "%s: empty-book sizing — %d contracts (worst-case 99¢ fill)",
                ticker, contracts,
            )
        else:
            contracts = int(bet_size / market_price)    # both in $
        if contracts < 1:
            log.info("Contract count rounds to 0 — skipping entry")
            return None

        # Step 4 — place market order; retry up to 4 times only on API failure.
        # If the order is placed but has no counterparty (resting), cancel and stop —
        # placing more identical market orders won't create buyers.
        _MAX_ATTEMPTS = 5   # 1 initial + 4 retries on API exception
        order_result: dict | None = None
        final_status  = "unknown"

        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(1.0)
                log.info(
                    "Market order retry %d/%d for %s (API failure)",
                    attempt, _MAX_ATTEMPTS - 1, ticker,
                )

            try:
                order_result = await self._kalshi.place_order(
                    ticker=ticker,
                    side=direction,
                    count=contracts,
                    order_type="market",
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                # Permanent errors — don't retry (retrying won't fix these)
                if any(kw in exc_str for kw in ("insufficient", "balance", "funds", "not enough")):
                    log.error(
                        "Market order failed (insufficient balance) for %s: %s — not retrying",
                        ticker, exc,
                    )
                    return None
                # Transient API / network failure — retry
                log.error(
                    "Market order attempt %d/%d failed for %s: %s",
                    attempt + 1, _MAX_ATTEMPTS, ticker, exc,
                )
                if attempt == _MAX_ATTEMPTS - 1:
                    await self._telegram.send_error(str(exc), "enter_trade")
                    return None
                continue

            # Order placed — check fill
            # Kalshi uses "executed" to mean fully filled (not "filled")
            final_status = order_result.get("status", "unknown")
            if final_status in ("filled", "partially_filled", "executed"):
                log.info(
                    "Market order filled — %s  attempt %d/%d  status=%s",
                    ticker, attempt + 1, _MAX_ATTEMPTS, final_status,
                )
                break

            # Order reached Kalshi but has no counterparty — cancel and stop.
            # Retrying won't help; no one is on the other side at any price.
            order_id = order_result.get("order_id", "")
            if order_id:
                try:
                    await self._kalshi.cancel_order(order_id)
                except Exception as cancel_exc:
                    log.warning("Failed to cancel order %s: %s", order_id, cancel_exc)
            log.warning(
                "Market order for %s placed but unfilled (status: %s) — "
                "no counter-party, not retrying",
                ticker, final_status,
            )
            await self._telegram.send_error(
                f"Market order for {ticker} ({direction.upper()} ×{contracts}) "
                f"could not fill (status: {final_status}) — no counter-party. "
                f"Order cancelled.",
                "Order unfilled",
            )
            return None

        if order_result is None:
            return None

        # Use the actual fill price when reported; fall back to our ask estimate
        filled_cents = order_result.get("filled_price") or ask_cents
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
            "Trade entered: id=%d [%s] %s %s ×%d @ %d¢ (market) | "
            "p_win=%.3f edge=%.3f conf=%.3f cost=$%.2f",
            trade_id, asset_symbol, direction.upper(), ticker, contracts,
            filled_cents, p_win, edge, ensemble_result.confidence, actual_cost,
        )

        # Step 7 — Place resting TP limit sell order so Kalshi fills it automatically
        # TP price: entry * (1 + TAKE_PROFIT_PCT), capped at 99¢
        tp_cents = min(99, int(filled_cents * (1.0 + settings.TAKE_PROFIT_PCT)))
        if tp_cents > filled_cents:
            try:
                tp_result = await self._kalshi.place_order(
                    ticker     = ticker,
                    side       = direction,
                    count      = contracts,
                    action     = "sell",
                    price      = tp_cents,
                    order_type = "limit",
                )
                tp_order_id = tp_result.get("order_id", "")
                if tp_order_id:
                    await self._db.update_bracket_orders(trade_id, tp_order_id=tp_order_id)
                    trade.tp_order_id = tp_order_id
                    log.info(
                        "Trade %d: TP resting sell placed at %d¢ (order %s)",
                        trade_id, tp_cents, tp_order_id[:8],
                    )
            except Exception as exc:
                log.warning("Trade %d: failed to place TP resting order: %s", trade_id, exc)

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
          3. pnl_pct >= TAKE_PROFIT_PCT (+55%) → take_profit
          4. Trailing stop activates at peak >= TRAILING_STOP_LOCK_PCT
          5. pnl_pct <= -STOP_LOSS_PCT (80%) → stop_loss
             e.g. entry 60¢ → triggers at 12¢ (60 × 0.20 = 12)
          6. current bid < CONFIDENCE_DECAY_EXIT × 100¢ (20¢) → decay

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

        # --- Trigger 0: check if resting TP limit order has already been filled ---
        if trade.tp_order_id:
            try:
                tp_order  = await self._kalshi.get_order(trade.tp_order_id)
                tp_status = tp_order.get("status", "unknown")
                if tp_status in ("filled", "executed", "partially_filled"):
                    # Use the price for OUR side of the trade
                    if trade.direction == "YES":
                        tp_price = tp_order.get("yes_price") or int(trade.entry_price)
                    else:
                        tp_price = tp_order.get("no_price") or int(trade.entry_price)
                    log.info(
                        "Trade %d: TP resting order %s filled at %d¢",
                        trade.id, trade.tp_order_id[:8], tp_price,
                    )
                    return await self._close_trade(trade, tp_price, "take_profit", skip_sell=True)
            except Exception as exc:
                log.warning("Trade %d: TP order status check failed: %s", trade.id, exc)

        # --- Trigger 1: market already resolved ---
        try:
            ob = await self._kalshi.get_order_book(ticker)
        except Exception as exc:
            # Lazy import avoids a circular reference at module level
            from kalshi_client import KalshiMarketClosedError
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
        *,
        skip_sell:   bool = False,  # True when resting order already filled (TP)
    ) -> TradeRow:
        """
        Place a market sell order with up to 4 retries (1 s between attempts).
        Uses actual fill price for P&L; falls back to current bid estimate.
        Sell failures are logged but never block the database update —
        the position resolves naturally at market expiry.
        """
        final_exit_price = exit_price   # updated to fill price if order confirms

        # Cancel the resting TP order if we're closing for a different reason
        # (SL/decay/expiry) to avoid a dangling resting order on Kalshi.
        if not skip_sell and trade.tp_order_id and exit_reason != "take_profit":
            try:
                await self._kalshi.cancel_order(trade.tp_order_id)
                log.info("Trade %d: cancelled resting TP order %s", trade.id, trade.tp_order_id[:8])
            except Exception as exc:
                log.warning("Trade %d: failed to cancel TP order %s: %s", trade.id, trade.tp_order_id[:8], exc)

        # Race condition guard: re-fetch from DB to ensure trade is still open.
        # Both the main cycle and exit monitor call check_exits independently —
        # without this check a trade could be sold twice if both calls race.
        fresh = await self._db.get_trade(trade.id)
        if fresh is None or fresh.status != "open":
            log.warning(
                "Trade %d: already closed/expired in DB (status=%s) — aborting sell (race guard)",
                trade.id, fresh.status if fresh else "not found",
            )
            return trade

        if not skip_sell and exit_reason != "expired":
            # Retry up to 4 times only on API exception.
            # YES and NO positions are symmetric: we always sell OUR side at market
            # (yes_bids for YES trades, no_bids for NO trades). P&L = (fill - entry).
            _MAX_ATTEMPTS = 5   # 1 initial + 4 retries on API exception
            for attempt in range(_MAX_ATTEMPTS):
                if attempt > 0:
                    await asyncio.sleep(1.0)
                    log.info(
                        "Market sell retry %d/%d — trade %d (%s)",
                        attempt, _MAX_ATTEMPTS - 1, trade.id, trade.market_ticker,
                    )

                try:
                    sell_result = await self._kalshi.place_order(
                        ticker=trade.market_ticker,
                        side=trade.direction.lower(),
                        count=trade.contracts,
                        action="sell",
                        order_type="market",
                    )
                except Exception as exc:
                    log.warning(
                        "Trade %d: market sell attempt %d/%d failed (%s)",
                        trade.id, attempt + 1, _MAX_ATTEMPTS, exc,
                    )
                    if attempt == _MAX_ATTEMPTS - 1:
                        await self._telegram.send_error(
                            f"Market sell failed for trade {trade.id} "
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
                        "Trade %d: market sell filled — attempt %d/%d  fill=%d¢",
                        trade.id, attempt + 1, _MAX_ATTEMPTS, final_exit_price,
                    )
                    break

                # Order placed but no counterparty — cancel and stop, don't retry
                order_id = sell_result.get("order_id", "")
                if order_id:
                    try:
                        await self._kalshi.cancel_order(order_id)
                    except Exception as cancel_exc:
                        log.warning("Failed to cancel sell order %s: %s", order_id, cancel_exc)
                log.warning(
                    "Trade %d: market sell for %s placed but unfilled (status: %s) "
                    "— no counter-party, recording at estimated price",
                    trade.id, trade.market_ticker, sell_status,
                )
                await self._telegram.send_error(
                    f"Market sell for trade {trade.id} ({trade.market_ticker}) "
                    f"could not fill (status: {sell_status}) — no counter-party. "
                    f"Position may still be open on Kalshi. Check manually.",
                    "sell_order_failed",
                )
                break  # record at estimated price and close in DB regardless

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

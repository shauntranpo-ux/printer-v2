"""
strategy.py — Kelly sizing, trade entry, and exit management

Owns the full lifecycle of a trade: sizing, entry, stop-loss/decay/expiry
exits. All database writes and Telegram alerts happen here so runner.py
only needs to call enter_trade() and check_exits().
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings
from database import TradeRow

log = logging.getLogger(__name__)

_MIN_BET_DOLLARS = 1.00     # never risk less than $1
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
        btc_price:       float | None = None,
        btc_momentum:    float | None = None,
    ) -> TradeRow | None:
        """
        Full entry flow:
          1. Calculate edge + bet size via Kelly
          2. Determine limit bid price (take ask liquidity)
          3. Derive contract count (floor)
          4. Place order via KalshiClient
          5. Log trade to database
          6. Send Telegram entry alert
          7. Return TradeRow (or None on any failure)
        """
        direction = ensemble_result.direction   # "yes" | "no" | "flat"
        if direction not in ("yes", "no"):
            return None

        # Step 1 — edge + Kelly size
        ask_cents   = market.get("yes_ask" if direction == "yes" else "no_ask", 50)
        market_price = ask_cents / 100.0
        edge         = ensemble_result.consensus_prob - market_price

        bet_size = await self.calculate_bet_size(edge, market_price, direction)
        if bet_size < _MIN_BET_DOLLARS:
            log.info(
                "Bet size $%.2f below minimum $%.2f — skipping entry",
                bet_size, _MIN_BET_DOLLARS,
            )
            return None

        # Step 2/3 — contracts (floor division; each contract costs ask_cents¢)
        contracts = int(bet_size / market_price)    # both in $
        if contracts < 1:
            log.info("Contract count rounds to 0 — skipping entry")
            return None

        actual_cost = contracts * market_price      # dollars

        # Step 4 — place order
        ticker = market.get("ticker", "")
        try:
            await self._kalshi.place_order(
                ticker=ticker,
                side=direction,
                price=ask_cents,
                count=contracts,
            )
        except Exception as exc:
            log.error("Order placement failed for %s: %s", ticker, exc)
            await self._telegram.send_error(str(exc), "enter_trade")
            return None

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
            entry_price         = float(ask_cents),
            size_dollars        = actual_cost,
            contracts           = contracts,
            edge                = edge,
            ensemble_confidence = ensemble_result.confidence,
            model_spread        = spread,
            btc_price_at_entry  = btc_price,
            btc_momentum        = btc_momentum,
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
            entry_price         = float(ask_cents),
            size_dollars        = actual_cost,
            contracts           = contracts,
            kelly_fraction      = None,
            edge                = edge,
            ensemble_confidence = ensemble_result.confidence,
            model_spread        = spread,
            btc_price_at_entry  = btc_price,
            btc_momentum        = btc_momentum,
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
            "Trade entered: id=%d %s %s ×%d @ %d¢ | "
            "edge=%.3f conf=%.3f cost=$%.2f",
            trade_id, direction.upper(), ticker, contracts,
            ask_cents, edge, ensemble_result.confidence, actual_cost,
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
          3. pnl_pct <= -STOP_LOSS_PCT (35%) → stop_loss
          4. current bid < CONFIDENCE_DECAY_EXIT × 100¢ → decay

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
            # Lazy import avoids a circular reference at module level
            from kalshi_client import KalshiMarketClosedError
            if isinstance(exc, KalshiMarketClosedError):
                log.info(
                    "Trade %d: %s resolved — marking expired", trade.id, ticker
                )
                return await self._close_trade(trade, int(trade.entry_price), "expired")
            log.warning(
                "Trade %d: orderbook fetch failed (%s) — skipping exit check",
                trade.id, exc,
            )
            return None

        # Current best bid for our side
        if trade.direction == "YES":
            bids        = ob.get("yes_bids", [])
            current_bid = bids[0]["price"] if bids else int(trade.entry_price)
        else:
            bids        = ob.get("no_bids", [])
            current_bid = bids[0]["price"] if bids else int(trade.entry_price)

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
        exit_price:  int,       # cents
        exit_reason: str,       # "stop_loss" | "decay" | "expired" | "manual"
    ) -> TradeRow:
        """
        Place a best-effort sell order, update the database, and alert.
        Sell failures are logged but do not block the database update
        (the position will resolve via market expiry).
        """
        if exit_reason != "expired":
            try:
                await self._kalshi.place_order(
                    ticker=trade.market_ticker,
                    side=trade.direction.lower(),
                    price=exit_price,
                    count=trade.contracts,
                    action="sell",
                )
            except Exception as exc:
                log.warning(
                    "Trade %d: sell order failed (%s) — "
                    "position remains open on Kalshi; recording estimated close",
                    trade.id, exc,
                )
                await self._telegram.send_error(
                    f"Sell failed for trade {trade.id} ({trade.market_ticker}): {exc}\n"
                    f"Position may still be open on Kalshi — check manually.",
                    "sell_order_failed",
                )

        pnl       = (exit_price - trade.entry_price) * trade.contracts / 100.0
        closed_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        db_status = "expired" if exit_reason == "expired" else "closed"

        await self._db.update_trade(
            trade.id,
            status      = db_status,
            exit_price  = float(exit_price),
            exit_reason = exit_reason,
            pnl_dollars = pnl,
            closed_at   = closed_ts,
        )
        await self._db.update_daily_stats()

        closed_trade = TradeRow(
            **{**trade.__dict__,
               "status":      db_status,
               "exit_price":  float(exit_price),
               "exit_reason": exit_reason,
               "pnl_dollars": pnl,
               "closed_at":   closed_ts},
        )
        await self._telegram.send_trade_exit(closed_trade, exit_reason)
        log.info(
            "Trade %d closed  ticker=%s  reason=%s  exit=%d¢  P&L=$%.2f",
            trade.id, trade.market_ticker, exit_reason, exit_price, pnl,
        )
        return closed_trade

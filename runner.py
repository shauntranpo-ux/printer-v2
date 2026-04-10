"""
runner.py — Main 15-minute trading loop
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from database import Database, TradeRow
from coinbase_feed import CoinbaseFeed, StaleDataError
from ensemble import EnsembleEngine, BtcData, Market
from risk_gates import RiskGates, BotState
from strategy import Strategy
from kalshi_client import (
    KalshiClient,
    KalshiAuthError,
    KalshiMarketClosedError,
    KalshiInsufficientFundsError,
)
from telegram_alerts import TelegramAlerter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("printer_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("runner")

TICK_INTERVAL_SEC = 15 * 60     # 15 minutes
EXIT_CHECK_SEC    = 60          # check open positions every 60s


# ---------------------------------------------------------------------------
# Clock alignment
# ---------------------------------------------------------------------------

async def sleep_until_next_tick(interval: int = TICK_INTERVAL_SEC) -> None:
    """Sleep until the next clock-aligned boundary (e.g., :00, :15, :30, :45)."""
    now = time.time()
    next_tick = (int(now // interval) + 1) * interval
    wait = next_tick - now
    log.info("Next tick in %.1fs (at %s UTC)", wait,
             datetime.fromtimestamp(next_tick, tz=timezone.utc).strftime("%H:%M:%S"))
    await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------

class BotRunner:
    def __init__(self):
        settings.validate()

        self.db      = Database(Path(settings.DB_PATH))
        self.feed    = CoinbaseFeed()
        self.ensemble = EnsembleEngine()
        self.risk = RiskGates(
            max_daily_drawdown_pct=settings.DAILY_LOSS_LIMIT / 10_000,  # rough pct; refined at runtime
            max_position_exposure=settings.MAX_OPEN_POSITIONS / 10.0,
            confidence_min=settings.MIN_CONFIDENCE,
            max_btc_vol_threshold=0.04,
            min_minutes_between_trades=15,
        )
        self.strategy = Strategy(
            kelly_fraction_multiplier=settings.KELLY_FRACTION,
            max_position_pct=min(settings.MAX_BET_SIZE / 1000, 0.10),
        )
        self.kalshi  = KalshiClient()
        self.telegram = TelegramAlerter()

        self._last_trade_ts: float = 0.0
        self._starting_balance_cents: int = 0
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        log.info("=== printer-v2 starting [%s] ===", settings.env)
        await self.db.connect()
        await self.feed.start()
        await self.telegram.start()

        # Wait for the first BTC price before doing anything
        log.info("Waiting for first BTC tick...")
        btc_price = await self.feed.wait_for_price(timeout=30.0)
        log.info("BTC price: $%.2f", btc_price)

        balance_dollars = await self.kalshi.get_balance()
        self._starting_balance_cents = int(balance_dollars * 100)
        log.info("Kalshi balance: $%.2f", balance_dollars)

        await self.db.log_event("startup", f"printer-v2 started [{settings.env}] BTC=${btc_price:,.2f}")
        await self.telegram.send_startup()
        self._running = True

    async def stop(self, reason: str = "clean exit") -> None:
        log.info("Shutting down: %s", reason)
        self._running = False
        await self.db.log_event("shutdown", reason)
        await self.telegram.send_kill_switch()
        await self.feed.stop()
        await self.telegram.stop()
        await self.kalshi.close()
        await self.db.close()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.start()

        monitor_task = asyncio.create_task(
            self._position_monitor_loop(), name="position-monitor"
        )

        try:
            while self._running:
                await sleep_until_next_tick()
                if self._running:
                    await self._run_tick()
        except asyncio.CancelledError:
            pass
        finally:
            monitor_task.cancel()
            await self.stop()

    # ------------------------------------------------------------------
    # 15-minute tick
    # ------------------------------------------------------------------

    async def _run_tick(self) -> None:
        log.info("--- tick ---")

        try:
            btc_price = self.feed.get_current_price()
        except StaleDataError as exc:
            log.warning("BTC feed stale — skipping tick: %s", exc)
            await self.telegram.send_error(str(exc), "stale_feed")
            return

        # 1. Find the nearest active BTC 15m market on Kalshi
        try:
            markets = await self.kalshi.get_btc_15m_markets()
            if not markets:
                log.warning("No active BTC 15m markets found on Kalshi")
                return
            market = markets[0]
            ticker = market["ticker"]
            log.info("Target market: %s  close=%s", ticker, market.get("close_time", "?"))
        except KalshiAuthError as exc:
            log.error("Kalshi auth failed — check API key: %s", exc)
            await self.telegram.send_error(str(exc), "kalshi_auth")
            return
        except Exception as exc:
            log.error("Failed to fetch Kalshi markets: %s", exc)
            await self.telegram.send_error(str(exc), "find_market")
            return

        # 2. Fetch orderbook early (needed for imbalance calc + order sizing later)
        try:
            orderbook = await self.kalshi.get_order_book(ticker)
        except KalshiMarketClosedError as exc:
            log.warning("Market %s closed before ensemble: %s", ticker, exc)
            return
        except Exception as exc:
            log.error("Failed to fetch orderbook for %s: %s", ticker, exc)
            return

        # 3. Build ensemble inputs
        bid_vol   = sum(l["size"] for l in orderbook.get("yes_bids", []))
        ask_vol   = sum(l["size"] for l in orderbook.get("yes_asks", []))
        imbalance = bid_vol / (ask_vol + 1e-9)

        try:
            close_dt = datetime.fromisoformat(
                market.get("close_time", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.error("Invalid close_time for market %s — skipping tick", ticker)
            return

        btc_data = BtcData(
            price     = btc_price,
            momentum  = self.feed.get_momentum(),
            candles   = self.feed.get_ohlcv(4),
            imbalance = imbalance,
        )
        market_obj = Market(
            ticker       = ticker,
            yes_price    = market.get("yes_ask", 50),
            no_price     = market.get("no_ask", 50),
            strike_price = float(market.get("strike_price") or 0),
            close_time   = close_dt,
        )

        # 4. Run ensemble
        try:
            signal = await self.ensemble.debate(btc_data, market_obj)
        except RuntimeError as exc:
            log.error("Ensemble failed (too few models responded): %s", exc)
            await self.telegram.send_error(str(exc), "ensemble_debate")
            return

        log.info(
            "Signal: %s  conf=%.3f  prob=%.3f  action=%s",
            signal.direction, signal.confidence, signal.raw_prob, signal.action,
        )

        # 5. Log ensemble decision
        await self.db.log_ensemble(
            ticker,
            consensus_prob = signal.raw_prob,
            model_spread   = signal.spread if signal.models else None,
            confidence     = signal.confidence,
            action         = signal.action,
            skip_reason    = signal.skip_reason,
        )

        # 6b. Build bot state for risk gates
        try:
            balance_dollars = await self.kalshi.get_balance()
            balance_cents   = int(balance_dollars * 100)
        except Exception as exc:
            log.error("Failed to fetch account state: %s", exc)
            await self.telegram.send_error(str(exc), "account_fetch")
            return

        open_trades = await self.db.get_open_trades()
        open_exposure_cents = sum(
            int(t.entry_price * t.contracts) for t in open_trades
        )
        today_stats = await self.db.get_daily_stats()

        state = BotState(
            balance_cents=balance_cents,
            starting_balance_cents=self._starting_balance_cents,
            daily_pnl_cents=int(today_stats.total_pnl * 100),
            open_exposure_cents=open_exposure_cents,
            last_trade_ts=self._last_trade_ts,
            candles_1h=self.feed.get_candles_1h(),
        )

        # 7. Risk gates
        gate_result = self.risk.check_all(signal, state)
        if not gate_result.passed:
            await self.db.log_event(
                "error",
                f"Gate {gate_result.failed_gate} [{gate_result.gate_name}] blocked: {gate_result.reason}",
            )

        if not gate_result.passed:
            if gate_result.failed_gate in (1, 2):
                await self.telegram.send_error(
                    gate_result.reason,
                    f"Gate {gate_result.failed_gate} [{gate_result.gate_name}]",
                )
            return

        # 8b. Build order
        order_params = self.strategy.build_order(
            signal, ticker, orderbook, balance_cents
        )
        if order_params is None:
            log.info("No order produced (signal too weak or size too small)")
            return

        # 8. Place order
        try:
            order = await self.kalshi.place_order(
                ticker=order_params.ticker,
                side=order_params.side,
                price=order_params.price_cents,
                count=order_params.contracts,
            )
            log.info("Order placed: %s  status=%s", order["order_id"], order["status"])
        except KalshiInsufficientFundsError as exc:
            log.warning("Insufficient funds — skipping trade: %s", exc)
            return
        except KalshiMarketClosedError as exc:
            log.warning("Market closed before order could be placed: %s", exc)
            return
        except Exception as exc:
            log.error("Order placement failed: %s", exc)
            await self.telegram.send_error(str(exc), "place_order")
            return

        # 9. Log trade + alert
        spread = (
            max(signal.models, key=lambda m: m.prob).prob
            - min(signal.models, key=lambda m: m.prob).prob
            if signal.models else None
        )
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        trade_id = await self.db.log_trade(
            market_ticker=ticker,
            direction=order_params.side.upper(),
            entry_price=order_params.price_cents,
            size_dollars=order_params.dollar_size,
            contracts=order_params.contracts,
            edge=signal.confidence,
            ensemble_confidence=signal.confidence,
            model_spread=spread,
            btc_price_at_entry=btc_price,
            timestamp=ts,
        )
        self._last_trade_ts = time.time()

        # Build a TradeRow for the alert without an extra DB roundtrip
        trade_row = TradeRow(
            id=trade_id, timestamp=ts,
            market_ticker=ticker, direction=order_params.side.upper(),
            entry_price=order_params.price_cents, size_dollars=order_params.dollar_size,
            contracts=order_params.contracts, kelly_fraction=None,
            edge=signal.confidence, ensemble_confidence=signal.confidence,
            model_spread=spread, btc_price_at_entry=btc_price,
            btc_momentum=self.feed.get_momentum(), status="open",
            exit_price=None, exit_reason=None, pnl_dollars=None, closed_at=None,
        )
        await self.telegram.send_trade_entry(trade_row)

    # ------------------------------------------------------------------
    # Position monitor (runs every 60s, independent of the 15m tick)
    # ------------------------------------------------------------------

    async def _position_monitor_loop(self) -> None:
        while self._running:
            await asyncio.sleep(EXIT_CHECK_SEC)
            if not self._running:
                return
            try:
                await self._check_exit_conditions()
            except Exception as exc:
                log.error("Position monitor error: %s", exc)

    async def _check_exit_conditions(self) -> None:
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            # Get current market price from the orderbook
            try:
                ob = await self.kalshi.get_order_book(trade.market_ticker)
            except KalshiMarketClosedError:
                # Market resolved — treat as expiry
                log.info("Trade %d: market %s resolved", trade.id, trade.market_ticker)
                await self._close_trade(trade, trade.entry_price, "expired")
                continue
            except Exception as exc:
                log.warning("Could not fetch orderbook for %s: %s", trade.market_ticker, exc)
                continue

            # Best ask on our side = current fair value to exit at
            if trade.direction == "YES":
                levels = ob.get("yes_asks") or ob.get("yes_bids") or []
            else:
                levels = ob.get("no_asks") or ob.get("no_bids") or []

            current_price = levels[0]["price"] if levels else trade.entry_price

            tp = trade.entry_price * 1.5
            sl = trade.entry_price * (1 - settings.STOP_LOSS_PCT)

            if current_price >= tp:
                exit_reason = "manual"
            elif current_price <= sl:
                exit_reason = "stop_loss"
            else:
                exit_reason = None

            if exit_reason:
                await self._close_trade(trade, current_price, exit_reason)

    async def _close_trade(self, trade, exit_price: float, exit_reason: str) -> None:
        pnl = (exit_price - trade.entry_price) * trade.contracts / 100
        closed_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        await self.db.update_trade(
            trade.id,
            status="closed",
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_dollars=pnl,
            closed_at=closed_ts,
        )
        await self.db.update_daily_stats()
        self.strategy.record_trade_outcome(
            int(trade.entry_price), int(exit_price), trade.contracts
        )
        self.ensemble.record_outcome(trade.entry_price, exit_price)
        closed_trade = TradeRow(
            **{**trade.__dict__,
               "exit_price": exit_price,
               "pnl_dollars": pnl,
               "closed_at": closed_ts}
        )
        await self.telegram.send_trade_exit(closed_trade, exit_reason)
        log.info("Trade %d closed  reason=%s  P&L=$%.2f", trade.id, exit_reason, pnl)

    async def _expire_trade(self, trade) -> None:
        await self.db.update_trade(
            trade.id,
            status="expired",
            exit_price=trade.entry_price,
            exit_reason="expired",
            pnl_dollars=0.0,
        )
        log.info("Trade %d marked expired", trade.id)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    bot = BotRunner()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop("signal")))

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())

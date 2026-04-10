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
from database import Database
from coinbase_feed import CoinbaseFeed
from ensemble import EnsembleEngine
from risk_gates import RiskGates, BotState
from strategy import Strategy
from kalshi_client import KalshiClient
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
        self.ensemble = EnsembleEngine(
            weights=settings.ensemble_weights,
            confidence_min=settings.MIN_CONFIDENCE,
        )
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
        self.kalshi  = KalshiClient(
            settings.KALSHI_API_KEY,
            settings.private_key_path,
            settings.KALSHI_BASE_URL,
        )
        self.telegram = TelegramAlerter(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)

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

        balance = await self.kalshi.get_balance()
        self._starting_balance_cents = balance.available_balance
        log.info("Kalshi balance: $%.2f", balance.available_balance / 100)

        await self.db.log_event("startup", f"printer-v2 started [{settings.env}] BTC=${btc_price:,.2f}")
        await self.telegram.alert_startup(settings.env, btc_price)
        self._running = True

    async def stop(self, reason: str = "clean exit") -> None:
        log.info("Shutting down: %s", reason)
        self._running = False
        await self.db.log_event("shutdown", reason)
        await self.telegram.alert_shutdown(reason)
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

        if self.feed.is_stale(max_age_sec=30):
            log.warning("BTC feed is stale — skipping tick")
            await self.telegram.alert_error("tick", "BTC feed stale — skipped tick")
            return

        # 1. Find the active BTC market on Kalshi
        try:
            market = await self.kalshi.find_btc_market()
            if not market:
                log.warning("No active BTC market found on Kalshi")
                return
            ticker = market["ticker"]
        except Exception as exc:
            log.error("Failed to fetch Kalshi market: %s", exc)
            await self.telegram.alert_error("find_market", str(exc))
            return

        # 2. Run ensemble
        candles_15m = self.feed.get_candles_15m()
        signal = self.ensemble.predict(candles_15m)
        log.info("Signal: %s  conf=%.3f  prob=%.3f", signal.direction, signal.confidence, signal.raw_prob)

        # 3. Log ensemble decision
        await self.db.log_ensemble(
            ticker,
            consensus_prob=signal.raw_prob,
            model_spread=max(signal.models, key=lambda m: m.prob).prob
                         - min(signal.models, key=lambda m: m.prob).prob
                         if signal.models else None,
            confidence=signal.confidence,
            action="TRADE" if signal.direction in ("yes", "no") else "SKIP",
            skip_reason=None if signal.direction in ("yes", "no") else "flat_signal",
        )

        # 4. Build bot state for risk gates
        try:
            balance = await self.kalshi.get_balance()
            positions = await self.kalshi.get_positions()
        except Exception as exc:
            log.error("Failed to fetch account state: %s", exc)
            await self.telegram.alert_error("account_fetch", str(exc))
            return

        open_trades = await self.db.get_open_trades()
        open_exposure_cents = sum(
            int(t.entry_price * t.contracts) for t in open_trades
        )
        today_stats = await self.db.get_daily_stats()

        state = BotState(
            balance_cents=balance.available_balance,
            starting_balance_cents=self._starting_balance_cents,
            daily_pnl_cents=int(today_stats.total_pnl * 100),
            open_exposure_cents=open_exposure_cents,
            last_trade_ts=self._last_trade_ts,
            candles_1h=self.feed.get_candles_1h(),
        )

        # 5. Risk gates
        gate_result = self.risk.check_all(signal, state)
        if not gate_result.passed:
            await self.db.log_event(
                "error",
                f"Gate {gate_result.failed_gate} [{gate_result.gate_name}] blocked: {gate_result.reason}",
            )

        if not gate_result.passed:
            if gate_result.failed_gate in (1, 2):
                await self.telegram.alert_gate_blocked(
                    gate_result.failed_gate, gate_result.gate_name, gate_result.reason
                )
            return

        # 6. Build order
        try:
            orderbook = await self.kalshi.get_orderbook(ticker, depth=5)
        except Exception as exc:
            log.error("Failed to fetch orderbook for %s: %s", ticker, exc)
            return

        order_params = self.strategy.build_order(
            signal, ticker, orderbook, balance.available_balance
        )
        if order_params is None:
            log.info("No order produced (signal too weak or size too small)")
            return

        # 7. Place order
        try:
            order = await self.kalshi.place_order(
                ticker=order_params.ticker,
                side=order_params.side,
                contracts=order_params.contracts,
                price=order_params.price_cents,
            )
            log.info("Order placed: %s", order.order_id)
        except Exception as exc:
            log.error("Order placement failed: %s", exc)
            await self.telegram.alert_error("place_order", str(exc))
            return

        # 8. Log trade + alert
        trade_id = await self.db.log_trade(
            market_ticker=ticker,
            direction=order_params.side.upper(),
            entry_price=order_params.price_cents,
            size_dollars=order_params.dollar_size,
            contracts=order_params.contracts,
            edge=signal.confidence,
            ensemble_confidence=signal.confidence,
            model_spread=max(signal.models, key=lambda m: m.prob).prob
                         - min(signal.models, key=lambda m: m.prob).prob
                         if signal.models else None,
            btc_price_at_entry=self.feed.get_price(),
        )
        self._last_trade_ts = time.time()

        await self.telegram.alert_trade_open(
            ticker=ticker,
            side=order_params.side,
            contracts=order_params.contracts,
            price_cents=order_params.price_cents,
            dollar_size=order_params.dollar_size,
            confidence=signal.confidence,
        )

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
            try:
                order = await self.kalshi.get_order(trade.kalshi_order_id)
            except Exception as exc:
                log.warning("Could not fetch order %s: %s", trade.kalshi_order_id, exc)
                continue

            # If order fully filled, get current market price for TP/SL check
            if order.status in ("filled", "partially_filled"):
                try:
                    ob = await self.kalshi.get_orderbook(trade.market_ticker, depth=1)
                except Exception:
                    continue

                current_price = (
                    ob.best_yes_ask() if trade.direction == "YES" else ob.best_no_ask()
                ) or trade.entry_price

                tp = trade.entry_price * 1.5
                sl = trade.entry_price * (1 - settings.STOP_LOSS_PCT)

                if current_price >= tp:
                    exit_reason = "manual"   # TP hit
                elif current_price <= sl:
                    exit_reason = "stop_loss"
                elif order.status == "filled":
                    exit_reason = "expired"
                else:
                    exit_reason = None

                if exit_reason:
                    pnl = (current_price - trade.entry_price) * trade.contracts / 100
                    await self.db.update_trade(
                        trade.id,
                        status="closed",
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl_dollars=pnl,
                    )
                    await self.db.update_daily_stats()
                    self.strategy.record_trade_outcome(
                        int(trade.entry_price), int(current_price), trade.contracts
                    )
                    self.ensemble.record_outcome(trade.entry_price, current_price)
                    await self.telegram.alert_trade_close(
                        ticker=trade.market_ticker,
                        side=trade.direction,
                        contracts=trade.contracts,
                        entry_cents=int(trade.entry_price),
                        exit_cents=int(current_price),
                        pnl=pnl,
                    )
                    log.info("Trade %d closed  reason=%s  P&L=$%.2f", trade.id, exit_reason, pnl)

            elif order.status == "canceled":
                await self.db.update_trade(
                    trade.id,
                    status="expired",
                    exit_price=trade.entry_price,
                    exit_reason="expired",
                    pnl_dollars=0.0,
                )


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

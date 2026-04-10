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

from config import cfg
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
        self.db      = Database(cfg.database_path)
        self.feed    = CoinbaseFeed(cfg.coinbase.api_key, cfg.coinbase.api_secret)
        self.ensemble = EnsembleEngine(
            weights={
                "trend":    cfg.ensemble.weight_trend,
                "mean_rev": cfg.ensemble.weight_mean_rev,
                "momentum": cfg.ensemble.weight_momentum,
                "vol":      cfg.ensemble.weight_vol,
            },
            confidence_min=cfg.ensemble.confidence_min,
        )
        self.risk = RiskGates(
            max_daily_drawdown_pct=cfg.risk.max_daily_drawdown_pct,
            max_position_exposure=cfg.risk.max_position_exposure,
            confidence_min=cfg.ensemble.confidence_min,
            max_btc_vol_threshold=cfg.risk.max_btc_vol_threshold,
            min_minutes_between_trades=cfg.risk.min_minutes_between_trades,
        )
        self.strategy = Strategy(
            kelly_fraction_multiplier=cfg.kelly.fraction,
            max_position_pct=cfg.kelly.max_position_pct,
        )
        self.kalshi  = KalshiClient(
            cfg.kalshi.api_key_id,
            cfg.kalshi.private_key_path,
            cfg.kalshi.base_url,
        )
        self.telegram = TelegramAlerter(cfg.telegram.token, cfg.telegram.chat_id)

        self._last_trade_ts: float = 0.0
        self._starting_balance_cents: int = 0
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        log.info("=== printer-v2 starting [%s] ===", cfg.env)
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

        await self.telegram.alert_startup(cfg.env, btc_price)
        self._running = True

    async def stop(self, reason: str = "clean exit") -> None:
        log.info("Shutting down: %s", reason)
        self._running = False
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

        # 3. Log signal
        signal_id = await self.db.log_signal(
            direction=signal.direction,
            confidence=signal.confidence,
            weights=signal.weights,
            btc_price=self.feed.get_price(),
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
        today_pnl = await self.db.get_today_pnl()

        state = BotState(
            balance_cents=balance.available_balance,
            starting_balance_cents=self._starting_balance_cents,
            daily_pnl_cents=int(today_pnl.net_pnl * 100),
            open_exposure_cents=open_exposure_cents,
            last_trade_ts=self._last_trade_ts,
            candles_1h=self.feed.get_candles_1h(),
        )

        # 5. Risk gates
        gate_result = self.risk.check_all(signal, state)
        await self.db.log_gate_event(
            gate_num=gate_result.failed_gate or 0,
            gate_name=gate_result.gate_name or "all",
            passed=gate_result.passed,
            reason=gate_result.reason,
            signal_id=signal_id,
        )

        if not gate_result.passed:
            if gate_result.failed_gate in (1, 2):
                # Critical gates — notify
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
            kalshi_order_id=order.order_id,
            market_ticker=ticker,
            direction=order_params.side,
            contracts=order_params.contracts,
            entry_price=order_params.price_cents,
            dollar_size=order_params.dollar_size,
            signal_id=signal_id,
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
                    ob.best_yes_ask() if trade.direction == "yes" else ob.best_no_ask()
                ) or trade.entry_price

                tp = trade.entry_price * 1.5   # simplified: +50% on entry price
                sl = trade.entry_price * 0.5   # simplified: -50% of entry price

                should_close = (
                    current_price >= tp or
                    current_price <= sl or
                    order.status == "filled"   # market already resolved
                )

                if should_close:
                    pnl = (current_price - trade.entry_price) * trade.contracts / 100
                    await self.db.update_trade(
                        trade.id,
                        close_price=current_price,
                        pnl=pnl,
                        status="closed",
                    )
                    await self.db.upsert_daily_pnl()
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
                    log.info("Trade %d closed  P&L=$%.2f", trade.id, pnl)

            elif order.status == "canceled":
                await self.db.update_trade(
                    trade.id, close_price=trade.entry_price, pnl=0.0, status="cancelled"
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

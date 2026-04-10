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
from coinbase_feed import CoinbaseFeed, StaleDataError
from ensemble import EnsembleEngine, BtcData, Market
from risk_gates import RiskGates
from strategy import Strategy
from kalshi_client import (
    KalshiClient,
    KalshiAuthError,
    KalshiMarketClosedError,
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

        self.db       = Database(Path(settings.DB_PATH))
        self.feed     = CoinbaseFeed()
        self.kalshi   = KalshiClient()
        self.ensemble = EnsembleEngine()
        self.risk = RiskGates(
            kalshi_client = self.kalshi,
            coinbase_feed = self.feed,
            database      = self.db,
        )
        self.telegram = TelegramAlerter()
        self.strategy = Strategy(
            kalshi_client = self.kalshi,
            database      = self.db,
            telegram      = self.telegram,
        )

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

        # 7. Risk gates
        gate_result = await self.risk.check_all(market, signal, settings.MAX_BET_SIZE)
        if not gate_result.passed:
            await self.db.log_event(
                "error",
                f"Gate [{gate_result.failed_gate}] blocked: {gate_result.reason}",
            )
            # Alert on critical gates (edge/liquidity/drawdown); not on conf/staleness noise
            if gate_result.failed_gate in ("edge", "liquidity", "drawdown"):
                await self.telegram.send_error(
                    gate_result.reason,
                    f"Gate [{gate_result.failed_gate}]",
                )
            return

        # 8b. Check position limit
        if not await self.strategy.can_open_position():
            return

        # 9. Enter trade (sizing, order placement, DB log, Telegram alert)
        trade = await self.strategy.enter_trade(
            market,
            signal,
            gate_result,
            btc_price    = btc_price,
            btc_momentum = self.feed.get_momentum(),
        )
        if trade:
            self._last_trade_ts = time.time()

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
        await self.strategy.check_exits(open_trades)



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

"""
runner.py — Main bot orchestrator

TradingBot owns the full lifecycle: startup, 15-minute cycle loop,
multi-market scanning, and graceful shutdown via STOP file.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from database import Database
from coinbase_feed import CoinbaseFeed
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

_CYCLE_SECONDS  = 15 * 60     # 15-minute interval
_CYCLE_BUFFER   = 10           # seconds after boundary before first tick
_MAX_MARKETS    = 3            # top N markets evaluated per cycle
_STOP_FILE      = Path("STOP")


# ---------------------------------------------------------------------------
# TradingBot
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self) -> None:
        settings.validate()

        self.db       = Database(Path(settings.DB_PATH))
        self.feed     = CoinbaseFeed()
        self.kalshi   = KalshiClient()
        self.ensemble = EnsembleEngine()
        self.risk     = RiskGates(
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

        # Tickers the ensemble returned WAIT on — re-evaluate next cycle
        self._wait_list: dict[str, float] = {}
        # Current UTC date string — used to detect midnight for daily summary
        self._last_day: str = ""
        # Streak protection: track last 5 trade outcomes (True=win)
        self._recent_results: list[bool] = []
        self._pause_cycles: int = 0        # cycles to skip after 5-loss streak

    # ------------------------------------------------------------------
    # Startup sequence
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        1. Config already validated in __init__
        2. Init database (creates tables)
        3. Connect Coinbase WebSocket, wait for first price
        4. Verify Kalshi connection (get balance)
        5. Start Telegram, send startup message
        6. Log startup event to database
        7. Check for STOP file — exit immediately if found
        """
        print("BOT STARTING - ENV CHECK")
        print(f"KALSHI_KEY SET: {bool(os.getenv('KALSHI_API_KEY'))}")
        print(f"KALSHI_KEY PREFIX: {os.getenv('KALSHI_API_KEY', '')[:8]}...")
        print(f"TELEGRAM SET: {bool(os.getenv('TELEGRAM_BOT_TOKEN'))}")
        _pkey_raw = os.getenv('KALSHI_PRIVATE_KEY', '')
        print(f"PRIVATE_KEY SET: {bool(_pkey_raw)}")
        print(f"PRIVATE_KEY LENGTH: {len(_pkey_raw)} chars")
        print(f"PRIVATE_KEY NEWLINES: {_pkey_raw.count(chr(10))} real, {_pkey_raw.count(chr(92)+'n')} literal \\n")

        log.info("=== printer-v2 starting [%s] ===", settings.env)

        # 2. Database
        await self.db.connect()

        # 3. Coinbase feed
        await self.feed.start()
        log.info("Waiting for first BTC tick (timeout=30s)...")
        btc_price = await self.feed.wait_for_price(timeout=30.0)
        log.info("BTC price: $%.2f", btc_price)

        # 4. Kalshi
        balance: float | None = None
        try:
            balance = await self.kalshi.get_balance()
            await self.db.set_balance(balance)
            print(f"Balance: ${balance:.2f}")
            log.info("Kalshi balance: $%.2f  mode: %s", balance, settings.env.upper())
        except KalshiAuthError as exc:
            print(f"[AUTH ERROR] {exc}")
            print(">>> ACTION REQUIRED: Kalshi credentials are invalid.")
            print(">>> 1. Log in to https://app.kalshi.com")
            print(">>> 2. Account → API → delete old key → Create new RSA key")
            print(">>> 3. Update KALSHI_API_KEY and KALSHI_PRIVATE_KEY in Railway Variables")
            log.error("Kalshi auth failed at startup (bot will still try to run): %s", exc)

        # 5. Telegram (start but don't send startup — notification fires from dashboard START button)
        await self.telegram.start()

        # 6. Database event
        balance_str = f"${balance:.2f}" if balance is not None else "auth_failed"
        await self.db.log_event(
            "startup",
            f"printer-v2 started [{settings.env}] "
            f"BTC=${btc_price:,.2f}  balance={balance_str}",
        )

        # 7. STOP file guard
        if _STOP_FILE.exists():
            log.warning("STOP file found — exiting immediately")
            await self.stop("STOP file present at startup")
            sys.exit(0)

        # Initialise day tracker so we don't send a summary on first cycle
        self._last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log.info("Startup complete — entering main loop")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self, reason: str = "clean exit") -> None:
        log.info("Shutting down: %s", reason)
        try:
            await self.db.log_event("shutdown", reason)
        except Exception:
            pass
        try:
            await self.telegram.send_kill_switch()
        except Exception:
            pass
        await self.feed.stop()
        await self.telegram.stop()
        await self.kalshi.close()
        await self.db.close()

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.start()
        await self.run_loop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """
        Runs forever until a STOP file appears or the process is killed.
        Catches all unexpected exceptions via handle_crash() which sleeps
        30 s and returns — the while-True then naturally restarts.
        """
        while True:
            try:
                # Kill switch
                if _STOP_FILE.exists():
                    log.warning("STOP file detected — halting")
                    await self.telegram.send_kill_switch()
                    await self.db.log_event("shutdown", "STOP file detected")
                    break

                # Trading cycle
                await self.run_cycle()

                # Daily summary at UTC midnight
                if self._is_new_day():
                    await self._send_daily_summary()

                # Sleep to next 15m boundary + buffer
                await self._sleep_until_next_cycle()

            except asyncio.CancelledError:
                raise   # don't swallow task cancellation

            except Exception as exc:
                await self.handle_crash(exc)

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> None:
        """
        One complete 15-minute evaluation pass:
          1. Check exits on existing positions
          2. Bail early if at position limit
          3. Collect BTC data
          4. Fetch active Kalshi markets (top 3 by volume)
          5. For each: ensemble → gates → momentum check → enter
        """
        cycle_start = datetime.now(timezone.utc)
        Path("heartbeat.txt").write_text(cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(f"=== CYCLE START === {cycle_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log.info("--- Cycle %s ---", cycle_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # Step 1 — exits (update streak from closed trades)
        open_trades = await self.db.get_open_trades()
        if open_trades:
            closed = await self.strategy.check_exits(open_trades)
            for t in closed:
                if t.pnl_dollars is not None:
                    self._recent_results.append(t.pnl_dollars > 0)
                    if len(self._recent_results) > 5:
                        self._recent_results.pop(0)
                    if len(self._recent_results) == 5 and not any(self._recent_results):
                        self._pause_cycles = 2
                        log.warning("5 loss streak — pausing 2 cycles")
                        await self.telegram.send_error(
                            "5 consecutive losses detected — pausing trading for 2 cycles",
                            "Loss streak protection",
                        )
            open_trades = await self.db.get_open_trades()

        # Streak hard pause
        if self._pause_cycles > 0:
            self._pause_cycles -= 1
            log.info("Loss streak pause: %d cycles remaining", self._pause_cycles + 1)
            return

        # Step 2 — position limit
        if not await self.strategy.can_open_position():
            log.info("Max positions open — skipping entry scan")
            return

        # Step 3 — BTC data
        if self.feed.is_stale():
            log.warning("BTC price data stale — skipping cycle")
            return

        btc_price = self.feed.get_current_price()
        momentum  = self.feed.get_momentum()
        ohlcv     = self.feed.get_ohlcv(4)
        log.info("BTC=$%.2f  momentum=%.3f  candles=%d", btc_price, momentum, len(ohlcv))

        # Balance — fetch once per cycle, persist to DB for dashboard
        balance = None
        try:
            balance = await self.kalshi.get_balance()
            await self.db.set_balance(balance)
        except Exception as exc:
            log.warning("Balance fetch failed: %s", exc)

        # Step 4 — markets
        try:
            markets = await self.kalshi.get_btc_15m_markets()
        except KalshiAuthError as exc:
            log.error("Kalshi auth error: %s", exc)
            await self.telegram.send_error(str(exc), "kalshi_auth")
            return
        except Exception as exc:
            log.error("Failed to fetch Kalshi markets: %s", exc)
            return

        print(f"BTC Price: ${btc_price:,.2f}")
        print(f"Balance: ${balance:.2f}" if balance is not None else "Balance: unavailable")
        print(f"Open positions: {len(open_trades)}")
        print(f"Markets found: {len(markets)}")

        if not markets:
            log.info("No active 15m BTC markets found")
            return

        # Step 5 — sort by volume, cap at 3
        markets = sorted(markets, key=lambda m: m.get("volume", 0), reverse=True)[:_MAX_MARKETS]
        log.info("Evaluating %d market(s): %s", len(markets), [m["ticker"] for m in markets])

        # Persist market list for dashboard cycle-watch box
        try:
            await self.db.set_market_watch({
                "cycle_ts":   cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "btc_price":  btc_price,
                "markets": [
                    {
                        "ticker":     m["ticker"],
                        "strike":     float(m.get("strike_price") or 0),
                        "close_time": m.get("close_time", ""),
                        "yes_ask":    m.get("yes_ask", 0),
                        "no_ask":     m.get("no_ask", 0),
                        "volume":     m.get("volume", 0),
                    }
                    for m in markets
                ],
                "last_signal": None,
            })
        except Exception as exc:
            log.debug("market_watch store failed: %s", exc)

        # Quick-lookup set of tickers we already hold
        open_tickers = {t.market_ticker for t in open_trades}

        # Step 6 — waited markets get re-evaluated FIRST, then new markets
        market_by_ticker = {m["ticker"]: m for m in markets}
        waited_tickers   = set(self._wait_list.keys())

        # Clear waited tickers that no longer appear in the market list (expired)
        for ticker in waited_tickers - market_by_ticker.keys():
            log.info("Waited market %s no longer available — clearing", ticker)
            self._wait_list.pop(ticker, None)

        # ── Time-based size multiplier (Upgrade 7) ───────────────────────────
        hour = cycle_start.hour
        if settings.RESPECT_TIME_FILTERS:
            if 13 <= hour < 21:          # US hours — full size
                time_mult = 1.0
            elif 0 <= hour < 4:          # Asia hours — 75%
                time_mult = 0.75
            else:                        # dead zone — 50%
                time_mult = 0.50
        else:
            time_mult = 1.0

        # ── Streak soft multiplier (Upgrade 6) ────────────────────────────────
        recent_losses = sum(1 for w in self._recent_results[-3:] if not w)
        if recent_losses >= 3:
            streak_mult = 0.25
            log.warning("Loss streak protection: bet size at 25%%")
        else:
            streak_mult = 1.0

        size_mult = time_mult * streak_mult

        # Re-evaluate waited markets
        for ticker in list(waited_tickers & market_by_ticker.keys()):
            self._wait_list.pop(ticker, None)
            if ticker in open_tickers:
                continue
            log.info("Re-evaluating waited market %s", ticker)
            await self._evaluate_market(
                market_by_ticker[ticker], btc_price, momentum, ohlcv,
                open_tickers, size_mult,
            )

        # Evaluate remaining new markets
        for market in markets:
            ticker = market["ticker"]
            if ticker in open_tickers or ticker in waited_tickers:
                continue
            await self._evaluate_market(
                market, btc_price, momentum, ohlcv, open_tickers, size_mult,
            )

    # ------------------------------------------------------------------
    # Per-market evaluation
    # ------------------------------------------------------------------

    async def _evaluate_market(
        self,
        market:       dict,
        btc_price:    float,
        momentum:     float,
        ohlcv:        list,
        open_tickers: set[str],
        size_mult:    float = 1.0,
    ) -> None:
        ticker = market["ticker"]

        # Orderbook for bid/ask imbalance (reused in liquidity gate)
        try:
            ob = await self.kalshi.get_order_book(ticker)
        except KalshiMarketClosedError:
            log.debug("Market %s already closed — skipping", ticker)
            return
        except Exception as exc:
            log.warning("Orderbook fetch failed for %s: %s", ticker, exc)
            return

        bid_vol   = sum(l["size"] for l in ob.get("yes_bids", []))
        ask_vol   = sum(l["size"] for l in ob.get("yes_asks", []))
        imbalance = bid_vol / (ask_vol + 1e-9)

        try:
            close_dt = datetime.fromisoformat(
                market.get("close_time", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.warning("Invalid close_time for %s — skipping", ticker)
            return

        btc_data   = BtcData(
            price     = btc_price,
            momentum  = momentum,
            candles   = ohlcv,
            imbalance = imbalance,
        )
        market_obj = Market(
            ticker       = ticker,
            yes_price    = market.get("yes_ask", 50),
            no_price     = market.get("no_ask", 50),
            strike_price = float(market.get("strike_price") or 0),
            close_time   = close_dt,
        )

        # --- Ensemble debate ---
        try:
            result = await self.ensemble.debate(btc_data, market_obj)
        except RuntimeError as exc:
            log.error("Ensemble failed for %s (too few models): %s", ticker, exc)
            return
        except Exception as exc:
            log.error("Unexpected ensemble error for %s: %s", ticker, exc)
            return

        log.info(
            "Ensemble [%s]: prob=%.3f conf=%.3f spread=%.3f action=%s",
            ticker, result.raw_prob, result.confidence, result.spread, result.action,
        )

        # Log every ensemble result regardless of action
        await self.db.log_ensemble(
            ticker,
            consensus_prob = result.raw_prob,
            model_spread   = result.spread if result.models else None,
            confidence     = result.confidence,
            action         = result.action,
            skip_reason    = result.skip_reason,
        )

        # --- Build trade-entry checklist (shown on dashboard) ---
        checks: list[dict] = []

        # Check 1: Direction (always has one based on consensus_prob)
        dir_label = "YES" if result.consensus_prob > 0.5 else "NO"
        checks.append({
            "id": "signal", "label": "Direction",
            "passed": True,
            "detail": f"{dir_label} · {result.consensus_prob*100:.0f}%",
        })

        # Check 2: Models agree (action != WAIT means spread is within MAX_MODEL_SPREAD)
        spread_ok = result.action != "WAIT"
        checks.append({
            "id": "spread", "label": "Models agree",
            "passed": spread_ok,
            "detail": f"spread {result.spread:.2f}",
        })

        # Check 3: Confidence (action == TRADE means >= MIN_CONFIDENCE; SKIP means failed)
        conf_ok = result.action == "TRADE"
        checks.append({
            "id": "confidence",
            "label": f"Confidence \u2265{settings.MIN_CONFIDENCE*100:.0f}%",
            "passed": conf_ok,
            "detail": f"{result.confidence*100:.0f}%",
        })

        # If WAIT or SKIP, fill remaining checks as not-evaluated and store
        _pending = [
            ("momentum",  "Momentum"),
            ("drawdown",  "Daily loss"),
            ("data_age",  "Data fresh"),
        ]
        if result.action in ("WAIT", "SKIP"):
            for cid, clabel in _pending:
                checks.append({"id": cid, "label": clabel, "passed": None, "detail": "—"})
            await self._store_last_signal(ticker, result, checks)
            if result.action == "WAIT":
                log.info("Models disagree on %s — adding to wait list (1 cycle pause)", ticker)
                self._wait_list[ticker] = time.time()
            else:
                log.info("Skipping %s: %s", ticker, result.skip_reason)
            return

        # Check 4: Momentum confirmation
        direction = result.direction   # "yes" | "no"
        if direction == "yes":
            mom_ok     = momentum >= -0.1
            mom_detail = f"{momentum:+.3f} (need \u2265 -0.10)"
        else:
            mom_ok     = momentum <= 0.1
            mom_detail = f"{momentum:+.3f} (need \u2264 +0.10)"
        checks.append({
            "id": "momentum", "label": "Momentum",
            "passed": mom_ok,
            "detail": mom_detail,
        })

        if not mom_ok:
            for cid, clabel in _pending[1:]:   # drawdown, data_age
                checks.append({"id": cid, "label": clabel, "passed": None, "detail": "—"})
            await self._store_last_signal(ticker, result, checks)
            log.info(
                "Momentum (%.2f) contradicts %s signal on %s — skipping",
                momentum, direction.upper(), ticker,
            )
            return

        # --- 5 risk gates ---
        gate_result = await self.risk.check_all(market, result, settings.MAX_BET_SIZE)

        # Checks 5-6: from gate results (edge/liquidity removed; skip internal confidence gate)
        for gate_key, chk_id, chk_label in [
            ("drawdown",  "drawdown",  "Daily loss"),
            ("staleness", "data_age",  "Data fresh"),
        ]:
            gd = gate_result.gate_details.get(gate_key)
            if gd is not None:
                checks.append({"id": chk_id, "label": chk_label,
                               "passed": gd["passed"], "detail": gd["reason"]})
            else:
                checks.append({"id": chk_id, "label": chk_label, "passed": None, "detail": "—"})

        await self._store_last_signal(ticker, result, checks)

        if not gate_result.passed:
            log.info("Gates failed on %s: %s", ticker, gate_result.reason)
            await self.db.log_event(
                "error",
                f"Gate [{gate_result.failed_gate}] blocked on {ticker}: "
                f"{gate_result.reason}",
            )
            if gate_result.failed_gate in ("edge", "liquidity", "drawdown"):
                await self.telegram.send_error(
                    gate_result.reason,
                    f"Gate [{gate_result.failed_gate}]",
                )
            return

        # --- Execute trade ---
        trade = await self.strategy.enter_trade(
            market,
            result,
            gate_result,
            btc_price        = btc_price,
            btc_momentum     = momentum,
            size_multiplier  = size_mult,
        )
        if trade:
            log.info(
                "✅ Trade entered: %s %s $%.2f",
                trade.market_ticker, trade.direction, trade.size_dollars,
            )
            open_tickers.add(ticker)   # guard against double-entry this cycle

    async def _store_last_signal(
        self, ticker: str, result: Any, checks: list[dict]
    ) -> None:
        """Persist the cycle-watch last_signal including the trade-entry checklist."""
        try:
            def _mdata(r: Any) -> dict | None:
                if r is None:
                    return None
                return {
                    "prob":      round(r.probability * 100, 1),
                    "direction": r.direction,
                    "reasoning": r.reasoning[:120],
                }

            watch = await self.db.get_market_watch() or {}
            watch["last_signal"] = {
                "ticker":      ticker,
                "direction":   result.direction.upper() if result.direction != "flat" else "FLAT",
                "action":      result.action,
                "prob":        round(result.raw_prob * 100, 1),
                "confidence":  round(result.confidence * 100, 1),
                "skip_reason": result.skip_reason,
                "checks":      checks,
                "models": {
                    "claude":   _mdata(result.claude),
                    "gpt":      _mdata(result.gpt),
                    "gemini":   _mdata(result.gemini),
                    "deepseek": _mdata(result.deepseek),
                },
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            await self.db.set_market_watch(watch)
        except Exception as exc:
            log.debug("market_watch signal update failed: %s", exc)

    # ------------------------------------------------------------------
    # Crash handler
    # ------------------------------------------------------------------

    async def handle_crash(self, error: Exception) -> None:
        """
        Log the full traceback, alert Telegram, sleep 30s.
        Returns normally so run_loop's while-True restarts the cycle.
        """
        tb_str = traceback.format_exc()
        log.error("Main loop crashed:\n%s", tb_str)

        try:
            await self.db.log_event("crash", str(error)[:500])
        except Exception:
            pass

        try:
            await self.telegram.send_error(str(error)[:350], "Main loop crashed")
        except Exception:
            pass

        log.info("Waiting 30s before restarting cycle...")
        await asyncio.sleep(30)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_new_day(self) -> bool:
        """True exactly once per UTC day (first call after midnight)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_day and today != self._last_day:
            self._last_day = today
            return True
        self._last_day = today
        return False

    async def _send_daily_summary(self) -> None:
        try:
            stats     = await self.db.get_daily_stats()
            recent    = await self.db.get_recent_trades(limit=50)
            pnls      = [t.pnl_dollars for t in recent if t.pnl_dollars is not None]
            best      = max(pnls) if pnls else None
            worst     = min(pnls) if pnls else None
            dir_stats = await self.db.get_win_rate_by_direction()
            await self.telegram.send_daily_summary(
                stats,
                best_trade   = best,
                worst_trade  = worst,
                win_rate_yes = dir_stats["YES"]["win_rate"],
                win_rate_no  = dir_stats["NO"]["win_rate"],
            )
            log.info("Daily summary sent")
        except Exception as exc:
            log.error("Daily summary failed: %s", exc)

    @staticmethod
    async def _sleep_until_next_cycle() -> None:
        """Sleep until the next 15-minute boundary plus a 10-second buffer."""
        now          = time.time()
        next_boundary = (int(now // _CYCLE_SECONDS) + 1) * _CYCLE_SECONDS
        wait          = next_boundary - now + _CYCLE_BUFFER
        log.info(
            "Next cycle in %.1fs  (at %s UTC + %ds buffer)",
            wait,
            datetime.fromtimestamp(next_boundary, tz=timezone.utc).strftime("%H:%M:%S"),
            _CYCLE_BUFFER,
        )
        await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    bot = TradingBot()

    loop = asyncio.get_running_loop()

    def _graceful_shutdown() -> None:
        log.info("Signal received — initiating graceful shutdown")
        asyncio.create_task(bot.stop("signal"))

    try:
        loop.add_signal_handler(signal.SIGINT,  _graceful_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _graceful_shutdown)
    except (NotImplementedError, OSError):
        # Windows does not support add_signal_handler for all signals
        pass

    try:
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await bot.stop("keyboard interrupt")


if __name__ == "__main__":
    asyncio.run(main())

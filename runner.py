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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
_CYCLE_BUFFER   = 60           # seconds after boundary before first tick
_MAX_MARKETS    = 10           # top N markets evaluated per cycle
_STOP_FILE      = Path("STOP")

# Per-asset bet size multipliers (applied on top of time/streak multipliers)
# BTC at 50% — fewer trades, lower % volatility vs alts; backtest showed 309 trades vs 1,250+ for alts
_ASSET_SIZE_OVERRIDES: dict[str, float] = {
    "BTC": 0.50,
}


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
        # All signals produced this cycle — accumulated for the dashboard
        self._cycle_signals: list[dict] = []
        # Trades actually placed this cycle
        self._cycle_trades_placed: int = 0
        # Markets found this cycle (non-zero = found but may have been empty)
        self._cycle_markets_found: int = 0
        # Markets that passed timing checks and were actually evaluated
        self._cycle_markets_evaluated: int = 0
        # Whether the bot-off gate fired for at least one market this cycle
        self._cycle_bot_off_hit: bool = False

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

        # Pre-populate candle history so AI models have data immediately
        log.info("Pre-fetching candle history...")
        await self.feed.prefetch_candle_history(n=10)
        log.info("Candle history loaded")

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
        # Always start in OFF mode after a deploy — user must press START
        await self.db.set_bot_enabled(False)
        log.info("Startup complete — entering main loop  [OFF MODE (press START on dashboard to enable trading)]")

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
        # Run exit monitor and main loop concurrently
        await asyncio.gather(
            self.run_loop(),
            self._exit_monitor_loop(),
        )

    # ------------------------------------------------------------------
    # Background exit monitor — checks SL/TP/decay every 60s independently
    # of the main entry cycle so positions don't wait up to 14 min for an exit
    # ------------------------------------------------------------------

    async def _exit_monitor_loop(self) -> None:
        """
        Runs forever alongside run_loop().
        Every 60 seconds: if bot is enabled and there are open trades,
        calls check_exits. This ensures SL/TP fires promptly even when
        the main cycle is sleeping between 15-minute windows.
        """
        _INTERVAL = 60
        while True:
            try:
                await asyncio.sleep(_INTERVAL)
                if not await self.db.get_bot_enabled():
                    continue
                open_trades = await self.db.get_open_trades()
                if open_trades:
                    log.info("Exit monitor: checking %d open trade(s)", len(open_trades))
                    await self.strategy.check_exits(open_trades)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Exit monitor error (non-fatal): %s", exc)

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

                # Within-window retry loop: keep re-evaluating every 30s while
                # still inside the entry window (< 660s in = 15min - 4min buffer).
                # Retry interval is capped to whatever time remains so we never
                # sleep past the cutoff.
                _MAX_TIME_IN     = 660   # stop retrying at 11 min in (last 4 min reserved)
                _RETRY_INTERVAL  = 10    # re-check every 10s within same window
                while True:
                    now_ts           = time.time()
                    boundary         = int(now_ts // _CYCLE_SECONDS) * _CYCLE_SECONDS
                    time_into_window = now_ts - boundary
                    if time_into_window >= _MAX_TIME_IN or self._cycle_trades_placed:
                        break   # past entry window, or a trade was placed
                    time_remaining = _MAX_TIME_IN - time_into_window
                    sleep_for = _RETRY_INTERVAL
                    log.info(
                        "Still in entry window (%.0fs in, %.0fs left) — re-evaluating in %.0fs",
                        time_into_window, time_remaining, sleep_for,
                    )
                    await asyncio.sleep(sleep_for)
                    await self.run_cycle()
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
        self._cycle_signals = []         # fresh slate for this cycle's dashboard signals
        self._cycle_trades_placed = 0    # actual orders placed this cycle
        self._cycle_markets_evaluated = 0  # markets that passed timing checks this cycle
        self._cycle_bot_off_hit = False  # reset bot-off gate flag
        _prev_wait_list = set(self._wait_list.keys())  # snapshot before reset
        self._wait_list.clear()          # fresh wait-list for this cycle
        Path("heartbeat.txt").write_text(cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(f"=== CYCLE START === {cycle_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log.info("--- Cycle %s ---", cycle_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # Step 1 — exits
        open_trades = await self.db.get_open_trades()
        if open_trades:
            await self.strategy.check_exits(open_trades)
            open_trades = await self.db.get_open_trades()

        # Step 2 — position limit
        if not await self.strategy.can_open_position():
            log.info("Max positions open — skipping entry scan")
            # Refresh cycle_ts in market_watch so the dashboard knows we ran a cycle
            # (without this, old signals age past ageMins threshold → "sleeping" banner)
            try:
                existing = await self.db.get_market_watch() or {}
                await self.db.set_market_watch({
                    **existing,
                    "cycle_ts":     cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "btc_price":    self.feed.get_price_for("BTC") or existing.get("btc_price"),
                    "cycle_status": "max_positions",
                    # Clear stale AI signals — no new ensemble ran this cycle
                    "signals":      [],
                    "last_signal":  None,
                })
                log.info("[DASHBOARD] market_watch refreshed (max_positions)")
            except Exception as exc:
                log.warning("market_watch refresh failed (max_positions): %s", exc)
            return

        # Step 3 — verify at least BTC feed is alive
        if self.feed.is_stale():
            log.warning("BTC price data stale — skipping cycle")
            try:
                existing = await self.db.get_market_watch() or {}
                await self.db.set_market_watch({
                    **existing,
                    "cycle_ts":    cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status":      "stale_feed",
                    # Clear stale AI signals — no new ensemble ran this cycle
                    "signals":     [],
                    "last_signal": None,
                })
            except Exception:
                pass
            return

        btc_price = self.feed.get_current_price()
        log.info("BTC=$%.2f", btc_price)

        # Balance — fetch once per cycle, persist to DB for dashboard
        balance = None
        try:
            balance = await self.kalshi.get_balance()
            await self.db.set_balance(balance)
        except Exception as exc:
            log.warning("Balance fetch failed: %s", exc)

        print(f"BTC Price: ${btc_price:,.2f}")
        print(f"Balance: ${balance:.2f}" if balance is not None else "Balance: unavailable")
        print(f"Open positions: {len(open_trades)}")

        size_mult = 1.0

        # Quick-lookup set of tickers we already hold
        open_tickers = {t.market_ticker for t in open_trades}

        # Step 4/5/6 — scan EVERY supported asset, evaluate its markets
        supported_assets: list[str] = settings.supported_assets_list
        all_markets_found: list[dict] = []

        for asset in supported_assets:
            # Skip assets whose feed is stale (but don't abort the whole cycle)
            if self.feed.is_stale_for(asset):
                log.debug("%s price stale — skipping this asset", asset)
                continue

            asset_price    = self.feed.get_price_for(asset)
            asset_momentum = self.feed.get_momentum_for(asset)
            asset_ohlcv    = self.feed.get_ohlcv_for(asset, 10)
            asset_cur_candle = self.feed.get_current_candle_for(asset)

            if asset_price <= 0:
                log.debug("%s price is 0 — skipping", asset)
                continue

            try:
                markets = await self.kalshi.get_crypto_15m_markets(asset)
            except KalshiAuthError as exc:
                log.error("Kalshi auth error: %s", exc)
                await self.telegram.send_error(str(exc), "kalshi_auth")
                return
            except Exception as exc:
                log.warning("Failed to fetch %s markets: %s", asset, exc)
                continue

            if not markets:
                log.debug("No active 15m %s markets found", asset)
                continue

            markets = sorted(markets, key=lambda m: m.get("volume", 0), reverse=True)[:_MAX_MARKETS]
            log.info(
                "%s=$%.4f  momentum=%.3f  markets=%d  [%s]",
                asset, asset_price, asset_momentum, len(markets),
                ", ".join(m["ticker"] for m in markets),
            )
            all_markets_found.extend(markets)

            # Apply per-asset size override on top of time/streak multiplier
            asset_size_mult = size_mult * _ASSET_SIZE_OVERRIDES.get(asset, 1.0)
            if asset in _ASSET_SIZE_OVERRIDES:
                log.debug(
                    "%s size multiplier: %.2f (base=%.2f × override=%.2f)",
                    asset, asset_size_mult, size_mult, _ASSET_SIZE_OVERRIDES[asset],
                )

            # Re-evaluate waited markets for this asset first.
            # _prev_wait_list is a snapshot taken before the clear at the top of
            # run_cycle() so waited tickers from the previous call are visible here.
            market_by_ticker = {m["ticker"]: m for m in markets}
            waited_tickers   = _prev_wait_list & set(market_by_ticker.keys())

            tasks = []
            for ticker in list(waited_tickers):
                if ticker in open_tickers:
                    continue
                log.info("Re-evaluating waited market %s", ticker)
                tasks.append(self._evaluate_market(
                    market_by_ticker[ticker], asset_price, asset_momentum,
                    asset_ohlcv, open_tickers, asset_size_mult,
                    asset=asset, current_candle=asset_cur_candle,
                ))

            # Evaluate remaining new markets for this asset in parallel
            for market in markets:
                ticker = market["ticker"]
                if ticker in open_tickers or ticker in self._wait_list:
                    continue
                tasks.append(self._evaluate_market(
                    market, asset_price, asset_momentum, asset_ohlcv,
                    open_tickers, asset_size_mult,
                    asset=asset, current_candle=asset_cur_candle,
                ))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        self._cycle_markets_found = len(all_markets_found)
        print(f"Total markets scanned: {len(all_markets_found)} across {len(supported_assets)} assets")

        # Persist market watch for dashboard — save all signals from this cycle
        log.info("[DASHBOARD] Storing market_watch: %d signal(s), %d market(s)",
                 len(self._cycle_signals), len(all_markets_found))
        if self._cycle_signals:
            log.info("[DASHBOARD] Signal keys: %s | tickers: %s",
                     list(self._cycle_signals[0].keys()),
                     [s["ticker"] for s in self._cycle_signals])
        try:
            last_sig = self._cycle_signals[-1] if self._cycle_signals else None
            if self._cycle_signals:
                cycle_status = "ok"
            elif self._cycle_bot_off_hit:
                cycle_status = "bot_off"    # markets were tradeable but bot is not started
            elif all_markets_found:
                cycle_status = "scanning"   # markets found but all skipped (order book / timing)
            else:
                cycle_status = "no_markets"
            await self.db.set_market_watch({
                "cycle_ts":     cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "btc_price":    btc_price,
                "cycle_status": cycle_status,
                "markets": [
                    {
                        "ticker":     m["ticker"],
                        "asset":      m.get("asset", "BTC"),
                        "strike":     float(m.get("strike_price") or 0),
                        "close_time": m.get("close_time", ""),
                        "yes_ask":    m.get("yes_ask", 0),
                        "no_ask":     m.get("no_ask", 0),
                        "volume":     m.get("volume", 0),
                        "title":      m.get("title", ""),
                    }
                    for m in all_markets_found
                ],
                "last_signal": last_sig,
                "signals":     self._cycle_signals,
            })
            log.info("[DASHBOARD] market_watch stored OK")
        except Exception as exc:
            log.warning("market_watch store failed: %s", exc)

    # ------------------------------------------------------------------
    # Per-market evaluation
    # ------------------------------------------------------------------

    async def _evaluate_market(
        self,
        market:          dict,
        btc_price:       float,
        momentum:        float,
        ohlcv:           list,
        open_tickers:    set[str],
        size_mult:       float = 1.0,
        asset:           str   = "BTC",
        current_candle:  dict | None = None,
    ) -> None:
        ticker = market["ticker"]

        imbalance = 0.0

        try:
            close_dt = datetime.fromisoformat(
                market.get("close_time", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.warning("Invalid close_time for %s — skipping", ticker)
            return

        # Time window guard: only enter between 3 min and 11 min into the 15m window.
        # Below 180s: market still opening, price discovery incomplete.
        # Below 240s remaining: too close to expiry, time-decay risk.
        now_utc      = datetime.now(timezone.utc)
        market_open  = close_dt - timedelta(minutes=15)
        time_in      = (now_utc - market_open).total_seconds()
        time_left    = (close_dt - now_utc).total_seconds()

        if time_in < 180:
            log.info(
                "Market %s too new (%.0fs in, need 180s) — skipping", ticker, time_in
            )
            return
        if time_left < 240:
            log.info(
                "Market %s too close to expiry (%.0fs left, need 240s) — skipping",
                ticker, time_left,
            )
            return

        # Market passed all timing checks — count it as evaluated
        self._cycle_markets_evaluated += 1

        # --- Bot enabled gate ---
        if not await self.db.get_bot_enabled():
            log.debug("Bot is OFF — skipping ensemble for %s", ticker)
            self._cycle_bot_off_hit = True
            return

        yes_ask = market.get("yes_ask") or 0
        no_ask  = market.get("no_ask")  or 0
        yes_bid = market.get("yes_bid") or 0
        no_bid  = market.get("no_bid")  or 0

        # Kalshi's /markets list endpoint frequently returns 0 for all price fields
        # even for actively traded markets. Fetch the live order book as a fallback.
        if yes_ask == 0 and no_ask == 0:
            try:
                ob = await self.kalshi.get_order_book(ticker, depth=3)
                yes_bids = ob.get("yes_bids", [])
                no_bids  = ob.get("no_bids",  [])
                yes_asks = ob.get("yes_asks", [])
                no_asks  = ob.get("no_asks",  [])
                if yes_asks:
                    yes_ask = yes_asks[0]["price"]
                elif no_bids:
                    yes_ask = 100 - no_bids[0]["price"]
                if no_asks:
                    no_ask = no_asks[0]["price"]
                elif yes_bids:
                    no_ask = 100 - yes_bids[0]["price"]
                if yes_bids and not yes_bid:
                    yes_bid = yes_bids[0]["price"]
                if no_bids and not no_bid:
                    no_bid = no_bids[0]["price"]
                if yes_ask or no_ask:
                    log.info(
                        "Market %s: order book \u2192 YES=%d\u00a2/%d\u00a2 NO=%d\u00a2/%d\u00a2 (ask/bid)",
                        ticker, yes_ask, yes_bid, no_ask, no_bid,
                    )
                    market = {**market, "yes_ask": yes_ask, "yes_bid": yes_bid,
                              "no_ask": no_ask, "no_bid": no_bid}
            except Exception as exc:
                log.debug("Order book fetch failed for %s: %s", ticker, exc)

        log.info(
            "Market %s: YES=%d\u00a2/%d\u00a2 NO=%d\u00a2/%d\u00a2 (ask/bid)",
            ticker, yes_ask, yes_bid, no_ask, no_bid,
        )

        # Skip markets with no real prices — cannot calculate EV or place orders.
        if yes_ask == 0 and no_ask == 0:
            log.info("Market %s: no ask prices from API, order book, or individual fetch — skipping (thin market)", ticker)
            return

        btc_data   = BtcData(
            price          = btc_price,
            momentum       = momentum,
            candles        = ohlcv,
            imbalance      = imbalance,
            symbol         = asset,
            current_candle = current_candle,
        )
        market_obj = Market(
            ticker       = ticker,
            yes_price    = yes_ask,
            no_price     = no_ask,
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
            claude_prob    = result.claude.probability   if result.claude   else None,
            gpt_prob       = result.gpt.probability      if result.gpt      else None,
            gemini_prob    = result.gemini.probability   if result.gemini   else None,
            deepseek_prob  = result.deepseek.probability if result.deepseek else None,
            consensus_prob = result.raw_prob,
            model_spread   = result.spread,
            confidence     = result.confidence,
            action         = result.action,
            skip_reason    = result.skip_reason,
        )

        # --- Build trade-entry checklist (shown on dashboard) ---
        checks: list[dict] = []

        # Check 1: Direction (always has one based on consensus_prob)
        dir_label = "UP" if result.consensus_prob > 0.5 else "DOWN"
        dir_pct   = result.consensus_prob * 100 if result.consensus_prob > 0.5 else (1 - result.consensus_prob) * 100
        checks.append({
            "id": "signal", "label": "Direction",
            "passed": True,
            "detail": f"{dir_label} · {dir_pct:.0f}%",
        })

        # Check 2: Models agree (action != WAIT means spread is within MAX_MODEL_SPREAD)
        spread_ok = result.action != "WAIT"
        checks.append({
            "id": "spread", "label": "Models agree",
            "passed": spread_ok,
            "detail": f"spread {result.spread:.2f}",
        })

        # Check 3: EV — expected value anchored to the live market ask price
        if result.direction == "yes":
            _ask_cents = market.get("yes_ask") or 0
            _ev = (result.consensus_prob - _ask_cents / 100.0) if _ask_cents else None
        elif result.direction == "no":
            _ask_cents = market.get("no_ask") or 0
            _ev = ((1.0 - result.consensus_prob) - _ask_cents / 100.0) if _ask_cents else None
        else:
            _ask_cents = 0
            _ev = None
        ev_ok = result.action == "TRADE" and _ev is not None and _ev >= settings.MIN_EV
        if _ev is None:
            _ev_detail = "no ask price"
        else:
            _ev_detail = f"{_ev*100:.1f}\u00a2 / need \u2265{settings.MIN_EV*100:.0f}\u00a2"
        checks.append({
            "id": "ev",
            "label": "EV",
            "passed": ev_ok,
            "detail": _ev_detail,
        })

        # If WAIT, fill remaining checks as not-evaluated and store
        _pending = [
            ("drawdown",  "Daily loss"),
            ("data_age",  "Data fresh"),
            ("spread",    "Spread"),
        ]
        if result.action == "WAIT":
            for cid, clabel in _pending:
                checks.append({"id": cid, "label": clabel, "passed": None, "detail": "—"})
            await self._store_last_signal(ticker, result, checks)
            log.info("Models disagree on %s — adding to wait list (1 cycle pause)", ticker)
            self._wait_list[ticker] = time.time()
            return

        # --- Risk gates ---
        gate_result = await self.risk.check_all(market, result, settings.MAX_BET_SIZE, asset=asset)

        # Checks 4-6: from gate results (ev gate already shown as Check 3 above)
        for gate_key, chk_id, chk_label in [
            ("drawdown",  "drawdown",  "Daily loss"),
            ("staleness", "data_age",  "Data fresh"),
            ("spread",    "spread",    "Spread"),
        ]:
            gd = gate_result.gate_details.get(gate_key)
            if gd is not None:
                # Gate reason strings include a verbose prefix (e.g. "Daily loss: $0.00 / $100")
                # that duplicates the label — strip everything up to and including the first ": "
                reason = gd["reason"]
                detail = reason.split(": ", 1)[-1] if ": " in reason else reason
                checks.append({"id": chk_id, "label": chk_label,
                               "passed": gd["passed"], "detail": detail})
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
            if gate_result.failed_gate == "drawdown":
                await self.telegram.send_error(
                    gate_result.reason,
                    f"Gate [{gate_result.failed_gate}]",
                )
            return

        trade = await self.strategy.enter_trade(
            market,
            result,
            gate_result,
            btc_price        = btc_price,
            btc_momentum     = momentum,
            asset_symbol     = asset,
            size_multiplier  = size_mult,
        )
        if trade:
            log.info(
                "✅ Trade entered: %s %s $%.2f",
                trade.market_ticker, trade.direction, trade.size_dollars,
            )
            self._cycle_trades_placed += 1
            open_tickers.add(ticker)   # guard against double-entry this cycle

    async def _store_last_signal(
        self, ticker: str, result: Any, checks: list[dict]
    ) -> None:
        """Accumulate this market's signal into _cycle_signals for the dashboard."""
        def _directional_prob(probability: float, direction: str) -> float:
            """Convert raw P(YES) to probability of the predicted direction."""
            p = probability * 100
            return round(100 - p if direction == "NO" else p, 1)

        def _mdata(r: Any) -> dict | None:
            if r is None:
                return None
            return {
                "prob":      _directional_prob(r.probability, r.direction),
                "direction": r.direction,
                "reasoning": r.reasoning[:120],
            }

        consensus_dir = result.direction.upper() if result.direction != "flat" else "FLAT"
        signal = {
            "ticker":      ticker,
            "direction":   consensus_dir,
            "action":      result.action,
            "prob":        _directional_prob(result.raw_prob, consensus_dir),
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
        # Replace existing entry for this ticker (re-evaluation) or append new
        self._cycle_signals = [s for s in self._cycle_signals if s["ticker"] != ticker]
        self._cycle_signals.append(signal)

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

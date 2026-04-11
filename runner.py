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
_CYCLE_BUFFER   = 180          # seconds after boundary before first tick (3-min buffer for order books to populate)
_MAX_MARKETS    = 10           # top N markets evaluated per cycle
_STOP_FILE      = Path("STOP")

# Per-asset bet size multipliers (applied on top of time/streak multipliers)
# BTC at 50% — fewer trades, lower % volatility vs alts; backtest showed 309 trades vs 1,250+ for alts
_ASSET_SIZE_OVERRIDES: dict[str, float] = {
    "BTC": 0.50,
}

# Per-asset minimum confidence thresholds (overrides global MIN_CONFIDENCE).
# SOL and XRP are highly choppy — require much stronger AI conviction before entering.
_ASSET_MIN_CONFIDENCE: dict[str, float] = {
    "SOL": 0.45,
    "XRP": 0.45,
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
        # Markets found this cycle (non-zero = found but may have been empty)
        self._cycle_markets_found: int = 0
        # Markets that passed timing checks and were actually evaluated
        self._cycle_markets_evaluated: int = 0

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
        bot_enabled  = await self.db.get_bot_enabled()
        trading_mode = "LIVE TRADING" if bot_enabled else "OFF MODE (analysis only — press START on dashboard to enable trading)"
        log.info("Startup complete — entering main loop  [%s]", trading_mode)

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

                # Short-cycle retry: when markets were found but ALL had empty order books,
                # wait 90s and scan again instead of skipping the whole 15-min window.
                # Kalshi late-night sessions can take 4-6 min for the first bids to appear.
                _EMPTY_RETRY_WAIT = 90
                _EMPTY_MAX_RETRIES = 3
                for _attempt in range(_EMPTY_MAX_RETRIES):
                    if self._cycle_signals or self._cycle_markets_evaluated == 0:
                        break   # signals generated, or no markets passed timing checks
                    log.info(
                        "All %d evaluated markets had empty order books — retrying in %ds (%d/%d)",
                        self._cycle_markets_evaluated, _EMPTY_RETRY_WAIT,
                        _attempt + 1, _EMPTY_MAX_RETRIES,
                    )
                    await asyncio.sleep(_EMPTY_RETRY_WAIT)
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
        self._cycle_markets_evaluated = 0  # markets that passed timing checks this cycle
        self._wait_list.clear()          # clear stale wait entries from previous 15m window
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
            return

        # Step 3 — verify at least BTC feed is alive
        if self.feed.is_stale():
            log.warning("BTC price data stale — skipping cycle")
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

            # Re-evaluate waited markets for this asset first
            market_by_ticker = {m["ticker"]: m for m in markets}
            waited_tickers   = set(self._wait_list.keys())

            for ticker in waited_tickers - market_by_ticker.keys():
                log.debug("Waited market %s expired — clearing", ticker)
                self._wait_list.pop(ticker, None)

            for ticker in list(waited_tickers & market_by_ticker.keys()):
                self._wait_list.pop(ticker, None)
                if ticker in open_tickers:
                    continue
                log.info("Re-evaluating waited market %s", ticker)
                await self._evaluate_market(
                    market_by_ticker[ticker], asset_price, asset_momentum,
                    asset_ohlcv, open_tickers, asset_size_mult,
                    asset=asset, current_candle=asset_cur_candle,
                )

            # Evaluate remaining new markets for this asset
            for market in markets:
                ticker = market["ticker"]
                if ticker in open_tickers or ticker in self._wait_list:
                    continue
                await self._evaluate_market(
                    market, asset_price, asset_momentum, asset_ohlcv,
                    open_tickers, asset_size_mult,
                    asset=asset, current_candle=asset_cur_candle,
                )

        self._cycle_markets_found = len(all_markets_found)
        print(f"Total markets scanned: {len(all_markets_found)} across {len(supported_assets)} assets")

        # Persist market watch for dashboard — save all signals from this cycle
        try:
            last_sig = self._cycle_signals[-1] if self._cycle_signals else None
            await self.db.set_market_watch({
                "cycle_ts":   cycle_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "btc_price":  btc_price,
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
        except Exception as exc:
            log.debug("market_watch store failed: %s", exc)

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

        # Refresh market ask prices from the live order book.
        # Kalshi only exposes bids; asks are derived from the opposite side's best bid.
        # yes_ask = 100 - best NO bid, no_ask = 100 - best YES bid (already done in client).
        yes_asks = ob.get("yes_asks", [])
        no_asks  = ob.get("no_asks",  [])
        yes_bids = ob.get("yes_bids", [])
        no_bids  = ob.get("no_bids",  [])
        if yes_asks:
            market["yes_ask"] = yes_asks[0]["price"]
        elif no_bids:
            market["yes_ask"] = 100 - no_bids[0]["price"]
        if no_asks:
            market["no_ask"] = no_asks[0]["price"]
        elif yes_bids:
            market["no_ask"] = 100 - yes_bids[0]["price"]
        # Derive the other side if still missing
        if market.get("yes_ask", 0) and not market.get("no_ask", 0):
            market["no_ask"] = 100 - market["yes_ask"]
        if market.get("no_ask", 0) and not market.get("yes_ask", 0):
            market["yes_ask"] = 100 - market["no_ask"]

        try:
            close_dt = datetime.fromisoformat(
                market.get("close_time", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.warning("Invalid close_time for %s — skipping", ticker)
            return

        # Time window guard
        # Only enter between 2 min and 5 min into the 15m window.
        # This prevents stale mid-session entries (e.g. bot restarted at minute 8).
        # Also blocks the last 3 min before expiry.
        now_utc      = datetime.now(timezone.utc)
        market_open  = close_dt - timedelta(minutes=15)
        time_in      = (now_utc - market_open).total_seconds()
        time_left    = (close_dt - now_utc).total_seconds()

        if time_in < 30:
            log.info(
                "Market %s too new (%.0fs in, need 30s) — skipping", ticker, time_in
            )
            return
        if time_in > 420:
            log.info(
                "Market %s too far into session (%.0fs in, max 420s) — skipping", ticker, time_in
            )
            return
        if time_left < 180:
            log.info(
                "Market %s too close to expiry (%.0fs left, need 180s) — skipping",
                ticker, time_left,
            )
            return

        # Market passed all timing checks — count it as evaluated
        self._cycle_markets_evaluated += 1

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
            yes_price    = market.get("yes_ask") or 50,
            no_price     = market.get("no_ask")  or 50,
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
        dir_label = "UP" if result.consensus_prob > 0.5 else "DOWN"
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

        # Check 3: Confidence — use asset-specific threshold for choppy assets
        asset_min_conf = _ASSET_MIN_CONFIDENCE.get(asset, settings.MIN_CONFIDENCE)
        conf_ok = result.action == "TRADE" and result.confidence >= asset_min_conf
        checks.append({
            "id": "confidence",
            "label": f"Confidence \u2265{asset_min_conf*100:.0f}%",
            "passed": conf_ok,
            "detail": f"{result.confidence*100:.0f}%",
        })

        # If WAIT or SKIP, fill remaining checks as not-evaluated and store
        _pending = [
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

        # Asset-specific confidence gate (SOL/XRP require higher conviction)
        if not conf_ok:
            for cid, clabel in _pending:
                checks.append({"id": cid, "label": clabel, "passed": None, "detail": "—"})
            await self._store_last_signal(ticker, result, checks)
            log.info(
                "Skipping %s [%s] — confidence %.3f below asset minimum %.2f (choppy asset)",
                ticker, asset, result.confidence, asset_min_conf,
            )
            return

        # --- Risk gates ---
        gate_result = await self.risk.check_all(market, result, settings.MAX_BET_SIZE, asset=asset)

        # Checks 4-5: from gate results
        for gate_key, chk_id, chk_label in [
            ("drawdown",  "drawdown",  "Daily loss"),
            ("staleness", "data_age",  "Data fresh"),
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
            if gate_result.failed_gate in ("edge", "liquidity", "drawdown"):
                await self.telegram.send_error(
                    gate_result.reason,
                    f"Gate [{gate_result.failed_gate}]",
                )
            return

        # --- Execute trade (only when bot is enabled — off mode skips this) ---
        if not await self.db.get_bot_enabled():
            log.info(
                "Bot in OFF mode — signal valid for %s but not trading",
                ticker,
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
            open_tickers.add(ticker)   # guard against double-entry this cycle

    async def _store_last_signal(
        self, ticker: str, result: Any, checks: list[dict]
    ) -> None:
        """Accumulate this market's signal into _cycle_signals for the dashboard."""
        def _mdata(r: Any) -> dict | None:
            if r is None:
                return None
            return {
                "prob":      round(r.probability * 100, 1),
                "direction": r.direction,
                "reasoning": r.reasoning[:120],
            }

        signal = {
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

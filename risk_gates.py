"""
risk_gates.py — 4-gate pre-trade risk filter

Active gates (in order):
  drawdown  — daily loss used must be below DAILY_LOSS_LIMIT
  ev        — expected value (consensus_prob - ask) must clear MIN_EV
  staleness — asset price feed must be fresh
  spread    — bid-ask spread must be ≤ MAX_SPREAD_CENTS; also computes
              effective EV adjusted for early-exit cost
All 4 gates must pass for a trade to execute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    passed:       bool
    failed_gate:  str | None        # name of first failed gate, or None
    reason:       str
    gate_details: dict              # {gate_name: {"passed": bool, "reason": str}}
    checked_at:   datetime


# ---------------------------------------------------------------------------
# RiskGates
# ---------------------------------------------------------------------------

class RiskGates:
    def __init__(self, kalshi_client, coinbase_feed, database) -> None:
        self._kalshi = kalshi_client
        self._feed   = coinbase_feed
        self._db     = database

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def check_all(
        self,
        market:          dict,
        ensemble_result,
        bet_size:        float,    # dollars
        asset:           str = "BTC",
    ) -> GateResult:
        """
        Run all 4 gates in order. Short-circuits on the first failure.
        Returns GateResult with passed=True only when ALL gates pass.
        """
        # Gate specs: (name, coroutine-factory, args)
        # Coroutines are created lazily so un-reached gates are never instantiated,
        # avoiding "coroutine was never awaited" RuntimeWarnings.
        gate_specs = [
            ("drawdown",  self._gate_drawdown,  ()),
            ("ev",        self._gate_ev,        (market, ensemble_result)),
            ("staleness", self._gate_staleness, (asset,)),
            ("spread",    self._gate_spread,    (market, ensemble_result)),
        ]

        details: dict = {}
        for name, fn, args in gate_specs:
            passed, reason = await fn(*args)
            details[name] = {"passed": passed, "reason": reason}
            if not passed:
                return GateResult(
                    passed       = False,
                    failed_gate  = name,
                    reason       = reason,
                    gate_details = details,
                    checked_at   = datetime.now(timezone.utc),
                )

        log.info("All 4 risk gates passed (drawdown / ev / staleness / spread)")
        return GateResult(
            passed       = True,
            failed_gate  = None,
            reason       = "all gates passed",
            gate_details = details,
            checked_at   = datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Gate 1 — Drawdown
    # ------------------------------------------------------------------

    async def _gate_drawdown(self) -> tuple[bool, str]:
        """
        Daily loss used must be below DAILY_LOSS_LIMIT.
        """
        daily_stats     = await self._db.get_daily_stats()
        daily_loss_used = daily_stats.daily_loss_used
        loss_limit      = settings.DAILY_LOSS_LIMIT

        if daily_loss_used >= loss_limit:
            reason = (
                f"Daily loss: ${daily_loss_used:.2f} / ${loss_limit:.0f} — "
                "limit reached, halting for the day"
            )
            log.info("Gate [drawdown]: FAIL — %s", reason)
            return False, reason

        reason = f"Daily loss: ${daily_loss_used:.2f} / ${loss_limit:.0f}"
        log.info("Gate [drawdown]: PASS — %s", reason)
        return True, reason

    # ------------------------------------------------------------------
    # Gate 2 — Expected Value
    # ------------------------------------------------------------------

    async def _gate_ev(
        self, market: dict, ensemble_result
    ) -> tuple[bool, str]:
        """
        Expected value of the trade must clear MIN_EV.

        For a Kalshi binary contract paying $1 at resolution:
          EV = consensus_prob - ask          (YES trade)
          EV = (1 - consensus_prob) - ask    (NO trade)

        This is dollars of expected profit per $1 of payout. It also
        checks that the ensemble action is TRADE (catches WAIT/SKIP from
        the spread-based model-agreement gate in ensemble.py).
        """
        action         = ensemble_result.action
        direction      = ensemble_result.direction   # "yes" | "no" | "flat"
        consensus_prob = ensemble_result.consensus_prob

        if action != "TRADE":
            reason = f"Ensemble action is '{action}' — not TRADE"
            log.info("Gate [ev]: FAIL — %s", reason)
            return False, reason

        if direction == "yes":
            ask_cents = market.get("yes_ask") or 0
        else:
            ask_cents = market.get("no_ask") or 0

        if ask_cents == 0:
            reason = f"No real ask price for {direction.upper()} — cannot calculate EV"
            log.info("Gate [ev]: FAIL — %s", reason)
            return False, reason

        ask = ask_cents / 100.0
        if direction == "yes":
            ev = consensus_prob - ask
        else:
            ev = (1.0 - consensus_prob) - ask

        min_ev = settings.MIN_EV
        passed = ev >= min_ev
        reason = f"EV: {ev:.1%} vs min {min_ev:.1%}"
        log.info("Gate [ev]: %s — %s", "PASS" if passed else "FAIL", reason)
        return passed, reason

    # ------------------------------------------------------------------
    # Gate 3 — Staleness
    # ------------------------------------------------------------------

    async def _gate_staleness(self, asset: str = "BTC") -> tuple[bool, str]:
        """
        Asset price data must be fresh (< PRICE_STALENESS_SECONDS old).
        Uses per-asset staleness check when available.
        """
        # Use per-asset method if available (multi-asset feed), else fall back
        if hasattr(self._feed, "is_stale_for"):
            stale = self._feed.is_stale_for(asset)
            state = getattr(self._feed, "_state", {}).get(asset)
            last  = state.last_update if state else None
        else:
            stale = self._feed.is_stale()
            last  = self._feed.last_update

        if last is not None:
            age     = (datetime.now(timezone.utc) - last).total_seconds()
            age_str = f"{age:.0f}s"
        else:
            age_str = "never"

        max_age = settings.PRICE_STALENESS_SECONDS

        if stale:
            reason = f"{asset} data age: {age_str} vs max {max_age}s — feed stale"
            log.info("Gate [staleness]: FAIL — %s", reason)
            return False, reason

        reason = f"{asset} data age: {age_str} vs max {max_age}s"
        log.info("Gate [staleness]: PASS — %s", reason)
        return True, reason

    # ------------------------------------------------------------------
    # Gate 4 — Bid-ask spread filter
    # ------------------------------------------------------------------

    async def _gate_spread(
        self, market: dict, ensemble_result
    ) -> tuple[bool, str]:
        """
        Gate 4 — Bid-ask spread filter.

        Rejects the trade when the spread on the relevant side is wider than
        MAX_SPREAD_CENTS.  Wide spreads mean the bot is immediately underwater
        if it needs to exit early (stop-loss): the fill is at the ask, but any
        sell must cross back through the spread to hit the bid.

        Spread calculation:
          YES trade: spread = yes_ask - yes_bid
          NO  trade: spread = no_ask  - no_bid
          If either bid or ask is zero (missing), spread = ∞ → gate fails.

        Effective EV:
          effective_ev = raw_ev − (spread_cents / 100) × EARLY_EXIT_PROBABILITY
          EARLY_EXIT_PROBABILITY (default 0.25) is the estimated fraction of
          trades that trigger a stop-loss exit before expiry.

        Returns tuple[bool, str] — same interface as all other gates.
        """
        direction      = ensemble_result.direction   # "yes" | "no"
        consensus_prob = ensemble_result.consensus_prob
        asset          = market.get("asset", "")
        ticker         = market.get("ticker", "")

        if direction == "yes":
            ask_cents = market.get("yes_ask") or 0
            bid_cents = market.get("yes_bid") or 0
        else:
            ask_cents = market.get("no_ask")  or 0
            bid_cents = market.get("no_bid")  or 0

        # Kalshi's /markets list endpoint frequently returns bid=0 for actively
        # traded markets.  When bid is missing, fetch the live order book to get
        # real prices before deciding whether to reject.
        if bid_cents == 0 and ticker:
            try:
                ob = await self._kalshi.get_order_book(ticker, depth=3)
                if direction == "yes":
                    live_bids = ob.get("yes_bids", [])
                    live_asks = ob.get("yes_asks", [])
                else:
                    live_bids = ob.get("no_bids", [])
                    live_asks = ob.get("no_asks", [])
                if live_bids:
                    bid_cents = live_bids[0]["price"]
                if live_asks and ask_cents == 0:
                    ask_cents = live_asks[0]["price"]
                log.debug(
                    "Gate [spread] %s %s: order book \u2192 bid=%d\u00a2 ask=%d\u00a2",
                    ticker, direction.upper(), bid_cents, ask_cents,
                )
            except Exception as exc:
                log.debug("Gate [spread]: order book fetch failed for %s: %s", ticker, exc)

        # Compute raw_ev with (possibly refreshed) ask_cents
        if direction == "yes":
            raw_ev = consensus_prob - ask_cents / 100.0
        else:
            raw_ev = (1.0 - consensus_prob) - ask_cents / 100.0

        # Gate fails immediately if either side of the market is still missing
        if ask_cents == 0 or bid_cents == 0:
            reason = (
                f"Spread: bid or ask missing for {direction.upper()} "
                f"(bid={bid_cents}\u00a2 ask={ask_cents}\u00a2) — "
                "cannot measure liquidity"
            )
            log.info("Gate [spread]: FAIL — %s", reason)
            return False, reason

        # Crossed market (bid ≥ ask) indicates bad data — reject rather than
        # compute a negative or zero spread that would incorrectly pass the gate.
        if bid_cents >= ask_cents:
            reason = (
                f"Spread: crossed market for {direction.upper()} "
                f"(bid={bid_cents}\u00a2 \u2265 ask={ask_cents}\u00a2) — "
                "invalid market data"
            )
            log.info("Gate [spread]: FAIL — %s", reason)
            return False, reason

        spread_cents = float(ask_cents - bid_cents)
        spread_pct   = (spread_cents / ask_cents) * 100.0

        # Effective EV accounts for early-exit cost (selling back at bid)
        p_exit       = settings.EARLY_EXIT_PROBABILITY
        effective_ev = raw_ev - (spread_cents / 100.0) * p_exit

        max_spread = settings.MAX_SPREAD_CENTS

        log.debug(
            "Gate [spread] %s %s: bid=%d\u00a2 ask=%d\u00a2 spread=%.0f\u00a2 "
            "(%.0f%% of ask)  raw_ev=%.1f\u00a2 eff_ev=%.1f\u00a2  "
            "p_exit=%.2f  max=%.0f\u00a2",
            asset, direction.upper(),
            bid_cents, ask_cents, spread_cents, spread_pct,
            raw_ev * 100, effective_ev * 100,
            p_exit, max_spread,
        )

        if spread_cents > max_spread:
            reason = (
                f"Spread too wide: {spread_cents:.0f}\u00a2 "
                f"(max {max_spread:.0f}\u00a2, {spread_pct:.0f}% of ask) "
                f"for {asset} {direction.upper()} — "
                f"raw_ev={raw_ev*100:.1f}\u00a2 eff_ev={effective_ev*100:.1f}\u00a2"
            )
            log.info("Gate [spread]: FAIL — %s", reason)
            return False, reason

        reason = (
            f"Spread: {spread_cents:.0f}\u00a2 ({spread_pct:.0f}% of ask) — "
            f"raw_ev={raw_ev*100:.1f}\u00a2 eff_ev={effective_ev*100:.1f}\u00a2"
        )
        log.info("Gate [spread]: PASS — %s", reason)
        return True, reason

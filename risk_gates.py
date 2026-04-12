"""
risk_gates.py — 3-gate pre-trade risk filter

Active gates: drawdown, ev, staleness.
  drawdown  — daily loss used must be below DAILY_LOSS_LIMIT
  ev        — expected value (consensus_prob - ask) must clear MIN_EV
  staleness — asset price feed must be fresh
All 3 gates must pass for a trade to execute.
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
        Run all 3 gates in order. Short-circuits on the first failure.
        Returns GateResult with passed=True only when ALL gates pass.
        """
        gates = [
            ("drawdown",  self._gate_drawdown()),
            ("ev",        self._gate_ev(market, ensemble_result)),
            ("staleness", self._gate_staleness(asset)),
        ]

        details: dict = {}
        for name, coro in gates:
            passed, reason = await coro
            details[name] = {"passed": passed, "reason": reason}
            if not passed:
                return GateResult(
                    passed       = False,
                    failed_gate  = name,
                    reason       = reason,
                    gate_details = details,
                    checked_at   = datetime.now(timezone.utc),
                )

        log.info("All 3 risk gates passed (drawdown / ev / staleness)")
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

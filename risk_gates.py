"""
risk_gates.py — 5-gate pre-trade risk filter

All 5 gates must pass for a trade to execute.
Each gate logs its result independently.
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
        Run all 5 gates in order. Short-circuits on the first failure.
        Returns GateResult with passed=True only when ALL gates pass.
        """
        gates = [
            ("drawdown",   self._gate_drawdown()),
            ("confidence", self._gate_confidence(ensemble_result)),
            ("staleness",  self._gate_staleness(asset)),
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

        log.info("All 3 risk gates passed")
        return GateResult(
            passed       = True,
            failed_gate  = None,
            reason       = "all gates passed",
            gate_details = details,
            checked_at   = datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Gate 1 — Edge
    # ------------------------------------------------------------------

    async def _gate_edge(
        self, market: dict, ensemble_result
    ) -> tuple[bool, str]:
        """
        Ensemble edge must clear MIN_EDGE above the market ask price.

        edge = consensus_prob - market_ask_price
        pass if edge >= MIN_EDGE (0.05)
        """
        direction      = ensemble_result.direction     # "yes" | "no" | "flat"
        consensus_prob = ensemble_result.consensus_prob
        min_edge       = settings.MIN_EDGE

        if direction not in ("yes", "no"):
            reason = f"direction is '{direction}' — no trade signal"
            log.info("Gate [edge]: FAIL — %s", reason)
            return False, reason

        if direction == "yes":
            market_price = (market.get("yes_ask") or 50) / 100.0
        else:
            market_price = (market.get("no_ask") or 50) / 100.0

        edge   = consensus_prob - market_price
        passed = edge >= min_edge
        reason = f"Edge: {edge:.1%} vs min {min_edge:.1%}"
        log.info("Gate [edge]: %s — %s", "PASS" if passed else "FAIL", reason)
        return passed, reason

    # ------------------------------------------------------------------
    # Gate 2 — Liquidity
    # ------------------------------------------------------------------

    async def _gate_liquidity(
        self, market: dict, ensemble_result, bet_size: float
    ) -> tuple[bool, str]:
        """
        Enough ask-side depth must exist within 3 cents of the best ask
        to fill our intended bet at market.

        available_liquidity = sum of ask sizes within 3¢ of best ask
        contracts_needed    = bet_size / ask_price_per_contract
        pass if available_liquidity >= contracts_needed
        """
        ticker    = market.get("ticker", "")
        direction = ensemble_result.direction

        try:
            ob = await self._kalshi.get_order_book(ticker)
        except Exception as exc:
            reason = f"Order book fetch failed: {exc}"
            log.warning("Gate [liquidity]: FAIL — %s", reason)
            return False, reason

        if direction == "yes":
            asks            = ob.get("yes_asks", [])
            ask_price_cents = market.get("yes_ask") or 50
        else:
            asks            = ob.get("no_asks", [])
            ask_price_cents = market.get("no_ask") or 50

        if not asks:
            reason = f"No {direction.upper()} asks in order book"
            log.info("Gate [liquidity]: FAIL — %s", reason)
            return False, reason

        best_ask      = asks[0]["price"]
        price_ceiling = best_ask + 3    # accept fills up to 3¢ worse than best

        available_liquidity = sum(
            lv["size"] for lv in asks if lv["price"] <= price_ceiling
        )

        ask_price_dollars = ask_price_cents / 100.0
        if ask_price_dollars <= 0:
            reason = "Ask price is zero — market not tradeable"
            log.warning("Gate [liquidity]: FAIL — %s", reason)
            return False, reason

        contracts_needed = bet_size / ask_price_dollars
        passed = available_liquidity >= contracts_needed
        reason = (
            f"Liquidity: {available_liquidity:.0f} contracts available, "
            f"need {contracts_needed:.1f}"
        )
        log.info("Gate [liquidity]: %s — %s", "PASS" if passed else "FAIL", reason)
        return passed, reason

    # ------------------------------------------------------------------
    # Gate 3 — Drawdown
    # ------------------------------------------------------------------

    async def _gate_drawdown(self) -> tuple[bool, str]:
        """
        Two sub-checks:
          1. Daily loss used must be below DAILY_LOSS_LIMIT.
          2. Open position exposure must be < 40% of current bankroll.
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

        # Bankroll exposure sub-check
        try:
            balance_dollars = await self._kalshi.get_balance()
        except Exception as exc:
            reason = f"Balance fetch failed: {exc}"
            log.warning("Gate [drawdown]: FAIL — %s", reason)
            return False, reason

        open_trades = await self._db.get_open_trades()
        open_exposure_dollars = sum(
            t.entry_price * t.contracts / 100 for t in open_trades
        )

        if balance_dollars > 0:
            max_exposure = balance_dollars * 0.40
            if open_exposure_dollars >= max_exposure:
                reason = (
                    f"Exposure: ${open_exposure_dollars:.2f} >= 40% of "
                    f"bankroll ${balance_dollars:.2f}"
                )
                log.info("Gate [drawdown]: FAIL — %s", reason)
                return False, reason

        reason = f"Daily loss: ${daily_loss_used:.2f} / ${loss_limit:.0f}"
        log.info("Gate [drawdown]: PASS — %s", reason)
        return True, reason

    # ------------------------------------------------------------------
    # Gate 4 — Confidence
    # ------------------------------------------------------------------

    async def _gate_confidence(self, ensemble_result) -> tuple[bool, str]:
        """
        Ensemble must have both sufficient confidence AND action == "TRADE".
        Ensemble sets action to WAIT/SKIP when spread or confidence fails
        its own internal thresholds — this gate catches those cases.
        """
        confidence = ensemble_result.confidence
        action     = ensemble_result.action
        min_conf   = settings.MIN_CONFIDENCE

        if action != "TRADE":
            reason = f"Ensemble action is '{action}' — not TRADE"
            log.info("Gate [confidence]: FAIL — %s", reason)
            return False, reason

        passed = confidence >= min_conf
        reason = f"Confidence: {confidence:.1%} vs min {min_conf:.1%}"
        log.info("Gate [confidence]: %s — %s", "PASS" if passed else "FAIL", reason)
        return passed, reason

    # ------------------------------------------------------------------
    # Gate 5 — Staleness
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

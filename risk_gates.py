"""
risk_gates.py — 5-gate pre-trade risk checks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from coinbase_feed import Candle
from ensemble import EnsembleSignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    passed: bool
    failed_gate: int | None     # 1-5, or None if all passed
    gate_name: str | None
    reason: str

    @classmethod
    def ok(cls) -> "GateResult":
        return cls(passed=True, failed_gate=None, gate_name=None, reason="all gates passed")

    @classmethod
    def fail(cls, gate: int, name: str, reason: str) -> "GateResult":
        return cls(passed=False, failed_gate=gate, gate_name=name, reason=reason)


# ---------------------------------------------------------------------------
# State that the gates inspect
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    """Snapshot of runtime state passed to gate checks each tick."""
    balance_cents: int              # current available balance
    starting_balance_cents: int     # balance at start of today
    daily_pnl_cents: int            # realised P&L today (can be negative)
    open_exposure_cents: int        # sum of all open position cost basis
    last_trade_ts: float            # unix timestamp of last trade (0 = never)
    candles_1h: list[Candle]        # recent 1h candles for vol calculation


# ---------------------------------------------------------------------------
# Individual gate implementations
# ---------------------------------------------------------------------------

def _gate1_drawdown(state: BotState, max_drawdown_pct: float) -> GateResult:
    """Daily P&L must not have fallen below -X% of starting balance."""
    if state.starting_balance_cents <= 0:
        return GateResult.fail(1, "drawdown", "starting balance is zero — cannot compute drawdown")

    drawdown_pct = -state.daily_pnl_cents / state.starting_balance_cents
    if drawdown_pct >= max_drawdown_pct:
        return GateResult.fail(
            1, "drawdown",
            f"daily drawdown {drawdown_pct:.1%} ≥ limit {max_drawdown_pct:.1%} — "
            f"halting for the day (P&L: ${state.daily_pnl_cents/100:.2f})",
        )
    return GateResult.ok()


def _gate2_exposure(state: BotState, max_exposure: float) -> GateResult:
    """Total open position cost must not exceed X% of balance."""
    if state.balance_cents <= 0:
        return GateResult.fail(2, "exposure", "zero balance")

    exposure_pct = state.open_exposure_cents / state.balance_cents
    if exposure_pct >= max_exposure:
        return GateResult.fail(
            2, "exposure",
            f"open exposure {exposure_pct:.1%} ≥ limit {max_exposure:.1%} "
            f"(${state.open_exposure_cents/100:.2f} of ${state.balance_cents/100:.2f})",
        )
    return GateResult.ok()


def _gate3_confidence(signal: EnsembleSignal, confidence_min: float) -> GateResult:
    """Ensemble confidence must exceed the minimum threshold."""
    if signal.direction == "flat":
        return GateResult.fail(3, "confidence", "ensemble returned flat signal")

    if signal.confidence < confidence_min:
        return GateResult.fail(
            3, "confidence",
            f"confidence {signal.confidence:.3f} < minimum {confidence_min:.3f}",
        )
    return GateResult.ok()


def _gate4_volatility(state: BotState, max_vol: float) -> GateResult:
    """BTC 1h realized volatility must be below the ceiling."""
    candles = state.candles_1h
    if len(candles) < 8:
        # Not enough data — allow trading but log a warning
        log.warning("Gate 4: fewer than 8 hourly candles available, skipping vol check")
        return GateResult.ok()

    import numpy as np
    closes = [c.close for c in candles[-8:]]
    returns = np.diff(np.log(closes))
    realized_vol = float(np.std(returns))   # 1-bar log return std

    if realized_vol >= max_vol:
        return GateResult.fail(
            4, "volatility",
            f"realized vol {realized_vol:.4f} ≥ ceiling {max_vol:.4f} — "
            "BTC too choppy to trade",
        )
    return GateResult.ok()


def _gate5_frequency(state: BotState, min_minutes: int) -> GateResult:
    """Minimum time between trades must have elapsed."""
    if state.last_trade_ts == 0:
        return GateResult.ok()

    elapsed_min = (time.time() - state.last_trade_ts) / 60.0
    if elapsed_min < min_minutes:
        return GateResult.fail(
            5, "frequency",
            f"only {elapsed_min:.1f}m since last trade — minimum is {min_minutes}m",
        )
    return GateResult.ok()


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

class RiskGates:
    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.05,
        max_position_exposure: float = 0.20,
        confidence_min: float = 0.60,
        max_btc_vol_threshold: float = 0.04,
        min_minutes_between_trades: int = 15,
    ):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_position_exposure = max_position_exposure
        self.confidence_min = confidence_min
        self.max_btc_vol_threshold = max_btc_vol_threshold
        self.min_minutes_between_trades = min_minutes_between_trades

    def check_all(self, signal: EnsembleSignal, state: BotState) -> GateResult:
        """Run all 5 gates in order. Returns on first failure."""
        gates = [
            lambda: _gate1_drawdown(state, self.max_daily_drawdown_pct),
            lambda: _gate2_exposure(state, self.max_position_exposure),
            lambda: _gate3_confidence(signal, self.confidence_min),
            lambda: _gate4_volatility(state, self.max_btc_vol_threshold),
            lambda: _gate5_frequency(state, self.min_minutes_between_trades),
        ]
        for gate_fn in gates:
            result = gate_fn()
            if not result.passed:
                log.warning(
                    "Gate %d [%s] BLOCKED: %s",
                    result.failed_gate, result.gate_name, result.reason,
                )
                return result

        log.info("All risk gates passed")
        return GateResult.ok()

"""
strategy.py — Kelly sizing + entry/exit logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ensemble import EnsembleSignal
from kalshi_client import OrderBook

log = logging.getLogger(__name__)

# Kalshi contracts are $1 face value, prices in cents (1–99).
# One contract costs `price` cents and pays $1 if it resolves YES.
CONTRACT_FACE_VALUE_CENTS = 100


# ---------------------------------------------------------------------------
# Order parameters
# ---------------------------------------------------------------------------

@dataclass
class OrderParams:
    ticker: str
    side: str           # "yes" | "no"
    contracts: int
    price_cents: int    # limit price in cents (1–99)
    take_profit_cents: int
    stop_loss_cents: int
    dollar_size: float  # nominal cost in dollars


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

def kelly_fraction(
    prob: float,
    *,
    win_payout: float = 1.0,    # net gain per dollar if correct
    loss_payout: float = 1.0,   # loss per dollar if wrong
) -> float:
    """
    Full Kelly fraction: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win_payout / loss_payout.

    Clamps to [0, 1].
    """
    if prob <= 0 or prob >= 1:
        return 0.0
    q = 1.0 - prob
    b = win_payout / loss_payout
    f = (prob * b - q) / b
    return max(0.0, min(f, 1.0))


def size_position(
    kelly: float,
    kelly_fraction_multiplier: float,   # e.g. 0.5 for half-Kelly
    balance_cents: int,
    max_position_pct: float,            # hard cap
) -> int:
    """Returns position size in cents (cost basis)."""
    fractional_kelly = kelly * kelly_fraction_multiplier
    raw_size_cents = int(balance_cents * fractional_kelly)
    max_size_cents = int(balance_cents * max_position_pct)
    size_cents = min(raw_size_cents, max_size_cents)
    return max(0, size_cents)


# ---------------------------------------------------------------------------
# Entry pricing
# ---------------------------------------------------------------------------

def get_entry_price(orderbook: OrderBook, side: str) -> int | None:
    """
    Returns the best available ask price for the desired side.
    We are always buyers, so we lift the ask.
    """
    if side == "yes":
        return orderbook.best_yes_ask()
    else:
        return orderbook.best_no_ask()


# ---------------------------------------------------------------------------
# Exit targets
# ---------------------------------------------------------------------------

def get_exit_targets(
    entry_cents: int,
    side: str,
    *,
    take_profit_mult: float = 1.5,  # TP at 1.5× the implied edge
    stop_loss_mult: float = 0.5,    # SL at 50% of cost
) -> tuple[int, int]:
    """
    Returns (take_profit_cents, stop_loss_cents) as market value targets.

    Since Kalshi pays $1 (100 cents) on resolution, our edge is:
      - YES side: we bought at `entry_cents`, potential gain = 100 - entry_cents
      - NO  side: we bought at `entry_cents`, potential gain = 100 - entry_cents

    Take profit: sell when the contract has appreciated by TP fraction of our edge.
    Stop loss:   sell when we've lost SL fraction of our entry cost.
    """
    edge = CONTRACT_FACE_VALUE_CENTS - entry_cents
    tp = min(entry_cents + int(edge * take_profit_mult), 99)
    sl = max(entry_cents - int(entry_cents * stop_loss_mult), 1)
    return tp, sl


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    def __init__(
        self,
        kelly_fraction_multiplier: float = 0.5,
        max_position_pct: float = 0.10,
        take_profit_mult: float = 1.5,
        stop_loss_mult: float = 0.5,
    ):
        self.kelly_fraction_multiplier = kelly_fraction_multiplier
        self.max_position_pct = max_position_pct
        self.take_profit_mult = take_profit_mult
        self.stop_loss_mult = stop_loss_mult

        # Rolling win rate for Kelly calibration
        self._wins = 0
        self._total = 0
        self._avg_win_cents = 50    # initial estimates
        self._avg_loss_cents = 25

    def build_order(
        self,
        signal: EnsembleSignal,
        ticker: str,
        orderbook: OrderBook,
        balance_cents: int,
    ) -> OrderParams | None:
        """
        Converts an ensemble signal into a concrete order.
        Returns None if the position would be too small to be worth placing.
        """
        side = signal.direction   # "yes" or "no"
        if side not in ("yes", "no"):
            return None

        entry_price = get_entry_price(orderbook, side)
        if entry_price is None:
            log.warning("No ask available on %s side for %s", side, ticker)
            return None

        # Kelly uses the win probability from the ensemble
        win_prob = signal.raw_prob if side == "yes" else (1.0 - signal.raw_prob)
        win_payout = (CONTRACT_FACE_VALUE_CENTS - entry_price) / entry_price
        loss_payout = 1.0

        kelly = kelly_fraction(win_prob, win_payout=win_payout, loss_payout=loss_payout)
        size_cents = size_position(
            kelly,
            self.kelly_fraction_multiplier,
            balance_cents,
            self.max_position_pct,
        )

        if size_cents < entry_price:
            # Can't even afford 1 contract
            log.info(
                "Position too small: size=$%.2f < 1 contract @ %dc",
                size_cents / 100, entry_price,
            )
            return None

        contracts = size_cents // entry_price
        actual_cost_cents = contracts * entry_price

        tp, sl = get_exit_targets(
            entry_price,
            side,
            take_profit_mult=self.take_profit_mult,
            stop_loss_mult=self.stop_loss_mult,
        )

        log.info(
            "Order built: %s %s × %d @ %dc | TP=%dc SL=%dc | Kelly=%.3f size=$%.2f",
            side, ticker, contracts, entry_price, tp, sl,
            kelly, actual_cost_cents / 100,
        )

        return OrderParams(
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=entry_price,
            take_profit_cents=tp,
            stop_loss_cents=sl,
            dollar_size=actual_cost_cents / 100,
        )

    def record_trade_outcome(self, entry_cents: int, exit_cents: int, contracts: int) -> None:
        """Update win/loss stats for Kelly calibration."""
        pnl = (exit_cents - entry_cents) * contracts
        self._total += 1
        if pnl > 0:
            self._wins += 1
            # Exponential moving average of win/loss size
            self._avg_win_cents = int(0.8 * self._avg_win_cents + 0.2 * pnl)
        else:
            self._avg_loss_cents = int(0.8 * self._avg_loss_cents + 0.2 * abs(pnl))

    @property
    def win_rate(self) -> float:
        return self._wins / self._total if self._total > 0 else 0.5

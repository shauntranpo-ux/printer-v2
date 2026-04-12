"""
test_spread_gate.py — Unit tests for Gate 4 (bid-ask spread filter)

Run with:
    python -m pytest test_spread_gate.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from risk_gates import RiskGates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gates() -> RiskGates:
    """RiskGates instance with mocked dependencies — only _gate_spread is exercised."""
    return RiskGates(
        kalshi_client = MagicMock(),
        coinbase_feed = MagicMock(),
        database      = MagicMock(),
    )


def _ensemble(direction: str = "yes", consensus_prob: float = 0.65) -> MagicMock:
    r = MagicMock()
    r.direction       = direction
    r.consensus_prob  = consensus_prob
    r.action          = "TRADE"
    return r


def _market(
    yes_ask: int = 45,
    yes_bid: int = 35,
    no_ask:  int = 55,
    no_bid:  int = 50,
    asset:   str = "BTC",
) -> dict:
    return {
        "yes_ask": yes_ask,
        "yes_bid": yes_bid,
        "no_ask":  no_ask,
        "no_bid":  no_bid,
        "asset":   asset,
    }


def run(coro):
    """Run a coroutine in the current event loop (or a new one)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Spread calculation — YES trade
# ---------------------------------------------------------------------------

class TestYesSpread:
    def test_tight_spread_passes(self):
        """YES spread of 5¢ passes MAX_SPREAD_CENTS=15."""
        gates  = _make_gates()
        market = _market(yes_ask=50, yes_bid=45)  # spread = 5¢
        passed, reason = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is True
        assert "5" in reason  # spread value in reason

    def test_wide_spread_fails(self):
        """YES spread of 20¢ exceeds MAX_SPREAD_CENTS=15 → gate fails."""
        gates  = _make_gates()
        market = _market(yes_ask=60, yes_bid=40)  # spread = 20¢
        passed, reason = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False
        assert "20" in reason
        assert "max" in reason.lower()

    def test_spread_exactly_at_limit_passes(self, monkeypatch):
        """Spread exactly equal to MAX_SPREAD_CENTS is allowed (≤, not <)."""
        from config import settings
        monkeypatch.setattr(settings, "MAX_SPREAD_CENTS", 15.0)
        gates  = _make_gates()
        market = _market(yes_ask=60, yes_bid=45)  # spread = 15¢ exactly
        passed, _ = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is True

    def test_spread_one_over_limit_fails(self, monkeypatch):
        """Spread of MAX_SPREAD_CENTS + 1 → gate fails."""
        from config import settings
        monkeypatch.setattr(settings, "MAX_SPREAD_CENTS", 15.0)
        gates  = _make_gates()
        market = _market(yes_ask=61, yes_bid=45)  # spread = 16¢
        passed, _ = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False


# ---------------------------------------------------------------------------
# Spread calculation — NO trade
# ---------------------------------------------------------------------------

class TestNoSpread:
    def test_no_tight_spread_passes(self):
        """NO spread of 5¢ passes."""
        gates  = _make_gates()
        market = _market(no_ask=55, no_bid=50)  # spread = 5¢
        passed, reason = run(gates._gate_spread(market, _ensemble("no", 0.35)))
        assert passed is True

    def test_no_wide_spread_fails(self):
        """NO spread of 20¢ fails."""
        gates  = _make_gates()
        market = _market(no_ask=65, no_bid=45)  # spread = 20¢
        passed, reason = run(gates._gate_spread(market, _ensemble("no", 0.35)))
        assert passed is False
        assert "20" in reason

    def test_no_side_ignored_for_yes_trade(self):
        """Wide NO spread does not block a YES trade — only YES spread matters."""
        gates  = _make_gates()
        market = _market(yes_ask=50, yes_bid=46, no_ask=65, no_bid=40)
        # YES spread = 4¢ (tight), NO spread = 25¢ (wide) — YES trade should pass
        passed, _ = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is True


# ---------------------------------------------------------------------------
# Missing bid / ask (zero values)
# ---------------------------------------------------------------------------

class TestMissingPrices:
    def test_missing_yes_bid_fails(self):
        """yes_bid = 0 → spread is undefined → gate fails."""
        gates  = _make_gates()
        market = _market(yes_ask=50, yes_bid=0)
        passed, reason = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False
        assert "missing" in reason.lower()

    def test_missing_yes_ask_fails(self):
        """yes_ask = 0 → gate fails (no price to buy at)."""
        gates  = _make_gates()
        market = _market(yes_ask=0, yes_bid=40)
        passed, reason = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False
        assert "missing" in reason.lower()

    def test_missing_no_bid_fails(self):
        """no_bid = 0 on a NO trade → gate fails."""
        gates  = _make_gates()
        market = _market(no_ask=55, no_bid=0)
        passed, reason = run(gates._gate_spread(market, _ensemble("no", 0.35)))
        assert passed is False
        assert "missing" in reason.lower()

    def test_both_sides_zero_fails(self):
        """All prices zero → gate fails."""
        gates  = _make_gates()
        market = _market(yes_ask=0, yes_bid=0, no_ask=0, no_bid=0)
        passed, _ = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False


# ---------------------------------------------------------------------------
# Effective EV calculation (encoded in reason string)
# ---------------------------------------------------------------------------

class TestEffectiveEv:
    def test_effective_ev_present_in_reason(self):
        """Reason string must mention both raw_ev and eff_ev."""
        gates  = _make_gates()
        # YES trade: ask=50, bid=46, spread=4¢, consensus_prob=0.60
        # raw_ev = 0.60 - 0.50 = 0.10 = 10¢
        # eff_ev = 0.10 - (4/100) * 0.25 = 0.10 - 0.01 = 0.09 = 9¢
        market = _market(yes_ask=50, yes_bid=46)
        passed, reason = run(gates._gate_spread(market, _ensemble("yes", 0.60)))
        assert passed is True
        assert "raw_ev" in reason
        assert "eff_ev" in reason

    def test_spread_pct_present_in_reason(self):
        """Spread percentage should be included in the pass reason."""
        gates  = _make_gates()
        market = _market(yes_ask=50, yes_bid=42)  # spread=8¢, 16% of ask
        passed, reason = run(gates._gate_spread(market, _ensemble("yes", 0.60)))
        assert passed is True
        assert "%" in reason


# ---------------------------------------------------------------------------
# Per-asset context in failure reason
# ---------------------------------------------------------------------------

class TestAssetLabel:
    def test_asset_name_in_fail_reason(self):
        """Failure reason should include the asset name for log clarity."""
        gates  = _make_gates()
        market = _market(yes_ask=70, yes_bid=40, asset="DOGE")  # spread=30¢
        passed, reason = run(gates._gate_spread(market, _ensemble("yes")))
        assert passed is False
        assert "DOGE" in reason


# ---------------------------------------------------------------------------
# check_all integration — spread is Gate 4
# ---------------------------------------------------------------------------

class TestCheckAllIntegration:
    """Verify the spread gate is wired into check_all() correctly."""

    def _mock_daily_stats(self):
        stats = MagicMock()
        stats.daily_loss_used = 0.0
        return stats

    def test_spread_gate_fires_after_other_gates(self, monkeypatch):
        """
        When drawdown/ev/staleness pass but spread fails,
        check_all() returns failed_gate='spread'.
        """
        from config import settings
        monkeypatch.setattr(settings, "MAX_SPREAD_CENTS", 10.0)
        monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT",  100.0)
        monkeypatch.setattr(settings, "MIN_EV",             0.05)
        monkeypatch.setattr(settings, "PRICE_STALENESS_SECONDS", 30)

        # Wire stubs
        db    = MagicMock()
        db.get_daily_stats = AsyncMock(return_value=self._mock_daily_stats())
        feed  = MagicMock()
        feed.is_stale_for  = MagicMock(return_value=False)
        feed.is_stale      = MagicMock(return_value=False)
        feed.last_update   = None
        feed._state        = {}

        gates = RiskGates(kalshi_client=MagicMock(), coinbase_feed=feed, database=db)

        # Market with 20¢ spread on YES — exceeds 10¢ max
        market = {
            "yes_ask": 60, "yes_bid": 40,
            "no_ask":  40, "no_bid":  35,
            "asset":   "BTC",
        }
        ensemble = _ensemble("yes", 0.70)   # raw_ev = 0.70 - 0.60 = 0.10 (passes ev gate)

        result = run(gates.check_all(market, ensemble, bet_size=5.0, asset="BTC"))

        assert result.passed is False
        assert result.failed_gate == "spread"
        assert "spread" in result.gate_details

    def test_all_four_gates_pass(self, monkeypatch):
        """
        When all 4 gates pass, check_all() returns passed=True
        and gate_details has entries for all 4 gates.
        """
        from config import settings
        monkeypatch.setattr(settings, "MAX_SPREAD_CENTS",   20.0)
        monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT",  100.0)
        monkeypatch.setattr(settings, "MIN_EV",             0.05)
        monkeypatch.setattr(settings, "PRICE_STALENESS_SECONDS", 30)
        monkeypatch.setattr(settings, "EARLY_EXIT_PROBABILITY", 0.25)

        db    = MagicMock()
        db.get_daily_stats = AsyncMock(return_value=self._mock_daily_stats())
        feed  = MagicMock()
        feed.is_stale_for  = MagicMock(return_value=False)
        feed.is_stale      = MagicMock(return_value=False)
        feed.last_update   = None
        feed._state        = {}

        gates = RiskGates(kalshi_client=MagicMock(), coinbase_feed=feed, database=db)

        # Tight spread (10¢) that passes
        market = {
            "yes_ask": 50, "yes_bid": 40,
            "no_ask":  50, "no_bid":  45,
            "asset":   "BTC",
        }
        ensemble = _ensemble("yes", 0.65)   # raw_ev = 0.65 - 0.50 = 0.15 (passes)

        result = run(gates.check_all(market, ensemble, bet_size=5.0, asset="BTC"))

        assert result.passed is True
        assert result.failed_gate is None
        assert set(result.gate_details.keys()) == {"drawdown", "ev", "staleness", "spread"}
        assert all(v["passed"] for v in result.gate_details.values())

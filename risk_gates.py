"""
risk_gates.py — 5-gate pre-trade risk checks

Gates (all must pass before any order is placed):
  Gate 1 — Max drawdown: daily P&L loss must be below threshold
  Gate 2 — Position limit: total open exposure must be below max
  Gate 3 — Confidence floor: ensemble confidence must exceed minimum
  Gate 4 — Volatility guard: BTC realized vol must be below ceiling
  Gate 5 — Trade frequency: minimum time between trades enforced

Responsibilities:
- Run all 5 gates sequentially and return pass/fail with reason
- Log all gate failures to database
"""

# TODO: implement RiskGates class
# TODO: implement check_drawdown(), check_position_limit()
# TODO: implement check_confidence(), check_volatility(), check_frequency()
# TODO: implement check_all(signal, state) -> GateResult(passed, failed_gate, reason)
# TODO: integrate with config thresholds from config.py

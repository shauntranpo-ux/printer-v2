"""
strategy.py — Kelly sizing + entry/exit logic

Responsibilities:
- Compute Kelly fraction from ensemble confidence and historical win rate
- Apply fractional Kelly (half-Kelly by default) for position sizing
- Determine entry price targets from orderbook
- Determine exit conditions: take-profit, stop-loss, time-based
- Translate signal + sizing into executable order parameters
"""

# TODO: implement Strategy class
# TODO: implement kelly_fraction(confidence, win_rate, avg_win, avg_loss) -> float
# TODO: implement size_position(kelly, balance, max_pct) -> dollar_size
# TODO: implement get_entry(orderbook, direction) -> price
# TODO: implement get_exit(entry, direction, atr) -> (take_profit, stop_loss)
# TODO: implement build_order(signal, sizing, entry, exit) -> OrderParams

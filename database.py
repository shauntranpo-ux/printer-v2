"""
database.py — SQLite trade logging

Tables:
  trades       — every order placed and its outcome
  signals      — ensemble signal + model weights per tick
  gate_events  — risk gate pass/fail log
  pnl_daily    — end-of-day P&L snapshots

Responsibilities:
- Initialize and migrate the SQLite schema on startup
- Provide async-friendly read/write methods (via aiosqlite)
- Query helpers for dashboard and risk gate checks
"""

# TODO: implement Database class with aiosqlite
# TODO: implement init_db() — create tables if not exist
# TODO: implement log_trade(order, signal, status)
# TODO: implement update_trade(trade_id, close_price, pnl, status)
# TODO: implement log_signal(signal, weights, confidence)
# TODO: implement log_gate_event(gate, passed, reason)
# TODO: implement get_daily_pnl(), get_recent_trades(n), get_open_positions()

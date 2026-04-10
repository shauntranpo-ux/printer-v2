"""
dashboard.py — Flask web dashboard

Responsibilities:
- Serve a live web UI showing bot status, P&L, and recent trades
- Expose REST endpoints for the frontend to poll
- Display: current signal, last trade, open position, daily P&L, gate status
- Display: per-model ensemble weights and confidence history
- Password-protect via env var DASHBOARD_PASSWORD
"""

from flask import Flask

app = Flask(__name__)

# TODO: implement GET /         — serve dashboard HTML
# TODO: implement GET /api/status — current bot state as JSON
# TODO: implement GET /api/trades — recent trade history from DB
# TODO: implement GET /api/pnl   — daily/weekly P&L summary
# TODO: implement GET /api/ensemble — current model weights + last signal
# TODO: add basic auth middleware using DASHBOARD_PASSWORD env var
# TODO: serve static dashboard HTML/JS (or render Jinja2 template)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

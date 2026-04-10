"""
dashboard.py — Flask web dashboard
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import date
from functools import wraps

from flask import Flask, jsonify, request, Response, render_template_string

from config import cfg
from database import Database

app = Flask(__name__)
_db = Database(cfg.database_path)

# ---------------------------------------------------------------------------
# Bootstrap: open the database in a background thread with its own event loop
# ---------------------------------------------------------------------------

_loop = asyncio.new_event_loop()

def _start_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_start_loop, args=(_loop,), daemon=True).start()

def _run(coro):
    """Run a coroutine on the dashboard's event loop and block until done."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=10)

@app.before_request
def _ensure_db():
    if _db._db is None:
        _run(_db.connect())


# ---------------------------------------------------------------------------
# Basic auth
# ---------------------------------------------------------------------------

def _require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.password != cfg.dashboard.password:
            return Response(
                "Unauthorized", 401,
                {"WWW-Authenticate": 'Basic realm="printer-v2"'},
            )
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@_require_auth
def index():
    return render_template_string(DASHBOARD_HTML)


@app.get("/api/status")
@_require_auth
def api_status():
    try:
        today = _run(_db.get_today_pnl())
        open_trades = _run(_db.get_open_trades())
        return jsonify({
            "status": "running",
            "env": cfg.env,
            "today": {
                "date": today.day,
                "trades": today.trades_count,
                "win_rate": (today.winning_trades / today.trades_count
                             if today.trades_count else 0),
                "net_pnl": today.net_pnl,
                "ending_balance": today.ending_balance,
            },
            "open_positions": len(open_trades),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/trades")
@_require_auth
def api_trades():
    try:
        n = min(int(request.args.get("n", 20)), 100)
        trades = _run(_db.get_recent_trades(n))
        return jsonify([
            {
                "id": t.id,
                "ticker": t.market_ticker,
                "direction": t.direction,
                "contracts": t.contracts,
                "entry_price": t.entry_price,
                "close_price": t.close_price,
                "dollar_size": t.dollar_size,
                "pnl": t.pnl,
                "status": t.status,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in trades
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/pnl")
@_require_auth
def api_pnl():
    try:
        n = min(int(request.args.get("days", 7)), 30)
        rows = _run(_db.get_daily_pnl(n))
        return jsonify([
            {
                "day": r.day,
                "trades": r.trades_count,
                "wins": r.winning_trades,
                "gross_pnl": r.gross_pnl,
                "net_pnl": r.net_pnl,
                "ending_balance": r.ending_balance,
            }
            for r in rows
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Minimal dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>printer-v2</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace; padding: 24px; }
    h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.4rem; letter-spacing: 2px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
    .card-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .card-value { font-size: 1.6rem; font-weight: bold; margin-top: 6px; }
    .pos { color: #3fb950; }
    .neg { color: #f85149; }
    .neutral { color: #e6edf3; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { color: #8b949e; text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; }
    td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
    tr:hover td { background: #161b22; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; }
    .badge-open   { background: #388bfd26; color: #58a6ff; }
    .badge-closed { background: #3fb95026; color: #3fb950; }
    .badge-error  { background: #f8514926; color: #f85149; }
    #refresh { color: #8b949e; font-size: 0.75rem; margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>&#9608; PRINTER-V2</h1>
  <div id="refresh">Loading...</div>
  <div class="grid" id="stats"></div>
  <div class="card">
    <div class="card-label" style="margin-bottom:12px">Recent Trades</div>
    <table>
      <thead><tr><th>Market</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Status</th><th>Opened</th></tr></thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
  <script>
    function fmt(v, prefix='$') {
      if (v == null) return '—';
      const s = prefix + Math.abs(v).toFixed(2);
      return v < 0 ? '-' + s : s;
    }
    function colorClass(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral'; }

    async function refresh() {
      const [status, trades] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/trades?n=30').then(r => r.json()),
      ]);

      document.getElementById('refresh').textContent =
        'Last updated: ' + new Date().toLocaleTimeString() + '  |  ENV: ' + (status.env || '?').toUpperCase();

      const t = status.today || {};
      const wr = t.win_rate != null ? (t.win_rate * 100).toFixed(0) + '%' : '—';
      const pnlClass = colorClass(t.net_pnl);
      document.getElementById('stats').innerHTML = `
        <div class="card"><div class="card-label">Daily P&L</div>
          <div class="card-value ${pnlClass}">${fmt(t.net_pnl)}</div></div>
        <div class="card"><div class="card-label">Trades Today</div>
          <div class="card-value neutral">${t.trades ?? 0}</div></div>
        <div class="card"><div class="card-label">Win Rate</div>
          <div class="card-value neutral">${wr}</div></div>
        <div class="card"><div class="card-label">Open Positions</div>
          <div class="card-value neutral">${status.open_positions ?? 0}</div></div>
        <div class="card"><div class="card-label">Balance</div>
          <div class="card-value neutral">${fmt(t.ending_balance)}</div></div>
      `;

      const tbody = document.getElementById('trades-body');
      tbody.innerHTML = (trades || []).map(t => {
        const pnl = t.pnl != null ? `<span class="${colorClass(t.pnl)}">${fmt(t.pnl)}</span>` : '—';
        const badge = `<span class="badge badge-${t.status}">${t.status}</span>`;
        const opened = t.opened_at ? t.opened_at.replace('T', ' ').slice(0, 16) : '—';
        return `<tr>
          <td>${t.ticker}</td>
          <td>${t.direction.toUpperCase()}</td>
          <td>${t.contracts}</td>
          <td>${t.entry_price}¢</td>
          <td>${t.close_price != null ? t.close_price + '¢' : '—'}</td>
          <td>${pnl}</td>
          <td>${badge}</td>
          <td>${opened}</td>
        </tr>`;
      }).join('');
    }

    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=cfg.dashboard.port, debug=False)

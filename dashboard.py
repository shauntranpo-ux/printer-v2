"""
dashboard.py — Flask web dashboard for printer-v2

Runs in a background thread alongside the trading bot.
All async DB/API calls are bridged via a dedicated event loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from flask import Flask, jsonify, Response

from config import settings
from database import Database
from kalshi_client import KalshiClient

app = Flask(__name__)

_STOP_FILE      = Path("STOP")
_HEARTBEAT_FILE = Path("heartbeat.txt")

# ---------------------------------------------------------------------------
# Async bridge — one background event loop for all async calls from Flask
# ---------------------------------------------------------------------------

_loop = asyncio.new_event_loop()

def _start_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_start_loop, args=(_loop,), daemon=True).start()


def _run(coro: Any, timeout: float = 10.0) -> Any:
    """Submit a coroutine to the dashboard event loop and block until done."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout=timeout)


# ---------------------------------------------------------------------------
# Shared clients (initialised once, reused across requests)
# ---------------------------------------------------------------------------

_db     = Database(Path(settings.DB_PATH))
_kalshi = KalshiClient()
_db_ok  = False


def _ensure_db() -> None:
    global _db_ok
    if not _db_ok:
        _run(_db.connect())
        _db_ok = True


# ---------------------------------------------------------------------------
# 30-second caches to avoid hammering external APIs
# ---------------------------------------------------------------------------

_btc_cache     : dict[str, Any] = {"price": None, "ts": 0.0}
_balance_cache : dict[str, Any] = {"value": None, "ts": 0.0}
_BTC_TTL       = 30.0
_BALANCE_TTL   = 60.0


def _get_btc_price() -> float | None:
    """Coinbase public REST — no auth required, cached 30s."""
    now = time.monotonic()
    if _btc_cache["price"] and now - _btc_cache["ts"] < _BTC_TTL:
        return _btc_cache["price"]
    try:
        resp = httpx.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5.0
        )
        price = float(resp.json()["data"]["amount"])
        _btc_cache.update(price=price, ts=now)
        return price
    except Exception:
        return _btc_cache.get("price")


def _get_balance() -> float | None:
    """Kalshi balance, cached 60s."""
    now = time.monotonic()
    if _balance_cache["value"] is not None and now - _balance_cache["ts"] < _BALANCE_TTL:
        return _balance_cache["value"]
    try:
        value = _run(_kalshi.get_balance())
        _balance_cache.update(value=value, ts=now)
        return value
    except Exception:
        return _balance_cache.get("value")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _bot_status_info() -> dict:
    """Derive bot state from DB (persists across container restarts)."""
    _ensure_db()
    stop  = _STOP_FILE.exists()
    start = _run(_db.get_bot_enabled())
    if stop:
        status = "stopped"
    elif start:
        status = "running"
    else:
        status = "waiting"
    return {"status": status, "stop_file": stop, "start_file": start}


async def _market_watch() -> dict | None:
    """Return the latest cycle-watch data from bot_kv."""
    return await _db.get_market_watch()


async def _last_ensemble() -> dict | None:
    """Return the most recent row from ensemble_log as a dict."""
    async with _db._conn() as db:
        cur = await db.execute(
            "SELECT * FROM ensemble_log ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def _current_position_price(ticker: str, direction: str) -> int | None:
    """Best bid on our side from Kalshi — used for live P&L on open positions."""
    try:
        ob = await _kalshi.get_order_book(ticker)
        bids = ob.get("yes_bids" if direction == "YES" else "no_bids", [])
        return bids[0]["price"] if bids else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/stats")
def api_stats():
    _ensure_db()
    try:
        today       = _run(_db.get_daily_stats())
        btc_price   = _get_btc_price()
        balance     = _get_balance()
        ensemble    = _run(_last_ensemble())
        bot_status  = _bot_status_info()["status"]

        pnl_pct = (
            (today.total_pnl / today.total_wagered * 100)
            if today.total_wagered else 0.0
        )

        return jsonify({
            "bot_status":  bot_status,
            "env":         settings.env,
            "btc_price":   btc_price,
            "balance":     balance,
            "pnl_pct":     round(pnl_pct, 2),
            "today": {
                "total_pnl":       today.total_pnl,
                "daily_loss_used": today.daily_loss_used,
                "daily_loss_limit": settings.DAILY_LOSS_LIMIT,
                "win_rate":        today.win_rate,
                "total_trades":    today.total_trades,
                "winning_trades":  today.winning_trades,
                "sharpe_ratio":    today.sharpe_ratio,
                "max_drawdown":    today.max_drawdown,
            },
            "last_ensemble": ensemble,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/positions")
def api_positions():
    _ensure_db()
    try:
        open_trades = _run(_db.get_open_trades())
        now_utc     = datetime.now(timezone.utc)
        out = []
        for t in open_trades:
            # Best-effort live price
            current = _run(_current_position_price(t.market_ticker, t.direction))
            entry   = t.entry_price
            pnl_pct = ((current - entry) / entry * 100) if (current and entry) else None

            # Age
            try:
                opened = datetime.strptime(t.timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                secs = int((now_utc - opened).total_seconds())
                h, r = divmod(secs, 3600)
                age  = f"{h}h {r//60}m" if h else f"{r//60}m"
            except Exception:
                age = "—"

            out.append({
                "id":            t.id,
                "ticker":        t.market_ticker,
                "direction":     t.direction,
                "size_dollars":  t.size_dollars,
                "contracts":     t.contracts,
                "entry_price":   entry,
                "current_price": current,
                "pnl_pct":       round(pnl_pct, 1) if pnl_pct is not None else None,
                "age":           age,
                "confidence":    t.ensemble_confidence,
            })
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/trades")
def api_trades():
    _ensure_db()
    try:
        trades = _run(_db.get_recent_trades(limit=20))
        return jsonify([
            {
                "id":          t.id,
                "timestamp":   t.timestamp,
                "ticker":      t.market_ticker,
                "direction":   t.direction,
                "size_dollars":t.size_dollars,
                "contracts":   t.contracts,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "pnl_dollars": t.pnl_dollars,
                "exit_reason": t.exit_reason,
                "confidence":  t.ensemble_confidence,
                "status":      t.status,
                "closed_at":   t.closed_at,
            }
            for t in trades
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/bot/status")
def api_bot_status():
    info = _bot_status_info()
    try:
        if _HEARTBEAT_FILE.exists():
            ts_str = _HEARTBEAT_FILE.read_text().strip()
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            secs = int((datetime.now(timezone.utc) - dt).total_seconds())
            info["last_cycle_secs"] = secs
            info["last_cycle_ts"]   = ts_str
        else:
            info["last_cycle_secs"] = None
            info["last_cycle_ts"]   = None
    except Exception:
        info["last_cycle_secs"] = None
        info["last_cycle_ts"]   = None
    return jsonify(info)


@app.get("/api/market_watch")
def api_market_watch():
    _ensure_db()
    try:
        data = _run(_market_watch())
        return jsonify(data or {})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/balance")
def api_balance():
    _ensure_db()
    try:
        balance = _run(_db.get_balance())
        return jsonify({"balance": balance})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/direction_stats")
def api_direction_stats():
    _ensure_db()
    try:
        return jsonify(_run(_db.get_win_rate_by_direction()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/model_performance")
def api_model_performance():
    _ensure_db()
    try:
        return jsonify(_run(_db.get_model_performance()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/daily_pnl")
def api_daily_pnl():
    _ensure_db()
    try:
        return jsonify(_run(_db.get_last_7_days_stats()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/bot/start")
def api_bot_start():
    _ensure_db()
    _STOP_FILE.unlink(missing_ok=True)
    _run(_db.set_bot_enabled(True))
    # Send Telegram notification — fires only when the dashboard START button is clicked
    try:
        token = settings.TELEGRAM_BOT_TOKEN
        chat_id = settings.TELEGRAM_CHAT_ID
        if token and chat_id:
            msg = "🟢 *printer-v2 STARTED*\nBot activated via dashboard."
            httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=5.0,
            )
    except Exception:
        pass
    return jsonify({"ok": True, "status": "running"})


@app.post("/api/bot/stop")
def api_bot_stop():
    _ensure_db()
    _STOP_FILE.touch()
    _run(_db.set_bot_enabled(False))
    return jsonify({"ok": True, "status": "stopped"})


@app.get("/")
def index():
    _ensure_db()
    return Response(_HTML, mimetype="text/html")


# ---------------------------------------------------------------------------
# Dashboard HTML — dark theme, monospace, 30s auto-refresh
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>printer-v2</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0d1117;--surface:#161b22;--border:#30363d;--dim:#21262d;
    --text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;
    --green:#3fb950;--red:#f85149;--yellow:#d29922;
    --font:'Courier New',Courier,monospace;
  }
  body{background:var(--bg);color:var(--text);font-family:var(--font);
       font-size:13px;padding:20px;min-height:100vh}

  /* Header */
  .header{display:flex;align-items:center;gap:12px;margin-bottom:24px;flex-wrap:wrap}
  .header h1{color:var(--blue);font-size:1.3rem;letter-spacing:3px;font-weight:700}
  .dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  .dot-green{background:var(--green);box-shadow:0 0 6px var(--green)}
  .dot-red{background:var(--red);box-shadow:0 0 6px var(--red)}
  .badge{padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;
         background:#388bfd26;color:var(--blue)}
  .badge-live{background:#3fb95026;color:var(--green)}
  .badge-demo{background:#d2992226;color:var(--yellow)}
  .refresh-ts{margin-left:auto;color:var(--muted);font-size:11px}

  /* Stat cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:12px;margin-bottom:20px}
  .card{background:var(--surface);border:1px solid var(--border);
        border-radius:8px;padding:14px 16px}
  .card-label{font-size:10px;color:var(--muted);text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:6px}
  .card-value{font-size:1.5rem;font-weight:700;line-height:1}
  .card-sub{font-size:11px;color:var(--muted);margin-top:4px}

  /* Progress bar */
  .pbar-track{background:var(--dim);border-radius:4px;height:6px;margin-top:8px;overflow:hidden}
  .pbar-fill{height:100%;border-radius:4px;background:var(--red);
             transition:width .4s ease}

  /* Section */
  .section{background:var(--surface);border:1px solid var(--border);
           border-radius:8px;padding:16px;margin-bottom:16px;overflow-x:auto}
  .section-title{font-size:11px;color:var(--muted);text-transform:uppercase;
                 letter-spacing:1.2px;margin-bottom:12px}

  /* Tables */
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{color:var(--muted);text-align:left;padding:6px 10px;
     border-bottom:1px solid var(--border);white-space:nowrap;font-weight:400}
  td{padding:6px 10px;border-bottom:1px solid var(--dim);white-space:nowrap}
  tr:last-child td{border-bottom:none}
  tbody tr:hover td{background:var(--dim)}

  /* Ensemble row */
  .ens-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
            gap:8px;margin-top:4px}
  .ens-cell{background:var(--dim);border-radius:6px;padding:10px 12px;text-align:center}
  .ens-model{font-size:10px;color:var(--muted);text-transform:uppercase;
             letter-spacing:.8px;margin-bottom:4px}
  .ens-prob{font-size:1.2rem;font-weight:700}
  .ens-action{font-size:1.1rem;font-weight:700;padding:8px 16px;
              border-radius:6px;display:inline-block;margin-left:8px}
  .action-TRADE{color:var(--green);background:#3fb95018}
  .action-SKIP{color:var(--muted);background:#8b949e18}
  .action-WAIT{color:var(--yellow);background:#d2992218}

  /* Colors */
  .pos{color:var(--green)}.neg{color:var(--red)}.neu{color:var(--text)}
  .dir-yes{color:var(--green)}.dir-no{color:var(--red)}
  .status-open{color:var(--blue)}.status-closed{color:var(--muted)}
  .status-expired{color:var(--yellow)}

  /* Cycle watch */
  .watch-meta{font-size:11px;color:var(--muted);margin-bottom:10px;display:flex;gap:16px;flex-wrap:wrap}
  .watch-signal{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
                margin-top:12px;padding-top:10px;border-top:1px solid var(--border)}
  .watch-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}

  /* Empty state */
  .empty{color:var(--muted);text-align:center;padding:20px;font-size:12px}

  /* Bot control buttons */
  .dot-yellow{background:var(--yellow);box-shadow:0 0 6px var(--yellow)}
  .btn-start{background:#238636;color:#fff;border:none;border-radius:6px;
             padding:6px 14px;font-family:var(--font);font-size:12px;
             cursor:pointer;font-weight:600;letter-spacing:.5px}
  .btn-start:hover{background:#2ea043}
  .btn-stop{background:#b62324;color:#fff;border:none;border-radius:6px;
            padding:6px 14px;font-family:var(--font);font-size:12px;
            cursor:pointer;font-weight:600;letter-spacing:.5px}
  .btn-stop:hover{background:#da3633}
  .btn-start:disabled,.btn-stop:disabled{opacity:.5;cursor:not-allowed}

  /* Model AI cards (per-model breakdown in cycle watch) */
  .model-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
  .model-card{background:var(--dim);border-radius:6px;padding:10px 12px}
  .model-card-name{font-size:10px;color:var(--muted);text-transform:uppercase;
                   letter-spacing:.8px;margin-bottom:4px}
  .model-card-prob{font-size:1.1rem;font-weight:700;margin-bottom:2px}
  .model-card-dir{font-size:10px;font-weight:600;margin-bottom:4px}
  .model-card-reasoning{font-size:10px;color:var(--muted);line-height:1.3;
                        overflow:hidden;display:-webkit-box;
                        -webkit-line-clamp:2;-webkit-box-orient:vertical}
  .model-card-null{opacity:.4;font-size:11px;color:var(--muted);margin-top:8px}

  /* AI Consensus panel */
  .consensus-panel{margin-top:14px;padding-top:14px;border-top:1px solid var(--border)}
  .consensus-title{font-size:10px;color:var(--muted);text-transform:uppercase;
                   letter-spacing:1.2px;margin-bottom:10px}
  .bot-vote-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
  .bot-vote-card{background:var(--dim);border-radius:6px;padding:12px;border:2px solid transparent}
  .bot-vote-card.vote-yes{border-color:#3fb95050}
  .bot-vote-card.vote-no{border-color:#f8514950}
  .bot-vote-card.vote-fail{opacity:.45}
  .bot-vote-name{font-size:10px;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.8px;margin-bottom:6px}
  .bot-vote-dir{font-size:1.4rem;font-weight:700;margin-bottom:2px;line-height:1}
  .bot-vote-prob{font-size:.95rem;font-weight:600;margin-bottom:6px}
  .bot-vote-reason{font-size:10px;color:var(--muted);line-height:1.4;
                   overflow:hidden;display:-webkit-box;
                   -webkit-line-clamp:3;-webkit-box-orient:vertical}
  .consensus-banner{display:flex;align-items:center;justify-content:center;
                    flex-wrap:wrap;gap:8px;padding:12px 16px;border-radius:8px;
                    font-weight:700;font-size:.95rem;text-align:center}
  .cbanner-yes{background:#3fb95018;border:1px solid #3fb95040;color:var(--green)}
  .cbanner-no{background:#f8514918;border:1px solid #f8514940;color:var(--red)}
  .cbanner-split{background:#d2992218;border:1px solid #d2992240;color:var(--yellow)}
  .cbanner-sub{font-size:11px;font-weight:400;color:var(--muted)}
  @media(max-width:600px){.bot-vote-cards{grid-template-columns:1fr 1fr}}

  /* Trade-entry checklist */
  .checklist{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;
             padding-top:10px;border-top:1px solid var(--border)}
  .chk-item{display:inline-flex;align-items:center;gap:5px;font-size:11px;
            background:var(--dim);border-radius:4px;padding:4px 8px;white-space:nowrap}
  .chk-icon{font-weight:700;font-size:12px;width:12px;text-align:center}
  .chk-pass .chk-icon{color:var(--green)}
  .chk-fail .chk-icon{color:var(--red)}
  .chk-skip .chk-icon{color:var(--muted);opacity:.45}
  .chk-label{color:var(--muted)}
  .chk-detail{color:var(--text);margin-left:3px;font-size:10px}
  .chk-fail .chk-label,.chk-fail .chk-detail{color:var(--red)}

  /* Direction stats & model performance */
  .dir-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .dir-cell{background:var(--dim);border-radius:6px;padding:12px 16px;text-align:center}
  .dir-label{font-size:10px;color:var(--muted);text-transform:uppercase;
             letter-spacing:1px;margin-bottom:6px}
  .dir-pct{font-size:1.6rem;font-weight:700}
  .dir-sub{font-size:11px;color:var(--muted);margin-top:4px}

  /* 7-day bar chart */
  .bar-chart{display:flex;align-items:flex-end;gap:6px;height:80px;margin-top:8px}
  .bar-wrap{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px}
  .bar{width:100%;border-radius:3px 3px 0 0;min-height:2px;transition:height .3s}
  .bar-pos{background:var(--green)}
  .bar-neg{background:var(--red)}
  .bar-label{font-size:9px;color:var(--muted);white-space:nowrap}
  .bar-val{font-size:9px;font-weight:600;white-space:nowrap}

  @media(max-width:600px){
    body{padding:12px}
    .card-value{font-size:1.2rem}
    .cards{grid-template-columns:1fr 1fr}
    .model-cards{grid-template-columns:1fr 1fr}
    .dir-grid{grid-template-columns:1fr 1fr}
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="dot" id="status-dot"></div>
  <h1>&#9608; PRINTER-V2</h1>
  <span id="mode-badge" class="badge">—</span>
  <span id="bot-status-text" style="font-size:12px;font-weight:600"></span>
  <span id="last-cycle" style="font-size:11px;color:var(--muted)"></span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <button id="btn-start" class="btn-start" onclick="botStart()">START BOT</button>
    <button id="btn-stop" class="btn-stop" onclick="botStop()">STOP BOT</button>
    <span class="refresh-ts" id="refresh-ts">Loading...</span>
  </div>
</div>

<!-- Stats cards -->
<div class="cards" id="cards"></div>

<!-- Cycle watch -->
<div class="section">
  <div class="section-title">Cycle Watch — What the Bot Is Looking At</div>
  <div id="watch-wrap"><div class="empty">Waiting for first cycle...</div></div>
</div>

<!-- Open positions -->
<div class="section">
  <div class="section-title">Open Positions</div>
  <div id="positions-wrap">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Dir</th><th>Size</th>
        <th>Entry</th><th>Current</th><th>P&L%</th><th>Age</th>
      </tr></thead>
      <tbody id="pos-body"></tbody>
    </table>
  </div>
</div>

<!-- Recent trades -->
<div class="section">
  <div class="section-title">Last 20 Trades</div>
  <table>
    <thead><tr>
      <th>Time</th><th>Ticker</th><th>Dir</th><th>Size</th>
      <th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th>
    </tr></thead>
    <tbody id="trades-body"></tbody>
  </table>
</div>

<!-- Last ensemble -->
<div class="section">
  <div class="section-title">Last Ensemble Decision</div>
  <div id="ensemble-wrap"><div class="empty">No data yet</div></div>
</div>

<!-- Win Rate by Direction -->
<div class="section">
  <div class="section-title">Win Rate by Direction (All Time)</div>
  <div id="direction-wrap"><div class="empty">No data yet</div></div>
</div>

<!-- Model Performance -->
<div class="section">
  <div class="section-title">Model Accuracy (All Time)</div>
  <div id="model-perf-wrap"><div class="empty">No data yet</div></div>
</div>

<!-- 7-Day P&L Chart -->
<div class="section">
  <div class="section-title">7-Day P&amp;L</div>
  <div id="daily-pnl-wrap"><div class="empty">No data yet</div></div>
</div>

<script>
'use strict';

/* ---- helpers ---- */
function $f(v, decimals=2, prefix='$') {
  if (v == null) return '—';
  const abs = Math.abs(v).toFixed(decimals);
  return (v < 0 ? '-' : '') + prefix + abs;
}
function pct(v, decimals=1) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(decimals) + '%';
}
function prob(v) {
  if (v == null) return '—';
  return (v * 100).toFixed(0) + '%';
}
function cc(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : 'neu'; }
function ts(s) {
  if (!s) return '—';
  return s.replace('T',' ').substring(0, 16);
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ---- data fetch ---- */
async function fetchAll() {
  const [stats, positions, trades, botStatus, watch, dirStats, modelPerf, dailyPnl] = await Promise.all([
    fetch('/api/stats').then(r => r.json()).catch(()=>({})),
    fetch('/api/positions').then(r => r.json()).catch(()=>[]),
    fetch('/api/trades').then(r => r.json()).catch(()=>[]),
    fetch('/api/bot/status').then(r => r.json()).catch(()=>({})),
    fetch('/api/market_watch').then(r => r.json()).catch(()=>null),
    fetch('/api/direction_stats').then(r => r.json()).catch(()=>({})),
    fetch('/api/model_performance').then(r => r.json()).catch(()=>({})),
    fetch('/api/daily_pnl').then(r => r.json()).catch(()=>[]),
  ]);
  return { stats, positions, trades, botStatus, watch, dirStats, modelPerf, dailyPnl };
}

/* ---- bot control ---- */
async function botStart() {
  const btn = document.getElementById('btn-start');
  btn.disabled = true;
  try { await fetch('/api/bot/start', {method:'POST'}); await refresh(); }
  finally { btn.disabled = false; }
}
async function botStop() {
  const btn = document.getElementById('btn-stop');
  btn.disabled = true;
  try { await fetch('/api/bot/stop', {method:'POST'}); await refresh(); }
  finally { btn.disabled = false; }
}

function cycleAgo(secs) {
  if (secs == null) return '—';
  if (secs < 60) return secs + 's ago';
  const m = Math.floor(secs / 60), s = secs % 60;
  return m + 'm ' + (s ? s + 's ' : '') + 'ago';
}

/* ---- render header ---- */
function renderHeader(stats, botStatus) {
  const status = botStatus?.status || stats.bot_status || 'waiting';
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('bot-status-text');

  if (status === 'running') {
    dot.className = 'dot dot-green';
    text.innerHTML = '<span style="color:var(--green)">&#11044; RUNNING</span>';
  } else if (status === 'stopped') {
    dot.className = 'dot dot-red';
    text.innerHTML = '<span style="color:var(--red)">&#11044; STOPPED</span>';
  } else {
    dot.className = 'dot dot-yellow';
    text.innerHTML = '<span style="color:var(--yellow)">&#11044; WAITING</span>';
  }

  const lc = document.getElementById('last-cycle');
  const secs = botStatus?.last_cycle_secs;
  lc.textContent = secs != null ? 'Last cycle: ' + cycleAgo(secs) : '';

  const mb = document.getElementById('mode-badge');
  const env = (stats.env || 'live').toUpperCase();
  mb.textContent = env;
  mb.className = 'badge ' + (env === 'DEMO' ? 'badge-demo' : 'badge-live');

  document.getElementById('refresh-ts').textContent =
    'Updated ' + new Date().toLocaleTimeString();
}

/* ---- render stat cards ---- */
function renderCards(stats) {
  const t = stats.today || {};
  const lossUsed = t.daily_loss_used ?? 0;
  const lossLimit = t.daily_loss_limit ?? 100;
  const lossPct = clamp(lossUsed / lossLimit * 100, 0, 100).toFixed(0);
  const wr = t.win_rate != null ? (t.win_rate * 100).toFixed(0) + '%' : '—';
  const pnl = t.total_pnl ?? 0;
  const pnlPct = stats.pnl_pct ?? 0;
  const btc = stats.btc_price != null
    ? '$' + stats.btc_price.toLocaleString('en-US', {maximumFractionDigits:0})
    : '—';
  const bal = stats.balance != null ? '$' + stats.balance.toFixed(2) : '—';

  document.getElementById('cards').innerHTML = `
    <div class="card">
      <div class="card-label">BTC Price</div>
      <div class="card-value neu">${btc}</div>
    </div>
    <div class="card">
      <div class="card-label">Balance</div>
      <div class="card-value neu">${bal}</div>
    </div>
    <div class="card">
      <div class="card-label">Today P&amp;L</div>
      <div class="card-value ${cc(pnl)}">${$f(pnl)}</div>
      <div class="card-sub ${cc(pnlPct)}">${pct(pnlPct)}</div>
    </div>
    <div class="card">
      <div class="card-label">Daily Loss Used</div>
      <div class="card-value ${lossUsed > 0 ? 'neg' : 'neu'}">${$f(lossUsed)} <span style="font-size:.8rem;color:var(--muted)">/ $${lossLimit.toFixed(0)}</span></div>
      <div class="pbar-track"><div class="pbar-fill" style="width:${lossPct}%"></div></div>
    </div>
    <div class="card">
      <div class="card-label">Win Rate</div>
      <div class="card-value neu">${wr}</div>
      <div class="card-sub">${t.winning_trades ?? 0} / ${t.total_trades ?? 0} trades</div>
    </div>
    <div class="card">
      <div class="card-label">Sharpe</div>
      <div class="card-value neu">${t.sharpe_ratio != null ? t.sharpe_ratio.toFixed(2) : '—'}</div>
    </div>
  `;
}

/* ---- render cycle watch ---- */
function renderWatchSection(w, positions) {
  const wrap = document.getElementById('watch-wrap');
  if (!w || !w.markets || w.markets.length === 0) {
    wrap.innerHTML = '<div class="empty">Waiting for first cycle...</div>';
    return;
  }

  const cycleTime = w.cycle_ts ? ts(w.cycle_ts) : '—';
  const btcStr   = w.btc_price
    ? '$' + w.btc_price.toLocaleString('en-US', {maximumFractionDigits: 0})
    : '—';

  // Parse ticker into a readable name: KXBTC15M-26APR101700-00 → "BTC · 17:00 UTC"
  function tickerLabel(ticker, asset) {
    const MONTHS = {JAN:'Jan',FEB:'Feb',MAR:'Mar',APR:'Apr',MAY:'May',JUN:'Jun',
                    JUL:'Jul',AUG:'Aug',SEP:'Sep',OCT:'Oct',NOV:'Nov',DEC:'Dec'};
    const NAMES  = {BTC:'Bitcoin',ETH:'Ethereum',SOL:'Solana',XRP:'XRP',
                    DOGE:'Dogecoin',HYPE:'HYPE'};
    const parts = (ticker || '').split('-');
    if (parts.length >= 2) {
      const dp = parts[1]; // e.g. 26APR101700
      const re = /^(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{4})$/;
      const m2 = dp.match(re);
      if (m2) {
        const mon  = MONTHS[m2[2]] || m2[2];
        const day  = parseInt(m2[3]);
        const hhmm = m2[4].substring(0,2) + ':' + m2[4].substring(2,4);
        const full = NAMES[asset] || asset || ticker;
        return { name: full, sub: mon + ' ' + day + ' · ' + hhmm + ' UTC' };
      }
    }
    return { name: asset || ticker, sub: ticker };
  }

  const rows = w.markets.map(m => {
    const strike = m.strike ? '$' + m.strike.toLocaleString('en-US', {maximumFractionDigits:4}) : '—';
    const yesAsk = m.yes_ask > 0 ? m.yes_ask + '¢' : '—';
    const noAsk  = m.no_ask  > 0 ? m.no_ask  + '¢' : (m.yes_ask > 0 ? (100 - m.yes_ask) + '¢' : '—');
    const lbl    = tickerLabel(m.ticker, m.asset);
    const titleTip = m.title ? ` title="${m.title.replace(/"/g,'&quot;')}"` : '';
    const closeIso = m.close_time || '';
    return `<tr${titleTip}>
      <td>
        <div style="color:var(--text);font-weight:600">${lbl.name}</div>
        <div style="font-size:10px;color:var(--muted)">${lbl.sub}</div>
        <div style="font-size:9px;color:var(--border)">${m.ticker}</div>
      </td>
      <td>${strike}</td>
      <td><span class="market-countdown" data-close="${closeIso}" style="font-variant-numeric:tabular-nums">—</span></td>
      <td class="dir-yes">${yesAsk}</td>
      <td class="dir-no">${noAsk}</td>
      <td style="color:var(--muted)">${(m.volume || 0).toLocaleString()}</td>
    </tr>`;
  }).join('');

  // ── Helper: render one AI vote panel for a single signal ───────────────
  const mdefs = [
    {key:'claude',   label:'Claude'},
    {key:'gpt',      label:'GPT-4o'},
    {key:'gemini',   label:'Gemini'},
    {key:'deepseek', label:'DeepSeek'},
  ];

  function renderSignalPanel(sig) {
    const sTicker = sig.ticker || '';
    const sAsset  = sTicker.replace(/^KX/,'').replace(/15M.*/,'');
    const sLbl    = tickerLabel(sTicker, sAsset);
    const sLabel  = sLbl.name + (sLbl.sub ? ' &middot; ' + sLbl.sub : '');
    const actCls  = 'ens-action action-' + (sig.action || 'SKIP');

    // Checklist
    let checklistHtml = '';
    if (sig.checks && sig.checks.length) {
      const items = sig.checks.map(c => {
        const icon = c.passed === true ? '&#10003;' : c.passed === false ? '&#10007;' : '&#8212;';
        const cls  = c.passed === true ? 'chk-pass' : c.passed === false ? 'chk-fail' : 'chk-skip';
        const det  = (c.detail && c.detail !== '—') ? `<span class="chk-detail">${c.detail}</span>` : '';
        return `<div class="chk-item ${cls}"><span class="chk-icon">${icon}</span><span class="chk-label">${c.label}</span>${det}</div>`;
      }).join('');
      checklistHtml = `<div class="checklist">${items}</div>`;
    }

    // Vote cards
    const models   = sig.models || {};
    const voteCards = mdefs.map(({key, label}) => {
      const m = models[key];
      if (!m) return `<div class="bot-vote-card vote-fail">
        <div class="bot-vote-name">${label}</div>
        <div class="bot-vote-dir" style="color:var(--muted);font-size:1.6rem;font-weight:700">&#10007;</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">OFFLINE</div>
      </div>`;
      // prob null or 50 means the model returned NO TRADE (no edge) — show WAITING state
      if (m.prob == null || m.prob === 50) {
        return `<div class="bot-vote-card" style="border-color:var(--border);opacity:.7">
          <div class="bot-vote-name">${label}</div>
          <div style="margin:6px 0 4px"><span style="color:var(--muted);font-size:1.1rem;font-weight:700;letter-spacing:1px">&#8212; WAITING</span></div>
          <div class="bot-vote-prob" style="color:var(--muted)">—</div>
          <div class="bot-vote-reason" style="color:var(--muted)">No edge detected<br><span style="opacity:.7">${m.reasoning || ''}</span></div>
        </div>`;
      }
      const isYes = m.direction === 'YES';
      const cls   = isYes ? 'vote-yes' : 'vote-no';
      const col   = isYes ? 'var(--green)' : 'var(--red)';
      const badge = isYes
        ? `<span style="background:#3fb95030;color:var(--green);border:1px solid #3fb95060;border-radius:4px;padding:3px 10px;font-size:1.3rem;font-weight:900;letter-spacing:1px">&#9650; UP</span>`
        : `<span style="background:#f8514930;color:var(--red);border:1px solid #f8514960;border-radius:4px;padding:3px 10px;font-size:1.3rem;font-weight:900;letter-spacing:1px">&#9660; DOWN</span>`;
      const what  = isYes ? 'Expects price <b>above</b> strike' : 'Expects price <b>below</b> strike';
      return `<div class="bot-vote-card ${cls}">
        <div class="bot-vote-name">${label}</div>
        <div style="margin:6px 0 4px">${badge}</div>
        <div class="bot-vote-prob" style="color:${col}">${m.prob}%</div>
        <div class="bot-vote-reason">${what}<br><span style="opacity:.7">${m.reasoning || ''}</span></div>
      </div>`;
    }).join('');

    // Consensus banner
    const working    = mdefs.map(d => models[d.key]).filter(Boolean);
    // Models with null or 50% prob returned NO TRADE — exclude from directional vote
    const withEdge   = working.filter(m => m.prob != null && m.prob !== 50);
    const waitCount  = working.length - withEdge.length;
    const yesCount   = withEdge.filter(m => m.direction === 'YES').length;
    const noCount    = withEdge.filter(m => m.direction === 'NO').length;
    const total      = working.length;
    const allWaiting = total > 0 && waitCount === total;
    const allAgree   = withEdge.length > 0 && (yesCount === withEdge.length || noCount === withEdge.length);
    const winner     = yesCount >= noCount ? 'YES' : 'NO';
    let bannerHtml;
    if (total === 0) {
      bannerHtml = `<div class="consensus-banner cbanner-split">All models offline &mdash; no trade</div>`;
    } else if (allWaiting) {
      bannerHtml = `<div class="consensus-banner cbanner-split">&#8212; WAITING &mdash; No edge detected across all models &mdash; skipping cycle</div>`;
    } else if (allAgree) {
      const bcls   = winner === 'YES' ? 'cbanner-yes' : 'cbanner-no';
      const barrow = winner === 'YES' ? '&#9650;' : '&#9660;';
      const bdir   = winner === 'YES' ? 'UP' : 'DOWN';
      const bwhat  = winner === 'YES' ? 'above strike' : 'below strike';
      bannerHtml = `<div class="consensus-banner ${bcls}">
        ${barrow}&nbsp;FULL CONSENSUS &mdash; <b>${bdir}</b>
        &nbsp;&middot;&nbsp; ${withEdge.length}/${total} bots agree &middot; all expect price to finish <b>${bwhat}</b>
        <span class="cbanner-sub">&nbsp;&#8594; trade if risk gates pass</span>
      </div>`;
    } else {
      bannerHtml = `<div class="consensus-banner cbanner-split">
        &#8764; SPLIT VOTE &mdash; ${yesCount} say UP &middot; ${noCount} say DOWN${waitCount > 0 ? ' &middot; ' + waitCount + ' waiting' : ''}
        &nbsp;&middot;&nbsp; leaning <b>${winner === 'YES' ? 'UP' : 'DOWN'}</b> (${Math.max(yesCount,noCount)}/${total})
        <span class="cbanner-sub">&nbsp;&#8594; models disagree, no trade</span>
      </div>`;
    }

    return `<div class="consensus-panel">
      <div class="consensus-title" style="display:flex;align-items:center;justify-content:space-between">
        <span>AI Bot Votes &mdash; <span style="color:var(--blue)">${sLabel}</span></span>
        <span class="${actCls}" style="font-size:11px">${sig.action}</span>
      </div>
      <div class="bot-vote-cards">${voteCards}</div>
      ${bannerHtml}
      ${checklistHtml}
    </div>`;
  }

  // ── Build all signal panels ──────────────────────────────────────────────
  // Build a quick lookup: ticker → open trade (from /api/positions)
  const openByTicker = {};
  (positions || []).forEach(p => { openByTicker[p.ticker] = p; });

  // Render a "ORDER FILLED" banner when there is already an open trade for a ticker
  function renderFilledPanel(sig, trade) {
    const sTicker = sig.ticker || '';
    const sAsset  = sTicker.replace(/^KX/,'').replace(/15M.*/,'');
    const sLbl    = tickerLabel(sTicker, sAsset);
    const sLabel  = sLbl.name + (sLbl.sub ? ' &middot; ' + sLbl.sub : '');
    const dirCls  = trade.direction === 'YES' ? 'dir-yes' : 'dir-no';
    const dirArrow = trade.direction === 'YES' ? '&#9650;' : '&#9660;';
    const dirWord  = trade.direction === 'YES' ? 'UP / YES' : 'DOWN / NO';
    const entryTs  = trade.timestamp ? ts(trade.timestamp) : '—';
    const entryPx  = trade.entry_price != null ? trade.entry_price + '¢' : '—';
    const curPx    = trade.current_price != null ? trade.current_price + '¢' : '—';
    const pnlPctStr = trade.pnl_pct != null
      ? `<span class="${trade.pnl_pct >= 0 ? 'dir-yes' : 'dir-no'}">${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct.toFixed(1)}%</span>`
      : '—';
    const costStr = trade.size_dollars != null ? '$' + trade.size_dollars.toFixed(2) : '—';
    return `<div class="consensus-panel">
      <div class="consensus-title" style="display:flex;align-items:center;justify-content:space-between">
        <span>Order Filled &mdash; <span style="color:var(--blue)">${sLabel}</span></span>
        <span class="ens-action action-TRADE" style="font-size:11px">FILLED</span>
      </div>
      <div style="display:flex;align-items:center;gap:18px;padding:14px 0 8px;flex-wrap:wrap">
        <div style="font-size:2.2rem;font-weight:900;class="${dirCls}"">${dirArrow} <span class="${dirCls}">${dirWord}</span></div>
        <div style="display:flex;gap:24px;flex-wrap:wrap">
          <div><div style="font-size:10px;color:var(--muted);text-transform:uppercase">Filled at</div><div style="font-size:1.3rem;font-weight:700">${entryPx}</div></div>
          <div><div style="font-size:10px;color:var(--muted);text-transform:uppercase">Cost</div><div style="font-size:1.3rem;font-weight:700">${costStr}</div></div>
          <div><div style="font-size:10px;color:var(--muted);text-transform:uppercase">Current</div><div style="font-size:1.3rem;font-weight:700">${curPx}</div></div>
          <div><div style="font-size:10px;color:var(--muted);text-transform:uppercase">P&amp;L</div><div style="font-size:1.3rem;font-weight:700">${pnlPctStr}</div></div>
          <div><div style="font-size:10px;color:var(--muted);text-transform:uppercase">Opened</div><div style="font-size:1.1rem;color:var(--muted)">${entryTs}</div></div>
        </div>
      </div>
      <div class="consensus-banner cbanner-yes" style="margin-top:4px">&#10003; Order executed &mdash; position is live. Monitoring for exit conditions.</div>
    </div>`;
  }

  const allSignals = w.signals && w.signals.length
    ? w.signals
    : (w.last_signal ? [w.last_signal] : []);

  let signalPanels;
  if (allSignals.length === 0) {
    const placeholders = mdefs.map(({label}) => `
      <div class="bot-vote-card vote-fail">
        <div class="bot-vote-name">${label}</div>
        <div class="bot-vote-dir" style="color:var(--muted);font-size:1.6rem;font-weight:700">?</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px">Waiting...</div>
      </div>`).join('');
    signalPanels = `<div class="consensus-panel">
      <div class="consensus-title">AI Bot Votes &mdash; What Each Model Is Watching</div>
      <div class="bot-vote-cards">${placeholders}</div>
      <div class="consensus-banner cbanner-split">Waiting for next cycle evaluation...</div>
    </div>`;
  } else {
    signalPanels = allSignals.map(sig => {
      const openTrade = openByTicker[sig.ticker];
      return openTrade ? renderFilledPanel(sig, openTrade) : renderSignalPanel(sig);
    }).join('');
  }

  wrap.innerHTML = `
    <div class="watch-meta">
      <span>Cycle: <b>${cycleTime}</b></span>
      <span>BTC: <b>${btcStr}</b></span>
      <span>${w.markets.length} market${w.markets.length !== 1 ? 's' : ''} scanned</span>
    </div>
    <table>
      <thead><tr>
        <th>Market</th><th>Strike</th><th>Expiry</th>
        <th>YES ask</th><th>NO ask</th><th>Volume</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${signalPanels}`;
}

/* ---- render open positions ---- */
function renderPositions(positions) {
  const body = document.getElementById('pos-body');
  if (!positions || positions.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="empty">No open positions</td></tr>';
    return;
  }
  body.innerHTML = positions.map(p => {
    const dirCls = p.direction === 'YES' ? 'dir-yes' : 'dir-no';
    const cur = p.current_price != null ? p.current_price + '¢' : '—';
    const pnlPctStr = p.pnl_pct != null
      ? `<span class="${cc(p.pnl_pct)}">${pct(p.pnl_pct,1)}</span>` : '—';
    return `<tr>
      <td>${p.ticker}</td>
      <td class="${dirCls}">${p.direction}</td>
      <td>$${p.size_dollars?.toFixed(2) ?? '—'}</td>
      <td>${p.entry_price}¢</td>
      <td>${cur}</td>
      <td>${pnlPctStr}</td>
      <td>${p.age ?? '—'}</td>
    </tr>`;
  }).join('');
}

/* ---- render recent trades ---- */
function renderTrades(trades) {
  const body = document.getElementById('trades-body');
  if (!trades || trades.length === 0) {
    body.innerHTML = '<tr><td colspan="8" class="empty">No trades yet</td></tr>';
    return;
  }
  body.innerHTML = trades.map(t => {
    const pnlEl = t.pnl_dollars != null
      ? `<span class="${cc(t.pnl_dollars)}">${$f(t.pnl_dollars)}</span>` : '—';
    const dirCls = t.direction === 'YES' ? 'dir-yes' : 'dir-no';
    const stCls = 'status-' + (t.status || 'closed');
    return `<tr>
      <td>${ts(t.timestamp)}</td>
      <td style="font-size:11px">${t.ticker}</td>
      <td class="${dirCls}">${t.direction}</td>
      <td>$${t.size_dollars?.toFixed(2) ?? '—'}</td>
      <td>${t.entry_price}¢</td>
      <td>${t.exit_price != null ? t.exit_price + '¢' : '—'}</td>
      <td>${pnlEl}</td>
      <td class="${stCls}">${t.exit_reason || t.status}</td>
    </tr>`;
  }).join('');
}

/* ---- render ensemble ---- */
function renderEnsemble(ens) {
  const wrap = document.getElementById('ensemble-wrap');
  if (!ens) { wrap.innerHTML = '<div class="empty">No ensemble data yet</div>'; return; }

  const models = [
    {label:'Claude',   val: ens.claude_prob},
    {label:'GPT-4o',   val: ens.gpt_prob},
    {label:'Gemini',   val: ens.gemini_prob},
    {label:'DeepSeek', val: ens.deepseek_prob},
    {label:'Consensus',val: ens.consensus_prob, bold:true},
  ];

  const action = ens.action || '—';
  const actionCls = 'ens-action action-' + action;
  const ticker = ens.market_ticker
    ? `<span style="color:var(--muted);font-size:11px">${ens.market_ticker}  ${ts(ens.timestamp)}</span>`
    : '';
  const skip = ens.skip_reason
    ? `<div style="color:var(--muted);font-size:11px;margin-top:6px">${ens.skip_reason}</div>` : '';

  wrap.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      ${ticker}
      <span class="${actionCls}">${action}</span>
      ${skip}
    </div>
    <div class="ens-grid">
      ${models.map(m => {
        const v = m.val;
        const vStr = v != null ? (v * 100).toFixed(0) + '%' : '—';
        const vCls = v == null ? 'neu' : v >= 0.5 ? 'pos' : 'neg';
        return `<div class="ens-cell">
          <div class="ens-model">${m.label}</div>
          <div class="ens-prob ${vCls}" ${m.bold ? 'style="font-size:1.4rem"':''}}>${vStr}</div>
        </div>`;
      }).join('')}
    </div>`;
}

/* ---- render direction stats ---- */
function renderDirectionStats(d) {
  const wrap = document.getElementById('direction-wrap');
  if (!d || (!d.YES && !d.NO)) { wrap.innerHTML = '<div class="empty">No closed trades yet</div>'; return; }
  const yes = d.YES || {}; const no = d.NO || {};
  const yrPct = yes.win_rate != null ? (yes.win_rate * 100).toFixed(0) + '%' : '—';
  const nrPct = no.win_rate  != null ? (no.win_rate  * 100).toFixed(0) + '%' : '—';
  const yCls  = yes.win_rate != null ? (yes.win_rate >= 0.5 ? 'pos' : 'neg') : 'neu';
  const nCls  = no.win_rate  != null ? (no.win_rate  >= 0.5 ? 'pos' : 'neg') : 'neu';
  wrap.innerHTML = `<div class="dir-grid">
    <div class="dir-cell">
      <div class="dir-label">YES trades</div>
      <div class="dir-pct ${yCls}">${yrPct}</div>
      <div class="dir-sub">${yes.wins ?? 0} W / ${yes.total ?? 0} total</div>
    </div>
    <div class="dir-cell">
      <div class="dir-label">NO trades</div>
      <div class="dir-pct ${nCls}">${nrPct}</div>
      <div class="dir-sub">${no.wins ?? 0} W / ${no.total ?? 0} total</div>
    </div>
  </div>`;
}

/* ---- render model performance ---- */
function renderModelPerformance(d) {
  const wrap = document.getElementById('model-perf-wrap');
  if (!d || !Object.keys(d).length) { wrap.innerHTML = '<div class="empty">No closed trades yet</div>'; return; }
  const models = [
    {key:'claude',   label:'Claude'},
    {key:'gpt',      label:'GPT-4o'},
    {key:'gemini',   label:'Gemini'},
    {key:'deepseek', label:'DeepSeek'},
  ];
  const rows = models.map(({key, label}) => {
    const m = d[key] || {};
    const acc = m.accuracy != null ? (m.accuracy * 100).toFixed(0) + '%' : '—';
    const cls = m.accuracy != null ? (m.accuracy >= 0.5 ? 'pos' : 'neg') : 'neu';
    return `<tr>
      <td>${label}</td>
      <td class="${cls}" style="font-weight:700">${acc}</td>
      <td style="color:var(--muted)">${m.correct ?? 0} / ${m.total ?? 0}</td>
    </tr>`;
  }).join('');
  wrap.innerHTML = `<table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>Correct / Total</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

/* ---- render 7-day P&L bar chart ---- */
function renderDailyPnl(days) {
  const wrap = document.getElementById('daily-pnl-wrap');
  if (!days || days.length === 0) { wrap.innerHTML = '<div class="empty">No daily stats yet</div>'; return; }
  const maxAbs = Math.max(...days.map(d => Math.abs(d.total_pnl || 0)), 0.01);
  const bars = days.map(d => {
    const v    = d.total_pnl || 0;
    const hPct = Math.round(Math.abs(v) / maxAbs * 100);
    const cls  = v >= 0 ? 'bar-pos' : 'bar-neg';
    const sign = v >= 0 ? '+' : '';
    const dt   = d.date ? d.date.slice(5) : ''; // MM-DD
    return `<div class="bar-wrap">
      <div class="bar-val ${v >= 0 ? 'pos' : 'neg'}">${sign}$${Math.abs(v).toFixed(0)}</div>
      <div class="bar ${cls}" style="height:${hPct}%"></div>
      <div class="bar-label">${dt}</div>
    </div>`;
  }).join('');
  wrap.innerHTML = `<div class="bar-chart">${bars}</div>`;
}

/* ---- live countdown ticker (updates every second) ---- */
function updateCountdowns() {
  const now = Date.now();
  document.querySelectorAll('.market-countdown').forEach(el => {
    const closeIso = el.dataset.close;
    if (!closeIso) { el.textContent = '—'; return; }
    const closeMs = new Date(closeIso).getTime();
    const secsLeft = Math.max(0, Math.floor((closeMs - now) / 1000));
    if (secsLeft === 0) {
      el.textContent = 'EXPIRED';
      el.style.color = 'var(--muted)';
      return;
    }
    const m = Math.floor(secsLeft / 60);
    const s = secsLeft % 60;
    const txt = m + ':' + String(s).padStart(2, '0');
    el.textContent = txt;
    // Color: green > 5min, yellow 2-5min, red < 2min
    el.style.color = secsLeft > 300 ? 'var(--green)'
                   : secsLeft > 120 ? 'var(--yellow)'
                   : 'var(--red)';
  });
}
setInterval(updateCountdowns, 1000);

/* ---- main refresh loop ---- */
async function refresh() {
  try {
    const { stats, positions, trades, botStatus, watch, dirStats, modelPerf, dailyPnl } = await fetchAll();
    renderHeader(stats, botStatus);
    renderCards(stats);
    renderWatchSection(watch, positions);
    renderPositions(positions);
    renderTrades(trades);
    renderEnsemble(stats.last_ensemble);
    renderDirectionStats(dirStats);
    renderModelPerformance(modelPerf);
    renderDailyPnl(dailyPnl);
  } catch(e) {
    document.getElementById('refresh-ts').textContent = 'Error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Thread launcher — call this from runner.py or __main__
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    """Start Flask in a daemon thread. Returns immediately."""
    host = getattr(settings, "DASHBOARD_HOST", "0.0.0.0")
    # Railway injects $PORT; fall back to settings value
    port = int(os.environ.get("PORT", getattr(settings, "DASHBOARD_PORT", 8080)))

    def _serve() -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.WARNING)
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_serve, name="dashboard", daemon=True)
    t.start()


if __name__ == "__main__":
    run_dashboard()
    # Block main thread so the daemon thread keeps running
    threading.Event().wait()

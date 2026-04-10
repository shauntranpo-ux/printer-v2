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
_START_FILE     = Path("START")
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
    """Derive bot state from sentinel files."""
    stop  = _STOP_FILE.exists()
    start = _START_FILE.exists()
    if stop:
        status = "stopped"
    elif start:
        status = "running"
    else:
        status = "waiting"
    return {"status": status, "stop_file": stop, "start_file": start}


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


@app.get("/api/balance")
def api_balance():
    _ensure_db()
    try:
        balance = _run(_db.get_balance())
        return jsonify({"balance": balance})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/bot/start")
def api_bot_start():
    _STOP_FILE.unlink(missing_ok=True)
    _START_FILE.touch()
    return jsonify({"ok": True, "status": "running"})


@app.post("/api/bot/stop")
def api_bot_stop():
    _STOP_FILE.touch()
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

  @media(max-width:600px){
    body{padding:12px}
    .card-value{font-size:1.2rem}
    .cards{grid-template-columns:1fr 1fr}
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
  const [stats, positions, trades, botStatus] = await Promise.all([
    fetch('/api/stats').then(r => r.json()).catch(()=>({})),
    fetch('/api/positions').then(r => r.json()).catch(()=>[]),
    fetch('/api/trades').then(r => r.json()).catch(()=>[]),
    fetch('/api/bot/status').then(r => r.json()).catch(()=>({})),
  ]);
  return { stats, positions, trades, botStatus };
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

/* ---- main refresh loop ---- */
async function refresh() {
  try {
    const { stats, positions, trades, botStatus } = await fetchAll();
    renderHeader(stats, botStatus);
    renderCards(stats);
    renderPositions(positions);
    renderTrades(trades);
    renderEnsemble(stats.last_ensemble);
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

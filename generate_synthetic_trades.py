"""
generate_synthetic_trades.py — Simulate 500 realistic trades for backtesting.

Generates synthetic trade history that faithfully mirrors the actual bot:
  - 4 AI models with actual weights (Claude 0.30, GPT 0.25, Gemini 0.25, DeepSeek 0.20)
  - 4 assets: BTC (100), ETH (130), SOL (130), XRP (140)
  - Realistic win rates (~55% overall) with documented regime/timing/distance correlations
  - Proper PnL = contracts × (exit_price - entry_price) / 100  (Kalshi mechanics)
  - Entry timing distributed across 0-660s window (runner._MAX_TIME_IN)
  - Market regimes inlined in the trade dict (used by backtest_regimes --synthetic)

Output: synthetic_trades.json
  {
    "meta":         {...generation params...},
    "trades":       [...500 trade dicts matching trades table schema + extra fields...],
    "ensemble_log": [...~1500 ensemble evaluation rows...]
  }

Usage:
    python generate_synthetic_trades.py
    python generate_synthetic_trades.py --n 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — mirrors actual bot config/settings.py
# ---------------------------------------------------------------------------

TOTAL_TRADES = 500
SEED = 42

ASSET_COUNTS = {"BTC": 100, "SOL": 130, "XRP": 140, "ETH": 130}
assert sum(ASSET_COUNTS.values()) == TOTAL_TRADES

# Bot model weights from config.py
MODEL_WEIGHTS = {"claude": 0.30, "gpt": 0.25, "gemini": 0.25, "deepseek": 0.20}

# Bot risk parameters from config.py
MIN_EV           = 0.05   # minimum edge required
KELLY_FRACTION   = 0.50
MAX_BET_SIZE     = 5.00
STOP_LOSS_PCT    = 0.70
TAKE_PROFIT_PCT  = 0.65
CONFIDENCE_DECAY = 0.20   # cents
TRAILING_LOCK    = 0.30

# Asset size overrides (from runner.py _ASSET_SIZE_OVERRIDES)
ASSET_SIZE_MULT  = {"BTC": 0.50, "ETH": 1.0, "SOL": 1.0, "XRP": 1.0}

# Simulation time window: Oct 2025 → Apr 2026
SIM_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
SIM_END   = datetime(2026, 4, 10, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Price model — approximate price trajectory for each asset
# ---------------------------------------------------------------------------

def _asset_price(ts: datetime, asset: str) -> float:
    """
    Simplified price model for Oct 2025 – Apr 2026.
    BTC: 65k → 108k peak (Jan 20) → 75k trough (Mar 10) → 85k recovery
    Other assets scale with BTC loosely.
    """
    t0  = SIM_START.timestamp()
    t_pk = datetime(2026, 1, 20, tzinfo=timezone.utc).timestamp()
    t_tr = datetime(2026, 3, 10, tzinfo=timezone.utc).timestamp()
    t_en = SIM_END.timestamp()
    t    = ts.timestamp()

    if t <= t_pk:
        frac = (t - t0) / (t_pk - t0)
        btc = 65_000 + frac * (108_000 - 65_000)
    elif t <= t_tr:
        frac = (t - t_pk) / (t_tr - t_pk)
        btc = 108_000 - frac * (108_000 - 75_000)
    else:
        frac = (t - t_tr) / (t_en - t_tr)
        btc = 75_000 + frac * (85_000 - 75_000)

    scales = {"BTC": 1.0, "ETH": 1/33, "SOL": 1/550, "XRP": 1/37_000}
    return btc * scales[asset]


def _btc_regime(ts: datetime) -> str:
    """Approximate historical BTC regime at a given timestamp."""
    t_pk  = datetime(2026, 1, 20, tzinfo=timezone.utc).timestamp()
    t_tr  = datetime(2026, 3, 10, tzinfo=timezone.utc).timestamp()
    t_nov = datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp()
    t     = ts.timestamp()

    if t < t_nov:
        return "sideways"
    if t_nov <= t < t_pk - 30 * 86400:
        return "bull"
    if t < t_pk:
        return "high_vol"
    if t < t_tr:
        return "bear"
    return "bull"


# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------

MONTH_ABBR = {1:"JAN", 2:"FEB", 3:"MAR", 4:"APR", 5:"MAY", 6:"JUN",
              7:"JUL", 8:"AUG", 9:"SEP", 10:"OCT", 11:"NOV", 12:"DEC"}

ASSET_PREFIXES = {"BTC": "KXBTC15M", "ETH": "KXETH15M",
                  "SOL": "KXSOL15M", "XRP": "KXXRP15M"}

ASSET_STRIKE_STEPS = {"BTC": 500, "ETH": 50, "SOL": 5, "XRP": 0.25}


def _format_strike(asset: str, strike: float) -> str:
    """Format strike into Kalshi ticker segment (T12345, T1p50, B85000 etc.)."""
    step = ASSET_STRIKE_STEPS[asset]
    snapped = round(strike / step) * step

    if asset == "XRP":
        # encode decimal as 'p': 2.50 → T2p50
        whole = int(snapped)
        frac  = round((snapped - whole) * 100)
        if frac == 0:
            return f"T{whole}"
        return f"T{whole}p{frac:02d}"

    if asset in ("ETH", "SOL"):
        return f"T{int(snapped)}"

    # BTC: use B for below-current, T for above-current (YES=price above strike)
    return f"T{int(snapped)}"


def _make_ticker(asset: str, ts: datetime, strike: float) -> str:
    """Build a Kalshi market ticker string."""
    prefix   = ASSET_PREFIXES[asset]
    date_str = f"{ts.day:02d}{MONTH_ABBR[ts.month]}{str(ts.year)[-2:]}"
    seg      = _format_strike(asset, strike)
    return f"{prefix}-{date_str}-{seg}"


# ---------------------------------------------------------------------------
# Win probability model
# ---------------------------------------------------------------------------

# Base win rate by asset
_BASE_WR = {"BTC": 0.52, "ETH": 0.56, "SOL": 0.57, "XRP": 0.55}

# Regime × direction effect on win probability
_REGIME_EFFECT: dict[tuple[str, str], float] = {
    ("bull",     "YES"): +0.08,  ("bull",     "NO"): -0.06,
    ("bear",     "YES"): -0.06,  ("bear",     "NO"): +0.08,
    ("sideways", "YES"):  0.00,  ("sideways", "NO"):  0.00,
    ("high_vol", "YES"): -0.10,  ("high_vol", "NO"): -0.10,
}

# Timing bucket effect
_TIMING_EFFECT = {
    "0-120s":   +0.04,
    "120-300s": +0.02,
    "300-480s":  0.00,
    "480-660s": -0.03,
}

# Strike distance effect (BTC trades only; others get 0)
_DISTANCE_EFFECT = {
    "0-0.5%": -0.06,
    "0.5-1%":  0.00,
    "1-2%":   +0.05,
    "2%+":    +0.02,
}


def _timing_bucket(secs: float) -> str:
    if secs < 120:   return "0-120s"
    if secs < 300:   return "120-300s"
    if secs < 480:   return "300-480s"
    return "480-660s"


def _distance_bucket(pct: float) -> str:
    if pct < 0.5:  return "0-0.5%"
    if pct < 1.0:  return "0.5-1%"
    if pct < 2.0:  return "1-2%"
    return "2%+"


def _win_prob(asset: str, direction: str, regime: str,
              secs: float, dist_pct: float | None) -> float:
    p = _BASE_WR[asset]
    p += _REGIME_EFFECT.get((regime, direction), 0.0)
    p += _TIMING_EFFECT[_timing_bucket(secs)]
    if dist_pct is not None:
        p += _DISTANCE_EFFECT[_distance_bucket(dist_pct)]
    return max(0.32, min(0.80, p))


# ---------------------------------------------------------------------------
# Model probability generation
# ---------------------------------------------------------------------------

# Claude: best calibrated.  GPT: consistent validator.
# Gemini: fast classifier (slightly noisy).  DeepSeek: adversarial skeptic.
_MODEL_STD    = {"claude": 0.06, "gpt": 0.07, "gemini": 0.09, "deepseek": 0.10}
_MODEL_BIAS   = {"claude": +0.02, "gpt": +0.01, "gemini": 0.0, "deepseek": -0.04}


def _model_probs(rng: random.Random, direction: str, won: bool,
                 entry_price: float) -> dict[str, float]:
    """
    Generate per-model P(YES) values.

    True signal strength: 0.62 for correct calls, 0.42 for wrong ones.
    Each model adds calibrated noise and a small bias.
    DeepSeek (adversarial) is most likely to disagree with the winner.
    """
    if direction == "YES":
        true_p = 0.62 if won else 0.42
    else:  # NO trade won means price ended below strike → P(YES) low
        true_p = 0.38 if won else 0.58

    probs: dict[str, float] = {}
    for model in ("claude", "gpt", "gemini", "deepseek"):
        raw = true_p + _MODEL_BIAS[model] + rng.gauss(0, _MODEL_STD[model])
        probs[model] = max(0.05, min(0.95, raw))

    return probs


def _consensus(probs: dict[str, float]) -> float:
    w = MODEL_WEIGHTS
    total = sum(w.values())
    return sum(probs[m] * w[m] / total for m in probs)


def _spread(probs: dict[str, float]) -> float:
    v = list(probs.values())
    return max(v) - min(v)


# ---------------------------------------------------------------------------
# PnL calculation
# ---------------------------------------------------------------------------

def _exit_info(rng: random.Random, won: bool,
               entry_price: float, contracts: int,
               entry_ts: datetime) -> tuple[float, str, str, float, float]:
    """Return (exit_price, exit_reason, status, pnl, peak_pnl_pct)."""

    if won:
        # Winning trades — 55% expire, 35% take-profit, 10% trailing-stop
        roll = rng.random()
        if roll < 0.55:
            reason, status = "expired", "expired"
            exit_price = 100.0
        elif roll < 0.90:
            reason, status = "take_profit", "closed"
            exit_price = min(97.0, round(entry_price * 1.65))
        else:
            reason, status = "trailing_stop", "closed"
            # Trailing stop activates at +30%, exits when drops to +20%
            exit_price = round(entry_price * rng.uniform(1.20, 1.45))
    else:
        # Losing trades — 45% expire, 35% stop_loss, 20% confidence_decay
        roll = rng.random()
        if roll < 0.45:
            reason, status = "expired", "expired"
            exit_price = 0.0
        elif roll < 0.80:
            reason, status = "stop_loss", "closed"
            exit_price = max(1.0, round(entry_price * 0.30))
        else:
            reason, status = "confidence_decay", "closed"
            exit_price = CONFIDENCE_DECAY * 100  # 20¢

    pnl = contracts * (exit_price - entry_price) / 100.0

    # Peak pnl pct: for winning trades, the peak was at least at exit
    peak_pnl_pct = (exit_price - entry_price) / entry_price if exit_price > entry_price else 0.0

    return exit_price, reason, status, pnl, peak_pnl_pct


# ---------------------------------------------------------------------------
# Trade generator
# ---------------------------------------------------------------------------

def _make_trade(rng: random.Random, trade_id: int, asset: str) -> dict:
    # 1. Entry timestamp — random 15-minute boundary in sim window
    total_secs = (SIM_END - SIM_START).total_seconds()
    t_boundary = SIM_START + timedelta(seconds=rng.uniform(0, total_secs - 900))
    # Snap to nearest 15-min boundary
    epoch = int(t_boundary.timestamp())
    boundary = (epoch // 900) * 900
    boundary_dt = datetime.fromtimestamp(boundary, tz=timezone.utc)

    # 2. Seconds into window (entry time within the window)
    #    Weight toward early entries (0-300s)
    if rng.random() < 0.60:
        secs = rng.uniform(30, 300)
    else:
        secs = rng.uniform(300, 650)
    entry_ts = boundary_dt + timedelta(seconds=secs)
    timing_bkt = _timing_bucket(secs)

    # 3. Asset price at entry
    asset_price = _asset_price(entry_ts, asset)
    # Add ±1% daily noise
    asset_price *= 1.0 + rng.gauss(0, 0.008)
    btc_price = _asset_price(entry_ts, "BTC") * (1.0 + rng.gauss(0, 0.005))

    # 4. Strike price — generate distance_pct from realistic distribution
    #    Log-normal centered at 1%, bounded [0.1%, 4%]
    raw_dist = math.exp(rng.gauss(-0.2, 0.7))  # log-normal ~ median 0.82%
    dist_pct = max(0.1, min(4.0, raw_dist))

    # Direction of distance: above or below strike
    above_strike = rng.random() < 0.50  # price equally likely above/below
    if above_strike:
        strike = asset_price * (1 + dist_pct / 100)
    else:
        strike = asset_price * (1 - dist_pct / 100)

    # 5. Direction — 60% YES, 40% NO
    direction = "YES" if rng.random() < 0.60 else "NO"

    # 6. Regime
    regime = _btc_regime(entry_ts)
    # Add small random variation per asset
    if rng.random() < 0.15:  # 15% chance asset regime differs from BTC
        regime = rng.choice(["bull", "bear", "sideways", "high_vol"])

    # 7. Win probability
    btc_dist_pct = dist_pct if asset == "BTC" else None
    p_win = _win_prob(asset, direction, regime, secs, btc_dist_pct)
    won   = rng.random() < p_win

    # 8. Entry price (cents) — the ask price paid
    #    YES trades: entry at 35-70¢ (bot buys yes contracts)
    #    NO trades: entry at 30-65¢
    if direction == "YES":
        entry_price = rng.uniform(35, 70)
    else:
        entry_price = rng.uniform(30, 65)
    entry_price = round(entry_price, 1)

    # 9. Model probabilities
    model_probs = _model_probs(rng, direction, won, entry_price)
    consensus   = _consensus(model_probs)
    spread_val  = _spread(model_probs)

    # 10. Edge and confidence
    if direction == "YES":
        edge = consensus - entry_price / 100.0
    else:
        edge = (1.0 - consensus) - entry_price / 100.0
    # Ensure edge >= MIN_EV (the bot's gate filters these out)
    edge = max(MIN_EV, edge)

    # Confidence: average of model confidences (rough estimate)
    # High-spread → penalized confidence
    avg_conf_raw = rng.uniform(0.45, 0.80)
    confidence   = avg_conf_raw * (0.80 if spread_val > 0.20 else 1.0)
    confidence   = round(max(0.25, min(0.90, confidence)), 3)

    # 11. Trade sizing — Kelly-based, capped at MAX_BET_SIZE * asset_mult
    max_size  = MAX_BET_SIZE * ASSET_SIZE_MULT.get(asset, 1.0)
    kelly_sz  = edge * KELLY_FRACTION * 10.0  # simplified Kelly in dollars
    size_dollars = min(max_size, max(0.50, round(kelly_sz * rng.uniform(0.7, 1.3), 2)))
    contracts = max(1, int(size_dollars * 100 / entry_price))
    # Recalc size from contracts
    size_dollars = round(contracts * entry_price / 100.0, 4)

    # 12. Exit
    exit_price, exit_reason, status, pnl, peak_pnl_pct = _exit_info(
        rng, won, entry_price, contracts, entry_ts
    )

    # 13. Closed timestamp (within or at end of 15-min window)
    window_end = boundary_dt + timedelta(seconds=900)
    if status == "expired":
        closed_at = window_end
    else:
        # Early exit somewhere between entry and window end
        remaining = (window_end - entry_ts).total_seconds()
        closed_at = entry_ts + timedelta(seconds=rng.uniform(60, remaining))

    # 14. BTC momentum (-1 to +1)
    btc_momentum = round(rng.gauss(0, 0.35), 3)
    btc_momentum = max(-1.0, min(1.0, btc_momentum))

    # 15. Build ticker
    ticker = _make_ticker(asset, entry_ts, strike)

    return {
        # ── Standard trades table columns ────────────────────────────────
        "id":                  trade_id,
        "timestamp":           entry_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "closed_at":           closed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_ticker":       ticker,
        "asset_symbol":        asset,
        "direction":           direction,
        "entry_price":         round(entry_price, 2),
        "exit_price":          round(exit_price, 2),
        "size_dollars":        round(size_dollars, 4),
        "contracts":           contracts,
        "kelly_fraction":      KELLY_FRACTION,
        "edge":                round(edge, 4),
        "ensemble_confidence": round(confidence, 4),
        "model_spread":        round(spread_val, 4),
        "btc_price_at_entry":  round(btc_price, 2),
        "btc_momentum":        btc_momentum,
        "status":              status,
        "exit_reason":         exit_reason,
        "pnl_dollars":         round(pnl, 4),
        "peak_pnl_pct":        round(peak_pnl_pct, 4),
        "claude_prob":         round(model_probs["claude"], 4),
        "gpt_prob":            round(model_probs["gpt"], 4),
        "gemini_prob":         round(model_probs["gemini"], 4),
        "deepseek_prob":       round(model_probs["deepseek"], 4),
        "tp_order_id":         None,
        "sl_order_id":         None,

        # ── Extra fields for --synthetic backtest mode ────────────────────
        "secs_into_window":    round(secs, 1),
        "timing_bucket":       timing_bkt,
        "regime_own":          regime,
        "regime_btc":          _btc_regime(entry_ts),
        "strike_distance_pct": round(dist_pct, 4),
        "distance_bucket":     _distance_bucket(dist_pct),
        "won":                 won,
    }


# ---------------------------------------------------------------------------
# Ensemble log generator
# ---------------------------------------------------------------------------

def _make_ensemble_rows(rng: random.Random, trades: list[dict]) -> list[dict]:
    """
    Generate ~3× as many ensemble_log rows as trades.
    Traded rows come from actual trade entries.
    WAIT/SKIP rows are synthetic evaluations that didn't convert.
    """
    rows: list[dict] = []
    eid = 1

    for t in trades:
        # Row corresponding to the actual trade placed (action=TRADE)
        rows.append({
            "id":            eid,
            "market_ticker": t["market_ticker"],
            "timestamp":     t["timestamp"],
            "action":        "TRADE",
            "consensus_prob": round(
                t["claude_prob"] * 0.30 + t["gpt_prob"] * 0.25 +
                t["gemini_prob"] * 0.25 + t["deepseek_prob"] * 0.20, 4
            ),
            "model_spread":  t["model_spread"],
            "confidence":    t["ensemble_confidence"],
            "claude_prob":   t["claude_prob"],
            "gpt_prob":      t["gpt_prob"],
            "gemini_prob":   t["gemini_prob"],
            "deepseek_prob": t["deepseek_prob"],
            "skip_reason":   None,
        })
        eid += 1

        # 2 extra WAIT or SKIP rows for every traded market
        for _ in range(2):
            is_wait = rng.random() < 0.55
            action  = "WAIT" if is_wait else "SKIP"

            c_p  = round(rng.uniform(0.30, 0.70), 4)
            g_p  = round(rng.uniform(0.30, 0.70), 4)
            ge_p = round(rng.uniform(0.30, 0.70), 4)
            ds_p = round(rng.uniform(0.30, 0.70), 4)
            cons = round(c_p * 0.30 + g_p * 0.25 + ge_p * 0.25 + ds_p * 0.20, 4)
            spr  = round(max(c_p, g_p, ge_p, ds_p) - min(c_p, g_p, ge_p, ds_p), 4)
            conf = round(rng.uniform(0.20, 0.55), 4)

            if is_wait:
                skip_reason = f"model spread {spr:.3f} > 0.50 with only 2/4 models agreeing"
            else:
                skip_reason = "confidence below threshold"

            # Offset timestamp slightly (different market, same cycle)
            ts_dt = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
            ts_dt += timedelta(seconds=rng.randint(5, 45))

            rows.append({
                "id":            eid,
                "market_ticker": _make_ticker(
                    rng.choice(list(ASSET_PREFIXES.keys())),
                    ts_dt,
                    rng.uniform(100, 200),
                ),
                "timestamp":     ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "action":        action,
                "consensus_prob": cons,
                "model_spread":  spr,
                "confidence":    conf,
                "claude_prob":   c_p,
                "gpt_prob":      g_p,
                "gemini_prob":   ge_p,
                "deepseek_prob": ds_p,
                "skip_reason":   skip_reason,
            })
            eid += 1

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n: int = TOTAL_TRADES, seed: int = SEED) -> dict:
    rng = random.Random(seed)

    # Build trade list in asset order (so distribution is exact)
    trade_id = 1
    trades: list[dict] = []
    for asset, count in ASSET_COUNTS.items():
        for _ in range(count):
            trade = _make_trade(rng, trade_id, asset)
            trades.append(trade)
            trade_id += 1

    # Shuffle so assets are intermixed chronologically
    rng.shuffle(trades)

    # Re-assign IDs after shuffle
    for i, t in enumerate(trades, 1):
        t["id"] = i

    ensemble_log = _make_ensemble_rows(rng, trades)

    # Compute summary stats
    pnls     = [t["pnl_dollars"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    total_pnl = sum(pnls)
    win_rate  = len(wins) / len(pnls) * 100

    by_asset: dict[str, dict] = {}
    for asset in ASSET_COUNTS:
        at = [t for t in trades if t["asset_symbol"] == asset]
        aw = [t for t in at if t["pnl_dollars"] > 0]
        by_asset[asset] = {
            "n":            len(at),
            "win_rate_pct": round(len(aw) / len(at) * 100, 1),
            "total_pnl":    round(sum(t["pnl_dollars"] for t in at), 2),
        }

    by_regime: dict[str, dict] = {}
    for regime in ("bull", "bear", "sideways", "high_vol"):
        rt = [t for t in trades if t["regime_btc"] == regime]
        rw = [t for t in rt if t["pnl_dollars"] > 0]
        if rt:
            by_regime[regime] = {
                "n":            len(rt),
                "win_rate_pct": round(len(rw) / len(rt) * 100, 1),
                "total_pnl":    round(sum(t["pnl_dollars"] for t in rt), 2),
            }

    by_bucket: dict[str, dict] = {}
    for bkt in ("0-120s", "120-300s", "300-480s", "480-660s"):
        bt = [t for t in trades if t["timing_bucket"] == bkt]
        bw = [t for t in bt if t["pnl_dollars"] > 0]
        if bt:
            by_bucket[bkt] = {
                "n":            len(bt),
                "win_rate_pct": round(len(bw) / len(bt) * 100, 1),
                "total_pnl":    round(sum(t["pnl_dollars"] for t in bt), 2),
            }

    return {
        "meta": {
            "n_trades":        n,
            "seed":            seed,
            "win_rate_pct":    round(win_rate, 2),
            "total_pnl":       round(total_pnl, 2),
            "sim_start":       SIM_START.strftime("%Y-%m-%d"),
            "sim_end":         SIM_END.strftime("%Y-%m-%d"),
            "n_ensemble_rows": len(ensemble_log),
            "by_asset":        by_asset,
            "by_btc_regime":   by_regime,
            "by_timing_bucket": by_bucket,
            "note": (
                "Synthetic data — generated from bot parameters to enable "
                "backtesting before live trade history accumulates. "
                "Win-rate correlations: early entries > late, "
                "1-2% strike distance > near knife-edge, "
                "bull+YES / bear+NO > sideways > high_vol."
            ),
        },
        "trades":       trades,
        "ensemble_log": ensemble_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic trades for printer-v2 backtesting"
    )
    parser.add_argument("--n",    type=int, default=TOTAL_TRADES, help="Number of trades")
    parser.add_argument("--seed", type=int, default=SEED,         help="RNG seed")
    parser.add_argument("--out",  default="synthetic_trades.json", help="Output file")
    args = parser.parse_args()

    print(f"Generating {args.n} synthetic trades (seed={args.seed}) ...")
    data = generate(n=args.n, seed=args.seed)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved -> {out_path}")

    m = data["meta"]
    print(f"\n  Total: {m['n_trades']} trades  |  WR={m['win_rate_pct']}%  |  P&L=${m['total_pnl']:+.2f}")
    print(f"  Ensemble log: {m['n_ensemble_rows']} rows  (TRADE + WAIT + SKIP)")
    print(f"\n  By asset:")
    for asset, s in m["by_asset"].items():
        print(f"    {asset:>4}  n={s['n']:>3}  WR={s['win_rate_pct']:>5.1f}%  P&L=${s['total_pnl']:>+7.2f}")
    print(f"\n  By BTC regime:")
    for regime, s in m["by_btc_regime"].items():
        print(f"    {regime:>10}  n={s['n']:>3}  WR={s['win_rate_pct']:>5.1f}%  P&L=${s['total_pnl']:>+7.2f}")
    print(f"\n  By timing bucket:")
    for bkt, s in m["by_timing_bucket"].items():
        print(f"    {bkt:>12}  n={s['n']:>3}  WR={s['win_rate_pct']:>5.1f}%  P&L=${s['total_pnl']:>+7.2f}")


if __name__ == "__main__":
    main()

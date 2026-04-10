"""
backtest.py — Backtest printer-v2 strategy on historical BTC 1m data.

Usage:
    python backtest.py --file btc_1m_data.xlsx
    python backtest.py --file data.csv --start 2020-01-01 --end 2023-12-31
    python backtest.py --file data.csv --bankroll 500 --max-bet 25 --tp 0.50 --sl 0.35

Dependencies (pip install if missing):
    pandas openpyxl scipy matplotlib numpy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# Strategy constants  (mirrors printer-v2 defaults)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BANKROLL       = 500.0
DEFAULT_MAX_BET        = 25.0
DEFAULT_TP             = 0.50   # take profit at +50 %
DEFAULT_SL             = 0.35   # stop loss at -35 %
MIN_EDGE               = 0.03   # lower than live (rule-based proxy has smaller edge)
MIN_CONF               = 0.18   # calibrated for rule-based proxy (max achievable ~0.60)
MAX_SPREAD             = 0.35
MAX_POSITIONS          = 3      # max concurrent trades per candle
DAILY_LOSS_LIMIT       = 100.0
KELLY_FRACTION         = 0.5
MIN_BET                = 1.0
STRIKE_OFFSETS         = [+0.005, 0.0, -0.005]   # +0.5 %, flat, -0.5 %

MODEL_WEIGHTS = {"claude": 0.30, "gpt": 0.25, "gemini": 0.25, "deepseek": 0.20}

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:          datetime
    exit_time:           datetime
    direction:           str       # "YES" | "NO"
    strike_price:        float
    entry_price:         float     # cents
    exit_price:          float     # cents
    bet_size:            float
    contracts:           int
    pnl_dollars:         float
    pnl_pct:             float
    exit_reason:         str
    momentum_at_entry:   float
    confidence_at_entry: float
    edge_at_entry:       float
    btc_price_at_entry:  float
    hold_candles:        int
    peak_pnl_pct:        float


@dataclass
class Position:
    entry_idx:           int
    entry_time:          datetime
    direction:           str
    strike_price:        float
    entry_price_cents:   float
    bet_size:            float
    contracts:           int
    momentum_at_entry:   float
    confidence_at_entry: float
    edge_at_entry:       float
    btc_price_at_entry:  float


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Data loading + resampling
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample(
    file_path: str,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    path = Path(file_path)
    print(f"Loading {path.name} ...")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.lower().strip() for c in df.columns]

    # Convert unix timestamp → UTC datetime
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        df["dt"] = pd.to_datetime(df["time"], utc=True)

    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype(float)

    print(f"  1m rows: {len(df):,}  ({df.index[0].date()} to {df.index[-1].date()})")

    # Resample 1m → 15m
    df15 = df.resample("15min").agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum"),
    ).dropna()

    # Drop zero/missing volume candles
    df15 = df15[df15["volume"] > 0]

    if start:
        df15 = df15[df15.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df15 = df15[df15.index <= pd.Timestamp(end, tz="UTC")]

    print(f"  15m candles: {len(df15):,}  ({df15.index[0].date()} to {df15.index[-1].date()})")
    return df15.reset_index()   # 'dt' becomes a regular column


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Precompute all indicators (vectorised — runs once before the main loop)
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Return arrays of equal length to df for every indicator used by the
    ensemble and by the market-pricing formula.
    """
    c = df["close"]
    o = df["open"]
    v = df["volume"]
    n = len(df)

    # ── Log returns ──────────────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1)).fillna(0.0).values

    # ── Rolling volatility (20-candle std of log returns) ────────────────────
    roll_vol = (
        pd.Series(log_ret).rolling(20, min_periods=2).std().fillna(0.01).values
    )

    # ── RSI-14 ───────────────────────────────────────────────────────────────
    delta = c.diff().fillna(0.0)
    gain  = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0.0, np.nan)
    rsi   = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).values

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    macd_line    = (ema12 - ema26).values
    signal_line  = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_above   = (macd_line > signal_line).astype(float)  # 1=bullish

    # ── Bollinger band %B (20-period, 2σ) ────────────────────────────────────
    sma20   = c.rolling(20, min_periods=2).mean()
    std20   = c.rolling(20, min_periods=2).std().fillna(0.0)
    bb_up   = sma20 + 2.0 * std20
    bb_lo   = sma20 - 2.0 * std20
    bb_pct  = ((c - bb_lo) / (bb_up - bb_lo).replace(0.0, np.nan)).fillna(0.5)
    bb_pct  = bb_pct.clip(0.0, 1.0).values

    # ── Momentum — exact coinbase_feed.py formula ─────────────────────────────
    # Direction: recency-weighted (0.1/0.2/0.3/0.4) tanh of body %
    body_pct  = np.tanh((c - o) / o.replace(0.0, np.nan) * 100.0).fillna(0.0)
    dir_score = (
        body_pct.shift(3).fillna(0.0) * 0.1
        + body_pct.shift(2).fillna(0.0) * 0.2
        + body_pct.shift(1).fillna(0.0) * 0.3
        + body_pct                       * 0.4
    ).values  # weights already sum to 1.0

    # Velocity: tanh of 4-candle price change
    c4     = c.shift(4).replace(0.0, np.nan)
    vel    = np.tanh((c - c4) / c4 * 100.0).fillna(0.0).values

    # Volume confirmation
    mean_v4   = v.rolling(4, min_periods=1).mean().replace(0.0, np.nan)
    vol_ratio = (v / mean_v4 - 1.0).fillna(0.0)
    dir_sign  = np.where(dir_score >= 0, 1.0, -1.0)
    vol_fac   = (np.tanh(vol_ratio.values) * dir_sign)

    momentum = np.clip(0.40 * dir_score + 0.40 * vel + 0.20 * vol_fac, -1.0, 1.0)

    return {
        "log_ret":   log_ret,
        "vol":       roll_vol,       # rolling 20-candle std of returns
        "rsi":       rsi,
        "macd_abv":  macd_above,
        "bb_pct":    bb_pct,
        "momentum":  momentum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Rule-based ensemble (proxy for 4-model AI debate)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_at(i: int, ind: dict[str, np.ndarray]) -> dict:
    """
    Return ensemble signal dict for candle index i.
    Mirrors the real ensemble's output structure used by runner.py.
    """
    momentum  = float(ind["momentum"][i])
    rsi       = float(ind["rsi"][i])
    macd_abv  = float(ind["macd_abv"][i])   # 1 = bullish, 0 = bearish
    bb_pct    = float(ind["bb_pct"][i])      # 0=lower, 1=upper band

    # ── Claude: momentum-based ───────────────────────────────────────────────
    # Probabilities calibrated to real AI output range (0.15–0.85)
    if   momentum >  0.3:  claude = 0.82
    elif momentum >  0.1:  claude = 0.64
    elif momentum < -0.3:  claude = 0.18
    elif momentum < -0.1:  claude = 0.36
    else:                  claude = 0.50

    # ── GPT: RSI-based (bullish lens) ────────────────────────────────────────
    if   rsi > 70:  gpt = 0.82
    elif rsi > 60:  gpt = 0.66
    elif rsi < 30:  gpt = 0.18
    elif rsi < 40:  gpt = 0.34
    else:           gpt = 0.50

    # ── Gemini: MACD-based ───────────────────────────────────────────────────
    gemini = (0.72 if macd_abv else 0.26)

    # ── DeepSeek: Bollinger-based (risk manager) ─────────────────────────────
    if   bb_pct < 0.20:  deepseek = 0.80
    elif bb_pct < 0.40:  deepseek = 0.64
    elif bb_pct > 0.80:  deepseek = 0.20
    elif bb_pct > 0.60:  deepseek = 0.36
    else:                deepseek = 0.50

    probs = {"claude": claude, "gpt": gpt, "gemini": gemini, "deepseek": deepseek}

    # Weighted consensus (renormalised to sum of weights)
    total_w   = sum(MODEL_WEIGHTS.values())
    consensus = sum(p * MODEL_WEIGHTS[m] for m, p in probs.items()) / total_w
    spread    = max(probs.values()) - min(probs.values())

    # Confidence = avg absolute distance from 0.5, penalised if spread > 0.20
    avg_conf   = sum(abs(p - 0.5) * 2.0 for p in probs.values()) / len(probs)
    confidence = avg_conf * 0.8 if spread > 0.20 else avg_conf

    if   spread    > MAX_SPREAD:  action = "WAIT"
    elif confidence < MIN_CONF:   action = "SKIP"
    else:                         action = "TRADE"

    return {
        "consensus":  consensus,
        "confidence": confidence,
        "spread":     spread,
        "action":     action,
        "direction":  "YES" if consensus > 0.5 else "NO",
        "momentum":   momentum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Market pricing
# ─────────────────────────────────────────────────────────────────────────────

def yes_price_cents(btc: float, strike: float, vol: float,
                    time_fraction: float = 1.0) -> float:
    """
    Black-Scholes-inspired probability for 'BTC above strike'.
    time_fraction: 1.0 = full period remaining, 0.0 = at expiry.
    Returns a price in cents [5, 95].
    """
    if vol <= 0.0:
        vol = 0.001
    adj_vol = vol * max(time_fraction, 0.001) ** 0.5
    distance = (strike - btc) / btc
    prob = float(norm.cdf(-distance / adj_vol))
    return max(5.0, min(95.0, prob * 100.0))



# ─────────────────────────────────────────────────────────────────────────────
# 5 · Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def kelly_size(edge: float, mkt_p: float, bankroll: float, max_bet: float) -> float:
    """Return bet size in dollars (rounded to nearest $0.50)."""
    if mkt_p <= 0.0 or mkt_p >= 1.0 or edge <= 0.0:
        return 0.0
    odds      = (1.0 - mkt_p) / mkt_p
    kelly_pct = (edge / odds) * KELLY_FRACTION
    if kelly_pct <= 0.0:
        return 0.0
    raw  = kelly_pct * bankroll
    size = min(raw, max_bet)
    size = math.floor(size / 0.5) * 0.5
    return size if size >= MIN_BET else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Main backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df:  pd.DataFrame,
    ind: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[Trade], pd.Series]:
    """
    Accurate Kalshi single-candle simulation.

    Each 15-min candle is one contract period.  Entry is at candle open;
    the contract resolves at candle close (binary: $1 win / $0 loss) unless
    TP or SL is hit intra-candle using candle high/low.

    P&L model (binary):
      WIN  → pnl = contracts × (1.00 − entry_price)          [+ve]
      LOSS → pnl = −contracts × entry_price                   [−ve]
      TP   → pnl = (tp_exit_cents − entry_cents) × contracts / 100
      SL   → pnl = (sl_exit_cents − entry_cents) × contracts / 100
    """
    bankroll  = args.bankroll
    max_bet   = args.max_bet
    tp        = args.tp
    sl        = args.sl

    opens     = df["open"].values
    highs     = df["high"].values
    lows      = df["low"].values
    closes    = df["close"].values
    times     = df["dt"].values
    n         = len(df)

    trades:      list[Trade] = []
    equity_vals: list[float] = []
    daily_loss:  dict        = {}

    WARMUP = 30

    for i in range(n):
        ts = pd.Timestamp(times[i]).to_pydatetime()
        equity_vals.append(bankroll)

        if i < WARMUP:
            continue

        btc_open  = opens[i]
        btc_high  = highs[i]
        btc_low   = lows[i]
        btc_close = closes[i]
        vol = max(float(ind["vol"][i]), 0.001)
        day = ts.date().isoformat()

        if daily_loss.get(day, 0.0) >= DAILY_LOSS_LIMIT:
            continue

        ens = ensemble_at(i, ind)
        if ens["action"] != "TRADE":
            continue

        direction = ens["direction"]
        momentum  = ens["momentum"]

        # Skip only on strong counter-momentum
        if direction == "YES" and momentum < -0.3:
            continue
        if direction == "NO"  and momentum >  0.3:
            continue

        candle_trades = 0

        for offset in STRIKE_OFFSETS:
            if candle_trades >= MAX_POSITIONS:
                break

            # Strike is based on open price (entry BTC)
            strike = btc_open * (1.0 + offset)

            # Entry price from Black-Scholes at candle open
            yes_entry_frac = yes_price_cents(btc_open, strike, vol) / 100.0

            if direction == "YES":
                entry_frac = yes_entry_frac
                edge       = ens["consensus"] - entry_frac
            else:
                entry_frac = 1.0 - yes_entry_frac
                edge       = (1.0 - ens["consensus"]) - entry_frac

            if edge < MIN_EDGE:
                continue

            bet = kelly_size(edge, entry_frac, bankroll, max_bet)
            if bet < MIN_BET:
                continue

            contracts = int(bet / entry_frac)
            if contracts < 1:
                continue

            entry_cents = entry_frac * 100.0
            cost        = contracts * entry_cents / 100.0   # actual dollars at risk

            # ── Intra-candle TP / SL using candle high / low ─────────────────
            # YES contract rises with BTC → check TP against HIGH, SL against LOW
            # NO  contract rises with BTC falling → check TP against LOW, SL against HIGH

            if direction == "YES":
                yes_at_best  = yes_price_cents(btc_high, strike, vol)
                yes_at_worst = yes_price_cents(btc_low,  strike, vol)
                contract_at_best  = yes_at_best
                contract_at_worst = yes_at_worst
            else:
                yes_at_best  = yes_price_cents(btc_low,  strike, vol)
                yes_at_worst = yes_price_cents(btc_high, strike, vol)
                contract_at_best  = 100.0 - yes_at_best
                contract_at_worst = 100.0 - yes_at_worst

            tp_threshold = entry_cents * (1.0 + tp)
            sl_threshold = entry_cents * (1.0 - sl)

            if contract_at_best >= tp_threshold:
                # TP hit intra-candle — sell at TP price
                reason      = "take_profit"
                exit_cents  = tp_threshold
                pnl         = (exit_cents - entry_cents) * contracts / 100.0
                pnl_pct     = pnl / cost

            elif contract_at_worst <= sl_threshold:
                # SL hit intra-candle — sell at SL price
                reason      = "stop_loss"
                exit_cents  = sl_threshold
                pnl         = (exit_cents - entry_cents) * contracts / 100.0
                pnl_pct     = pnl / cost

            else:
                # Contract expires at candle close — binary resolution
                if direction == "YES":
                    won = btc_close >= strike
                else:
                    won = btc_close < strike

                reason     = "expired"
                exit_cents = 100.0 if won else 0.0

                if won:
                    # Receive $1 per contract, subtract cost
                    pnl     = contracts * (100.0 - entry_cents) / 100.0
                else:
                    # Lose entire stake
                    pnl     = -cost
                pnl_pct = pnl / cost

            if pnl < 0:
                daily_loss[day] = daily_loss.get(day, 0.0) + abs(pnl)
            bankroll        += pnl
            equity_vals[-1]  = bankroll

            trades.append(Trade(
                entry_time          = ts,
                exit_time           = ts,   # same 15-min candle
                direction           = direction,
                strike_price        = round(strike, 2),
                entry_price         = round(entry_cents, 2),
                exit_price          = round(exit_cents, 2),
                bet_size            = round(bet, 2),
                contracts           = contracts,
                pnl_dollars         = round(pnl, 4),
                pnl_pct             = round(pnl_pct * 100.0, 2),
                exit_reason         = reason,
                momentum_at_entry   = round(momentum, 4),
                confidence_at_entry = round(ens["confidence"], 4),
                edge_at_entry       = round(edge, 4),
                btc_price_at_entry  = round(btc_open, 2),
                hold_candles        = 1,
                peak_pnl_pct        = round(max(pnl_pct * 100.0, 0.0), 2),
            ))

            candle_trades += 1

    equity_series = pd.Series(equity_vals, index=df["dt"])
    return trades, equity_series


# ─────────────────────────────────────────────────────────────────────────────
# 7 · Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    trades: list[Trade],
    equity: pd.Series,
    starting_bankroll: float,
) -> dict:
    if not trades:
        return {}

    pnls   = [t.pnl_dollars for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(trades)

    win_rate     = len(wins) / n * 100
    total_pnl    = sum(pnls)
    avg_win      = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss     = sum(losses) / len(losses) if losses else 0.0
    best         = max(pnls)
    worst        = min(pnls)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    arr = np.array(pnls)
    sharpe = float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0

    # Max drawdown from equity curve
    eq    = equity.values
    peak  = np.maximum.accumulate(eq)
    dd    = (peak - eq) / np.where(peak > 0, peak, 1.0) * 100.0
    max_dd = float(dd.max())

    avg_hold = np.mean([t.hold_candles * 15 for t in trades])

    win_pcts  = [t.pnl_pct for t in trades if t.pnl_dollars > 0]
    loss_pcts = [t.pnl_pct for t in trades if t.pnl_dollars <= 0]

    # Exit breakdown
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # Direction breakdown
    yes_t  = [t for t in trades if t.direction == "YES"]
    no_t   = [t for t in trades if t.direction == "NO"]
    yes_wr = sum(1 for t in yes_t if t.pnl_dollars > 0) / len(yes_t) * 100 if yes_t else 0.0
    no_wr  = sum(1 for t in no_t  if t.pnl_dollars > 0) / len(no_t)  * 100 if no_t  else 0.0

    # Monthly breakdown
    monthly: dict[str, dict] = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly[key]["trades"] += 1
        if t.pnl_dollars > 0:
            monthly[key]["wins"] += 1
        monthly[key]["pnl"] += t.pnl_dollars
    for v in monthly.values():
        v["win_rate"] = v["wins"] / v["trades"] * 100 if v["trades"] else 0.0

    return {
        "total_trades":    n,
        "win_rate":        round(win_rate,    2),
        "total_pnl":       round(total_pnl,   4),
        "return_pct":      round(total_pnl / starting_bankroll * 100, 2),
        "max_drawdown":    round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 4),
        "profit_factor":   round(profit_factor, 4) if profit_factor != float("inf") else "∞",
        "avg_win":         round(avg_win,  4),
        "avg_loss":        round(avg_loss, 4),
        "avg_win_pct":     round(np.mean(win_pcts)  if win_pcts  else 0.0, 2),
        "avg_loss_pct":    round(np.mean(loss_pcts) if loss_pcts else 0.0, 2),
        "best_trade":      round(best,  4),
        "worst_trade":     round(worst, 4),
        "avg_hold_min":    round(float(avg_hold), 1),
        "exit_reasons":    reasons,
        "yes_count":       len(yes_t),
        "no_count":        len(no_t),
        "yes_win_rate":    round(yes_wr, 2),
        "no_win_rate":     round(no_wr,  2),
        "monthly":         monthly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8 · Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    m:                 dict,
    df:                pd.DataFrame,
    starting_bankroll: float,
) -> None:
    n  = m["total_trades"]
    rs = m["exit_reasons"]

    def rl(key, label):
        cnt = rs.get(key, 0)
        pct = cnt / n * 100 if n else 0.0
        return f"  {label:<16} {cnt:>5}  ({pct:>5.1f}%)"

    monthly  = m["monthly"]
    m_lines  = "\n".join(
        f"  {k}:  {v['trades']:>4} trades | {v['win_rate']:>5.1f}% WR | "
        f"${v['pnl']:>+9.2f}"
        for k, v in sorted(monthly.items())
    )
    best_m  = max(monthly, key=lambda k: monthly[k]["pnl"]) if monthly else "—"
    worst_m = min(monthly, key=lambda k: monthly[k]["pnl"]) if monthly else "—"
    best_p  = monthly[best_m]["pnl"]  if best_m  != "—" else 0.0
    worst_p = monthly[worst_m]["pnl"] if worst_m != "—" else 0.0

    SEP = "=" * 47

    print(f"""
{SEP}
        PRINTER V2 BACKTEST RESULTS
{SEP}
  Period:          {df['dt'].iloc[0].date()} to {df['dt'].iloc[-1].date()}
  Total candles:   {len(df):,}
  Total trades:    {n}
{SEP}
  PERFORMANCE
  Win rate:        {m['win_rate']:.1f}%
  Total P&L:       ${m['total_pnl']:+.2f}
  Return:          {m['return_pct']:+.2f}% on starting ${starting_bankroll:.0f}
  Max drawdown:    {m['max_drawdown']:.2f}%
  Sharpe ratio:    {m['sharpe_ratio']:.3f}
  Profit factor:   {m['profit_factor']}
{SEP}
  TRADE BREAKDOWN
  Avg win:         ${m['avg_win']:+.2f}  ({m['avg_win_pct']:+.1f}%)
  Avg loss:        ${m['avg_loss']:+.2f}  ({m['avg_loss_pct']:+.1f}%)
  Best trade:      ${m['best_trade']:+.2f}
  Worst trade:     ${m['worst_trade']:+.2f}
  Hold time:       <=15 min (single candle)
{SEP}
  EXIT REASONS  (single 15-min candle per trade)
{rl('take_profit', 'Take profit:')}
{rl('stop_loss',   'Stop loss:')}
{rl('expired',     'Expired (binary):')}
{SEP}
  DIRECTION BREAKDOWN
  YES trades:      {m['yes_count']}  |  WR: {m['yes_win_rate']:.1f}%
  NO trades:       {m['no_count']}  |  WR: {m['no_win_rate']:.1f}%
{SEP}
  MONTHLY BREAKDOWN
{m_lines}
{SEP}
  BEST MONTH:      {best_m}  ${best_p:+.2f}
  WORST MONTH:     {worst_m}  ${worst_p:+.2f}
{SEP}""")


# ─────────────────────────────────────────────────────────────────────────────
# 9 · Charts
# ─────────────────────────────────────────────────────────────────────────────

_BG      = "#0d1117"
_SURFACE = "#161b22"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_BLUE    = "#58a6ff"
_YELLOW  = "#d29922"
_MUTED   = "#8b949e"


def _ax_style(ax):
    ax.set_facecolor(_SURFACE)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SURFACE)


def chart_equity(trades: list[Trade], equity: pd.Series, starting_bankroll: float) -> None:
    eq  = equity.values
    idx = np.arange(len(eq))

    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.where(peak > 0, peak, 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=_BG,
    )
    fig.suptitle("Equity Curve", color="white", fontsize=14, fontweight="bold")

    _ax_style(ax1)
    ax1.plot(idx, eq, color=_BLUE, linewidth=1.1, label="Bankroll")
    ax1.fill_between(idx, eq, starting_bankroll,
                     where=(eq < starting_bankroll), color=_RED, alpha=0.12)
    ax1.axhline(starting_bankroll, color=_MUTED, linewidth=0.7, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Bankroll ($)", color="white")

    # Mark trade entries and exits on the equity curve
    ts_index = equity.index
    for t in trades:
        try:
            ei = ts_index.get_indexer([t.entry_time], method="nearest")[0]
            xi = ts_index.get_indexer([t.exit_time],  method="nearest")[0]
            ax1.scatter(ei, eq[min(ei, len(eq)-1)], color=_BLUE,  s=6, zorder=4, alpha=0.45)
            ax1.scatter(xi, eq[min(xi, len(eq)-1)],
                        color=_GREEN if t.pnl_dollars > 0 else _RED,
                        s=9, zorder=5, alpha=0.6)
        except Exception:
            pass
    ax1.legend(facecolor=_SURFACE, labelcolor="white", fontsize=9)

    _ax_style(ax2)
    ax2.fill_between(idx, dd, color=_RED, alpha=0.45)
    ax2.plot(idx, dd, color=_RED, linewidth=0.7)
    ax2.set_ylabel("Drawdown %", color="white")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print("Saved equity_curve.png")


def chart_monthly_pnl(trades: list[Trade]) -> None:
    monthly: dict[str, float] = {}
    for t in trades:
        k = t.entry_time.strftime("%Y-%m")
        monthly[k] = monthly.get(k, 0.0) + t.pnl_dollars

    months = sorted(monthly)
    pnls   = [monthly[m] for m in months]
    colors = [_GREEN if p >= 0 else _RED for p in pnls]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BG)
    _ax_style(ax)
    ax.bar(range(len(months)), pnls, color=colors, alpha=0.82, width=0.72)
    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("P&L ($)", color="white")
    ax.set_title("Monthly P&L", color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("monthly_pnl.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print("Saved monthly_pnl.png")


def chart_win_rate(trades: list[Trade]) -> None:
    monthly_wins:   dict[str, int] = {}
    monthly_total:  dict[str, int] = {}
    for t in trades:
        k = t.entry_time.strftime("%Y-%m")
        monthly_total[k] = monthly_total.get(k, 0) + 1
        if t.pnl_dollars > 0:
            monthly_wins[k] = monthly_wins.get(k, 0) + 1

    months = sorted(monthly_total)
    rates  = [monthly_wins.get(m, 0) / monthly_total[m] * 100 for m in months]
    x      = range(len(months))

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BG)
    _ax_style(ax)
    ax.plot(x, rates, color=_BLUE, linewidth=1.5, marker="o", markersize=4)
    ax.fill_between(x, rates, 50,
                    where=[r >= 50 for r in rates], color=_GREEN, alpha=0.10)
    ax.fill_between(x, rates, 50,
                    where=[r <  50 for r in rates], color=_RED,   alpha=0.10)
    ax.axhline(55, color=_YELLOW, linewidth=1.0, linestyle="--", label="55% target")
    ax.axhline(50, color=_MUTED,  linewidth=0.7, linestyle="--", alpha=0.5, label="50% breakeven")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Win Rate %", color="white")
    ax.set_title("Win Rate by Month", color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor=_SURFACE, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig("win_rate_by_month.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print("Saved win_rate_by_month.png")


def chart_distribution(trades: list[Trade]) -> None:
    pnls     = [t.pnl_dollars for t in trades]
    pos_pnl  = [p for p in pnls if p >= 0]
    neg_pnl  = [p for p in pnls if p < 0]
    mean_p   = float(np.mean(pnls))
    std_p    = float(np.std(pnls))

    bins = min(60, max(10, len(pnls) // 8))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
    _ax_style(ax)
    if pos_pnl:
        ax.hist(pos_pnl, bins=bins // 2 or 5, color=_GREEN, alpha=0.75, label=f"Wins ({len(pos_pnl)})")
    if neg_pnl:
        ax.hist(neg_pnl, bins=bins // 2 or 5, color=_RED,   alpha=0.75, label=f"Losses ({len(neg_pnl)})")
    ax.axvline(mean_p, color=_YELLOW, linewidth=1.5, linestyle="--",
               label=f"Mean ${mean_p:+.2f}")
    ax.axvline(mean_p + std_p, color=_MUTED, linewidth=1.0, linestyle=":",
               label=f"±1σ ${std_p:.2f}")
    ax.axvline(mean_p - std_p, color=_MUTED, linewidth=1.0, linestyle=":")
    ax.axvline(0, color="white", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("P&L ($)", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.set_title("Trade P&L Distribution", color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor=_SURFACE, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig("trade_distribution.png", dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print("Saved trade_distribution.png")


def generate_charts(
    trades: list[Trade],
    equity: pd.Series,
    df:     pd.DataFrame,
    starting_bankroll: float,
) -> None:
    plt.style.use("dark_background")
    chart_equity(trades, equity, starting_bankroll)
    chart_monthly_pnl(trades)
    chart_win_rate(trades)
    chart_distribution(trades)


# ─────────────────────────────────────────────────────────────────────────────
# 10 · Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(trades: list[Trade]) -> None:
    rows = [
        {
            "entry_time":          t.entry_time.isoformat(),
            "exit_time":           t.exit_time.isoformat(),
            "direction":           t.direction,
            "strike_price":        t.strike_price,
            "entry_price_cents":   t.entry_price,
            "exit_price_cents":    t.exit_price,
            "bet_size":            t.bet_size,
            "contracts":           t.contracts,
            "pnl_dollars":         t.pnl_dollars,
            "pnl_pct":             t.pnl_pct,
            "exit_reason":         t.exit_reason,
            "momentum_at_entry":   t.momentum_at_entry,
            "confidence_at_entry": t.confidence_at_entry,
            "edge_at_entry":       t.edge_at_entry,
            "btc_price_at_entry":  t.btc_price_at_entry,
            "hold_candles":        t.hold_candles,
            "peak_pnl_pct":        t.peak_pnl_pct,
        }
        for t in trades
    ]
    pd.DataFrame(rows).to_csv("backtest_results.csv", index=False)
    print("Saved backtest_results.csv")


def save_json(metrics: dict, args: argparse.Namespace) -> None:
    def _ser(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        return obj

    payload = {
        "config": {
            "file":     args.file,
            "start":    args.start,
            "end":      args.end,
            "bankroll": args.bankroll,
            "max_bet":  args.max_bet,
            "tp":       args.tp,
            "sl":       args.sl,
        },
        "summary": {k: v for k, v in metrics.items() if k != "monthly"},
        "monthly":  metrics.get("monthly", {}),
    }
    with open("backtest_summary.json", "w") as fh:
        json.dump(_ser(payload), fh, indent=2, default=str)
    print("Saved backtest_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# 11 · CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest printer-v2 strategy on historical BTC OHLCV data."
    )
    p.add_argument("--file",     default="btc_1m_data.xlsx",
                   help="Path to 1m CSV or Excel file (default: btc_1m_data.xlsx)")
    p.add_argument("--start",    default=None, metavar="YYYY-MM-DD",
                   help="Filter start date (inclusive)")
    p.add_argument("--end",      default=None, metavar="YYYY-MM-DD",
                   help="Filter end date (inclusive)")
    p.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                   help=f"Starting bankroll $ (default: {DEFAULT_BANKROLL})")
    p.add_argument("--max-bet",  type=float, default=DEFAULT_MAX_BET,
                   help=f"Max bet size $ (default: {DEFAULT_MAX_BET})")
    p.add_argument("--tp",       type=float, default=DEFAULT_TP,
                   help=f"Take profit fraction (default: {DEFAULT_TP})")
    p.add_argument("--sl",       type=float, default=DEFAULT_SL,
                   help=f"Stop loss fraction (default: {DEFAULT_SL})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_and_resample(args.file, args.start, args.end)
    if len(df) < 50:
        print("ERROR: fewer than 50 candles after filtering — nothing to backtest.")
        sys.exit(1)

    print("Computing indicators ...")
    ind = compute_indicators(df)

    print(f"Running backtest on {len(df):,} 15m candles ...")
    trades, equity = run_backtest(df, ind, args)

    if not trades:
        print("No trades generated. Try relaxing filters or using more data.")
        sys.exit(0)

    metrics = compute_metrics(trades, equity, args.bankroll)
    print_report(metrics, df, args.bankroll)
    save_csv(trades)
    save_json(metrics, args)
    generate_charts(trades, equity, df, args.bankroll)

    print(f"\nFinished. {len(trades)} trades | Final bankroll: "
          f"${equity.iloc[-1]:.2f} | P&L: ${metrics['total_pnl']:+.2f}")


if __name__ == "__main__":
    main()

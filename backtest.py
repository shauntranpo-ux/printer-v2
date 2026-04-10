"""
backtest.py — Backtest printer-v2 strategy on historical BTC 1m data.

Usage:
    python backtest.py --file btc_1m_data.xlsx
    python backtest.py --file data.csv --start 2020-01-01 --end 2023-12-31
    python backtest.py --file data.csv --bankroll 500 --max-bet 25 --tp 0.55 --sl 0.80

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
DEFAULT_TP             = 0.55   # take profit at +55 % (or hold to expiry if bid ≥ 75¢)
DEFAULT_SL             = 0.80   # stop loss at -80 %
MIN_EDGE               = 0.03
MIN_EDGE_CHOPPY        = 0.08   # higher bar for mean reversion
MIN_CONF               = 0.18   # calibrated for rule-based proxy
MIN_CONF_CHOPPY        = 0.28   # mean reversion needs stronger conviction
MAX_SPREAD             = 0.35
MAX_POSITIONS          = 3
DAILY_LOSS_LIMIT       = 100.0
KELLY_FRACTION         = 0.5
MIN_BET                = 0.50
STRIKE_OFFSETS         = [+0.005, 0.0, -0.005]

# ── High-frequency / high-quality mode (toggled via --hf flag) ──────────────
# Drops TRENDING (low WR), expands to 7 strikes, adds volume+momentum scoring.
HF_MIN_CONF            = 0.14   # lower base bar — scoring filter compensates
HF_MAX_SPREAD          = 0.28   # tighter model agreement
HF_STRIKE_OFFSETS      = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]
HF_MAX_POSITIONS       = 7
HF_MIN_MOMENTUM        = 0.20   # |momentum| must be at least this strong
HF_VOL_SURGE           = 1.30   # volume must be 1.3× rolling average
HF_MIN_SIGNAL_SCORE    = 3      # need 3+ of 5 confirmation signals to trade

# Regime thresholds
ADX_VOLATILE           = 25
ADX_TRENDING           = 15     # was 20 — more candles classified as TRENDING
ATR_VOLATILE           = 0.002  # was 0.003 — lower bar for VOLATILE regime
ATR_TRENDING           = 0.002

# Entry filters  (volume filter removed; EMA filter moved to CHOPPY only)
BODY_RATIO_MIN         = 0.25   # body/range — removes extreme doji candles

# Streak protection
STREAK_SOFT            = 3      # 3 losses → 25% bet size, raise conf
STREAK_HARD            = 5      # 5 losses → skip 2 candles

# Time-based filters (UTC hours)
HIGH_ACTIVITY_START    = 13
HIGH_ACTIVITY_END      = 21
MED_ACTIVITY_START     = 0
MED_ACTIVITY_END       = 4

# Model weights by regime (Upgrade 5)
WEIGHTS_TREND = {"claude": 0.35, "gpt": 0.25, "gemini": 0.25, "deepseek": 0.15}
WEIGHTS_CHOPPY = {"claude": 0.15, "gpt": 0.20, "gemini": 0.20, "deepseek": 0.45}

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
    regime:              str   # "VOLATILE" | "TRENDING" | "CHOPPY"


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
    body_pct  = np.tanh((c - o) / o.replace(0.0, np.nan) * 100.0).fillna(0.0)
    dir_score = (
        body_pct.shift(3).fillna(0.0) * 0.1
        + body_pct.shift(2).fillna(0.0) * 0.2
        + body_pct.shift(1).fillna(0.0) * 0.3
        + body_pct                       * 0.4
    ).values

    c4     = c.shift(4).replace(0.0, np.nan)
    vel    = np.tanh((c - c4) / c4 * 100.0).fillna(0.0).values
    mean_v4   = v.rolling(4, min_periods=1).mean().replace(0.0, np.nan)
    vol_ratio = (v / mean_v4 - 1.0).fillna(0.0)
    dir_sign  = np.where(dir_score >= 0, 1.0, -1.0)
    vol_fac   = (np.tanh(vol_ratio.values) * dir_sign)
    momentum = np.clip(0.40 * dir_score + 0.40 * vel + 0.20 * vol_fac, -1.0, 1.0)

    # ── ATR-14 (Wilder smoothing — com = period − 1 = 13) ────────────────────
    h        = df["high"]
    lo       = df["low"]
    prev_c   = c.shift(1).fillna(c)
    tr       = pd.concat([(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()],
                         axis=1).max(axis=1)
    atr14    = tr.ewm(com=13, adjust=False).mean()
    atr_pct  = (atr14 / c.replace(0, np.nan)).fillna(0.0).values

    # ── ADX-14 ───────────────────────────────────────────────────────────────
    h_diff   = h.diff().fillna(0.0)
    l_diff   = (-lo.diff()).fillna(0.0)
    dm_plus  = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
    dm_minus = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)
    sm_tr    = tr.ewm(com=13, adjust=False).mean().replace(0, np.nan)
    di_plus  = 100.0 * pd.Series(dm_plus).ewm(com=13, adjust=False).mean() / sm_tr
    di_minus = 100.0 * pd.Series(dm_minus).ewm(com=13, adjust=False).mean() / sm_tr
    di_sum   = (di_plus + di_minus).replace(0, np.nan)
    dx       = 100.0 * (di_plus - di_minus).abs() / di_sum
    adx14    = dx.ewm(com=13, adjust=False).mean().fillna(0.0).values

    # ── EMA9 and EMA21 ────────────────────────────────────────────────────────
    ema9  = c.ewm(span=9,  adjust=False).mean().values
    ema21 = c.ewm(span=21, adjust=False).mean().values

    # ── 10-candle average volume (for volume confirmation filter) ─────────────
    vol_ma10 = v.rolling(10, min_periods=1).mean().replace(0, np.nan).values

    return {
        "log_ret":   log_ret,
        "vol":       roll_vol,
        "rsi":       rsi,
        "macd_abv":  macd_above,
        "bb_pct":    bb_pct,
        "momentum":  momentum,
        "atr_pct":   atr_pct,
        "adx":       adx14,
        "ema9":      ema9,
        "ema21":     ema21,
        "vol_ma10":  vol_ma10,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Regime classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(adx: float, atr_pct: float) -> str:
    """
    Classify current market into VOLATILE / TRENDING / CHOPPY.
      VOLATILE  — high ADX + wide ATR (best for momentum trades)
      TRENDING  — moderate trend, lower volatility
      CHOPPY    — low directionality (use mean-reversion instead)
    """
    if adx > ADX_VOLATILE and atr_pct > ATR_VOLATILE:
        return "VOLATILE"
    if adx > ADX_TRENDING and atr_pct > ATR_TRENDING:
        return "TRENDING"
    return "CHOPPY"


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Rule-based ensemble (proxy for 4-model AI debate)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_at(i: int, ind: dict[str, np.ndarray], regime: str = "VOLATILE") -> dict:
    """
    Rule-based ensemble proxy.  Model weights and confidence thresholds
    adapt to the current market regime (Upgrades 1 & 5).
    """
    momentum = float(ind["momentum"][i])
    rsi      = float(ind["rsi"][i])
    macd_abv = float(ind["macd_abv"][i])
    bb_pct   = float(ind["bb_pct"][i])

    # ── Claude: momentum-based ───────────────────────────────────────────────
    if   momentum >  0.3: claude = 0.82
    elif momentum >  0.1: claude = 0.64
    elif momentum < -0.3: claude = 0.18
    elif momentum < -0.1: claude = 0.36
    else:                 claude = 0.50

    # ── GPT: RSI-based (bullish lens) ────────────────────────────────────────
    if   rsi > 70: gpt = 0.82
    elif rsi > 60: gpt = 0.66
    elif rsi < 30: gpt = 0.18
    elif rsi < 40: gpt = 0.34
    else:          gpt = 0.50

    # ── Gemini: MACD-based ───────────────────────────────────────────────────
    gemini = 0.72 if macd_abv else 0.26

    # ── DeepSeek: Bollinger mean-reversion lens ───────────────────────────────
    if   bb_pct < 0.20: deepseek = 0.80
    elif bb_pct < 0.40: deepseek = 0.64
    elif bb_pct > 0.80: deepseek = 0.20
    elif bb_pct > 0.60: deepseek = 0.36
    else:               deepseek = 0.50

    probs = {"claude": claude, "gpt": gpt, "gemini": gemini, "deepseek": deepseek}

    # ── Regime-adaptive weights (Upgrade 5) ──────────────────────────────────
    w = WEIGHTS_CHOPPY if regime == "CHOPPY" else WEIGHTS_TREND
    total_w   = sum(w.values())
    consensus = sum(probs[m] * w[m] for m in probs) / total_w
    spread    = max(probs.values()) - min(probs.values())

    avg_conf   = sum(abs(p - 0.5) * 2.0 for p in probs.values()) / len(probs)
    confidence = avg_conf * 0.8 if spread > 0.20 else avg_conf

    min_conf_use = MIN_CONF_CHOPPY if regime == "CHOPPY" else MIN_CONF

    if   spread     > MAX_SPREAD:    action = "WAIT"
    elif confidence < min_conf_use:  action = "SKIP"
    else:                            action = "TRADE"

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
    Accurate Kalshi single-candle simulation with all 7 upgrades:
      1. Market regime detection (VOLATILE / TRENDING / CHOPPY)
      2. Mean reversion for CHOPPY markets
      3. Stronger entry filters (RSI range, volume, candle body, EMA trend)
      4. Dynamic ATR-based TP/SL per regime
      5. Regime-adaptive ensemble weights
      6. Streak protection (3-loss and 5-loss triggers)
      7. Time-based activity filters (UTC hours)

    P&L model (binary Kalshi):
      WIN  → pnl = contracts × (1.00 − entry_price)
      LOSS → pnl = −contracts × entry_price
      TP   → pnl = (tp_exit_cents − entry_cents) × contracts / 100
      SL   → pnl = (sl_exit_cents − entry_cents) × contracts / 100
    """
    bankroll  = args.bankroll
    max_bet   = args.max_bet
    tp_base   = args.tp
    sl_base   = args.sl
    hf_mode   = getattr(args, "hf", False)

    # Apply high-frequency overrides
    strike_offsets  = HF_STRIKE_OFFSETS if hf_mode else STRIKE_OFFSETS
    max_positions   = HF_MAX_POSITIONS  if hf_mode else MAX_POSITIONS

    opens     = df["open"].values
    highs     = df["high"].values
    lows      = df["low"].values
    closes    = df["close"].values
    times     = df["dt"].values
    volumes   = df["volume"].values
    n         = len(df)

    trades:      list[Trade] = []
    equity_vals: list[float] = []
    daily_loss:  dict        = {}

    # Streak protection state
    recent_results: list[bool] = []   # True=win, False=loss (last 5)
    pause_candles = 0                 # candles to skip after 5-loss streak

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
        vol_stat  = max(float(ind["vol"][i]), 0.001)   # rolling-std volatility
        day       = ts.date().isoformat()

        if daily_loss.get(day, 0.0) >= DAILY_LOSS_LIMIT:
            continue

        # ── Upgrade 6: streak protection (hard pause) ────────────────────────
        if pause_candles > 0:
            pause_candles -= 1
            continue

        # ── Upgrade 1: regime classification ─────────────────────────────────
        adx     = float(ind["adx"][i])
        atr_pct = float(ind["atr_pct"][i])
        regime  = classify_regime(adx, atr_pct)

        # ── Upgrade 7: time-based sizing multiplier ───────────────────────────
        # Dead-zone min_conf gate removed — off-hours just use reduced size
        hour = ts.hour
        if HIGH_ACTIVITY_START <= hour < HIGH_ACTIVITY_END:
            time_mult     = 1.0
        elif MED_ACTIVITY_START <= hour < MED_ACTIVITY_END:
            time_mult     = 0.75
        else:
            time_mult     = 0.50
        time_min_conf = MIN_CONF    # same confidence bar for all hours

        # ── Upgrade 6: streak soft trigger ───────────────────────────────────
        recent_losses = sum(1 for w in recent_results[-3:] if not w)
        if recent_losses >= STREAK_SOFT:
            # Reduce size to 25%; slightly raise confidence but stay achievable
            streak_mult     = 0.25
            streak_min_conf = MIN_CONF + 0.05   # e.g. 0.23 — still reachable
        else:
            streak_mult     = 1.0
            streak_min_conf = 0.0

        effective_min_conf = max(time_min_conf, streak_min_conf)

        # ── Regime branch: CHOPPY mean reversion vs TRENDING/VOLATILE ────────
        if regime == "CHOPPY":
            # Upgrade 2: mean reversion entry
            bb    = float(ind["bb_pct"][i])
            rsi   = float(ind["rsi"][i])
            if bb < 0.15 and rsi < 40:
                mr_direction = "YES"
            elif bb > 0.85 and rsi > 60:
                mr_direction = "NO"
            else:
                continue   # no mean reversion trigger

            # EMA alignment applied in CHOPPY to confirm mean-reversion bias
            ema9_val = float(ind["ema9"][i])
            if mr_direction == "YES" and btc_close <= ema9_val:
                continue
            if mr_direction == "NO"  and btc_close >= ema9_val:
                continue

            ens = ensemble_at(i, ind, "CHOPPY")
            if ens["action"] == "WAIT":
                continue
            if ens["confidence"] < max(MIN_CONF_CHOPPY, effective_min_conf):
                continue

            direction  = mr_direction
            tp_trade   = 0.25
            sl_trade   = 0.20
            size_mult  = 0.50 * time_mult * streak_mult
            min_edge_use = MIN_EDGE_CHOPPY

        else:
            # TRENDING or VOLATILE
            if hf_mode:
                # ── High-frequency mode: VOLATILE-only + signal scoring ───────
                if regime != "VOLATILE":
                    continue   # skip TRENDING entirely (lower WR)

                ens = ensemble_at(i, ind, regime)
                if ens["action"] != "TRADE":
                    continue
                if ens["confidence"] < HF_MIN_CONF:
                    continue
                if ens["spread"] > HF_MAX_SPREAD:
                    continue

                direction = ens["direction"]
                momentum  = ens["momentum"]
                rsi       = float(ind["rsi"][i])
                vol_ma    = float(ind["vol_ma10"][i])
                macd_abv  = float(ind["macd_abv"][i])
                ema9_v    = float(ind["ema9"][i])

                # ── Signal scoring: require 3+ of 5 confirmations ─────────────
                # Each indicator must independently agree with the trade direction.
                score = 0

                # 1. Strong momentum in direction
                if direction == "YES" and momentum >= HF_MIN_MOMENTUM:
                    score += 1
                elif direction == "NO" and momentum <= -HF_MIN_MOMENTUM:
                    score += 1

                # 2. Volume surge (heightened activity = real move)
                if volumes[i] >= vol_ma * HF_VOL_SURGE:
                    score += 1

                # 3. RSI confirms direction (not at extreme against trade)
                if direction == "YES" and 40 <= rsi <= 75:
                    score += 1
                elif direction == "NO" and 25 <= rsi <= 60:
                    score += 1

                # 4. MACD aligns with direction
                if direction == "YES" and macd_abv >= 0.5:
                    score += 1
                elif direction == "NO" and macd_abv < 0.5:
                    score += 1

                # 5. EMA9 price alignment (price above EMA for YES, below for NO)
                if direction == "YES" and btc_close > ema9_v:
                    score += 1
                elif direction == "NO" and btc_close < ema9_v:
                    score += 1

                if score < HF_MIN_SIGNAL_SCORE:
                    continue

                # ── Candle body filter (remove doji noise) ────────────────────
                candle_range = btc_high - btc_low
                candle_body  = abs(btc_close - btc_open)
                if candle_range > 0 and candle_body / candle_range < BODY_RATIO_MIN:
                    continue

                # ── Counter-momentum hard block ───────────────────────────────
                if direction == "YES" and momentum < -0.10:
                    continue
                if direction == "NO"  and momentum >  0.10:
                    continue

                tp_trade  = max(tp_base, atr_pct * 15)
                sl_trade  = max(sl_base, atr_pct * 10)
                size_mult = 1.0 * time_mult * streak_mult
                min_edge_use = MIN_EDGE

            else:
                # ── Standard mode: TRENDING + VOLATILE ───────────────────────
                ens = ensemble_at(i, ind, regime)
                if ens["action"] != "TRADE":
                    continue
                if ens["confidence"] < effective_min_conf:
                    continue

                direction = ens["direction"]
                momentum  = ens["momentum"]

                # Skip on strong counter-momentum
                if direction == "YES" and momentum < -0.3:
                    continue
                if direction == "NO"  and momentum >  0.3:
                    continue

                # ── Upgrade 3a: RSI range filter ──────────────────────────────
                rsi = float(ind["rsi"][i])
                if direction == "YES" and not (30 <= rsi <= 80):
                    continue
                if direction == "NO"  and not (20 <= rsi <= 70):
                    continue

                # ── Upgrade 3c: candle body filter (remove extreme doji) ──────
                candle_range = btc_high - btc_low
                candle_body  = abs(btc_close - btc_open)
                if candle_range > 0 and candle_body / candle_range < BODY_RATIO_MIN:
                    continue

                # ── Upgrade 4: dynamic TP/SL ──────────────────────────────────
                if regime == "VOLATILE":
                    tp_trade  = max(tp_base, atr_pct * 15)
                    sl_trade  = max(sl_base, atr_pct * 10)
                    size_mult = 1.0 * time_mult * streak_mult
                else:   # TRENDING
                    tp_trade  = 0.45
                    sl_trade  = 0.30
                    size_mult = 0.75 * time_mult * streak_mult

                min_edge_use = MIN_EDGE

        candle_trades = 0

        for offset in strike_offsets:
            if candle_trades >= max_positions:
                break

            strike         = btc_open * (1.0 + offset)
            yes_entry_frac = yes_price_cents(btc_open, strike, vol_stat) / 100.0

            if direction == "YES":
                entry_frac = yes_entry_frac
                edge       = ens["consensus"] - entry_frac
            else:
                entry_frac = 1.0 - yes_entry_frac
                edge       = (1.0 - ens["consensus"]) - entry_frac

            if edge < min_edge_use:
                continue

            bet = kelly_size(edge, entry_frac, bankroll, max_bet) * size_mult
            if bet < MIN_BET:
                continue

            contracts = int(bet / entry_frac)
            if contracts < 1:
                continue

            entry_cents = entry_frac * 100.0
            cost        = contracts * entry_cents / 100.0

            # ── Intra-candle TP / SL ──────────────────────────────────────────
            if direction == "YES":
                yes_at_best  = yes_price_cents(btc_high, strike, vol_stat)
                yes_at_worst = yes_price_cents(btc_low,  strike, vol_stat)
                contract_at_best, contract_at_worst = yes_at_best, yes_at_worst
            else:
                yes_at_best  = yes_price_cents(btc_low,  strike, vol_stat)
                yes_at_worst = yes_price_cents(btc_high, strike, vol_stat)
                contract_at_best  = 100.0 - yes_at_best
                contract_at_worst = 100.0 - yes_at_worst

            tp_threshold = entry_cents * (1.0 + tp_trade)
            sl_threshold = entry_cents * (1.0 - sl_trade)

            if contract_at_best >= tp_threshold:
                if contract_at_best >= 75.0:
                    # Market bid ≥ 75¢ at TP level → hold to expiry.
                    # At 75%+ implied probability, expected expiry payout
                    # beats exiting now — let it settle at 100¢ or 0¢.
                    won        = (btc_close >= strike) if direction == "YES" else (btc_close < strike)
                    reason     = "expired_tp"
                    exit_cents = 100.0 if won else 0.0
                    pnl        = contracts * (100.0 - entry_cents) / 100.0 if won else -cost
                    pnl_pct    = pnl / cost
                else:
                    reason     = "take_profit"
                    exit_cents = tp_threshold
                    pnl        = (exit_cents - entry_cents) * contracts / 100.0
                    pnl_pct    = pnl / cost
            elif contract_at_worst <= sl_threshold:
                reason     = "stop_loss"
                exit_cents = sl_threshold
                pnl        = (exit_cents - entry_cents) * contracts / 100.0
                pnl_pct    = pnl / cost
            else:
                won = (btc_close >= strike) if direction == "YES" else (btc_close < strike)
                reason     = "expired"
                exit_cents = 100.0 if won else 0.0
                pnl        = contracts * (100.0 - entry_cents) / 100.0 if won else -cost
                pnl_pct    = pnl / cost

            if pnl < 0:
                daily_loss[day] = daily_loss.get(day, 0.0) + abs(pnl)
            bankroll       += pnl
            equity_vals[-1] = bankroll

            # ── Streak tracking ───────────────────────────────────────────────
            recent_results.append(pnl > 0)
            if len(recent_results) > 5:
                recent_results.pop(0)

            # Hard pause after 5 consecutive losses; clear streak so it doesn't
            # re-trigger immediately on the next trade after the pause ends
            if len(recent_results) == 5 and not any(recent_results):
                pause_candles = 2
                recent_results.clear()

            trades.append(Trade(
                entry_time          = ts,
                exit_time           = ts,
                direction           = direction,
                strike_price        = round(strike, 2),
                entry_price         = round(entry_cents, 2),
                exit_price          = round(exit_cents, 2),
                bet_size            = round(bet, 2),
                contracts           = contracts,
                pnl_dollars         = round(pnl, 4),
                pnl_pct             = round(pnl_pct * 100.0, 2),
                exit_reason         = reason,
                momentum_at_entry   = round(ens["momentum"], 4),
                confidence_at_entry = round(ens["confidence"], 4),
                edge_at_entry       = round(edge, 4),
                btc_price_at_entry  = round(btc_open, 2),
                hold_candles        = 1,
                peak_pnl_pct        = round(max(pnl_pct * 100.0, 0.0), 2),
                regime              = regime,
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

    # Regime breakdown
    regime_stats: dict[str, dict] = {}
    for r in ("VOLATILE", "TRENDING", "CHOPPY"):
        rt = [t for t in trades if t.regime == r]
        regime_stats[r] = {
            "count":    len(rt),
            "pct":      round(len(rt) / n * 100, 1) if n else 0.0,
            "win_rate": round(sum(1 for t in rt if t.pnl_dollars > 0) / len(rt) * 100, 1) if rt else 0.0,
            "pnl":      round(sum(t.pnl_dollars for t in rt), 2),
        }

    return {
        "total_trades":    n,
        "win_rate":        round(win_rate,    2),
        "total_pnl":       round(total_pnl,   4),
        "return_pct":      round(total_pnl / starting_bankroll * 100, 2),
        "max_drawdown":    round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 4),
        "profit_factor":   round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "avg_win":         round(avg_win,  4),
        "avg_loss":        round(avg_loss, 4),
        "avg_win_pct":     round(np.mean(win_pcts)  if win_pcts  else 0.0, 2),
        "avg_loss_pct":    round(np.mean(loss_pcts) if loss_pcts else 0.0, 2),
        "best_trade":      round(best,  4),
        "worst_trade":     round(worst, 4),
        "avg_hold_min":    15.0,
        "exit_reasons":    reasons,
        "yes_count":       len(yes_t),
        "no_count":        len(no_t),
        "yes_win_rate":    round(yes_wr, 2),
        "no_win_rate":     round(no_wr,  2),
        "monthly":         monthly,
        "regime_stats":    regime_stats,
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

    mode_label = m.get("mode", "STANDARD")
    print(f"""
{SEP}
     PRINTER V2 BACKTEST  [{mode_label}]
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
{rl('expired_tp',  'Held->expiry (TP):')}
{rl('stop_loss',   'Stop loss:')}
{rl('expired',     'Expired (binary):')}
{SEP}
  REGIME BREAKDOWN
  VOLATILE:  {m['regime_stats']['VOLATILE']['count']:>5} trades  {m['regime_stats']['VOLATILE']['pct']:>5.1f}%  WR:{m['regime_stats']['VOLATILE']['win_rate']:>5.1f}%  P&L:${m['regime_stats']['VOLATILE']['pnl']:>+9.2f}
  TRENDING:  {m['regime_stats']['TRENDING']['count']:>5} trades  {m['regime_stats']['TRENDING']['pct']:>5.1f}%  WR:{m['regime_stats']['TRENDING']['win_rate']:>5.1f}%  P&L:${m['regime_stats']['TRENDING']['pnl']:>+9.2f}
  CHOPPY:    {m['regime_stats']['CHOPPY']['count']:>5} trades  {m['regime_stats']['CHOPPY']['pct']:>5.1f}%  WR:{m['regime_stats']['CHOPPY']['win_rate']:>5.1f}%  P&L:${m['regime_stats']['CHOPPY']['pnl']:>+9.2f}
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
    p.add_argument("--hf",       action="store_true", default=False,
                   help="High-frequency mode: VOLATILE-only, 7 strikes, volume+momentum scoring")
    return p.parse_args()


def main() -> None:
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    metrics["mode"] = "HIGH-FREQ / VOLATILE-ONLY" if args.hf else "STANDARD"
    print_report(metrics, df, args.bankroll)
    save_csv(trades)
    save_json(metrics, args)
    generate_charts(trades, equity, df, args.bankroll)

    print(f"\nFinished. {len(trades)} trades | Final bankroll: "
          f"${equity.iloc[-1]:.2f} | P&L: ${metrics['total_pnl']:+.2f}")


if __name__ == "__main__":
    main()

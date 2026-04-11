"""
backtest_regimes.py — Win rate and P&L by market regime, per asset.

Classifies each trade into one of four regimes using 4-hour candles of the
underlying asset's price (BTC for BTC markets, SOL for SOL markets, etc.)
plus a separate BTC-only pass that gives an overall crypto-market context.

Regimes (mutually exclusive, applied in priority order):
  high_vol  — 14-period ATR / close > 2.5%   (elevated volatility)
  bull      — EMA20 > EMA50 by > 0.5%         (uptrend)
  bear      — EMA20 < EMA50 by > 0.5%         (downtrend)
  sideways  — EMAs within ±0.5%               (range-bound)

Price data source (in order of preference per asset):
  1. Local 1-minute CSV  ({ASSET}USDT_1m.csv)  resampled to 4H  — fast, offline
  2. Binance public API  (/api/v3/klines, no auth)               — Railway fallback

Output: backtest_regimes.json

Usage:
    python backtest_regimes.py
    python backtest_regimes.py --db path/to/printer_v2.db [--csv-dir .]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------

EMA_FAST         = 20     # 4H bars  →  80 h  ≈ 3.3 days
EMA_SLOW         = 50     # 4H bars  → 200 h  ≈ 8.3 days
ATR_PERIOD       = 14     # bars
HIGH_VOL_ATR_PCT = 0.025  # 2.5%  ATR/close threshold for high_vol
SIDEWAYS_BAND    = 0.005  # ±0.5% EMA20/EMA50 separation → sideways


# ---------------------------------------------------------------------------
# Asset / ticker maps
# ---------------------------------------------------------------------------

# Kalshi series ticker prefix → underlying asset
# Longer prefixes first so KXBTC15M matches before KXBTC
_KALSHI_PREFIXES: list[tuple[str, str]] = sorted([
    ("KXBTC15M", "BTC"), ("KXBTC",  "BTC"),
    ("KXETH15M", "ETH"), ("KXETH",  "ETH"),
    ("KXSOL15M", "SOL"), ("KXSOL",  "SOL"),
    ("KXXRP15M", "XRP"), ("KXXRP",  "XRP"),
    ("KXDOGE15M","DOGE"),("KXDOGE", "DOGE"),
    ("KXHYPE15M","HYPE"),("KXHYPE", "HYPE"),
], key=lambda x: -len(x[0]))

_BINANCE_SYM = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT",
                "XRP":"XRPUSDT","DOGE":"DOGEUSDT"}
_CSV_NAME    = {"BTC":"BTCUSDT_1m.csv","ETH":"ETHUSDT_1m.csv",
                "SOL":"SOLUSDT_1m.csv","XRP":"XRPUSDT_1m.csv",
                "DOGE":"DOGEUSDT_1m.csv"}


def _asset(ticker: str) -> str:
    t = ticker.upper()
    for prefix, asset in _KALSHI_PREFIXES:
        if t.startswith(prefix):
            return asset
    return "OTHER"


# ---------------------------------------------------------------------------
# Price data loading
# ---------------------------------------------------------------------------

def _load_csv_4h(path: Path) -> pd.DataFrame:
    """Read a 1-minute OHLCV CSV and resample to 4-hour candles."""
    df = pd.read_csv(
        path,
        names=["time", "open", "high", "low", "close", "volume"],
        skiprows=1,
        dtype={"time": "int64", "open": float, "high": float,
               "low": float, "close": float, "volume": float},
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df4 = df.resample("4h", origin="epoch").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close", "last"),
        volume=("volume","sum"),
    ).dropna(subset=["close"])
    return df4


def _fetch_binance_4h(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Pull 4H klines from Binance public API with automatic pagination.
    Each request returns ≤1000 bars; loops until full range is covered.
    """
    all_rows: list = []
    cur = start_ms
    while cur < end_ms:
        r = _req.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "4h",
                    "startTime": cur, "endTime": end_ms, "limit": 1000},
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        cur = int(batch[-1][0]) + 1
        if len(batch) < 1000:
            break

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qvol","ntrades","tbvol","tbqvol","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("time")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def _get_4h_ohlcv(
    asset: str,
    start_dt: datetime,
    end_dt: datetime,
    csv_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """
    Return (4H OHLCV DataFrame, source_label).
    Applies a 30-day warmup buffer before start_dt for EMA stability.
    """
    def _to_utc(dt: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    buf_start = _to_utc(start_dt) - pd.Timedelta(days=30)
    buf_end   = _to_utc(end_dt)   + pd.Timedelta(days=1)

    # 1. Local CSV
    csv_path = csv_dir / _CSV_NAME.get(asset, "")
    if csv_path.exists():
        try:
            df = _load_csv_4h(csv_path)
            df = df[(df.index >= buf_start) & (df.index <= buf_end)]
            if len(df) >= EMA_SLOW:
                return df, f"csv:{csv_path.name}"
        except Exception as e:
            print(f"  [warn] CSV load failed for {asset}: {e}")

    # 2. Binance API
    sym = _BINANCE_SYM.get(asset)
    if sym and _HAS_REQUESTS:
        try:
            start_ms = int(buf_start.timestamp() * 1000)
            end_ms   = int(buf_end.timestamp()   * 1000)
            print(f"  Fetching {sym} 4H candles from Binance ...")
            df = _fetch_binance_4h(sym, start_ms, end_ms)
            if len(df) >= EMA_SLOW:
                return df, "binance_api"
        except Exception as e:
            print(f"  [warn] Binance API failed for {asset}: {e}")

    return pd.DataFrame(), "unavailable"


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def _classify(df4: pd.DataFrame) -> pd.Series:
    """
    Vectorised regime labels for each 4H candle.

    Priority (first match wins):
      unknown  — insufficient data for indicators
      high_vol — ATR14/close > HIGH_VOL_ATR_PCT
      bull     — (EMA20 - EMA50) / EMA50 >  SIDEWAYS_BAND
      bear     — (EMA20 - EMA50) / EMA50 < -SIDEWAYS_BAND
      sideways — everything else
    """
    if df4.empty:
        return pd.Series(dtype=str)

    c        = df4["close"]
    ema_fast = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=EMA_SLOW, adjust=False).mean()
    atr      = (df4["high"] - df4["low"]).rolling(ATR_PERIOD).mean()
    atr_pct  = atr / c
    trend    = (ema_fast - ema_slow) / ema_slow

    has_data = atr_pct.notna() & trend.notna()

    labels = np.select(
        [~has_data,
         atr_pct > HIGH_VOL_ATR_PCT,
         trend    > SIDEWAYS_BAND,
         trend    < -SIDEWAYS_BAND],
        ["unknown", "high_vol", "bull", "bear"],
        default="sideways",
    )
    return pd.Series(labels, index=df4.index, name="regime")


def _regime_at(ts_str: str, reg: pd.Series) -> str:
    """Look up the regime label for a trade timestamp string."""
    if reg.empty:
        return "unknown"
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        floored = pd.Timestamp(ts).floor("4h")
        val = reg.asof(floored)
        return val if isinstance(val, str) and val not in ("", "nan") else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _pct(n, d):
    return round(n / d * 100, 2) if d else None


def _sharpe(pnls: list[float]) -> float | None:
    n = len(pnls)
    if n < 2:
        return None
    mean = sum(pnls) / n
    var  = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std  = math.sqrt(var)
    return round(mean / std, 4) if std > 0 else None


def _agg(records: list[dict]) -> dict:
    """Aggregate trades into win-rate / P&L statistics."""
    pnls   = [r["pnl"] for r in records]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "n":             len(pnls),
        "win_rate_pct":  _pct(len(wins), len(pnls)),
        "total_pnl":     round(sum(pnls), 4),
        "avg_pnl":       round(sum(pnls) / len(pnls), 4),
        "avg_win":       round(sum(wins)   / len(wins),   4) if wins   else None,
        "avg_loss":      round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor": (round(-sum(wins) / sum(losses), 4)
                          if losses and sum(losses) != 0 else None),
        "sharpe":        _sharpe(pnls),
    }


def _by_regime(records: list[dict], regime_key: str = "regime") -> dict:
    """Group records by regime and return aggregated stats per regime."""
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r[regime_key]].append(r)
    return {regime: _agg(recs) for regime, recs in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: Path, csv_dir: Path) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    raw = con.execute(
        """
        SELECT id, market_ticker, timestamp, direction,
               entry_price, pnl_dollars, exit_reason,
               btc_price_at_entry, btc_momentum
        FROM   trades
        WHERE  status IN ('closed', 'expired')
          AND  pnl_dollars IS NOT NULL
        ORDER  BY timestamp ASC
        """
    ).fetchall()
    con.close()

    if not raw:
        return {"error": "No closed trades in database."}

    # ── 1. Attach asset label to every trade ─────────────────────────────────
    records = [
        {
            "id":          r["id"],
            "ticker":      r["market_ticker"],
            "asset":       _asset(r["market_ticker"]),
            "timestamp":   r["timestamp"],
            "pnl":         r["pnl_dollars"],
            "direction":   r["direction"],
            "entry_price": r["entry_price"],
            "exit_reason": r["exit_reason"],
        }
        for r in raw
    ]

    # ── 2. Determine overall time range ──────────────────────────────────────
    def _parse(ts: str) -> datetime:
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts[:26])   # trim sub-microseconds
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    all_dt  = [_parse(r["timestamp"]) for r in records]
    min_dt  = min(all_dt)
    max_dt  = max(all_dt)

    # ── 3. Load price data and compute regimes for every relevant asset ───────
    assets_seen = sorted({r["asset"] for r in records} - {"OTHER"})
    asset_regime: dict[str, pd.Series] = {}
    data_sources: dict[str, str] = {}

    print(f"\nLoading price data for: {assets_seen}")
    for asset in assets_seen:
        df4, source = _get_4h_ohlcv(asset, min_dt, max_dt, csv_dir)
        data_sources[asset] = source
        if df4.empty:
            print(f"  {asset}: no data — trades will be labelled 'unknown'")
            asset_regime[asset] = pd.Series(dtype=str)
        else:
            reg = _classify(df4)
            asset_regime[asset] = reg
            counts = reg.value_counts().to_dict()
            print(f"  {asset} ({source}): {len(df4)} 4H bars  {counts}")

    # ── 4. BTC regime as market-wide context (always computed) ────────────────
    if "BTC" not in asset_regime:
        print("  Loading BTC separately for market-context regime ...")
        df4_btc, src_btc = _get_4h_ohlcv("BTC", min_dt, max_dt, csv_dir)
        data_sources["BTC"] = src_btc
        asset_regime["BTC"] = _classify(df4_btc) if not df4_btc.empty else pd.Series(dtype=str)

    btc_reg = asset_regime.get("BTC", pd.Series(dtype=str))

    # ── 5. Label each trade with its own-asset regime + BTC regime ───────────
    for r in records:
        own_reg = asset_regime.get(r["asset"], pd.Series(dtype=str))
        r["regime_own"] = _regime_at(r["timestamp"], own_reg)
        r["regime_btc"] = _regime_at(r["timestamp"], btc_reg)

    # ── 6. Overall BTC-regime breakdown (all trades) ──────────────────────────
    for r in records:
        r["regime"] = r["regime_btc"]   # alias for _by_regime()
    summary_btc_regime = _by_regime(records)

    # ── 7. Per-asset breakdowns ───────────────────────────────────────────────
    by_asset_raw: dict[str, list] = defaultdict(list)
    for r in records:
        by_asset_raw[r["asset"]].append(r)

    by_asset: dict[str, dict] = {}
    for asset in sorted(by_asset_raw):
        recs = by_asset_raw[asset]
        for r in recs:
            r["regime"] = r["regime_own"]   # own-asset regime for grouping
        by_asset[asset] = {
            "n":              len(recs),
            "price_source":   data_sources.get(asset, "unavailable"),
            "by_own_regime":  _by_regime(recs, "regime_own"),
            "by_btc_regime":  _by_regime(recs, "regime_btc"),
        }

    # ── 8. BTC regime candle distribution ────────────────────────────────────
    # Count how many 4H bars fall in each regime across the full trade period
    btc_candle_counts: dict[str, int] = {}
    if not btc_reg.empty:
        def _ts_utc(dt: datetime) -> pd.Timestamp:
            ts = pd.Timestamp(dt)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        period_mask = (btc_reg.index >= _ts_utc(min_dt)) & \
                      (btc_reg.index <= _ts_utc(max_dt))
        in_period = btc_reg[period_mask]
        btc_candle_counts = in_period.value_counts().to_dict()

    # ── 9. Best regime × asset combinations ──────────────────────────────────
    combos = []
    for asset, stats in by_asset.items():
        for regime, agg in stats["by_own_regime"].items():
            if agg["n"] >= 3 and regime not in ("unknown",):
                combos.append({
                    "asset":         asset,
                    "regime":        regime,
                    "n":             agg["n"],
                    "win_rate_pct":  agg["win_rate_pct"],
                    "total_pnl":     agg["total_pnl"],
                    "avg_pnl":       agg["avg_pnl"],
                    "sharpe":        agg["sharpe"],
                })
    combos.sort(key=lambda x: (x["win_rate_pct"] or 0), reverse=True)

    # ── 10. Assembly ──────────────────────────────────────────────────────────
    total = len(records)
    return {
        "meta": {
            "db_path":        str(db_path.resolve()),
            "total_trades":   total,
            "trade_period":   f"{min_dt.date()} to {max_dt.date()}",
            "assets_traded":  assets_seen,
            "price_sources":  data_sources,
            "regime_params": {
                "ema_fast":            EMA_FAST,
                "ema_slow":            EMA_SLOW,
                "atr_period":          ATR_PERIOD,
                "high_vol_atr_pct":    HIGH_VOL_ATR_PCT * 100,
                "sideways_band_pct":   SIDEWAYS_BAND    * 100,
                "note": (
                    "Regime priority: high_vol > bull > bear > sideways. "
                    "ATR = mean(high-low) over ATR_PERIOD 4H bars. "
                    "EMA warmup: 30-day buffer applied before first trade."
                ),
            },
        },
        "btc_regime_candle_distribution": btc_candle_counts,
        "all_trades_by_btc_regime":       summary_btc_regime,
        "by_asset":                       by_asset,
        "best_regime_asset_combos":       combos[:20],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    W   = 65
    SEP = "-" * W
    meta = results["meta"]
    print(f"\n{SEP}")
    print(f"  REGIME BACKTEST  trades={meta['total_trades']}  "
          f"period={meta['trade_period']}")
    print(SEP)

    print(f"\n  ALL TRADES by BTC regime:")
    for regime, s in results["all_trades_by_btc_regime"].items():
        pf_str = f"PF={s['profit_factor']}" if s["profit_factor"] else "PF=N/A"
        print(f"  {regime:>10}  n={s['n']:>3}  WR={s['win_rate_pct']:>5}%  "
              f"P&L=${s['total_pnl']:>+8.2f}  avg=${s['avg_pnl']:>+6.2f}  {pf_str}")

    print(f"\n  PER-ASSET breakdown (own-asset regime):")
    for asset, data in results["by_asset"].items():
        print(f"\n  [{asset}]  n={data['n']}  src={data['price_source']}")
        for regime, s in data["by_own_regime"].items():
            print(f"    {regime:>10}  n={s['n']:>3}  WR={s['win_rate_pct']:>5}%  "
                  f"P&L=${s['total_pnl']:>+8.2f}  avg=${s['avg_pnl']:>+6.2f}")

    print(f"\n  TOP REGIME x ASSET combos (>= 3 trades, by win rate):")
    for c in results["best_regime_asset_combos"][:10]:
        print(f"  {c['asset']:>5} / {c['regime']:>10}  "
              f"n={c['n']:>3}  WR={c['win_rate_pct']:>5}%  "
              f"P&L=${c['total_pnl']:>+8.2f}")
    print(f"{SEP}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime backtest from printer_v2.db")
    parser.add_argument("--db",      default="printer_v2.db")
    parser.add_argument("--out",     default="backtest_regimes.json")
    parser.add_argument("--csv-dir", default=".", help="Directory containing *USDT_1m.csv files")
    args = parser.parse_args()

    db_path  = Path(args.db)
    out_path = Path(args.out)
    csv_dir  = Path(args.csv_dir)

    if not db_path.exists():
        print(f"[error] Database not found: {db_path}")
        return

    print(f"Reading {db_path} ...")
    results = run(db_path, csv_dir)

    if "error" in results:
        print(f"[error] {results['error']}")
        return

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out_path}")
    _print_summary(results)


if __name__ == "__main__":
    main()

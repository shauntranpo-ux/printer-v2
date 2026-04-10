"""
walkforward.py — Walk-Forward Analysis for printer-v2

Rolls 18-month In-Sample + 3-month Out-of-Sample windows across the full
BTC history. Optimises on Sharpe ratio in IS, evaluates on OOS, then chains
all OOS windows into a single realistic equity curve.

Usage:
    py walkforward.py --file "C:\\path\\to\\binance_api_BTCUSDT_1m.csv"
    py walkforward.py --file data.csv --bankroll 500 --max-bet 25

    # More trades: scan 9 Kalshi strike markets instead of 3
    py walkforward.py --file data.csv --n-strikes 9

    # Compound growth: max-bet scales as 5% of current bankroll
    py walkforward.py --file data.csv --compound

    # Both — the path to $50k/year:
    py walkforward.py --file data.csv --n-strikes 9 --compound

Outputs (written to ./wfa_output/):
    wfa_results.csv    — one row per window (params, IS/OOS metrics)
    wfa_equity.csv     — chained OOS equity curve
    wfa_summary.json   — aggregate stats + WFA efficiency ratio
    wfa_equity.png     — chained equity vs buy-and-hold
    wfa_windows.png    — IS vs OOS win-rate per window
    wfa_params.png     — parameter selection frequency
    wfa_winrate.png    — IS vs OOS win-rate scatter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Import backtest as a module so we can patch its globals ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

IS_MONTHS   = 18    # in-sample window length
OOS_MONTHS  =  3    # out-of-sample window length
STEP_MONTHS =  3    # slide step (equals OOS so windows don't overlap on OOS)

DEFAULT_BANKROLL    = 500.0
DEFAULT_MAX_BET     =  25.0
DEFAULT_N_STRIKES   =   3   # mirrors STRIKE_OFFSETS in backtest.py
DEFAULT_COMPOUND_PCT = 5.0  # max_bet = this % of bankroll when --compound is used

# Parameter grid  (4 × 3 × 3 = 36 combos per IS window)
PARAM_GRID = {
    "min_conf": [0.10, 0.14, 0.18, 0.22],
    "tp":       [0.45, 0.55, 0.65],
    "sl":       [0.70, 0.80, 0.90],
}


def _build_strike_offsets(n: int) -> list[float]:
    """
    Build n evenly-spaced strike offsets centred on 0.
    Step is fixed at 0.5% so adjacent strikes are always 0.5% apart.

    n=3  → [-0.005,  0.000, +0.005]  (current default)
    n=5  → [-0.010, -0.005, 0.000, +0.005, +0.010]
    n=7  → [-0.015, ..., +0.015]
    n=9  → [-0.020, ..., +0.020]
    n=11 → [-0.025, ..., +0.025]
    """
    half = n // 2
    step = 0.005
    return [round(i * step, 4) for i in range(-half, half + 1)]

MIN_IS_TRADES  = 20   # ignore IS run if fewer trades (params likely degenerate)
MIN_OOS_TRADES =  3   # flag (but keep) OOS windows with very few trades


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_args(bankroll: float, tp: float, sl: float) -> argparse.Namespace:
    return argparse.Namespace(
        bankroll=bankroll,
        max_bet=DEFAULT_MAX_BET,
        tp=tp,
        sl=sl,
        hf=False,
    )


def _slice_data(
    df:  pd.DataFrame,
    ind: dict[str, np.ndarray],
    start_dt: pd.Timestamp,
    end_dt:   pd.Timestamp,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Return (df_slice, ind_slice) for [start_dt, end_dt)."""
    mask    = (df["dt"] >= start_dt) & (df["dt"] < end_dt)
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return None, None
    s, e = int(indices[0]), int(indices[-1]) + 1
    df_s  = df.iloc[s:e].reset_index(drop=True)
    ind_s = {k: v[s:e] for k, v in ind.items()}
    return df_s, ind_s


def _run(
    df_s:        pd.DataFrame,
    ind_s:       dict,
    bankroll:    float,
    tp:          float,
    sl:          float,
    min_conf:    float,
    max_bet:     float = DEFAULT_MAX_BET,
    n_strikes:   int   = DEFAULT_N_STRIKES,
) -> tuple[list, dict]:
    """Patch backtest globals, run, return (trades, metrics)."""
    bt.MIN_CONF        = min_conf
    bt.MIN_CONF_CHOPPY = min(min_conf + 0.10, 0.50)
    bt.STRIKE_OFFSETS  = _build_strike_offsets(n_strikes)
    bt.MAX_POSITIONS   = n_strikes   # allow one trade per strike per candle
    args = _make_args(bankroll, tp, sl)
    args.max_bet = max_bet
    try:
        trades, equity = bt.run_backtest(df_s, ind_s, args)
    except Exception as exc:
        print(f"    [warn] run_backtest raised: {exc}")
        return [], {}
    if not trades:
        return [], {}
    metrics = bt.compute_metrics(trades, equity, bankroll)
    return trades, metrics


def _sharpe(metrics: dict) -> float:
    if not metrics:
        return -999.0
    s = metrics.get("sharpe_ratio", -999.0)
    return float(s) if s == s else -999.0   # NaN guard


def _pf(metrics: dict) -> float:
    """Profit factor; return 0 if empty or 'inf' string."""
    if not metrics:
        return 0.0
    pf = metrics.get("profit_factor", 0.0)
    if pf == "inf" or pf == float("inf"):
        return 10.0   # cap at 10 for efficiency ratio maths
    return float(pf) if pf == pf else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Window generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(
    df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (is_start, is_end, oos_start, oos_end) tuples."""
    data_start = df["dt"].min()
    data_end   = df["dt"].max()
    windows    = []
    t = data_start

    while True:
        is_start  = t
        is_end    = t  + pd.DateOffset(months=IS_MONTHS)
        oos_start = is_end
        oos_end   = is_end + pd.DateOffset(months=OOS_MONTHS)

        if oos_end > data_end:
            break

        windows.append((is_start, is_end, oos_start, oos_end))
        t += pd.DateOffset(months=STEP_MONTHS)

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Grid search on IS window
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_is(
    df_is:     pd.DataFrame,
    ind_is:    dict,
    n_strikes: int = DEFAULT_N_STRIKES,
) -> tuple[dict | None, dict]:
    """
    Try all 36 parameter combos on the IS slice.
    Rank by Sharpe ratio; require >= MIN_IS_TRADES.
    Returns (best_params_dict, best_metrics_dict).
    """
    best_sharpe  = -999.0
    best_params  = None
    best_metrics = {}

    combos = list(product(
        PARAM_GRID["min_conf"],
        PARAM_GRID["tp"],
        PARAM_GRID["sl"],
    ))

    for min_conf, tp, sl in combos:
        _, m = _run(
            df_is, ind_is, DEFAULT_BANKROLL, tp, sl, min_conf,
            n_strikes=n_strikes,
        )
        s = _sharpe(m)
        n = m.get("total_trades", 0)
        if n >= MIN_IS_TRADES and s > best_sharpe:
            best_sharpe  = s
            best_params  = {"min_conf": min_conf, "tp": tp, "sl": sl}
            best_metrics = m

    return best_params, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis — printer-v2")
    parser.add_argument("--file",         required=True,  help="Path to BTC 1m CSV")
    parser.add_argument("--bankroll",     type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--max-bet",      type=float, default=DEFAULT_MAX_BET,
                        help="Hard cap on bet size (overridden by --compound)")
    parser.add_argument("--n-strikes",    type=int,   default=DEFAULT_N_STRIKES,
                        help="Number of Kalshi strike offsets per candle (3/5/7/9/11). "
                             "Higher = more trades. Use 9 to target $50k/yr.")
    parser.add_argument("--compound",     action="store_true",
                        help="Scale max-bet as a %% of current bankroll each OOS window "
                             "(compound growth mode). Targets $50k+/yr in later years.")
    parser.add_argument("--compound-pct", type=float, default=DEFAULT_COMPOUND_PCT,
                        help="Max-bet as %% of bankroll when --compound is active (default 5)")
    parser.add_argument("--out",          default="wfa_output", help="Output directory")
    args = parser.parse_args()

    # Validate n-strikes: must be odd (so there's a centre strike at 0)
    if args.n_strikes % 2 == 0:
        args.n_strikes += 1
        print(f"[info] --n-strikes rounded up to {args.n_strikes} (must be odd)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + resample (once) ────────────────────────────────────────────
    print("\n=== Walk-Forward Analysis — printer-v2 ===\n")
    print(f"Loading data from: {args.file}")
    df = bt.load_and_resample(args.file)

    print("\nComputing indicators (vectorised)...")
    ind = bt.compute_indicators(df)

    # ── 2. Generate rolling windows ──────────────────────────────────────────
    windows = generate_windows(df)
    n_win   = len(windows)
    strikes_list = _build_strike_offsets(args.n_strikes)
    compound_mode = args.compound

    print(f"\nWindows: {n_win}  ({IS_MONTHS}mo IS + {OOS_MONTHS}mo OOS, step={STEP_MONTHS}mo)")
    print(f"Strike offsets ({args.n_strikes}): {strikes_list}")
    print(f"Compound mode: {'ON (' + str(args.compound_pct) + '% of bankroll per window)' if compound_mode else 'OFF (fixed $' + str(args.max_bet) + ')'}")
    print(f"Parameter combos per IS window: "
          f"{len(PARAM_GRID['min_conf'])} x {len(PARAM_GRID['tp'])} x "
          f"{len(PARAM_GRID['sl'])} = "
          f"{len(PARAM_GRID['min_conf'])*len(PARAM_GRID['tp'])*len(PARAM_GRID['sl'])}")

    if n_win == 0:
        print("ERROR: Not enough data for even one window. Exiting.")
        sys.exit(1)

    # ── 3. Walk-forward loop ─────────────────────────────────────────────────
    wfa_rows: list[dict]     = []
    oos_equity_chain: list   = []   # (timestamp, bankroll) pairs
    oos_bankroll = args.bankroll    # chains across windows

    is_pf_list:  list[float] = []
    oos_pf_list: list[float] = []

    t_start = time.time()

    for w_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows, 1):
        elapsed = time.time() - t_start
        eta     = (elapsed / w_idx) * (n_win - w_idx + 1)
        print(
            f"\n[{w_idx:02d}/{n_win}]  "
            f"IS {is_start.date()} → {is_end.date()}  |  "
            f"OOS {oos_start.date()} → {oos_end.date()}  "
            f"(elapsed {elapsed:.0f}s  eta ~{eta:.0f}s)"
        )

        # ── Slice data ───────────────────────────────────────────────────────
        df_is,  ind_is  = _slice_data(df, ind, is_start,  is_end)
        df_oos, ind_oos = _slice_data(df, ind, oos_start, oos_end)

        if df_is is None or len(df_is) < 100:
            print("  SKIP — IS slice too small")
            continue
        if df_oos is None or len(df_oos) < 30:
            print("  SKIP — OOS slice too small")
            continue

        print(f"  IS candles: {len(df_is):,}   OOS candles: {len(df_oos):,}")

        # ── Grid search on IS ────────────────────────────────────────────────
        print("  Grid-searching IS (36 combos)...", end="", flush=True)
        best_params, is_metrics = grid_search_is(df_is, ind_is, n_strikes=args.n_strikes)
        print(f"  done — IS trades: {is_metrics.get('total_trades', 0)}")

        if best_params is None:
            print("  SKIP — no IS combo met min-trade threshold")
            # Use default fallback params so OOS still runs
            best_params = {"min_conf": 0.18, "tp": 0.55, "sl": 0.80}
            is_metrics  = {}

        print(
            f"  Best IS params: min_conf={best_params['min_conf']}  "
            f"tp={best_params['tp']}  sl={best_params['sl']}  "
            f"Sharpe={_sharpe(is_metrics):.3f}  "
            f"WR={is_metrics.get('win_rate', 0):.1f}%"
        )

        # ── Determine max-bet for this OOS window ────────────────────────────
        if compound_mode:
            window_max_bet = max(args.max_bet, oos_bankroll * args.compound_pct / 100.0)
        else:
            window_max_bet = args.max_bet

        # ── Evaluate OOS with best params ────────────────────────────────────
        oos_trades, oos_metrics = _run(
            df_oos, ind_oos,
            oos_bankroll,
            best_params["tp"],
            best_params["sl"],
            best_params["min_conf"],
            max_bet   = window_max_bet,
            n_strikes = args.n_strikes,
        )

        oos_n = oos_metrics.get("total_trades", 0)
        flag  = " [LOW TRADES]" if oos_n < MIN_OOS_TRADES else ""
        print(
            f"  OOS result:  trades={oos_n}{flag}  "
            f"WR={oos_metrics.get('win_rate', 0):.1f}%  "
            f"PnL=${oos_metrics.get('total_pnl', 0):.2f}  "
            f"Sharpe={_sharpe(oos_metrics):.3f}"
        )

        # ── Collect equity for chaining ──────────────────────────────────────
        if oos_trades:
            pnl_running = oos_bankroll
            for t in oos_trades:
                pnl_running += t.pnl_dollars
                oos_equity_chain.append((t.entry_time, pnl_running))
            oos_bankroll = pnl_running

        is_pf_list.append(_pf(is_metrics))
        oos_pf_list.append(_pf(oos_metrics))

        # ── Build results row ────────────────────────────────────────────────
        row = {
            "window":        w_idx,
            "is_start":      is_start.date().isoformat(),
            "is_end":        is_end.date().isoformat(),
            "oos_start":     oos_start.date().isoformat(),
            "oos_end":       oos_end.date().isoformat(),
            # Best params
            "min_conf":      best_params["min_conf"],
            "tp":            best_params["tp"],
            "sl":            best_params["sl"],
            "max_bet_used":  round(window_max_bet, 2),
            # IS metrics
            "is_trades":     is_metrics.get("total_trades", 0),
            "is_wr":         is_metrics.get("win_rate", 0),
            "is_pnl":        is_metrics.get("total_pnl", 0),
            "is_sharpe":     _sharpe(is_metrics),
            "is_pf":         _pf(is_metrics),
            "is_maxdd":      is_metrics.get("max_drawdown", 0),
            # OOS metrics
            "oos_trades":    oos_metrics.get("total_trades", 0),
            "oos_wr":        oos_metrics.get("win_rate", 0),
            "oos_pnl":       oos_metrics.get("total_pnl", 0),
            "oos_sharpe":    _sharpe(oos_metrics),
            "oos_pf":        _pf(oos_metrics),
            "oos_maxdd":     oos_metrics.get("max_drawdown", 0),
            "oos_bankroll":  oos_bankroll,
        }
        wfa_rows.append(row)

    # ── 4. Compute WFA efficiency ratio ──────────────────────────────────────
    valid_pairs = [
        (i, o) for i, o in zip(is_pf_list, oos_pf_list) if i > 0
    ]
    if valid_pairs:
        avg_is_pf  = np.mean([p[0] for p in valid_pairs])
        avg_oos_pf = np.mean([p[1] for p in valid_pairs])
        wfa_eff    = avg_oos_pf / avg_is_pf if avg_is_pf > 0 else 0.0
    else:
        avg_is_pf  = 0.0
        avg_oos_pf = 0.0
        wfa_eff    = 0.0

    # ── 5. Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(wfa_rows)

    if not results_df.empty:
        total_oos_trades = int(results_df["oos_trades"].sum())
        avg_oos_wr   = float(results_df[results_df["oos_trades"] > 0]["oos_wr"].mean())
        total_oos_pnl = float(results_df["oos_pnl"].sum())
        final_bankroll = float(oos_bankroll)

        # Most-selected params
        param_freq = {
            "min_conf": results_df["min_conf"].value_counts().to_dict(),
            "tp":       results_df["tp"].value_counts().to_dict(),
            "sl":       results_df["sl"].value_counts().to_dict(),
        }
    else:
        total_oos_trades = 0
        avg_oos_wr       = 0.0
        total_oos_pnl    = 0.0
        final_bankroll   = args.bankroll
        param_freq       = {}

    # Annualised stats (OOS spans 28 windows × 3mo = 84 months = 7 years from window 1)
    data_years = len(wfa_rows) * OOS_MONTHS / 12.0
    annual_trades = total_oos_trades / data_years if data_years > 0 else 0
    annual_pnl    = total_oos_pnl    / data_years if data_years > 0 else 0

    summary = {
        "windows_total":     len(windows),
        "windows_completed": len(wfa_rows),
        "is_months":         IS_MONTHS,
        "oos_months":        OOS_MONTHS,
        "step_months":       STEP_MONTHS,
        "n_strikes":         args.n_strikes,
        "compound_mode":     compound_mode,
        "compound_pct":      args.compound_pct if compound_mode else None,
        "starting_bankroll": args.bankroll,
        "final_bankroll":    round(final_bankroll, 2),
        "total_oos_trades":  total_oos_trades,
        "avg_oos_win_rate":  round(avg_oos_wr, 2),
        "total_oos_pnl":     round(total_oos_pnl, 2),
        "annual_trades_avg": round(annual_trades, 0),
        "annual_pnl_avg":    round(annual_pnl, 2),
        "avg_is_pf":         round(avg_is_pf, 4),
        "avg_oos_pf":        round(avg_oos_pf, 4),
        "wfa_efficiency":    round(wfa_eff, 4),
        "wfa_efficiency_ok": wfa_eff >= 0.70,
        "param_frequency":   param_freq,
    }

    # ── 6. Print summary ─────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n\n{sep}")
    print("  WALK-FORWARD ANALYSIS — SUMMARY")
    print(sep)
    print(f"  Windows completed       : {len(wfa_rows)} / {len(windows)}")
    print(f"  IS / OOS / Step         : {IS_MONTHS}mo / {OOS_MONTHS}mo / {STEP_MONTHS}mo")
    print(f"  Strike markets / candle : {args.n_strikes}  "
          f"(offsets: {_build_strike_offsets(args.n_strikes)})")
    print(f"  Compound mode           : {'ON (' + str(args.compound_pct) + '% of bankroll)' if compound_mode else 'OFF (fixed $' + str(args.max_bet) + ')'}")
    print(sep)
    print(f"  Total OOS trades        : {total_oos_trades:,}")
    print(f"  Avg OOS win rate        : {avg_oos_wr:.1f}%")
    print(f"  Total OOS P&L           : ${total_oos_pnl:+.2f}")
    print(f"  Avg trades / year       : {annual_trades:,.0f}")
    print(f"  Avg P&L / year          : ${annual_pnl:+,.2f}")
    print(sep)
    print(f"  Starting bankroll       : ${args.bankroll:.2f}")
    print(f"  Final bankroll (chained): ${final_bankroll:,.2f}")
    print(f"  Return on capital       : {(final_bankroll/args.bankroll - 1)*100:+.1f}%")
    print(sep)
    print(f"  Avg IS  profit factor   : {avg_is_pf:.3f}")
    print(f"  Avg OOS profit factor   : {avg_oos_pf:.3f}")
    print(f"  WFA Efficiency Ratio    : {wfa_eff:.3f}  "
          f"({'PASS (>=0.70)' if wfa_eff >= 0.70 else 'FAIL (<0.70)'})")
    print(sep)

    if not results_df.empty:
        most_conf = results_df["min_conf"].mode()[0]
        most_tp   = results_df["tp"].mode()[0]
        most_sl   = results_df["sl"].mode()[0]
        print(f"\n  Most-selected params (deploy these live):")
        print(f"    MIN_CONFIDENCE = {most_conf}")
        print(f"    TAKE_PROFIT    = {most_tp}")
        print(f"    STOP_LOSS      = {most_sl}")

    # ── Annual projection ────────────────────────────────────────────────────
    if not compound_mode and avg_oos_wr > 55:
        avg_profit_per_trade = (total_oos_pnl / total_oos_trades) if total_oos_trades else 0
        needed_trades = 50_000 / avg_profit_per_trade if avg_profit_per_trade > 0 else 0
        print(f"\n  --- $50k/year projection (fixed ${args.max_bet} max bet) ---")
        print(f"  Avg profit/trade        : ${avg_profit_per_trade:.2f}")
        print(f"  Trades needed/year      : {needed_trades:,.0f}")
        print(f"  Current trades/year     : {annual_trades:,.0f}")
        if args.n_strikes < 9:
            print(f"  --> Re-run with --n-strikes 9 --compound to project $50k+/yr")
        else:
            print(f"  --> Re-run with --compound to model compound growth to $50k+/yr")

    print(sep)

    # ── 7. Write output files ────────────────────────────────────────────────
    if not results_df.empty:
        csv_path = out_dir / "wfa_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    if oos_equity_chain:
        eq_df = pd.DataFrame(oos_equity_chain, columns=["timestamp", "bankroll"])
        eq_path = out_dir / "wfa_equity.csv"
        eq_df.to_csv(eq_path, index=False)
        print(f"Saved: {eq_path}")

    summary_path = out_dir / "wfa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    # ── 8. Charts ────────────────────────────────────────────────────────────
    if results_df.empty or len(results_df) < 2:
        print("\nNot enough windows for charts. Done.")
        return

    print("\nGenerating charts...")
    _plot_equity(oos_equity_chain, args.bankroll, out_dir)
    _plot_windows(results_df, out_dir)
    _plot_params(results_df, out_dir)
    _plot_winrate_scatter(results_df, out_dir)
    print("All charts saved.")
    print(f"\nAll output written to: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart functions
# ─────────────────────────────────────────────────────────────────────────────

def _plot_equity(
    oos_equity_chain: list,
    starting_bankroll: float,
    out_dir: Path,
) -> None:
    if not oos_equity_chain:
        return

    times = [x[0] for x in oos_equity_chain]
    vals  = [x[1] for x in oos_equity_chain]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, vals, color="#00c896", linewidth=1.2, label="Strategy (chained OOS)")
    ax.axhline(starting_bankroll, color="#888", linewidth=0.8, linestyle="--",
               label=f"Starting bankroll (${starting_bankroll:.0f})")
    ax.fill_between(times, starting_bankroll, vals,
                    where=[v > starting_bankroll for v in vals],
                    color="#00c896", alpha=0.15)
    ax.fill_between(times, starting_bankroll, vals,
                    where=[v <= starting_bankroll for v in vals],
                    color="#ff4455", alpha=0.15)
    ax.set_title("Walk-Forward — Chained OOS Equity Curve", fontsize=13, pad=10)
    ax.set_ylabel("Bankroll ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "wfa_equity.png", dpi=150)
    plt.close(fig)
    print("  wfa_equity.png")


def _plot_windows(results_df: pd.DataFrame, out_dir: Path) -> None:
    n   = len(results_df)
    w   = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))

    is_wr  = results_df["is_wr"].values
    oos_wr = results_df["oos_wr"].values

    bars_is  = ax.bar(w - 0.2, is_wr,  0.38, label="IS win rate",  color="#4488ff", alpha=0.85)
    bars_oos = ax.bar(w + 0.2, oos_wr, 0.38, label="OOS win rate", color="#00c896", alpha=0.85)
    ax.axhline(50, color="#888", linewidth=0.8, linestyle="--")

    labels = [f"W{i+1}\n{r['oos_start'][:7]}" for i, r in results_df.iterrows()]
    ax.set_xticks(w)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("IS vs OOS Win Rate per Window", fontsize=13, pad=10)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "wfa_windows.png", dpi=150)
    plt.close(fig)
    print("  wfa_windows.png")


def _plot_params(results_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    param_cfg = [
        ("min_conf", "MIN_CONFIDENCE", PARAM_GRID["min_conf"]),
        ("tp",       "TAKE_PROFIT",    PARAM_GRID["tp"]),
        ("sl",       "STOP_LOSS",      PARAM_GRID["sl"]),
    ]

    for ax, (col, title, grid_vals) in zip(axes, param_cfg):
        counts = results_df[col].value_counts().reindex(grid_vals, fill_value=0)
        colors = ["#00c896" if c == counts.max() else "#4488ff" for c in counts.values]
        ax.bar([str(v) for v in counts.index], counts.values, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Times selected")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Parameter Selection Frequency (green = most common)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "wfa_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wfa_params.png")


def _plot_winrate_scatter(results_df: pd.DataFrame, out_dir: Path) -> None:
    valid = results_df[results_df["oos_trades"] >= MIN_OOS_TRADES]
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        valid["is_wr"], valid["oos_wr"],
        c=valid["oos_pnl"],
        cmap="RdYlGn",
        s=80, edgecolors="k", linewidths=0.4, alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="OOS P&L ($)")

    ax.plot([0, 100], [0, 100], color="#888", linewidth=0.8, linestyle="--")
    ax.axhline(50, color="#aaa", linewidth=0.6, linestyle=":")
    ax.axvline(50, color="#aaa", linewidth=0.6, linestyle=":")

    # Annotate each point with window number
    for _, row in valid.iterrows():
        ax.annotate(
            f"W{int(row['window'])}",
            (row["is_wr"], row["oos_wr"]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points",
        )

    corr = np.corrcoef(valid["is_wr"], valid["oos_wr"])[0, 1]
    ax.set_title(f"IS vs OOS Win Rate  (r={corr:.2f})", fontsize=13, pad=10)
    ax.set_xlabel("IS Win Rate (%)")
    ax.set_ylabel("OOS Win Rate (%)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "wfa_winrate.png", dpi=150)
    plt.close(fig)
    print("  wfa_winrate.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

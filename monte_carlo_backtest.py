#!/usr/bin/env python3
"""
monte_carlo_backtest.py
=======================
Monte Carlo simulation for the printer-v2 Kalshi 15-minute binary strategy.

Tests 4 configurations:
  A  SL=70%  TP=65%  $5 bet
  B  SL=70%  TP=65%  $50 bet
  C  No SL   TP=65%  $5 bet
  D  No SL   TP=65%  $50 bet

Price path model:
  Each trade is a 15-minute Kalshi binary contract bought at ENTRY_CENTS.
  The contract resolves WIN (100¢) with probability WIN_PROB, or LOSE (0¢).
  The mid-session price follows a Brownian bridge from entry → terminal,
  letting the bot hit SL/TP before expiry if the path warrants it.

Usage (PowerShell):
  python monte_carlo_backtest.py
  python monte_carlo_backtest.py --win-prob 0.58
  python monte_carlo_backtest.py --win-prob 0.55 --simulations 20000 --trades 200
  python monte_carlo_backtest.py --sweep          # sweep all win probabilities
"""

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these or pass via --flags)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WIN_PROB      = 0.55      # assumed AI ensemble accuracy
DEFAULT_N_SIMULATIONS = 10_000    # Monte Carlo runs per config
DEFAULT_N_TRADES      = 100       # trades per simulation run

ENTRY_CENTS    = 50.0   # contract entry price (¢) — 50¢ = neutral market
TP_PCT         = 0.65   # take-profit at +65% from entry
SL_PCT         = 0.70   # stop-loss at -70% from entry
BET_SIZES      = [5.0, 50.0]

# Price path noise (¢ per minute). ~4¢/min matches typical Kalshi 15m volatility.
PATH_VOLATILITY = 4.0

# Win probabilities for the --sweep mode
SWEEP_PROBS = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]


# ─────────────────────────────────────────────────────────────────────────────
# PRICE PATH SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_price_path(
    entry_cents: float,
    win:         bool,
    rng:         random.Random,
) -> list[float]:
    """
    Brownian bridge from entry_cents → 0 or 100 over 15 steps (1 per minute).
    Noise decays as time-to-expiry shrinks so the path converges at expiry.
    Returns list of 15 prices (minute 1 … 15).
    """
    terminal = 100.0 if win else 0.0
    n_steps  = 15
    prices: list[float] = []
    price = entry_cents

    for step in range(1, n_steps + 1):
        remaining = n_steps - step + 1
        drift     = (terminal - price) / remaining
        time_frac = step / n_steps
        # Noise envelope: full at start, near-zero at expiry
        noise_sd  = PATH_VOLATILITY * math.sqrt(max(0, 1.0 - time_frac))
        price     = price + drift + rng.gauss(0, noise_sd)
        price     = max(0.5, min(99.5, price))
        prices.append(price)

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRADE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trade(
    entry_cents:  float,
    win_prob:     float,
    sl_pct:       Optional[float],   # None = no stop-loss
    tp_pct:       float,
    bet_dollars:  float,
    rng:          random.Random,
) -> tuple[float, str]:
    """
    Simulate one binary contract trade.
    Returns (pnl_dollars, exit_reason).

    Exit reasons: take_profit | stop_loss | expired_win | expired_loss
    """
    win       = rng.random() < win_prob
    contracts = max(1, int(bet_dollars / (entry_cents / 100.0)))

    tp_trigger = min(99.0, entry_cents * (1.0 + tp_pct))
    sl_trigger = (entry_cents * (1.0 - sl_pct)) if sl_pct is not None else None

    prices = simulate_price_path(entry_cents, win, rng)

    for price in prices:
        if price >= tp_trigger:
            pnl = (tp_trigger - entry_cents) / 100.0 * contracts
            return pnl, "take_profit"
        if sl_trigger is not None and price <= sl_trigger:
            pnl = (sl_trigger - entry_cents) / 100.0 * contracts
            return pnl, "stop_loss"

    # Expiry
    final = 100.0 if win else 0.0
    pnl   = (final - entry_cents) / 100.0 * contracts
    return pnl, "expired_win" if win else "expired_loss"


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUN (one full series of N trades)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    final_pnl:       float
    max_drawdown:    float
    exit_counts:     dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return sum(self.exit_counts.values())

    @property
    def win_count(self) -> int:
        return self.exit_counts.get("take_profit", 0) + self.exit_counts.get("expired_win", 0)

    @property
    def win_rate(self) -> float:
        n = self.n_trades
        return self.win_count / n if n > 0 else 0.0

    def pct(self, key: str) -> float:
        n = self.n_trades
        return self.exit_counts.get(key, 0) / n if n > 0 else 0.0


def run_simulation(
    win_prob:    float,
    sl_pct:      Optional[float],
    tp_pct:      float,
    bet_dollars: float,
    n_trades:    int,
    rng:         random.Random,
) -> RunResult:
    pnl     = 0.0
    peak    = 0.0
    max_dd  = 0.0
    exits: dict[str, int] = {}

    for _ in range(n_trades):
        trade_pnl, reason = simulate_trade(
            ENTRY_CENTS, win_prob, sl_pct, tp_pct, bet_dollars, rng
        )
        pnl += trade_pnl
        exits[reason] = exits.get(reason, 0) + 1
        if pnl > peak:
            peak = pnl
        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd

    return RunResult(final_pnl=pnl, max_drawdown=max_dd, exit_counts=exits)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATS ACROSS ALL SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggStats:
    config_name:     str
    win_prob:        float
    bet_dollars:     float
    sl_pct:          Optional[float]
    n_simulations:   int
    n_trades:        int
    pct_profitable:  float   # % of simulations that ended with positive P&L
    avg_final_pnl:   float
    median_final_pnl: float
    avg_max_drawdown: float
    avg_win_rate:    float
    avg_tp_rate:     float
    avg_sl_rate:     float
    avg_expiry_win:  float
    avg_expiry_loss: float
    sharpe:          float   # avg_pnl / std_pnl (across simulations)
    worst_pnl:       float
    best_pnl:        float


def aggregate(
    config_name: str,
    win_prob:    float,
    sl_pct:      Optional[float],
    tp_pct:      float,
    bet_dollars: float,
    n_sims:      int,
    n_trades:    int,
    rng:         random.Random,
) -> AggStats:
    results: list[RunResult] = []

    for _ in range(n_sims):
        r = run_simulation(win_prob, sl_pct, tp_pct, bet_dollars, n_trades, rng)
        results.append(r)

    pnls     = sorted(r.final_pnl for r in results)
    avg_pnl  = sum(pnls) / len(pnls)
    median   = pnls[len(pnls) // 2]
    worst    = pnls[0]
    best     = pnls[-1]

    variance = sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)
    std_pnl  = math.sqrt(variance) if variance > 0 else 1e-9
    sharpe   = avg_pnl / std_pnl

    profitable   = sum(1 for p in pnls if p > 0)
    avg_dd       = sum(r.max_drawdown for r in results) / len(results)
    avg_win_rate = sum(r.win_rate for r in results) / len(results)
    avg_tp       = sum(r.pct("take_profit") for r in results) / len(results)
    avg_sl       = sum(r.pct("stop_loss") for r in results) / len(results)
    avg_exp_win  = sum(r.pct("expired_win") for r in results) / len(results)
    avg_exp_loss = sum(r.pct("expired_loss") for r in results) / len(results)

    return AggStats(
        config_name      = config_name,
        win_prob         = win_prob,
        bet_dollars      = bet_dollars,
        sl_pct           = sl_pct,
        n_simulations    = n_sims,
        n_trades         = n_trades,
        pct_profitable   = profitable / n_sims * 100,
        avg_final_pnl    = avg_pnl,
        median_final_pnl = median,
        avg_max_drawdown = avg_dd,
        avg_win_rate     = avg_win_rate * 100,
        avg_tp_rate      = avg_tp * 100,
        avg_sl_rate      = avg_sl * 100,
        avg_expiry_win   = avg_exp_win * 100,
        avg_expiry_loss  = avg_exp_loss * 100,
        sharpe           = sharpe,
        worst_pnl        = worst,
        best_pnl         = best,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_header(win_prob: float, n_sims: int, n_trades: int) -> None:
    print()
    print("=" * 90)
    print("  MONTE CARLO BACKTEST — printer-v2  |  Kalshi 15-min Binary Strategy")
    print("=" * 90)
    print(f"  Win probability : {win_prob*100:.1f}%   "
          f"Entry price: {ENTRY_CENTS:.0f}¢   "
          f"Simulations: {n_sims:,}   "
          f"Trades/sim: {n_trades}")
    print(f"  TP threshold    : +{TP_PCT*100:.0f}% from entry  "
          f"SL threshold: -{SL_PCT*100:.0f}% from entry")
    print("-" * 90)


def print_table(stats_list: list[AggStats]) -> None:
    hdr = (
        f"  {'Config':<22} {'Profit%':>7} {'WinRate':>7} "
        f"{'TP%':>6} {'SL%':>6} {'ExpW%':>6} {'ExpL%':>6} "
        f"{'AvgPnL':>8} {'MedPnL':>8} {'MaxDD':>8} {'Sharpe':>7} "
        f"{'Worst':>8} {'Best':>8}"
    )
    print(hdr)
    print("  " + "-" * 86)
    for s in stats_list:
        sl_label = f"SL={SL_PCT*100:.0f}%" if s.sl_pct is not None else "No SL "
        print(
            f"  {s.config_name:<22} "
            f"{s.pct_profitable:>6.1f}% "
            f"{s.avg_win_rate:>6.1f}% "
            f"{s.avg_tp_rate:>5.1f}% "
            f"{s.avg_sl_rate:>5.1f}% "
            f"{s.avg_expiry_win:>5.1f}% "
            f"{s.avg_expiry_loss:>5.1f}% "
            f"${s.avg_final_pnl:>7.2f} "
            f"${s.median_final_pnl:>7.2f} "
            f"${s.avg_max_drawdown:>7.2f} "
            f"{s.sharpe:>6.3f} "
            f"${s.worst_pnl:>7.2f} "
            f"${s.best_pnl:>7.2f}"
        )
    print()


def export_csv(stats_list: list[AggStats], filename: str) -> None:
    fields = [
        "config_name", "win_prob", "bet_dollars", "sl_pct",
        "n_simulations", "n_trades",
        "pct_profitable", "avg_final_pnl", "median_final_pnl",
        "avg_max_drawdown", "avg_win_rate",
        "avg_tp_rate", "avg_sl_rate", "avg_expiry_win", "avg_expiry_loss",
        "sharpe", "worst_pnl", "best_pnl",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats_list:
            writer.writerow({k: getattr(s, k) for k in fields})
    print(f"  CSV saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_configs(bet_dollars: float) -> list[tuple[str, Optional[float]]]:
    """Return (name, sl_pct) pairs for each config at this bet size."""
    return [
        (f"SL={SL_PCT*100:.0f}%  ${bet_dollars:.0f} bet",  SL_PCT),
        (f"No SL    ${bet_dollars:.0f} bet",                None),
    ]


def run_all(win_prob: float, n_sims: int, n_trades: int, rng: random.Random) -> list[AggStats]:
    all_stats: list[AggStats] = []
    configs = [(b, name, sl) for b in BET_SIZES for name, sl in build_configs(b)]
    total   = len(configs)

    for i, (bet, name, sl) in enumerate(configs, 1):
        label = f"  [{i}/{total}] Running: {name} ..."
        print(label, end="", flush=True)
        t0 = time.perf_counter()
        stats = aggregate(name, win_prob, sl, TP_PCT, bet, n_sims, n_trades, rng)
        elapsed = time.perf_counter() - t0
        print(f"\r{label}  done ({elapsed:.1f}s)")
        all_stats.append(stats)

    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo backtest — printer-v2 strategy")
    parser.add_argument("--win-prob",    type=float, default=DEFAULT_WIN_PROB,
                        help=f"AI win probability (default {DEFAULT_WIN_PROB})")
    parser.add_argument("--simulations", type=int,   default=DEFAULT_N_SIMULATIONS,
                        help=f"Monte Carlo runs per config (default {DEFAULT_N_SIMULATIONS})")
    parser.add_argument("--trades",      type=int,   default=DEFAULT_N_TRADES,
                        help=f"Trades per simulation run (default {DEFAULT_N_TRADES})")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("--sweep",       action="store_true",
                        help="Sweep all win probabilities and export full CSV")
    parser.add_argument("--no-csv",      action="store_true",
                        help="Skip CSV export")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.sweep:
        # ── Sweep mode: run all win probabilities, export one big CSV ──────
        all_stats: list[AggStats] = []
        for wp in SWEEP_PROBS:
            print_header(wp, args.simulations, args.trades)
            stats = run_all(wp, args.simulations, args.trades, rng)
            print_table(stats)
            all_stats.extend(stats)
        if not args.no_csv:
            export_csv(all_stats, "monte_carlo_sweep.csv")
    else:
        # ── Single win probability ──────────────────────────────────────────
        print_header(args.win_prob, args.simulations, args.trades)
        stats = run_all(args.win_prob, args.simulations, args.trades, rng)
        print_table(stats)
        if not args.no_csv:
            ts = time.strftime("%Y%m%d_%H%M%S")
            export_csv(stats, f"monte_carlo_{ts}.csv")


if __name__ == "__main__":
    main()

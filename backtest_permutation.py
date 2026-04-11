"""
backtest_permutation.py — Permutation significance test for printer-v2 ensemble edge.

Null hypothesis: trade direction labels have no predictive power.
Wins and losses are no better than a coin flip — the ensemble just got lucky.

Method: for each of 10,000 permutations, randomly flip the sign on each
trade's P&L (equivalent to randomly guessing direction). Compare the actual
observed statistic against this null distribution.

  p-value = fraction of permuted runs that meet or exceed the observed value.
  p < 0.05 → reject null → edge is statistically significant.

Tests run:
  1. Portfolio win rate       — is our win rate above random?
  2. Portfolio total P&L      — is cumulative P&L above random?
  3. Portfolio Sharpe ratio   — is risk-adjusted return above random?
  4. Per-model accuracy       — is each model's direction call above random?
  5. Agreement delta          — when a model agrees with ensemble, does WR improve?
  6. Profit factor            — is PF > 1 statistically above random?

Usage:
    python backtest_permutation.py
    python backtest_permutation.py --db printer_v2.db --n 10000 --out backtest_permutation.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> float | None:
    return round(n / d * 100, 2) if d else None


def _sharpe(pnls: list[float]) -> float | None:
    n = len(pnls)
    if n < 2:
        return None
    mean = sum(pnls) / n
    var  = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std  = math.sqrt(var)
    return mean / std if std > 0 else None


def _profit_factor(pnls: list[float]) -> float | None:
    gains  = sum(p for p in pnls if p > 0)
    losses = sum(p for p in pnls if p < 0)
    if losses == 0:
        return None   # infinite — not testable
    return gains / abs(losses)


def _p_value(observed: float, null_dist: list[float], tail: str = "right") -> float:
    """
    Fraction of the null distribution that is >= observed (right tail)
    or <= observed (left tail). Always returns a value in [0, 1].
    """
    n = len(null_dist)
    if n == 0:
        return 1.0
    if tail == "right":
        return sum(1 for v in null_dist if v >= observed) / n
    return sum(1 for v in null_dist if v <= observed) / n


def _ci(dist: list[float], alpha: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) percentile confidence interval."""
    s = sorted(dist)
    lo = s[int(len(s) * (1 - alpha) / 2)]
    hi = s[int(len(s) * (1 - (1 - alpha) / 2))]
    return round(lo, 6), round(hi, 6)


class Trade(NamedTuple):
    pnl:          float
    direction:    str           # "YES" | "NO"
    claude_prob:  float | None
    gpt_prob:     float | None
    gemini_prob:  float | None
    deepseek_prob: float | None


# ---------------------------------------------------------------------------
# Load trades from DB
# ---------------------------------------------------------------------------

MODELS = ("claude", "gpt", "gemini", "deepseek")
MODEL_PROB_COLS = {
    "claude":   "claude_prob",
    "gpt":      "gpt_prob",
    "gemini":   "gemini_prob",
    "deepseek": "deepseek_prob",
}


def load_trades(db_path: Path) -> list[Trade]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT pnl_dollars, direction,
               claude_prob, gpt_prob, gemini_prob, deepseek_prob
        FROM   trades
        WHERE  status IN ('closed', 'expired')
          AND  pnl_dollars IS NOT NULL
        ORDER  BY timestamp ASC
        """
    ).fetchall()
    con.close()
    return [
        Trade(
            pnl           = r["pnl_dollars"],
            direction     = r["direction"],
            claude_prob   = r["claude_prob"],
            gpt_prob      = r["gpt_prob"],
            gemini_prob   = r["gemini_prob"],
            deepseek_prob = r["deepseek_prob"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Permutation core
# ---------------------------------------------------------------------------

def _permuted_stats(pnl_abs: list[float], n_perm: int, rng: random.Random) -> dict:
    """
    Generate null distribution by randomly flipping signs on |P&L| values.
    Returns dict of {stat_name: [n_perm floats]}.
    """
    n = len(pnl_abs)
    null_wr:  list[float] = []
    null_pnl: list[float] = []
    null_sh:  list[float] = []
    null_pf:  list[float] = []

    for _ in range(n_perm):
        perm = [abs(p) * (1 if rng.random() < 0.5 else -1) for p in pnl_abs]
        wins = sum(1 for v in perm if v > 0)
        null_wr.append(wins / n)
        null_pnl.append(sum(perm))
        sh = _sharpe(perm)
        null_sh.append(sh if sh is not None else 0.0)
        pf = _profit_factor(perm)
        null_pf.append(pf if pf is not None else 0.0)

    return {
        "win_rate": null_wr,
        "total_pnl": null_pnl,
        "sharpe": null_sh,
        "profit_factor": null_pf,
    }


def _two_sample_perm(
    group_a: list[float],   # P&L when model agreed
    group_b: list[float],   # P&L when model disagreed
    n_perm:  int,
    rng:     random.Random,
) -> list[float]:
    """
    Null distribution for (WR_a - WR_b) under random label assignment.
    Pool both groups, randomly split into same sizes, measure WR difference.
    """
    na, nb = len(group_a), len(group_b)
    if na == 0 or nb == 0:
        return []
    pool = group_a + group_b
    diffs: list[float] = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        perm_a = pool[:na]
        perm_b = pool[na:]
        wr_a = sum(1 for v in perm_a if v > 0) / na
        wr_b = sum(1 for v in perm_b if v > 0) / nb if nb else 0.0
        diffs.append(wr_a - wr_b)
    return diffs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(trades: list[Trade], n_perm: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    n   = len(trades)

    pnls     = [t.pnl for t in trades]
    pnl_abs  = [abs(p) for p in pnls]
    wins     = sum(1 for p in pnls if p > 0)
    obs_wr   = wins / n
    obs_pnl  = sum(pnls)
    obs_sh   = _sharpe(pnls) or 0.0
    obs_pf   = _profit_factor(pnls) or 0.0

    print(f"  trades={n}  wins={wins}  WR={obs_wr*100:.1f}%  "
          f"P&L=${obs_pnl:+.2f}  Sharpe={obs_sh:.3f}  PF={obs_pf:.3f}")
    print(f"  Running {n_perm:,} permutations...")

    # ── 1. Portfolio permutation tests ────────────────────────────────────────
    null = _permuted_stats(pnl_abs, n_perm, rng)

    port_tests = {
        "win_rate": {
            "observed":        round(obs_wr * 100, 4),
            "null_mean":       round(sum(null["win_rate"]) / n_perm * 100, 4),
            "null_std":        round(
                math.sqrt(sum((v - sum(null["win_rate"]) / n_perm) ** 2
                              for v in null["win_rate"]) / (n_perm - 1)) * 100, 4
            ),
            "null_95ci_pct":   [round(v * 100, 2) for v in _ci(null["win_rate"])],
            "p_value":         round(_p_value(obs_wr, null["win_rate"]), 6),
            "significant":     _p_value(obs_wr, null["win_rate"]) < 0.05,
            "unit":            "%",
        },
        "total_pnl": {
            "observed":        round(obs_pnl, 4),
            "null_mean":       round(sum(null["total_pnl"]) / n_perm, 4),
            "null_std":        round(
                math.sqrt(sum((v - sum(null["total_pnl"]) / n_perm) ** 2
                              for v in null["total_pnl"]) / (n_perm - 1)), 4
            ),
            "null_95ci":       [round(v, 2) for v in _ci(null["total_pnl"])],
            "p_value":         round(_p_value(obs_pnl, null["total_pnl"]), 6),
            "significant":     _p_value(obs_pnl, null["total_pnl"]) < 0.05,
            "unit":            "USD",
        },
        "sharpe": {
            "observed":        round(obs_sh, 4),
            "null_mean":       round(sum(null["sharpe"]) / n_perm, 4),
            "null_std":        round(
                math.sqrt(sum((v - sum(null["sharpe"]) / n_perm) ** 2
                              for v in null["sharpe"]) / (n_perm - 1)), 4
            ),
            "null_95ci":       [round(v, 4) for v in _ci(null["sharpe"])],
            "p_value":         round(_p_value(obs_sh, null["sharpe"]), 6),
            "significant":     _p_value(obs_sh, null["sharpe"]) < 0.05,
        },
        "profit_factor": {
            "observed":        round(obs_pf, 4),
            "null_mean":       round(sum(null["profit_factor"]) / n_perm, 4),
            "null_std":        round(
                math.sqrt(sum((v - sum(null["profit_factor"]) / n_perm) ** 2
                              for v in null["profit_factor"]) / (n_perm - 1)), 4
            ),
            "null_95ci":       [round(v, 4) for v in _ci(null["profit_factor"])],
            "p_value":         round(_p_value(obs_pf, null["profit_factor"]), 6),
            "significant":     _p_value(obs_pf, null["profit_factor"]) < 0.05,
        },
    }

    # ── 2. Per-model accuracy permutation test ────────────────────────────────
    # Null: direction accuracy = 50% (random guess).
    # Permutation: randomly reassign win/loss labels, recalculate model accuracy.
    # A model is "correct" when: (model agreed with trade direction) == (trade won).
    model_tests: dict[str, dict] = {}

    for model in MODELS:
        col = f"{model}_prob"
        model_trades = [t for t in trades if getattr(t, col) is not None]
        nm = len(model_trades)
        if nm < 10:
            model_tests[model] = {"error": f"only {nm} trades with data — skip"}
            continue

        # Observed: how many times was this model correct?
        correct = 0
        model_pnls: list[float] = []
        agreed_pnls: list[float] = []
        disagreed_pnls: list[float] = []

        for t in model_trades:
            prob_yes = getattr(t, col)
            won      = t.pnl > 0
            model_said_yes  = prob_yes > 0.5
            trade_is_yes    = t.direction == "YES"
            model_agreed    = model_said_yes == trade_is_yes   # voted with ensemble

            is_correct = (model_agreed and won) or (not model_agreed and not won)
            if is_correct:
                correct += 1

            model_pnls.append(t.pnl)
            if model_agreed:
                agreed_pnls.append(t.pnl)
            else:
                disagreed_pnls.append(t.pnl)

        obs_acc = correct / nm

        # Null distribution for accuracy: shuffle win/loss labels
        pnl_abs_m = [abs(p) for p in model_pnls]

        # Re-derive model agreement flags (these don't change under permutation)
        agreed_flags = []
        for t in model_trades:
            prob_yes       = getattr(t, col)
            model_said_yes = prob_yes > 0.5
            trade_is_yes   = t.direction == "YES"
            agreed_flags.append(model_said_yes == trade_is_yes)

        null_acc: list[float] = []
        for _ in range(n_perm):
            # Randomly flip win/loss for each trade
            perm_wins = [rng.random() < 0.5 for _ in range(nm)]
            perm_correct = sum(
                1 for ag, won in zip(agreed_flags, perm_wins)
                if (ag and won) or (not ag and not won)
            )
            null_acc.append(perm_correct / nm)

        p_acc = _p_value(obs_acc, null_acc)

        # Two-sample test: WR when agreed vs disagreed
        obs_wr_agreed    = (sum(1 for p in agreed_pnls if p > 0) / len(agreed_pnls)
                            if agreed_pnls else None)
        obs_wr_disagreed = (sum(1 for p in disagreed_pnls if p > 0) / len(disagreed_pnls)
                            if disagreed_pnls else None)

        if obs_wr_agreed is not None and obs_wr_disagreed is not None:
            obs_delta  = obs_wr_agreed - obs_wr_disagreed
            null_delta = _two_sample_perm(agreed_pnls, disagreed_pnls, n_perm, rng)
            p_delta    = round(_p_value(obs_delta, null_delta), 6) if null_delta else None
        else:
            obs_delta  = None
            p_delta    = None
            null_delta = []

        model_tests[model] = {
            "trades_with_data": nm,

            "accuracy": {
                "observed_pct":  round(obs_acc * 100, 2),
                "null_mean_pct": round(sum(null_acc) / n_perm * 100, 2),
                "null_95ci_pct": [round(v * 100, 2) for v in _ci(null_acc)],
                "p_value":       round(p_acc, 6),
                "significant":   p_acc < 0.05,
            },

            # Two-sample test: does agreeing with ensemble improve WR?
            "agreement_delta": {
                "agreed_n":           len(agreed_pnls),
                "disagreed_n":        len(disagreed_pnls),
                "agreed_wr_pct":      round(obs_wr_agreed * 100, 2) if obs_wr_agreed is not None else None,
                "disagreed_wr_pct":   round(obs_wr_disagreed * 100, 2) if obs_wr_disagreed is not None else None,
                "observed_delta_pct": round(obs_delta * 100, 2) if obs_delta is not None else None,
                "null_95ci_pct":      [round(v * 100, 2) for v in _ci(null_delta)] if null_delta else None,
                "p_value":            p_delta,
                "significant":        (p_delta < 0.05) if p_delta is not None else None,
                "interpretation":     (
                    "Model agreement with ensemble is associated with significantly higher WR"
                    if (p_delta is not None and p_delta < 0.05 and obs_delta is not None and obs_delta > 0)
                    else "Model agreement shows no significant WR improvement over disagreement"
                    if p_delta is not None
                    else "insufficient data"
                ),
            },
        }

        sig_mark = "✓ SIGNIFICANT" if p_acc < 0.05 else "✗ not significant"
        print(f"  {model:>10}  acc={obs_acc*100:.1f}%  p={p_acc:.4f}  {sig_mark}")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    all_sig = [
        port_tests["win_rate"]["significant"],
        port_tests["total_pnl"]["significant"],
        port_tests["sharpe"]["significant"],
        *[
            model_tests[m]["accuracy"]["significant"]
            for m in MODELS
            if "accuracy" in model_tests.get(m, {})
        ],
    ]
    n_sig = sum(1 for v in all_sig if v)
    overall_verdict = (
        "EDGE IS STATISTICALLY SIGNIFICANT"
        if port_tests["win_rate"]["significant"] and port_tests["total_pnl"]["significant"]
        else "EDGE IS NOT YET PROVEN — need more trades or WR is too close to 50%"
    )

    return {
        "meta": {
            "n_trades":      n,
            "n_permutations": n_perm,
            "rng_seed":      seed,
            "alpha":         0.05,
        },
        "verdict": overall_verdict,
        "tests_significant": f"{n_sig}/{len(all_sig)}",
        "portfolio": port_tests,
        "models":    model_tests,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation significance test for printer-v2")
    parser.add_argument("--db",  default="printer_v2.db",            help="SQLite DB path")
    parser.add_argument("--n",   type=int, default=10_000,           help="Permutation count")
    parser.add_argument("--out", default="backtest_permutation.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42,              help="RNG seed (reproducibility)")
    args = parser.parse_args()

    db_path  = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"[error] Database not found: {db_path}")
        return

    print(f"Loading trades from {db_path} ...")
    trades = load_trades(db_path)
    n = len(trades)

    if n == 0:
        print("[error] No closed trades found in database.")
        return

    print(f"Loaded {n} closed trades.\n")

    results = run(trades, n_perm=args.n, seed=args.seed)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    # ── Quick summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  VERDICT: {results['verdict']}")
    print(f"  Tests significant: {results['tests_significant']} (p < 0.05)")
    print(f"{'─'*60}")
    pt = results["portfolio"]
    for stat, label in [
        ("win_rate",     "Win Rate     "),
        ("total_pnl",    "Total P&L    "),
        ("sharpe",       "Sharpe       "),
        ("profit_factor","Profit Factor"),
    ]:
        t = pt[stat]
        sig = "✓" if t["significant"] else "✗"
        print(f"  {sig} {label}  obs={t['observed']}  p={t['p_value']:.4f}")
    print(f"{'─'*60}")
    for model in MODELS:
        mt = results["models"].get(model, {})
        if "error" in mt:
            print(f"  ? {model:>10}  {mt['error']}")
            continue
        acc = mt["accuracy"]
        sig = "✓" if acc["significant"] else "✗"
        print(f"  {sig} {model:>10}  acc={acc['observed_pct']}%  p={acc['p_value']:.4f}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()

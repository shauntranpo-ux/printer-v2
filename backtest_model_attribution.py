"""
backtest_model_attribution.py — Per-model P&L, win rate, and accuracy from live trade history.

Reads printer_v2.db and answers:
  1. On trades where this model voted, how often was it right?
  2. On trades where this model AGREED with the consensus, did P&L improve?
  3. On trades where this model DISAGREED, did P&L suffer?
  4. How well-calibrated was each model's probability?

Output: backtest_model_attribution.json

Usage:
    python backtest_model_attribution.py
    python backtest_model_attribution.py --db path/to/printer_v2.db
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


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
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance)
    return round(mean / std, 4) if std > 0 else None


def _calibration_bins(probs: list[float], outcomes: list[bool]) -> list[dict]:
    """
    Group predictions into 10% probability buckets and compare predicted
    vs actual win rate — shows whether each model over/under-estimates edge.
    """
    bins: dict[int, list[bool]] = defaultdict(list)
    for p, win in zip(probs, outcomes):
        bucket = min(int(p * 10), 9)   # 0–9 maps to [0,10)…[90,100)
        bins[bucket].append(win)

    result = []
    for b in range(10):
        items = bins[b]
        if not items:
            continue
        result.append({
            "bucket":        f"{b*10}–{b*10+9}%",
            "predicted_pct": b * 10 + 5,          # bucket midpoint
            "actual_pct":    _pct(sum(items), len(items)),
            "n":             len(items),
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = ("claude", "gpt", "gemini", "deepseek")
MODEL_PROB_COLS = {
    "claude":   "claude_prob",
    "gpt":      "gpt_prob",
    "gemini":   "gemini_prob",
    "deepseek": "deepseek_prob",
}


def run(trades_raw: list, ensemble_raw: list, source: str = "local") -> dict:
    # ── 1. Closed / expired trades ────────────────────────────────────────────
    trades = trades_raw

    # ── 2. All ensemble_log rows (includes SKIP / WAIT, not just TRADE) ───────
    ensemble_rows = ensemble_raw

    total_trades = len(trades)
    if total_trades == 0:
        return {"error": "No closed trades found in database."}

    # ── 3. Overall portfolio stats ────────────────────────────────────────────
    all_pnls   = [t["pnl_dollars"] for t in trades]
    wins       = [p for p in all_pnls if p > 0]
    losses     = [p for p in all_pnls if p < 0]

    exit_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t["exit_reason"] or "unknown"] += 1

    portfolio = {
        "total_trades":   total_trades,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   _pct(len(wins), total_trades),
        "total_pnl":      round(sum(all_pnls), 4),
        "avg_pnl":        round(sum(all_pnls) / total_trades, 4),
        "avg_win":        round(sum(wins)   / len(wins),   4) if wins   else None,
        "avg_loss":       round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor":  round(-sum(wins) / sum(losses), 4) if losses and sum(losses) != 0 else None,
        "sharpe":         _sharpe(all_pnls),
        "exit_reasons":   dict(exit_counts),
    }

    # ── 4. Per-model trade-level stats ────────────────────────────────────────
    #
    # For each trade we know:
    #   - The actual outcome (pnl > 0 = win)
    #   - Each model's P(YES) at entry
    #   - The trade direction (YES / NO)
    #
    # Model "voted correctly" = its directional call matched the trade outcome.
    #   e.g. model_prob > 0.5 → "YES vote"; trade direction=YES, trade won → correct
    #
    # Segments:
    #   "agreed"    — model voted same direction as consensus and trade was placed
    #   "disagreed" — model voted opposite to consensus (minority vote)
    #
    model_stats: dict[str, dict] = {}

    for model in MODELS:
        col = MODEL_PROB_COLS[model]

        # Trades where this model gave a probability
        model_trades = [t for t in trades if t[col] is not None]
        n = len(model_trades)
        if n == 0:
            model_stats[model] = {"error": "no probability data in trades table"}
            continue

        # ── Direction call: model's predicted direction for the trade ──────────
        # Model prob is P(YES). If direction=YES we use prob directly;
        # if direction=NO we use 1-prob so "probability of winning" is uniform.
        direction_correct = 0
        pnls_when_agreed   = []
        pnls_when_disagreed = []
        probs_for_calib    = []
        outcomes_for_calib = []

        pnls_all_model    = []
        wins_model        = 0
        avg_edge_correct  = []
        avg_edge_wrong    = []

        for t in model_trades:
            prob_yes   = t[col]           # model's P(YES), in [0,1]
            direction  = t["direction"]   # "YES" or "NO" (what we actually traded)
            won        = t["pnl_dollars"] > 0
            pnl        = t["pnl_dollars"]

            # Model's directional call for this specific trade direction
            if direction == "YES":
                model_agreed_with_trade = prob_yes > 0.5
                prob_of_winning = prob_yes
            else:  # direction == "NO"
                model_agreed_with_trade = prob_yes <= 0.5
                prob_of_winning = 1.0 - prob_yes

            # Was the model's call correct? (agreed with trade AND trade won, or
            # disagreed with trade AND trade lost)
            model_correct = (model_agreed_with_trade and won) or (not model_agreed_with_trade and not won)
            if model_correct:
                direction_correct += 1

            pnls_all_model.append(pnl)
            if won:
                wins_model += 1

            # Calibration uses the probability the model assigned to the trade winning
            probs_for_calib.append(prob_of_winning)
            outcomes_for_calib.append(won)

            # Edge of the model's estimate vs actual
            if model_agreed_with_trade:
                pnls_when_agreed.append(pnl)
                if t["edge"] is not None:
                    avg_edge_correct.append(t["edge"])
            else:
                pnls_when_disagreed.append(pnl)
                if t["edge"] is not None:
                    avg_edge_wrong.append(t["edge"])

        wins_agreed   = [p for p in pnls_when_agreed    if p > 0]
        wins_disagreed = [p for p in pnls_when_disagreed if p > 0]

        model_stats[model] = {
            # How many trades had this model's probability recorded
            "trades_with_data": n,

            # Raw accuracy: did model's direction call match actual outcome?
            "direction_accuracy_pct": _pct(direction_correct, n),

            # Of trades where model agreed with the ensemble direction
            "agreed": {
                "n":          len(pnls_when_agreed),
                "pct_of_trades": _pct(len(pnls_when_agreed), n),
                "win_rate_pct":  _pct(len(wins_agreed), len(pnls_when_agreed)),
                "total_pnl":     round(sum(pnls_when_agreed), 4) if pnls_when_agreed else 0,
                "avg_pnl":       round(sum(pnls_when_agreed) / len(pnls_when_agreed), 4) if pnls_when_agreed else None,
                "sharpe":        _sharpe(pnls_when_agreed),
            },

            # Of trades where model DISAGREED with the ensemble direction
            "disagreed": {
                "n":          len(pnls_when_disagreed),
                "pct_of_trades": _pct(len(pnls_when_disagreed), n),
                "win_rate_pct":  _pct(len(wins_disagreed), len(pnls_when_disagreed)),
                "total_pnl":     round(sum(pnls_when_disagreed), 4) if pnls_when_disagreed else 0,
                "avg_pnl":       round(sum(pnls_when_disagreed) / len(pnls_when_disagreed), 4) if pnls_when_disagreed else None,
                "sharpe":        _sharpe(pnls_when_disagreed),
            },

            # Calibration: how well did model probabilities match actual win rates?
            "calibration_bins": _calibration_bins(probs_for_calib, outcomes_for_calib),

            # Average model-assigned edge on correct vs wrong calls
            "avg_prob_on_correct_calls": round(
                sum(p for p, o in zip(probs_for_calib, outcomes_for_calib) if o) /
                max(sum(outcomes_for_calib), 1), 4
            ),
            "avg_prob_on_wrong_calls": round(
                sum(p for p, o in zip(probs_for_calib, outcomes_for_calib) if not o) /
                max(sum(1 for o in outcomes_for_calib if not o), 1), 4
            ),
        }

    # ── 5. Ensemble signal stats (all cycles, not just placed trades) ──────────
    total_signals  = len(ensemble_rows)
    traded_signals = sum(1 for r in ensemble_rows if r["action"] == "TRADE")
    wait_signals   = sum(1 for r in ensemble_rows if r["action"] == "WAIT")
    skip_signals   = sum(1 for r in ensemble_rows if r["action"] == "SKIP")

    # Per-model signal-level stats (on all ensemble_log rows, not just placed trades)
    model_signal_stats: dict[str, dict] = {}
    for model in MODELS:
        col = MODEL_PROB_COLS[model]
        rows_with_data = [r for r in ensemble_rows if r[col] is not None]
        n_sig = len(rows_with_data)
        if n_sig == 0:
            model_signal_stats[model] = {"error": "no data"}
            continue

        yes_votes = sum(1 for r in rows_with_data if r[col] > 0.5)
        no_votes  = n_sig - yes_votes

        # Disagreement with consensus: model direction != consensus direction
        disagreements = sum(
            1 for r in rows_with_data
            if r["consensus_prob"] is not None
            and (r[col] > 0.5) != (r["consensus_prob"] > 0.5)
        )
        rows_with_consensus = sum(1 for r in rows_with_data if r["consensus_prob"] is not None)

        avg_prob    = sum(r[col] for r in rows_with_data) / n_sig
        avg_abs_edge = sum(abs(r[col] - 0.5) for r in rows_with_data) / n_sig

        model_signal_stats[model] = {
            "total_signals":          n_sig,
            "yes_votes":              yes_votes,
            "no_votes":               no_votes,
            "yes_vote_pct":           _pct(yes_votes, n_sig),
            "avg_probability":        round(avg_prob, 4),
            "avg_conviction":         round(avg_abs_edge * 200, 2),  # 0–100 scale
            "disagreed_with_consensus_pct": _pct(
                disagreements, rows_with_consensus
            ) if rows_with_consensus else None,
        }

    # ── 6. Model ranking (by direction accuracy on placed trades) ─────────────
    ranked = sorted(
        [
            (m, model_stats[m].get("direction_accuracy_pct") or 0)
            for m in MODELS
            if "error" not in model_stats.get(m, {})
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    # ── 7. Assembly ───────────────────────────────────────────────────────────
    return {
        "meta": {
            "source":         source,
            "total_closed_trades": total_trades,
            "total_ensemble_signals": total_signals,
        },
        "portfolio": portfolio,
        "model_ranking": [
            {"model": m, "direction_accuracy_pct": round(acc, 2)}
            for m, acc in ranked
        ],
        "models": model_stats,
        "ensemble_signals": {
            "total":  total_signals,
            "traded": traded_signals,
            "wait":   wait_signals,
            "skip":   skip_signals,
            "trade_rate_pct": _pct(traded_signals, total_signals),
            "per_model": model_signal_stats,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_data(args) -> tuple[list, list, str]:
    if getattr(args, "synthetic", False):
        p = Path(getattr(args, "synthetic_file", "synthetic_trades.json"))
        if not p.exists():
            raise FileNotFoundError(f"Synthetic data not found: {p}. Run generate_synthetic_trades.py first.")
        data = json.loads(p.read_text())
        trades   = data.get("trades", []) if isinstance(data, dict) else data
        ensemble = data.get("ensemble_log", []) if isinstance(data, dict) else []
        return trades, ensemble, str(p)
    if args.url:
        if not _HAS_REQUESTS:
            raise RuntimeError("pip install requests to use --url")
        url = args.url.rstrip("/")
        print(f"Fetching trades from {url}/api/backtest/trades ...")
        resp = _req.get(f"{url}/api/backtest/trades", timeout=60)
        resp.raise_for_status()
        trades = resp.json()
        print(f"Fetching ensemble_log from {url}/api/backtest/ensemble_log ...")
        resp2 = _req.get(f"{url}/api/backtest/ensemble_log", timeout=60)
        resp2.raise_for_status()
        ensemble = resp2.json()
        return trades, ensemble, url
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    trades = [dict(r) for r in con.execute(
        "SELECT * FROM trades WHERE status IN ('closed','expired') AND pnl_dollars IS NOT NULL ORDER BY timestamp ASC"
    ).fetchall()]
    ensemble = [dict(r) for r in con.execute(
        "SELECT * FROM ensemble_log ORDER BY timestamp ASC"
    ).fetchall()]
    con.close()
    return trades, ensemble, str(db_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-model attribution from printer_v2.db")
    parser.add_argument("--db",  default="printer_v2.db", help="Path to SQLite database")
    parser.add_argument("--url", default=None, help="Live Railway URL (e.g. https://printerv2.up.railway.app)")
    parser.add_argument("--synthetic", action="store_true", help="Load from synthetic_trades.json")
    parser.add_argument("--synthetic-file", default="synthetic_trades.json", dest="synthetic_file")
    parser.add_argument("--out", default="backtest_model_attribution.json", help="Output file")
    args = parser.parse_args()

    out_path = Path(args.out)

    try:
        trades, ensemble, source = _load_data(args)
    except Exception as exc:
        print(f"[error] {exc}")
        return

    print(f"Loaded {len(trades)} trades, {len(ensemble)} ensemble rows from {source}")
    results = run(trades, ensemble, source=source)

    if "error" in results:
        print(f"[error] {results['error']}")
        return

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out_path}")

    # ── Quick summary to stdout ──────────────────────────────────────────────
    p = results["portfolio"]
    print(f"\n{'-'*55}")
    print(f"  PORTFOLIO  trades={p['total_trades']}  "
          f"WR={p['win_rate_pct']}%  P&L=${p['total_pnl']:+.2f}  "
          f"PF={p['profit_factor']}")
    print(f"{'-'*55}")
    print(f"  MODEL RANKING (direction accuracy on placed trades)")
    for r in results["model_ranking"]:
        m   = results["models"][r["model"]]
        agg = m.get("agreed",    {})
        dis = m.get("disagreed", {})
        print(
            f"  {r['model']:>10}  acc={r['direction_accuracy_pct']:>5.1f}%  "
            f"agreed_WR={agg.get('win_rate_pct') or '-':>5}%  "
            f"disagreed_WR={dis.get('win_rate_pct') or '-':>5}%  "
            f"disagreed_n={dis.get('n', 0)}"
        )
    print(f"{'-'*55}")
    sigs = results["ensemble_signals"]
    print(f"  SIGNALS  total={sigs['total']}  "
          f"traded={sigs['traded']}  wait={sigs['wait']}  "
          f"trade_rate={sigs['trade_rate_pct']}%")
    print(f"{'-'*55}")


if __name__ == "__main__":
    main()

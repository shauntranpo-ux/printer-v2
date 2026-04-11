"""
backtest_liquidity.py — Slippage-adjusted P&L using a parametric bid/ask spread model.

Since printer_v2.db stores actual fill prices (not order book snapshots), bid/ask
spreads are estimated from a price-dependent half-spread function calibrated to
typical Kalshi market liquidity.

Key facts baked in:
  - entry_price  = actual ask fill (cents); no order entry before fill
  - exit_price   = actual bid fill for early exits; 100/0 for expired (no spread)
  - edge field   = consensus_prob - ask/100  →  fair_value ≈ entry_price + edge*100
  - base_slippage = (entry_hs + exit_hs) × contracts / 100  (dollars)
  - mid_pnl      = actual_pnl + base_slippage  (theoretical at midpoint)

Scenarios (applied to base_slippage):
  optimistic   (0.5×) — tighter book than modeled
  base         (1.0×) — reproduces actual_pnl exactly (sanity check)
  conservative (2.0×) — wider book, worse fills

Half-spread model:
  35–65¢   → 3.0¢  (near-50¢, widest uncertainty)
  20–35¢ / 65–80¢ → 2.0¢
  10–20¢ / 80–90¢ → 1.5¢
  <10¢   / >90¢   → 1.0¢

Output: backtest_liquidity.json

Usage:
    python backtest_liquidity.py
    python backtest_liquidity.py --db path/to/printer_v2.db
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
# Spread model
# ---------------------------------------------------------------------------

def _half_spread(price_cents: float) -> float:
    """Return estimated half bid/ask spread at a given price (0–100 cents)."""
    dist = abs(min(max(price_cents, 0.0), 100.0) - 50.0)
    if dist < 15:   return 3.0   # 35–65¢  — near 50¢, widest
    if dist < 30:   return 2.0   # 20–35¢ / 65–80¢
    if dist < 40:   return 1.5   # 10–20¢ / 80–90¢
    return 1.0                    # <10¢   / >90¢  — extreme, narrowest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: float, d: float) -> float | None:
    return round(n / d * 100, 2) if d else None


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 2:
        return None
    mean = sum(pnls) / len(pnls)
    var  = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std  = math.sqrt(var)
    return round(mean / std, 4) if std > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("optimistic",   0.5),
    ("base",         1.0),
    ("conservative", 2.0),
]

PRICE_BUCKETS = [
    ("<20¢",   0,   20),
    ("20–34¢", 20,  35),
    ("35–49¢", 35,  50),
    ("50–64¢", 50,  65),
    ("65–79¢", 65,  80),
    ("≥80¢",   80, 101),
]


def run(trades_raw: list, source: str = "local") -> dict:
    trades = [t for t in trades_raw if t.get("entry_price") is not None]

    if not trades:
        return {"error": "No closed trades with entry_price in database."}

    # ── Per-trade slippage ────────────────────────────────────────────────────
    records: list[dict] = []
    for t in trades:
        entry_p   = float(t["entry_price"])
        contracts = t["contracts"] or 0
        pnl       = float(t["pnl_dollars"])
        reason    = (t["exit_reason"] or "unknown").lower()

        entry_hs = _half_spread(entry_p)

        # Expired trades settle at exact 100 or 0 — no spread on exit
        if reason == "expired":
            exit_hs = 0.0
        elif t["exit_price"] is not None:
            exit_hs = _half_spread(float(t["exit_price"]))
        else:
            # exit_price missing for a non-expired trade — conservative fallback
            exit_hs = entry_hs

        base_slip = (entry_hs + exit_hs) * contracts / 100.0  # dollars
        mid_pnl   = pnl + base_slip

        records.append({
            "id":          t["id"],
            "direction":   t["direction"],
            "entry_price": entry_p,
            "exit_price":  t["exit_price"],
            "contracts":   contracts,
            "actual_pnl":  pnl,
            "reason":      reason,
            "entry_hs":    entry_hs,
            "exit_hs":     exit_hs,
            "base_slip":   base_slip,
            "mid_pnl":     mid_pnl,
        })

    n                = len(records)
    total_contracts  = sum(r["contracts"]  for r in records)
    actual_total     = sum(r["actual_pnl"] for r in records)
    mid_total        = sum(r["mid_pnl"]    for r in records)
    total_base_slip  = sum(r["base_slip"]  for r in records)
    actual_wins      = sum(1 for r in records if r["actual_pnl"] > 0)
    gross_wins       = sum(r["actual_pnl"] for r in records if r["actual_pnl"] > 0)
    avg_entry_hs     = sum(r["entry_hs"]   for r in records) / n

    # ── Scenarios ─────────────────────────────────────────────────────────────
    # scenario_pnl = mid_pnl - base_slip * mult
    # When mult=1.0: scenario_pnl = actual_pnl  (identity — confirms model)
    scenario_out: dict[str, dict] = {}
    for sc_name, mult in SCENARIOS:
        sc_pnls  = [r["mid_pnl"] - r["base_slip"] * mult for r in records]
        sc_slip  = total_base_slip * mult
        wins     = [p for p in sc_pnls if p > 0]
        losses   = [p for p in sc_pnls if p < 0]
        scenario_out[sc_name] = {
            "spread_multiplier":      mult,
            "total_slippage":         round(sc_slip, 4),
            "avg_slippage_per_trade": round(sc_slip / n, 4),
            "total_pnl":              round(sum(sc_pnls), 4),
            "avg_pnl":                round(sum(sc_pnls) / n, 4),
            "win_rate_pct":           _pct(len(wins), n),
            "profit_factor":          (
                round(-sum(wins) / sum(losses), 4)
                if losses and sum(losses) != 0 else None
            ),
            "sharpe":                 _sharpe(sc_pnls),
            "is_profitable":          sum(sc_pnls) > 0,
        }

    # ── Break-even ────────────────────────────────────────────────────────────
    # Find the slippage multiplier at which total_pnl = 0:
    #   mid_total - mult * total_base_slip = 0  →  mult = mid_total / total_base_slip
    #
    # Then approximate the corresponding half-spread in cents:
    #   break_even_hs ≈ avg_entry_hs × break_even_mult
    if total_base_slip > 0 and mid_total > 0:
        be_mult = round(mid_total / total_base_slip, 3)
        be_hs   = round(avg_entry_hs * be_mult, 2)
    else:
        be_mult = None
        be_hs   = None

    cons_pass = scenario_out["conservative"]["is_profitable"]
    if be_mult is None:
        be_verdict = "Cannot compute break-even (no positive mid P&L or no slippage)."
    elif be_mult > 2.0:
        be_verdict = (
            f"Very robust: strategy remains profitable up to {be_mult:.1f}× the modeled "
            f"spread (approx {be_hs}¢ half-spread). Conservative scenario passes."
        )
    elif be_mult > 1.0:
        be_verdict = (
            f"Profitable at current fills; breaks even at {be_mult:.1f}× modeled spread "
            f"(approx {be_hs}¢ half-spread). "
            f"Conservative 2× scenario {'passes' if cons_pass else 'fails'}."
        )
    else:
        be_verdict = (
            f"Model spreads alone make the strategy unprofitable (break-even at {be_mult:.1f}×). "
            "Actual fills may be worse than the parametric model assumes."
        )

    # ── By exit type ──────────────────────────────────────────────────────────
    by_exit: dict[str, list] = defaultdict(list)
    for r in records:
        by_exit[r["reason"]].append(r)

    exit_stats: dict[str, dict] = {}
    for reason in sorted(by_exit):
        rows    = by_exit[reason]
        nr      = len(rows)
        slip_s  = sum(r["base_slip"]  for r in rows)
        pnl_s   = sum(r["actual_pnl"] for r in rows)
        mid_s   = sum(r["mid_pnl"]    for r in rows)
        wins_n  = sum(1 for r in rows if r["actual_pnl"] > 0)
        exit_stats[reason] = {
            "n":                      nr,
            "avg_entry_hs_cents":     round(sum(r["entry_hs"] for r in rows) / nr, 2),
            "avg_exit_hs_cents":      round(sum(r["exit_hs"]  for r in rows) / nr, 2),
            "avg_slippage":           round(slip_s / nr, 4),
            "total_slippage":         round(slip_s, 4),
            "actual_total_pnl":       round(pnl_s, 4),
            "mid_total_pnl":          round(mid_s, 4),
            "win_rate_pct":           _pct(wins_n, nr),
        }

    # ── By entry price bucket ─────────────────────────────────────────────────
    price_stats: dict[str, dict] = {}
    for label, lo, hi in PRICE_BUCKETS:
        rows = [r for r in records if lo <= r["entry_price"] < hi]
        if not rows:
            continue
        nr     = len(rows)
        wins_n = sum(1 for r in rows if r["actual_pnl"] > 0)
        price_stats[label] = {
            "n":                   nr,
            "avg_entry_hs_cents":  round(sum(r["entry_hs"]    for r in rows) / nr, 2),
            "total_slippage":      round(sum(r["base_slip"]    for r in rows), 4),
            "actual_total_pnl":    round(sum(r["actual_pnl"]   for r in rows), 4),
            "win_rate_pct":        _pct(wins_n, nr),
        }

    # ── Assembly ──────────────────────────────────────────────────────────────
    return {
        "meta": {
            "source":         source,
            "total_trades":   n,
            "total_contracts": total_contracts,
            "spread_model":   "parametric price-dependent half-spread (no order-book data)",
            "model_schedule": "3¢ near 50¢ | 2¢ mid-range | 1.5¢ far | 1¢ extreme",
            "note": (
                "entry_price = actual ask fill; "
                "exit_price  = actual bid fill (or 100/0 for expired, zero exit spread). "
                "base scenario (1×) reproduces actual_pnl exactly."
            ),
        },
        "actual": {
            "total_pnl":    round(actual_total, 4),
            "avg_pnl":      round(actual_total / n, 4),
            "win_rate_pct": _pct(actual_wins, n),
        },
        "slippage_summary": {
            "total_base_slippage":         round(total_base_slip, 4),
            "avg_slippage_per_trade":      round(total_base_slip / n, 4),
            "slippage_pct_of_gross_wins":  _pct(total_base_slip, gross_wins) if gross_wins else None,
            "mid_total_pnl":               round(mid_total, 4),
            "avg_entry_half_spread_cents": round(avg_entry_hs, 2),
        },
        "scenarios":   scenario_out,
        "break_even": {
            "multiplier_at_zero_pnl":   be_mult,
            "approx_half_spread_cents": be_hs,
            "verdict":                  be_verdict,
        },
        "by_exit_type":    exit_stats,
        "by_entry_price":  price_stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_trades(args) -> tuple[list, str]:
    if getattr(args, "synthetic", False):
        p = Path(getattr(args, "synthetic_file", "synthetic_trades.json"))
        if not p.exists():
            raise FileNotFoundError(f"Synthetic data not found: {p}. Run generate_synthetic_trades.py first.")
        data = json.loads(p.read_text())
        trades = data if isinstance(data, list) else data.get("trades", [])
        return trades, str(p)
    if args.url:
        if not _HAS_REQUESTS:
            raise RuntimeError("pip install requests to use --url")
        url = args.url.rstrip("/")
        print(f"Fetching trades from {url}/api/backtest/trades ...")
        resp = _req.get(f"{url}/api/backtest/trades", timeout=60)
        resp.raise_for_status()
        return resp.json(), url
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM trades WHERE status IN ('closed','expired') AND pnl_dollars IS NOT NULL ORDER BY timestamp ASC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows], str(db_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slippage-adjusted P&L backtest from printer_v2.db"
    )
    parser.add_argument("--db",  default="printer_v2.db", help="Path to SQLite database")
    parser.add_argument("--url", default=None, help="Live Railway URL (e.g. https://printerv2.up.railway.app)")
    parser.add_argument("--synthetic", action="store_true", help="Load from synthetic_trades.json")
    parser.add_argument("--synthetic-file", default="synthetic_trades.json", dest="synthetic_file")
    parser.add_argument("--out", default="backtest_liquidity.json", help="Output JSON path")
    args = parser.parse_args()

    out_path = Path(args.out)

    try:
        trades, source = _load_trades(args)
    except Exception as exc:
        print(f"[error] {exc}")
        return

    print(f"Loaded {len(trades)} trades from {source}")
    results = run(trades, source=source)

    if "error" in results:
        print(f"[error] {results['error']}")
        return

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out_path}")

    # ── Quick summary ──────────────────────────────────────────────────────────
    a  = results["actual"]
    sl = results["slippage_summary"]
    be = results["break_even"]
    sc = results["scenarios"]
    W  = 62

    print(f"\n{'-'*W}")
    print(f"  ACTUAL   trades={results['meta']['total_trades']}  "
          f"WR={a['win_rate_pct']}%  P&L=${a['total_pnl']:+.2f}")
    print(f"  SLIPPAGE base=${sl['total_base_slippage']:+.2f}  "
          f"avg/trade=${sl['avg_slippage_per_trade']:.4f}  "
          f"avg_entry_hs={sl['avg_entry_half_spread_cents']}c")
    print(f"  MID P&L  ${sl['mid_total_pnl']:+.2f}  (theoretical at midpoint)")
    print(f"{'-'*W}")
    for sc_name, _ in SCENARIOS:
        s    = sc[sc_name]
        mark = "OK" if s["is_profitable"] else "--"
        print(f"  {sc_name:>12}  ({s['spread_multiplier']:.1f}x)  "
              f"P&L=${s['total_pnl']:+.2f}  "
              f"WR={s['win_rate_pct']}%  "
              f"slip=${s['total_slippage']:.2f}  [{mark}]")
    print(f"{'-'*W}")
    print(f"  BREAK-EVEN  mult={be['multiplier_at_zero_pnl']}x  "
          f"hs~={be['approx_half_spread_cents']}c")
    print(f"  {be['verdict']}")
    print(f"{'-'*W}")

    print(f"\n  BY EXIT TYPE")
    for reason, stats in results["by_exit_type"].items():
        print(f"  {reason:>16}  n={stats['n']:>3}  "
              f"slip=${stats['total_slippage']:+.2f}  "
              f"pnl=${stats['actual_total_pnl']:+.2f}  "
              f"WR={stats['win_rate_pct']}%")

    print(f"\n  BY ENTRY PRICE")
    for bucket, stats in results["by_entry_price"].items():
        print(f"  {bucket:>8}  n={stats['n']:>3}  "
              f"hs={stats['avg_entry_hs_cents']}c  "
              f"slip=${stats['total_slippage']:+.2f}  "
              f"pnl=${stats['actual_total_pnl']:+.2f}")
    print()


if __name__ == "__main__":
    main()

"""
backtest_timing.py — Win rate and P&L by seconds-into-window at trade entry.

Each Kalshi 15-minute market window starts at a 900-second epoch boundary
(:00, :15, :30, :45 of each hour).  runner.py computes this identically:

    boundary = int(now_ts // 900) * 900

The bot retries entry every ~60s and cuts off at 660s (runner._MAX_TIME_IN),
so all trades should fall within 0–660s.  The four buckets cover that range
completely.

Buckets (user-specified):
    0–120s     first 2 min   (initial evaluation + first retry)
  120–300s     2–5 min       (2nd–4th retry)
  300–480s     5–8 min       (5th–6th retry)
  480–660s     8–11 min      (7th–8th retry, last entry window)

Also computed per bucket:
  - avg entry price, avg edge, avg ensemble confidence
  - exit-reason distribution
  - YES/NO direction split
  - per-asset breakdown
  - "entry attempt" estimate (which ~60s slot within the window)

Output: backtest_timing.json

Usage:
    python backtest_timing.py
    python backtest_timing.py --db path/to/printer_v2.db
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Window / bucket config
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 900   # 15-minute Kalshi windows
MAX_ENTRY_SECS = 660   # runner._MAX_TIME_IN — hard cut-off

BUCKETS: list[tuple[int, int, str]] = [
    (0,   120, "0-120s"),
    (120, 300, "120-300s"),
    (300, 480, "300-480s"),
    (480, 660, "480-660s"),
]
BUCKET_LABELS = [label for _, _, label in BUCKETS]

# Approximate entry attempt number within each bucket
# (retries every ~60s starting from ~0s)
BUCKET_ATTEMPT = {
    "0-120s":   "1st-2nd",
    "120-300s": "2nd-4th",
    "300-480s": "5th-6th",
    "480-660s": "7th-8th",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    ts = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts[:26])
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _secs_into_window(ts: datetime) -> float:
    """Seconds elapsed since the most recent 15-min epoch boundary."""
    epoch = ts.timestamp()
    boundary = (epoch // WINDOW_SECONDS) * WINDOW_SECONDS
    return epoch - boundary


def _bucket(secs: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= secs < hi:
            return label
    # Outside 0–660 range: classify by direction
    return ">=660s" if secs >= 660 else "<0s"


def _pct(n: int | float, d: int | float) -> float | None:
    return round(n / d * 100, 2) if d else None


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 2:
        return None
    mean = sum(pnls) / len(pnls)
    var  = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std  = math.sqrt(var)
    return round(mean / std, 4) if std > 0 else None


def _agg(records: list[dict]) -> dict:
    """Standard win-rate / P&L stats for a group of trade records."""
    pnls   = [r["pnl"] for r in records]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    edges  = [r["edge"] for r in records if r["edge"] is not None]
    prices = [r["entry_price"] for r in records if r["entry_price"] is not None]
    confs  = [r["conf"] for r in records if r["conf"] is not None]

    exit_counts: dict[str, int] = defaultdict(int)
    for r in records:
        exit_counts[r["exit_reason"] or "unknown"] += 1

    direction_counts: dict[str, int] = defaultdict(int)
    for r in records:
        direction_counts[r["direction"] or "unknown"] += 1

    return {
        "n":                   len(records),
        "win_rate_pct":        _pct(len(wins), len(records)),
        "total_pnl":           round(sum(pnls), 4),
        "avg_pnl":             round(sum(pnls) / len(records), 4),
        "avg_win":             round(sum(wins)   / len(wins),   4) if wins   else None,
        "avg_loss":            round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor":       (
            round(-sum(wins) / sum(losses), 4)
            if losses and sum(losses) != 0 else None
        ),
        "sharpe":              _sharpe(pnls),
        "avg_entry_price":     round(sum(prices) / len(prices), 2) if prices else None,
        "avg_edge":            round(sum(edges)  / len(edges),  4) if edges  else None,
        "avg_confidence":      round(sum(confs)  / len(confs),  4) if confs  else None,
        "exit_reasons":        dict(exit_counts),
        "direction_split":     dict(direction_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    raw = con.execute(
        """
        SELECT id, market_ticker, timestamp, direction, entry_price,
               pnl_dollars, exit_reason, edge, ensemble_confidence,
               btc_price_at_entry
        FROM   trades
        WHERE  status IN ('closed', 'expired')
          AND  pnl_dollars IS NOT NULL
          AND  timestamp   IS NOT NULL
        ORDER  BY timestamp ASC
        """
    ).fetchall()
    con.close()

    if not raw:
        return {"error": "No closed trades in database."}

    # ── Enrich each trade with timing fields ─────────────────────────────────
    records: list[dict] = []
    for r in raw:
        ts   = _parse_ts(r["timestamp"])
        secs = _secs_into_window(ts)
        bkt  = _bucket(secs)

        records.append({
            "id":          r["id"],
            "ticker":      r["market_ticker"],
            "timestamp":   r["timestamp"],
            "ts":          ts,
            "direction":   r["direction"],
            "entry_price": r["entry_price"],
            "pnl":         r["pnl_dollars"],
            "exit_reason": r["exit_reason"],
            "edge":        r["edge"],
            "conf":        r["ensemble_confidence"],
            "secs":        round(secs, 1),
            "bucket":      bkt,
        })

    total = len(records)

    # ── Sanity check: are any trades outside the expected 0–660s range? ───────
    outside_window = [r for r in records if r["bucket"] not in BUCKET_LABELS]

    # ── Per-bucket stats ──────────────────────────────────────────────────────
    by_bucket: dict[str, list] = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)

    bucket_stats: dict[str, dict] = {}
    for label in BUCKET_LABELS:
        recs = by_bucket.get(label, [])
        if not recs:
            bucket_stats[label] = {"n": 0}
            continue
        stats = _agg(recs)
        stats["entry_attempt"] = BUCKET_ATTEMPT[label]
        stats["secs_remaining_at_mid"] = WINDOW_SECONDS - (BUCKETS[BUCKET_LABELS.index(label)][0] +
                                          BUCKETS[BUCKET_LABELS.index(label)][1]) // 2
        stats["avg_secs_into_window"] = round(
            sum(r["secs"] for r in recs) / len(recs), 1
        )
        bucket_stats[label] = stats

    # ── Seconds distribution (fine-grained histogram, 60s bins) ───────────────
    # Keys are integers (bin start in seconds) for reliable numeric sorting.
    hist_60s: dict[int, int] = defaultdict(int)
    for r in records:
        bin_start = int(r["secs"] // 60) * 60
        hist_60s[bin_start] += 1

    # ── Per-asset × bucket matrix ─────────────────────────────────────────────
    def _asset(ticker: str) -> str:
        t = ticker.upper()
        for prefix, asset in [
            ("KXBTC", "BTC"), ("KXETH", "ETH"),
            ("KXSOL", "SOL"), ("KXXRP", "XRP"),
            ("KXDOGE","DOGE"),("KXHYPE","HYPE"),
        ]:
            if t.startswith(prefix):
                return asset
        return "OTHER"

    for r in records:
        r["asset"] = _asset(r["ticker"])

    by_asset_bucket: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_asset_bucket[r["asset"]][r["bucket"]].append(r)

    asset_bucket_stats: dict[str, dict] = {}
    for asset in sorted(by_asset_bucket):
        asset_bucket_stats[asset] = {}
        for label in BUCKET_LABELS:
            recs = by_asset_bucket[asset].get(label, [])
            asset_bucket_stats[asset][label] = (
                {"n": len(recs), "win_rate_pct": _pct(sum(1 for r in recs if r["pnl"] > 0), len(recs)),
                 "total_pnl": round(sum(r["pnl"] for r in recs), 4),
                 "avg_pnl":   round(sum(r["pnl"] for r in recs) / len(recs), 4)}
                if recs else {"n": 0}
            )

    # ── Best and worst bucket ─────────────────────────────────────────────────
    ranked = sorted(
        [(lbl, bucket_stats[lbl]) for lbl in BUCKET_LABELS if bucket_stats[lbl].get("n", 0) > 0],
        key=lambda x: x[1].get("win_rate_pct") or 0,
        reverse=True,
    )

    # ── Overall stats ─────────────────────────────────────────────────────────
    overall = _agg(records)

    # ── Assembly ──────────────────────────────────────────────────────────────
    result = {
        "meta": {
            "db_path":           str(db_path.resolve()),
            "total_trades":      total,
            "window_seconds":    WINDOW_SECONDS,
            "max_entry_seconds": MAX_ENTRY_SECS,
            "outside_window_n":  len(outside_window),
            "note": (
                "window_start = floor(trade_timestamp, 900s), matching runner.py "
                "boundary = int(now_ts // 900) * 900. "
                "Retry interval ~60s; bucket '0-120s' = 1st-2nd attempt, "
                "'480-660s' = 7th-8th attempt."
            ),
        },
        "overall":         overall,
        "buckets":         bucket_stats,
        "bucket_ranking":  [
            {"bucket": lbl, "win_rate_pct": s.get("win_rate_pct"), "n": s.get("n"),
             "avg_pnl": s.get("avg_pnl"), "total_pnl": s.get("total_pnl")}
            for lbl, s in ranked
        ],
        "histogram_60s":   {
            f"{k}-{k+59}s": v
            for k, v in sorted(hist_60s.items())
        },
        "by_asset":        asset_bucket_stats,
    }

    if outside_window:
        result["outside_window_trades"] = [
            {"id": r["id"], "ticker": r["ticker"],
             "timestamp": r["timestamp"], "secs": r["secs"], "bucket": r["bucket"]}
            for r in outside_window
        ]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    SEP = "-" * 70
    meta = results["meta"]
    ov   = results["overall"]
    print(f"\n{SEP}")
    print(f"  TIMING BACKTEST  trades={meta['total_trades']}  "
          f"WR={ov['win_rate_pct']}%  P&L=${ov['total_pnl']:+.2f}")
    if meta["outside_window_n"]:
        print(f"  WARNING: {meta['outside_window_n']} trades outside 0-660s window")
    print(SEP)
    print(f"  {'Bucket':>12}  {'n':>4}  {'WR%':>6}  {'TotalP&L':>10}  "
          f"{'AvgP&L':>8}  {'AvgEntry':>8}  {'AvgEdge':>8}  Attempt")
    print(f"  {'-'*12}  {'-'*4}  {'-'*6}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  -------")
    for label in BUCKET_LABELS:
        s = results["buckets"].get(label, {})
        n = s.get("n", 0)
        if n == 0:
            print(f"  {label:>12}  {'0':>4}  {'—':>6}  {'—':>10}  {'—':>8}  {'—':>8}  {'—':>8}")
            continue
        wr  = f"{s['win_rate_pct']:.1f}" if s["win_rate_pct"] is not None else "—"
        ep  = f"{s['avg_entry_price']:.1f}" if s["avg_entry_price"] else "—"
        eg  = f"{s['avg_edge']:.3f}" if s["avg_edge"] else "—"
        att = s.get("entry_attempt", "—")
        print(f"  {label:>12}  {n:>4}  {wr:>6}  "
              f"${s['total_pnl']:>+9.2f}  ${s['avg_pnl']:>+7.2f}  "
              f"{ep:>8}  {eg:>8}  {att}")
    print(SEP)
    print(f"\n  BUCKET RANKING (by win rate):")
    for r in results["bucket_ranking"]:
        print(f"  {r['bucket']:>12}  WR={r['win_rate_pct']:>5.1f}%  "
              f"n={r['n']}  avg=${r['avg_pnl']:+.2f}")
    print(f"\n  ENTRY TIME HISTOGRAM (60s bins):")
    for bin_lbl, count in results["histogram_60s"].items():
        bar = "#" * min(count, 40)
        print(f"  {bin_lbl:>12}  {bar:<40}  {count}")
    print(f"{SEP}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trade entry timing analysis from printer_v2.db"
    )
    parser.add_argument("--db",  default="printer_v2.db")
    parser.add_argument("--out", default="backtest_timing.json")
    args = parser.parse_args()

    db_path  = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"[error] Database not found: {db_path}")
        return

    print(f"Reading {db_path} ...")
    results = run(db_path)

    if "error" in results:
        print(f"[error] {results['error']}")
        return

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out_path}")
    _print_summary(results)


if __name__ == "__main__":
    main()

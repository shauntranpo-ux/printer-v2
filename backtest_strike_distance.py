"""
backtest_strike_distance.py — Win rate and P&L by BTC-price distance from strike.

For every closed/expired trade, computes:

    distance_pct = |btc_price_at_entry - strike| / strike * 100

where 'strike' is parsed from the Kalshi market_ticker (the last hyphen-separated
segment after stripping any letter prefix: KXBTC15M-25APR14-B65000 → 65000).

For non-BTC markets (SOL, ETH, XRP) the strike is that asset's threshold, not a
BTC price level, so those trades are included with an 'asset_mismatch' flag and
broken out separately.

Buckets (user-specified):
    0-0.5%     BTC within 0.5% of strike   (near the knife-edge)
    0.5-1%     BTC within 0.5–1%
    1-2%       BTC 1–2% from strike
    2%+        BTC more than 2% from strike (clearly in/out of money)

Also tracked per bucket:
    in_the_money   / out_of_the_money split
        YES trade: ITM when btc_price > strike
        NO  trade: ITM when btc_price < strike
    avg entry price, avg edge, avg confidence
    exit-reason distribution

Schema used (from database.py / PRAGMA inspection):
    trades: id, market_ticker, timestamp, direction, entry_price,
            pnl_dollars, exit_reason, edge, ensemble_confidence,
            btc_price_at_entry

Output: backtest_strike_distance.json

Usage:
    python backtest_strike_distance.py
    python backtest_strike_distance.py --db path/to/printer_v2.db
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Bucket config
# ---------------------------------------------------------------------------

BUCKETS: list[tuple[float, float, str]] = [
    (0.0,  0.5,  "0-0.5%"),
    (0.5,  1.0,  "0.5-1%"),
    (1.0,  2.0,  "1-2%"),
    (2.0,  1e9,  "2%+"),
]
BUCKET_LABELS = [label for _, _, label in BUCKETS]

# Assets whose trades have a meaningful btc_price_at_entry vs own_strike comparison
BTC_ASSET_PREFIXES = ("KXBTC",)

KALSHI_PREFIXES = [
    ("KXBTC",  "BTC"),
    ("KXETH",  "ETH"),
    ("KXSOL",  "SOL"),
    ("KXXRP",  "XRP"),
    ("KXDOGE", "DOGE"),
    ("KXHYPE", "HYPE"),
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _asset(ticker: str) -> str:
    t = ticker.upper()
    # Sort longest prefix first to avoid KXBTC matching KXBTC15M early
    for prefix, asset in sorted(KALSHI_PREFIXES, key=lambda x: -len(x[0])):
        if t.startswith(prefix):
            return asset
    return "OTHER"


def _parse_strike(ticker: str) -> float | None:
    """
    Extract the numeric strike/target price from the final '-'-delimited
    segment of a Kalshi market ticker.

    Examples:
        KXBTC15M-25APR14-B65000      → 65000.0
        KXBTC15M-25APR14-T83000      → 83000.0
        KXBTCUSD-25APR11T1400-T83000 → 83000.0
        KXSOL15M-25APR14-T150        → 150.0
        KXXRP15M-25APR14-T1p50       → 1.50  (dot encoded as 'p')
    """
    if not ticker:
        return None
    last_seg = ticker.split("-")[-1]          # e.g. "B65000", "T83000", "T1p50"
    # Normalise decimal-as-p encoding (Kalshi sometimes uses 'p' for '.')
    last_seg = last_seg.replace("p", ".")
    m = re.match(r"[A-Za-z]*(\d+(?:\.\d+)?)", last_seg)
    if m:
        val = float(m.group(1))
        return val if val > 0 else None
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket(dist: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= dist < hi:
            return label
    return "2%+"        # catch-all for anything >= 2


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
    pnls   = [r["pnl"] for r in records]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    edges  = [r["edge"]  for r in records if r["edge"]  is not None]
    prices = [r["entry_price"] for r in records if r["entry_price"] is not None]
    confs  = [r["conf"]  for r in records if r["conf"]  is not None]

    exit_counts: dict[str, int] = defaultdict(int)
    for r in records:
        exit_counts[r["exit_reason"] or "unknown"] += 1

    return {
        "n":               len(records),
        "win_rate_pct":    _pct(len(wins), len(records)),
        "total_pnl":       round(sum(pnls), 4),
        "avg_pnl":         round(sum(pnls) / len(records), 4),
        "avg_win":         round(sum(wins)   / len(wins),   4) if wins   else None,
        "avg_loss":        round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor":   (
            round(-sum(wins) / sum(losses), 4)
            if losses and sum(losses) != 0 else None
        ),
        "sharpe":          _sharpe(pnls),
        "avg_entry_price": round(sum(prices) / len(prices), 2) if prices else None,
        "avg_edge":        round(sum(edges)  / len(edges),  4) if edges  else None,
        "avg_confidence":  round(sum(confs)  / len(confs),  4) if confs  else None,
        "exit_reasons":    dict(exit_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(trades_raw: list, source: str = "local") -> dict:
    raw = list(trades_raw)

    if not raw:
        return {"error": "No closed trades found in database."}

    has_btc_price = any(t.get("btc_price_at_entry") is not None for t in raw)
    schema_note = (
        "btc_price_at_entry present" if has_btc_price
        else "btc_price_at_entry column missing — strike distance unavailable"
    )

    # ── Enrich each trade ──────────────────────────────────────────────────────
    records: list[dict] = []
    skipped_no_btc   = 0
    skipped_no_strike = 0

    for r in raw:
        ticker     = r["market_ticker"]
        asset      = _asset(ticker)
        strike     = _parse_strike(ticker)
        btc_price  = r["btc_price_at_entry"] if has_btc_price else None

        # Compute distance and moneyness only when we have both values
        distance_pct : float | None = None
        in_the_money : bool  | None = None
        btc_match    : bool         = asset == "BTC"

        if btc_price is None:
            skipped_no_btc += 1
        elif strike is None or strike == 0:
            skipped_no_strike += 1
        else:
            distance_pct = abs(btc_price - strike) / strike * 100

            if r["direction"] == "YES":
                in_the_money = btc_price > strike
            else:
                in_the_money = btc_price < strike

        records.append({
            "id":           r["id"],
            "ticker":       ticker,
            "asset":        asset,
            "btc_match":    btc_match,
            "direction":    r["direction"],
            "entry_price":  r["entry_price"],
            "pnl":          r["pnl_dollars"],
            "exit_reason":  r["exit_reason"],
            "edge":         r["edge"],
            "conf":         r["ensemble_confidence"],
            "btc_price":    btc_price,
            "strike":       strike,
            "distance_pct": distance_pct,
            "in_the_money": in_the_money,
            "bucket": (
                _bucket(distance_pct)
                if distance_pct is not None else "unknown"
            ),
        })

    total = len(records)

    # ── Primary analysis: BTC trades only ────────────────────────────────────
    btc_records = [r for r in records if r["btc_match"]]

    def _bucket_breakdown(recs: list[dict]) -> dict:
        """Split into per-bucket stats plus ITM/OTM sub-groups."""
        by_bucket: dict[str, list] = defaultdict(list)
        for r in recs:
            by_bucket[r["bucket"]].append(r)

        out: dict[str, dict] = {}
        for label in BUCKET_LABELS + ["unknown"]:
            group = by_bucket.get(label, [])
            if not group:
                out[label] = {"n": 0}
                continue

            itm  = [r for r in group if r["in_the_money"] is True]
            otm  = [r for r in group if r["in_the_money"] is False]
            unkn = [r for r in group if r["in_the_money"] is None]

            s = _agg(group)
            if itm:
                s["in_the_money"]     = _agg(itm)
            if otm:
                s["out_of_the_money"] = _agg(otm)
            if unkn:
                s["moneyness_unknown"] = {"n": len(unkn)}

            # Avg distance within bucket
            dists = [r["distance_pct"] for r in group if r["distance_pct"] is not None]
            s["avg_distance_pct"] = round(sum(dists) / len(dists), 3) if dists else None

            out[label] = s

        return out

    btc_buckets = _bucket_breakdown(btc_records)

    # ── All-trades breakdown (includes non-BTC with caveat) ───────────────────
    # For non-BTC trades the "distance" is btc_price vs asset_strike (cross-asset),
    # which can still reveal macro correlations but is labelled accordingly.
    all_buckets = _bucket_breakdown(records)

    # ── Bucket ranking ────────────────────────────────────────────────────────
    def _rank(buckets: dict) -> list[dict]:
        return sorted(
            [
                {
                    "bucket":        lbl,
                    "n":             s.get("n", 0),
                    "win_rate_pct":  s.get("win_rate_pct"),
                    "avg_pnl":       s.get("avg_pnl"),
                    "total_pnl":     s.get("total_pnl"),
                    "avg_distance":  s.get("avg_distance_pct"),
                }
                for lbl, s in buckets.items()
                if s.get("n", 0) > 0 and lbl not in ("unknown",)
            ],
            key=lambda x: x["win_rate_pct"] or 0,
            reverse=True,
        )

    # ── Per-asset summary ──────────────────────────────────────────────────────
    by_asset: dict[str, list] = defaultdict(list)
    for r in records:
        by_asset[r["asset"]].append(r)

    asset_summary: dict[str, dict] = {}
    for asset, recs in sorted(by_asset.items()):
        ab = _bucket_breakdown(recs)
        asset_summary[asset] = {
            "n":               len(recs),
            "btc_distance_meaningful": asset == "BTC",
            "note": (
                None if asset == "BTC"
                else f"Strike is {asset} price, not BTC — distance is cross-asset"
            ),
            "buckets": {
                lbl: {"n": s.get("n", 0),
                      "win_rate_pct": s.get("win_rate_pct"),
                      "avg_pnl": s.get("avg_pnl"),
                      "avg_distance_pct": s.get("avg_distance_pct")}
                for lbl, s in ab.items() if s.get("n", 0) > 0
            },
        }

    # ── Strike parsing diagnostics ────────────────────────────────────────────
    parsed_ok     = sum(1 for r in records if r["strike"] is not None)
    parsed_failed = sum(1 for r in records if r["strike"] is None)

    sample_tickers = list({r["ticker"] for r in records})[:10]

    # ── Assembly ──────────────────────────────────────────────────────────────
    return {
        "meta": {
            "source":               source,
            "total_trades":         total,
            "btc_trades":           len(btc_records),
            "schema_note":          schema_note,
            "strike_parsed_ok":     parsed_ok,
            "strike_parse_failed":  parsed_failed,
            "skipped_no_btc_price": skipped_no_btc,
            "sample_tickers":       sample_tickers,
            "distance_formula":     "|btc_price_at_entry - strike| / strike * 100",
            "moneyness": {
                "YES_ITM": "btc_price > strike (BTC already above target)",
                "NO_ITM":  "btc_price < strike (BTC already below target)",
            },
        },
        "overall":                  _agg(records),
        "btc_trades_only": {
            "n":              len(btc_records),
            "buckets":        btc_buckets,
            "bucket_ranking": _rank(btc_buckets),
        },
        "all_trades": {
            "buckets":        all_buckets,
            "bucket_ranking": _rank(all_buckets),
            "note":           (
                "Includes non-BTC trades where distance = |btc_price - asset_strike|"
                " (cross-asset, less interpretable than BTC-only view)."
            ),
        },
        "by_asset": asset_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    SEP = "-" * 72
    meta = results["meta"]
    ov   = results["overall"]

    print(f"\n{SEP}")
    print(f"  STRIKE DISTANCE  total={meta['total_trades']}  "
          f"btc={meta['btc_trades']}  "
          f"WR={ov['win_rate_pct']}%  P&L=${ov['total_pnl']:+.2f}")
    print(f"  Strike parse: {meta['strike_parsed_ok']} ok / "
          f"{meta['strike_parse_failed']} failed  |  {meta['schema_note']}")
    print(SEP)

    for section_label, section_key in [
        ("BTC TRADES ONLY (btc_price vs btc_strike)", "btc_trades_only"),
        ("ALL TRADES (cross-asset flagged)",          "all_trades"),
    ]:
        section = results[section_key]
        if section.get("n", section.get("buckets", {})) == 0:
            continue
        print(f"\n  {section_label}:")
        print(f"  {'Bucket':>10}  {'n':>4}  {'WR%':>6}  {'TotalP&L':>10}  "
              f"{'AvgP&L':>8}  {'AvgDist%':>9}  {'AvgEntry':>9}")
        print(f"  {'-'*10}  {'-'*4}  {'-'*6}  {'-'*10}  "
              f"{'-'*8}  {'-'*9}  {'-'*9}")
        for label in BUCKET_LABELS:
            s = section["buckets"].get(label, {})
            n = s.get("n", 0)
            if n == 0:
                continue
            wr   = f"{s['win_rate_pct']:.1f}" if s.get("win_rate_pct") is not None else "—"
            dist = f"{s['avg_distance_pct']:.2f}" if s.get("avg_distance_pct") is not None else "—"
            ep   = f"{s['avg_entry_price']:.1f}" if s.get("avg_entry_price") else "—"
            print(f"  {label:>10}  {n:>4}  {wr:>6}  "
                  f"${s['total_pnl']:>+9.2f}  ${s['avg_pnl']:>+7.2f}  "
                  f"{dist:>9}  {ep:>9}")

        print(f"\n  Ranking: ", end="")
        for r in section["bucket_ranking"]:
            print(f"{r['bucket']} WR={r['win_rate_pct']}% n={r['n']}", end="  ")
        print()

    print(f"\n  BY ASSET:")
    for asset, data in results["by_asset"].items():
        caveat = "" if data["btc_distance_meaningful"] else "  [cross-asset]"
        print(f"  {asset}  n={data['n']}{caveat}")
        for lbl, s in data["buckets"].items():
            if s.get("n", 0):
                dist = f"{s['avg_distance_pct']:.2f}%" if s.get("avg_distance_pct") else "—"
                print(f"    {lbl:>10}  n={s['n']:>3}  WR={s['win_rate_pct']}%  "
                      f"avg=${s['avg_pnl']:+.2f}  dist={dist}")
    print(f"{SEP}\n")


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
        description="Strike-distance win rate analysis from printer_v2.db"
    )
    parser.add_argument("--db",  default="printer_v2.db")
    parser.add_argument("--url", default=None, help="Live Railway URL (e.g. https://printerv2.up.railway.app)")
    parser.add_argument("--synthetic", action="store_true", help="Load from synthetic_trades.json")
    parser.add_argument("--synthetic-file", default="synthetic_trades.json", dest="synthetic_file")
    parser.add_argument("--out", default="backtest_strike_distance.json")
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
    _print_summary(results)


if __name__ == "__main__":
    main()

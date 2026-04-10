"""
download_data.py — Download Coinbase Exchange 1m OHLCV data for WFA/backtest
(Public API, no auth, no geo-restrictions — US-accessible)

Supports: BTC, ETH, SOL, XRP, DOGE
BNB and HYPE are not listed on Coinbase — skip for backtest.

Usage:
    py download_data.py               # all 5 assets, 2 years
    py download_data.py --asset BTC   # single asset
    py download_data.py --years 3     # longer lookback
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

COINBASE_REST = "https://api.exchange.coinbase.com/products/{product_id}/candles"

# Coinbase product IDs per asset
PRODUCTS = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
}

_BATCH  = 300   # max candles per Coinbase request
_SLEEP  = 0.15  # seconds between requests (~6 req/s, under 10/s limit)


def download(product_id: str, asset: str, years: int, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{asset}USDT_1m.csv"   # keep same naming as before

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=years * 365)
    total_min = int((end_dt - start_dt).total_seconds() / 60)

    print(f"\n{asset} ({product_id}): downloading ~{total_min:,} candles ({years}y)...")

    rows: list[list] = []
    cursor_end = end_dt

    while cursor_end > start_dt:
        cursor_start = max(cursor_end - timedelta(minutes=_BATCH), start_dt)

        resp = requests.get(
            COINBASE_REST.format(product_id=product_id),
            params={
                "start":       cursor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":         cursor_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "granularity": 60,   # 1-minute candles
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            cursor_end = cursor_start - timedelta(seconds=1)
            continue

        for k in batch:
            # Coinbase returns [time, low, high, open, close, volume]
            # time is already unix seconds
            rows.append([
                int(k[0]),      # time (unix seconds)
                float(k[3]),    # open
                float(k[2]),    # high
                float(k[1]),    # low
                float(k[4]),    # close
                float(k[5]),    # volume
            ])

        cursor_end = cursor_start - timedelta(seconds=1)
        done = total_min - int((cursor_end - start_dt).total_seconds() / 60)
        pct  = min(done / total_min * 100, 100)
        print(f"  {pct:5.1f}%  {len(rows):,} rows", end="\r", flush=True)
        time.sleep(_SLEEP)

    # Sort ascending, deduplicate
    rows.sort(key=lambda r: r[0])
    seen: set[int] = set()
    deduped = []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(deduped)

    print(f"\n  Saved {len(deduped):,} rows -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Coinbase Exchange 1m data for backtest/WFA"
    )
    parser.add_argument("--asset",  default="ALL",
                        help="BTC/ETH/SOL/XRP/DOGE or ALL (BNB/HYPE not on Coinbase)")
    parser.add_argument("--years",  type=int, default=2,
                        help="Years of history (default 2)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    targets = (
        list(PRODUCTS.keys())
        if args.asset.upper() == "ALL"
        else [args.asset.upper()]
    )

    for asset in targets:
        product_id = PRODUCTS.get(asset)
        if not product_id:
            print(f"WARNING: {asset} not available on Coinbase Exchange — skipping")
            continue
        try:
            download(product_id, asset, args.years, out_dir)
        except Exception as exc:
            print(f"\n  ERROR downloading {asset}: {exc}")

    print("\nDone. Run WFA with:")
    for asset in targets:
        if asset in PRODUCTS:
            csv_path = Path(args.outdir) / f"{asset}USDT_1m.csv"
            print(f"  py walkforward.py --file {csv_path} --n-strikes 9 --compound")


if __name__ == "__main__":
    main()

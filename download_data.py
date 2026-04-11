"""
download_data.py — Download 1m OHLCV data for WFA/backtest.

Source: Coinbase Exchange (public REST, no auth)
Assets: BTC, ETH, SOL, XRP, DOGE, HYPE

Usage:
    py download_data.py               # all 6 assets, 2 years
    py download_data.py --asset ETH   # single asset
    py download_data.py --years 3     # longer lookback
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ── Coinbase Exchange (public REST, no auth) ─────────────────────────────────
COINBASE_REST = "https://api.exchange.coinbase.com/products/{product_id}/candles"
COINBASE_PRODUCTS = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
    "HYPE": "HYPE-USD",   # listed on Coinbase Jan 2025
}
_CB_BATCH = 300    # max candles per Coinbase request
_CB_SLEEP = 0.15   # ~6 req/s

ALL_ASSETS = list(COINBASE_PRODUCTS.keys())


# ── Coinbase downloader ───────────────────────────────────────────────────────

def _download_coinbase(product_id: str, asset: str, years: int, out_dir: Path) -> Path:
    out_path = out_dir / f"{asset}USDT_1m.csv"
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=years * 365)
    total_min = int((end_dt - start_dt).total_seconds() / 60)

    print(f"\n{asset} (Coinbase {product_id}): ~{total_min:,} candles ({years}y)...")

    rows: list[list] = []
    cursor_end = end_dt

    while cursor_end > start_dt:
        cursor_start = max(cursor_end - timedelta(minutes=_CB_BATCH), start_dt)
        resp = requests.get(
            COINBASE_REST.format(product_id=product_id),
            params={
                "start":       cursor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":         cursor_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "granularity": 60,
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
            # Coinbase: [time, low, high, open, close, volume]
            rows.append([int(k[0]), float(k[3]), float(k[2]),
                         float(k[1]), float(k[4]), float(k[5])])

        cursor_end = cursor_start - timedelta(seconds=1)
        pct = min((1 - (cursor_end - start_dt).total_seconds() /
                   (end_dt - start_dt).total_seconds()) * 100, 100)
        print(f"  {pct:5.1f}%  {len(rows):,} rows", end="\r", flush=True)
        time.sleep(_CB_SLEEP)

    return _save(rows, out_path, asset)


# ── Shared save helper ────────────────────────────────────────────────────────

def _save(rows: list[list], out_path: Path, asset: str) -> Path:
    rows.sort(key=lambda r: r[0])
    seen: set[int] = set()
    deduped = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]  # type: ignore[func-returns-value]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(deduped)

    print(f"\n  Saved {len(deduped):,} rows → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 1m OHLCV data for backtest/WFA (Coinbase Exchange)"
    )
    parser.add_argument("--asset",  default="ALL",
                        help=f"Asset or ALL. Options: {', '.join(ALL_ASSETS)}")
    parser.add_argument("--years",  type=int, default=2,
                        help="Years of history (default 2)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current dir)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)

    targets = ALL_ASSETS if args.asset.upper() == "ALL" else [args.asset.upper()]

    downloaded: list[Path] = []
    for asset in targets:
        try:
            if asset not in COINBASE_PRODUCTS:
                print(f"WARNING: {asset} not supported — skipping")
                continue
            p = _download_coinbase(COINBASE_PRODUCTS[asset], asset, args.years, out_dir)
            downloaded.append(p)
        except Exception as exc:
            print(f"\n  ERROR downloading {asset}: {exc}")

    if downloaded:
        print("\nDone. Run sweep with:")
        files = " ".join(p.name for p in downloaded)
        print(f"  py sweep.py --files {files} --wfa")


if __name__ == "__main__":
    main()

"""
download_data.py — Download Bybit 1m OHLCV data for WFA/backtest
(Bybit public API, no auth, no geo-restrictions)

Usage:
    py download_data.py               # downloads all 7 assets, 2 years
    py download_data.py --asset BTC   # single asset
    py download_data.py --years 3     # extend lookback
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

BYBIT_REST = "https://api.bybit.com/v5/market/kline"

# Bybit linear perpetual / spot symbols for each asset
SYMBOLS = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "XRP":  "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}


def download(symbol: str, years: int, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{symbol}_1m.csv"

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000

    rows: list[list] = []
    # Bybit returns results in DESCENDING order — walk backwards from end
    cursor = end_ms
    total  = (end_ms - start_ms) // (60 * 1000)

    print(f"\n{symbol}: downloading ~{total:,} candles ({years}y)...")

    while cursor > start_ms:
        resp = requests.get(
            BYBIT_REST,
            params={
                "category": "linear",
                "symbol":   symbol,
                "interval": "1",        # 1m
                "start":    start_ms,
                "end":      cursor,
                "limit":    1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            print(f"\n  API error: {data.get('retMsg')} — trying spot category")
            resp2 = requests.get(
                BYBIT_REST,
                params={
                    "category": "spot",
                    "symbol":   symbol,
                    "interval": "1",
                    "start":    start_ms,
                    "end":      cursor,
                    "limit":    1000,
                },
                timeout=15,
            )
            resp2.raise_for_status()
            data = resp2.json()

        batch = data.get("result", {}).get("list", [])
        if not batch:
            break

        # Each entry: [startTime(ms), open, high, low, close, volume, turnover]
        for k in batch:
            t = int(k[0])
            if t < start_ms:
                continue
            rows.append([
                t // 1000,      # time (unix seconds)
                float(k[1]),    # open
                float(k[2]),    # high
                float(k[3]),    # low
                float(k[4]),    # close
                float(k[5]),    # volume
            ])

        # Advance cursor backwards (Bybit returns newest first)
        oldest_in_batch = int(batch[-1][0])
        if oldest_in_batch >= cursor:
            break
        cursor = oldest_in_batch - 1

        pct = (end_ms - cursor) / (end_ms - start_ms) * 100
        print(f"  {min(pct,100):5.1f}%  {len(rows):,} rows", end="\r", flush=True)
        time.sleep(0.05)

    # Sort ascending by time
    rows.sort(key=lambda r: r[0])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)

    print(f"\n  Saved {len(rows):,} rows -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bybit 1m data for backtest/WFA")
    parser.add_argument("--asset",  default="ALL", help="BTC/ETH/SOL/XRP/DOGE/BNB/HYPE or ALL")
    parser.add_argument("--years",  type=int, default=2, help="Years of history (default 2)")
    parser.add_argument("--outdir", default=".",  help="Output directory (default: current)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    targets = (
        list(SYMBOLS.keys())
        if args.asset.upper() == "ALL"
        else [args.asset.upper()]
    )

    for asset in targets:
        sym = SYMBOLS.get(asset)
        if not sym:
            print(f"WARNING: {asset} not supported — skipping")
            continue
        try:
            download(sym, args.years, out_dir)
        except Exception as exc:
            print(f"\n  ERROR downloading {asset}: {exc}")

    print("\nDone. Run WFA with:")
    for asset in targets:
        sym = SYMBOLS.get(asset)
        if sym:
            csv_path = out_dir / f"{sym}_1m.csv"
            print(f"  py walkforward.py --file {csv_path} --n-strikes 9 --compound")


if __name__ == "__main__":
    main()

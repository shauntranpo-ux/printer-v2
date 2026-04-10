"""
download_data.py — Download Binance 1m OHLCV data for WFA/backtest

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

BINANCE_REST = "https://api.binance.com/api/v3/klines"

# Binance symbols for each asset
SYMBOLS = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "XRP":  "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "BNB":  "BNBUSDT",
    # HYPE is not listed on Binance — skip
}


def download(symbol: str, years: int, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{symbol}_1m.csv"

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000

    rows: list[list] = []
    cursor = start_ms
    total  = (end_ms - start_ms) // (60 * 1000)   # approximate total 1m candles

    print(f"\n{symbol}: downloading ~{total:,} candles ({years}y)...")

    while cursor < end_ms:
        resp = requests.get(
            BINANCE_REST,
            params={
                "symbol":    symbol,
                "interval":  "1m",
                "startTime": cursor,
                "endTime":   end_ms,
                "limit":     1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for k in batch:
            # k[0] = open_time ms → convert to unix seconds
            rows.append([
                int(k[0]) // 1000,  # time (seconds)
                float(k[1]),        # open
                float(k[2]),        # high
                float(k[3]),        # low
                float(k[4]),        # close
                float(k[5]),        # volume
            ])

        cursor = int(batch[-1][0]) + 60_000   # advance past last candle
        pct = (cursor - start_ms) / (end_ms - start_ms) * 100
        print(f"  {pct:5.1f}%  {len(rows):,} rows", end="\r", flush=True)
        time.sleep(0.08)   # stay well under Binance rate limit

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)

    print(f"\n  Saved {len(rows):,} rows → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance 1m data for backtest/WFA")
    parser.add_argument("--asset",  default="ALL", help="BTC/ETH/SOL/XRP/DOGE/BNB or ALL")
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
            print(f"WARNING: {asset} not in Binance symbol map — skipping")
            continue
        download(sym, args.years, out_dir)

    print("\nDone. Run WFA with:")
    for asset in targets:
        sym = SYMBOLS.get(asset)
        if sym:
            print(f"  py walkforward.py --file {out_dir / (sym + '_1m.csv')} --n-strikes 9 --compound")


if __name__ == "__main__":
    main()

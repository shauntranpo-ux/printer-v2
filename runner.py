"""
runner.py — Main 15-minute trading loop

Responsibilities:
- Orchestrate all components: feed, ensemble, risk, strategy, client
- Run on a 15-minute cadence aligned to the clock (e.g., :00, :15, :30, :45)
- On each tick: fetch candles → run ensemble → check risk gates → size + place order
- Track open positions and check exit conditions every minute
- Handle exceptions gracefully and send Telegram alerts on errors
- Maintain bot state across restarts via database
"""

import asyncio

# TODO: implement BotRunner class
# TODO: implement run_tick() — one full 15m decision cycle
# TODO: implement monitor_positions() — 1m exit condition checker
# TODO: implement clock_align() — sleep until next 15m boundary
# TODO: implement graceful shutdown on SIGINT/SIGTERM
# TODO: wire up: CoinbaseFeed + EnsembleEngine + RiskGates + Strategy + KalshiClient

async def main():
    # TODO: initialize all components from config
    # TODO: start feed, then enter main loop
    pass

if __name__ == "__main__":
    asyncio.run(main())

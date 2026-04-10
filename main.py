"""
main.py — Unified entry point: TradingBot + Flask dashboard in one process.

Both share the same SQLite file (printer_v2.db), so market_watch,
balance, and trade data written by the bot are immediately visible
on the dashboard.

Deploy as a single Railway 'web' service:
    Start command: python main.py
    -- then disable/delete the separate worker service in Railway --
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import threading
from pathlib import Path

# Set up logging before any module imports so runner.py's basicConfig is a no-op
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("printer_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

_STOP_FILE = Path("STOP")


# ---------------------------------------------------------------------------
# Bot background thread
# ---------------------------------------------------------------------------

def _bot_thread() -> None:
    """
    Run TradingBot in a daemon thread with its own asyncio event loop.

    - Starts automatically unless STOP file is present.
    - Restarts after crashes (30s delay).
    - Pauses while STOP file exists; resumes when deleted (START button).
    """
    while True:
        # Wait while explicitly stopped
        if _STOP_FILE.exists():
            time.sleep(3)
            continue

        log.info("[bot] Starting TradingBot...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from runner import TradingBot
            bot = TradingBot()
            loop.run_until_complete(bot.run())
            log.info("[bot] TradingBot exited cleanly")
        except SystemExit:
            # STOP file triggered sys.exit(0) in runner start()
            log.info("[bot] Bot exited via SystemExit (STOP file at startup)")
        except Exception as exc:
            log.error("[bot] Bot crashed: %s", exc, exc_info=True)
            log.info("[bot] Waiting 30s before restart...")
            time.sleep(30)
        finally:
            try:
                loop.close()
            except Exception:
                pass

        log.info("[bot] Bot stopped — will restart unless STOP file present")
        time.sleep(5)


# Start the bot in a background daemon thread immediately at import time
# (daemon=True means it is killed automatically when the main process exits)
_bot = threading.Thread(target=_bot_thread, daemon=True, name="trading-bot")
_bot.start()

# Import dashboard — registers all Flask routes, exposes `app`
from dashboard import app  # noqa: E402  (import after thread start is intentional)


# ---------------------------------------------------------------------------
# Entry point (python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log.info("Dashboard starting on 0.0.0.0:%d", port)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,   # MUST be False — reloader forks and starts bot twice
        threaded=True,
    )

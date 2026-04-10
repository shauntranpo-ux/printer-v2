"""
telegram_alerts.py — Telegram notification system
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ---------------------------------------------------------------------------
# Message queue + sender
# ---------------------------------------------------------------------------

class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._http = httpx.AsyncClient(timeout=10.0)

    async def start(self) -> None:
        if not self._enabled:
            log.info("Telegram alerts disabled (no token/chat_id configured)")
            return
        self._task = asyncio.create_task(self._sender_loop(), name="telegram-sender")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._http.aclose()

    async def send(self, text: str) -> None:
        """Enqueue a message — never blocks the caller."""
        if self._enabled:
            await self._queue.put(text)

    async def _sender_loop(self) -> None:
        url = TELEGRAM_API.format(token=self._token)
        while True:
            text = await self._queue.get()
            try:
                await self._http.post(url, json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                })
            except Exception as exc:
                log.warning("Telegram send failed: %s", exc)
            finally:
                self._queue.task_done()
            await asyncio.sleep(0.3)   # stay well under 30 msg/s limit

    # ------------------------------------------------------------------
    # Typed alert helpers
    # ------------------------------------------------------------------

    async def alert_trade_open(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        dollar_size: float,
        confidence: float,
    ) -> None:
        emoji = "📈" if side == "yes" else "📉"
        await self.send(
            f"{emoji} <b>TRADE OPEN</b>\n"
            f"Market: <code>{ticker}</code>\n"
            f"Side: <b>{side.upper()}</b>  ×{contracts} contracts @ {price_cents}¢\n"
            f"Cost: <b>${dollar_size:.2f}</b>  |  Confidence: {confidence:.0%}"
        )

    async def alert_trade_close(
        self,
        ticker: str,
        side: str,
        contracts: int,
        entry_cents: int,
        exit_cents: int,
        pnl: float,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        sign = "+" if pnl >= 0 else ""
        await self.send(
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"Market: <code>{ticker}</code>  {side.upper()} ×{contracts}\n"
            f"Entry: {entry_cents}¢  →  Exit: {exit_cents}¢\n"
            f"P&amp;L: <b>{sign}${pnl:.2f}</b>"
        )

    async def alert_gate_blocked(self, gate_num: int, gate_name: str, reason: str) -> None:
        await self.send(
            f"🚧 <b>Gate {gate_num} [{gate_name}] blocked trade</b>\n"
            f"{reason}"
        )

    async def alert_error(self, context: str, error: str) -> None:
        await self.send(
            f"🔴 <b>ERROR</b> [{context}]\n"
            f"<code>{error[:300]}</code>"
        )

    async def alert_daily_summary(
        self,
        trades: int,
        wins: int,
        net_pnl: float,
        balance: float,
    ) -> None:
        win_rate = wins / trades if trades else 0.0
        emoji = "💰" if net_pnl >= 0 else "📉"
        sign = "+" if net_pnl >= 0 else ""
        await self.send(
            f"{emoji} <b>Daily Summary</b>\n"
            f"Trades: {trades}  |  Win rate: {win_rate:.0%}\n"
            f"Net P&amp;L: <b>{sign}${net_pnl:.2f}</b>\n"
            f"Balance: ${balance:.2f}"
        )

    async def alert_startup(self, env: str, btc_price: float) -> None:
        await self.send(
            f"🟢 <b>printer-v2 started</b>  [{env}]\n"
            f"BTC: ${btc_price:,.2f}"
        )

    async def alert_shutdown(self, reason: str = "clean exit") -> None:
        await self.send(f"🔴 <b>printer-v2 stopped</b>  [{reason}]")

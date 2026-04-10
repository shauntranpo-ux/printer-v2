"""
telegram_alerts.py — Telegram notification system
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from config import settings
from database import DailyStats, TradeRow

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_RETRIES = 3
_RETRY_DELAYS = (2.0, 4.0)   # seconds between attempt 1→2 and 2→3


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(value: float | None, scale: float = 1.0) -> str:
    """Format a fraction as a percentage string, e.g. 0.735 → '73.5%'."""
    if value is None:
        return "—"
    return f"{value * scale:.1f}%"


def _usd(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.2f}"


def _momentum_emoji(btc_momentum: float | None) -> str:
    if btc_momentum is None:
        return "➡️"
    if btc_momentum > 0.001:
        return "📈"
    if btc_momentum < -0.001:
        return "📉"
    return "➡️"


def _hold_time(opened: str | None, closed: str | None) -> str:
    """Return a human-readable hold duration, e.g. '1h 23m' or '8m'."""
    if not opened or not closed:
        return "—"
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        t0 = datetime.strptime(opened, fmt).replace(tzinfo=timezone.utc)
        t1 = datetime.strptime(closed, fmt).replace(tzinfo=timezone.utc)
        secs = int((t1 - t0).total_seconds())
        if secs < 0:
            return "—"
        h, rem = divmod(secs, 3600)
        m = rem // 60
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"
    except (ValueError, TypeError):
        return "—"


def _direction_line(direction: str) -> str:
    arrow = "🟢" if direction == "YES" else "🔴"
    return f"{arrow} <b>{direction}</b>"


# ---------------------------------------------------------------------------
# TelegramAlerter
# ---------------------------------------------------------------------------

class TelegramAlerter:
    def __init__(self) -> None:
        self._token    = settings.TELEGRAM_BOT_TOKEN
        self._chat_id  = settings.TELEGRAM_CHAT_ID
        self._enabled  = settings.telegram_enabled
        self._url      = _API_BASE.format(token=self._token)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._http = httpx.AsyncClient(timeout=10.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled:
            log.info("Telegram disabled — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
            return
        self._task = asyncio.create_task(self._sender_loop(), name="telegram-sender")
        log.info("Telegram alerter started (chat_id=%s)", self._chat_id)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal queue + retry sender
    # ------------------------------------------------------------------

    async def _enqueue(self, text: str) -> None:
        """Put a message on the queue. Always returns immediately."""
        if self._enabled:
            await self._queue.put(text)

    async def _sender_loop(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                await self._send_with_retry(text)
            except Exception as exc:
                # Already exhausted retries inside _send_with_retry;
                # this is a belt-and-suspenders catch so the loop never dies.
                log.error("Telegram sender loop unexpected error: %s", exc)
            finally:
                self._queue.task_done()
            await asyncio.sleep(0.35)   # ~2-3 msg/s, well under Telegram's 30/s limit

    async def _send_with_retry(self, text: str) -> None:
        payload = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await self._http.post(self._url, json=payload)
                if resp.status_code == 429:
                    # Telegram rate-limit: honour Retry-After header
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    log.warning("Telegram rate-limited — waiting %.0fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return   # success
            except httpx.HTTPStatusError as exc:
                log.warning("Telegram HTTP %d on attempt %d: %s",
                            exc.response.status_code, attempt, exc)
            except Exception as exc:
                log.warning("Telegram send attempt %d failed: %s", attempt, exc)

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAYS[attempt - 1])

        log.warning("Telegram: all %d attempts failed — message dropped (silent fail)", _MAX_RETRIES)

    # ------------------------------------------------------------------
    # Public alert methods
    # ------------------------------------------------------------------

    async def send_trade_entry(self, trade: TradeRow) -> None:
        mom_emoji = _momentum_emoji(trade.btc_momentum)
        mom_score = f"{trade.btc_momentum:+.4f}" if trade.btc_momentum is not None else "—"
        btc = f"${trade.btc_price_at_entry:,.2f}" if trade.btc_price_at_entry else "—"
        entry_cents = int(trade.entry_price)

        await self._enqueue(
            f"🟢 <b>TRADE ENTERED</b>\n"
            f"Market: <code>{trade.market_ticker}</code>\n"
            f"Direction: {_direction_line(trade.direction)}\n"
            f"Size: <b>${trade.size_dollars:.2f}</b> ({trade.contracts} contracts)\n"
            f"Entry Price: <b>{entry_cents}¢</b>\n"
            f"Edge: <b>{_pct(trade.edge, 100)}</b>\n"
            f"Confidence: <b>{_pct(trade.ensemble_confidence, 100)}</b>\n"
            f"BTC Price: {btc}\n"
            f"Momentum: {mom_emoji} {mom_score}"
        )

    async def send_trade_exit(self, trade: TradeRow, reason: str) -> None:
        pnl     = trade.pnl_dollars or 0.0
        pnl_pct = (pnl / trade.size_dollars * 100) if trade.size_dollars else 0.0
        hold    = _hold_time(trade.timestamp, trade.closed_at)
        sign    = "+" if pnl >= 0 else ""

        _icons: dict[str, tuple[str, str]] = {
            "take_profit":   ("💰", "TAKE PROFIT"),
            "trailing_stop": ("📈", "TRAILING STOP"),
            "stop_loss":     ("🛑", "STOP LOSS"),
            "decay":         ("📉", "DECAY EXIT"),
            "expired":       ("⏰", "EXPIRED"),
            "manual":        ("🔧", "MANUAL CLOSE"),
        }
        icon, label = _icons.get(reason, ("❌", reason.upper()))

        lines = [
            f"{icon} <b>{label}</b>",
            f"Market: <code>{trade.market_ticker}</code>",
            f"Direction: {_direction_line(trade.direction)}",
            f"P&amp;L: <b>{sign}${pnl:.2f}</b> ({sign}{pnl_pct:.1f}%)",
            f"Hold Time: {hold}",
        ]

        if reason == "trailing_stop" and trade.peak_pnl_pct is not None:
            lines.append(f"Peak P&amp;L reached: +{trade.peak_pnl_pct * 100:.1f}%")

        if reason in ("stop_loss", "decay"):
            entry = trade.entry_price
            exit_ = trade.exit_price or entry
            lines.append(
                f"Entry → Exit: {int(entry)}¢ → {int(exit_)}¢"
            )

        await self._enqueue("\n".join(lines))

    async def send_daily_summary(
        self,
        stats: DailyStats,
        best_trade:   float | None = None,
        worst_trade:  float | None = None,
        win_rate_yes: float | None = None,
        win_rate_no:  float | None = None,
    ) -> None:
        wr         = _pct(stats.win_rate, 100)
        pnl_emoji  = "💰" if stats.total_pnl >= 0 else "📉"
        loss_limit = settings.DAILY_LOSS_LIMIT

        wr_yes = _pct(win_rate_yes, 100) if win_rate_yes is not None else "—"
        wr_no  = _pct(win_rate_no,  100) if win_rate_no  is not None else "—"

        await self._enqueue(
            f"{pnl_emoji} <b>DAILY SUMMARY</b>\n"
            f"Date: {stats.date}\n"
            f"Trades: {stats.total_trades}  |  Win Rate: <b>{wr}</b>\n"
            f"YES: {wr_yes}  ·  NO: {wr_no}\n"
            f"P&amp;L: <b>{_usd(stats.total_pnl)}</b>\n"
            f"Daily Loss Used: ${stats.daily_loss_used:.2f} / ${loss_limit:.0f}\n"
            f"Best Trade: {_usd(best_trade)}\n"
            f"Worst Trade: {_usd(worst_trade)}"
        )

    async def send_error(self, error_msg: str, context: str) -> None:
        # Truncate long errors so the message doesn't get rejected (4096 char limit)
        truncated = error_msg[:350] + ("…" if len(error_msg) > 350 else "")
        await self._enqueue(
            f"⚠️ <b>BOT ERROR</b>\n"
            f"Error: <code>{truncated}</code>\n"
            f"Context: {context}\n"
            f"Action: Auto-restarting..."
        )

    async def send_startup(self) -> None:
        models = (
            f"Claude ({settings.CLAUDE_MODEL.split('-')[-1]}) / "
            f"GPT-4o / "
            f"Gemini ({settings.GEMINI_MODEL.split('-')[-1]}) / "
            f"DeepSeek"
        )
        await self._enqueue(
            f"🚀 <b>PRINTER V2 STARTED</b>\n"
            f"Mode: <b>{'DEMO' if settings.KALSHI_DEMO else 'LIVE'}</b>\n"
            f"Max Bet: <b>${settings.MAX_BET_SIZE:.0f}</b>\n"
            f"Daily Stop: <b>${settings.DAILY_LOSS_LIMIT:.0f}</b>\n"
            f"Models: {models}\n"
            f"Status: Scanning 24/7"
        )

    async def send_kill_switch(self) -> None:
        await self._enqueue("☠️ <b>KILL SWITCH ACTIVATED</b> — Bot stopped")

"""
ensemble.py — 4-model AI ensemble engine

Runs Claude, GPT-4o, Gemini, and DeepSeek in parallel via asyncio.gather().
Each model receives identical market context and returns a JSON prediction.
Results are weighted into a consensus signal with spread/confidence gating.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types

from config import settings
from coinbase_feed import Candle

log = logging.getLogger(__name__)

_CALL_TIMEOUT = 30.0   # seconds before a model is marked as failed


# ---------------------------------------------------------------------------
# System prompts  (asset symbol is injected at call time)
# ---------------------------------------------------------------------------

def _system_prompt(symbol: str) -> str:
    return (
        f"You are an expert {symbol} short-term trader specializing in 15-minute binary outcomes. "
        f"You analyze price action, momentum, RSI, candle patterns, and order flow to make "
        f"high-conviction directional calls. "
        f"You are NOT a hedge — you commit to a view. When data supports a direction, "
        f"your probability should reflect that conviction (0.65–0.85 for strong signals, "
        f"0.55–0.65 for moderate signals). "
        f"Only return 0.45–0.55 when data is genuinely contradictory with no clear edge. "
        f"Respond in JSON only."
    )

_JSON_SCHEMA_HINT = (
    '\n\nRespond with ONLY this JSON — no other text:\n'
    '{"direction": "YES" or "NO", "probability": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "one sentence"}\n'
    'probability = P(YES) = probability price closes ABOVE strike.\n'
    'direction "YES" → probability > 0.50 | direction "NO" → probability < 0.50\n'
    'COMMIT to a view. If trend/momentum is clear, show it: 0.68, 0.72, 0.30, 0.25 etc.\n'
    'DO NOT default to 0.50 unless signals are genuinely mixed.'
)


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BtcData:
    price:          float           # current asset price in USD
    momentum:       float           # -1.0 to +1.0 from CoinbaseFeed.get_momentum_for()
    candles:        list[Candle]    # last 10 completed 15m candles
    imbalance:      float           # bid_vol / ask_vol from order book
    symbol:         str = "BTC"     # asset symbol — used in AI prompts
    current_candle: dict | None = None  # in-progress candle (live, incomplete)


@dataclass
class Market:
    ticker:       str
    yes_price:    int          # cents (1–99)
    no_price:     int          # cents (1–99)
    strike_price: float        # BTC USD threshold
    close_time:   datetime     # UTC expiry


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    direction:  str            # "YES" | "NO"
    probability: float         # P(YES) in [0.0, 1.0]
    confidence:  float         # model's self-reported confidence [0.0, 1.0]
    reasoning:   str
    latency_ms:  float

    @property
    def prob(self) -> float:
        """Alias for probability — backward compat with runner.py."""
        return self.probability


@dataclass
class EnsembleResult:
    consensus_prob: float      # weighted P(YES)
    confidence:     float      # penalized average confidence
    spread:         float      # max(probs) - min(probs)
    action:         str        # "TRADE" | "SKIP" | "WAIT"
    skip_reason:    str | None
    timestamp:      datetime
    claude:   ModelResult | None = field(default=None)
    gpt:      ModelResult | None = field(default=None)
    gemini:   ModelResult | None = field(default=None)
    deepseek: ModelResult | None = field(default=None)

    # ------------------------------------------------------------------
    # Compatibility properties (used by risk_gates.py / strategy.py)
    # ------------------------------------------------------------------

    @property
    def direction(self) -> str:
        """
        Lowercase direction for downstream consumers.
        Returns "flat" when action is not TRADE so risk_gates gate 3
        correctly blocks the trade without any code changes.
        """
        if self.action != "TRADE":
            return "flat"
        return "yes" if self.consensus_prob > 0.5 else "no"

    @property
    def raw_prob(self) -> float:
        """Alias for consensus_prob — used by strategy.py."""
        return self.consensus_prob

    @property
    def models(self) -> list[ModelResult]:
        """Non-None results — used by runner.py spread calculation."""
        return [m for m in (self.claude, self.gpt, self.gemini, self.deepseek)
                if m is not None]


# ---------------------------------------------------------------------------
# EnsembleEngine
# ---------------------------------------------------------------------------

class EnsembleEngine:
    def __init__(self) -> None:
        # Lazy-initialised SDK clients
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        self._openai_client:    openai.AsyncOpenAI | None = None
        self._deepseek_client:  openai.AsyncOpenAI | None = None
        self._gemini_client:    genai.Client | None = None

    # ------------------------------------------------------------------
    # Client initialisation (lazy — avoids failures at import time)
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        if not self._anthropic_client:
            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
        if not self._openai_client:
            self._openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )
        if not self._deepseek_client:
            self._deepseek_client = openai.AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
            )
        if not self._gemini_client:
            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(btc_data: BtcData, market: Market) -> str:
        sym      = btc_data.symbol
        price    = btc_data.price
        strike   = market.strike_price
        now_utc  = datetime.now(timezone.utc)
        mins_left = max(0, int((market.close_time - now_utc).total_seconds() / 60))
        candles  = btc_data.candles  # up to 10 completed 15m candles

        # ── Strike distance ───────────────────────────────────────────────
        dist_pct = ((strike - price) / price * 100) if price > 0 else 0.0
        if abs(dist_pct) < 0.01:
            strike_note = f"price is AT the strike (coin-flip territory unless momentum is strong)"
        elif dist_pct > 0:
            strike_note = f"price must rise {dist_pct:.3f}% to hit YES"
        else:
            strike_note = f"price must fall {abs(dist_pct):.3f}% to hit NO (currently {abs(dist_pct):.3f}% above strike)"

        # ── Multi-timeframe price change ──────────────────────────────────
        def pct_chg(old: float, new: float) -> str:
            if old <= 0:
                return "n/a"
            return f"{(new - old) / old * 100:+.3f}%"

        chg_15m = pct_chg(candles[-2].close, candles[-1].close) if len(candles) >= 2 else "n/a"
        chg_60m = pct_chg(candles[-5].close, candles[-1].close) if len(candles) >= 5 else "n/a"
        chg_2h  = pct_chg(candles[-9].close, candles[-1].close) if len(candles) >= 9 else "n/a"

        # ── RSI (14-period approx on available candles) ───────────────────
        rsi_str = "n/a"
        if len(candles) >= 3:
            moves = [(c.close - c.open) for c in candles]
            gains  = [m for m in moves if m > 0]
            losses = [-m for m in moves if m < 0]
            avg_g = sum(gains)  / len(gains)  if gains  else 0.0
            avg_l = sum(losses) / len(losses) if losses else 1e-9
            rsi   = 100 - (100 / (1 + avg_g / avg_l))
            rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "neutral"
            rsi_str = f"{rsi:.0f} ({rsi_label})"

        # ── Trend & candle bias ───────────────────────────────────────────
        if len(candles) >= 4:
            first_c = candles[-4].close
            last_c  = candles[-1].close
            trend_pct = (last_c - first_c) / first_c * 100 if first_c > 0 else 0
            trend = "UPTREND" if trend_pct > 0.15 else "DOWNTREND" if trend_pct < -0.15 else "SIDEWAYS"
            bull_count = sum(1 for c in candles[-4:] if c.close >= c.open)
            candle_bias = f"{bull_count}/4 bullish candles"
        else:
            trend = "UNKNOWN"
            candle_bias = "insufficient data"

        # ── Order book signal ─────────────────────────────────────────────
        imb = btc_data.imbalance
        ob_signal = (
            f"{imb:.2f}x — strong BUY pressure" if imb > 1.5 else
            f"{imb:.2f}x — mild BUY pressure"   if imb > 1.1 else
            f"{imb:.2f}x — strong SELL pressure" if imb < 0.67 else
            f"{imb:.2f}x — mild SELL pressure"   if imb < 0.9 else
            f"{imb:.2f}x — balanced"
        )

        # ── Completed candles table ───────────────────────────────────────
        if candles:
            candle_rows = []
            for c in candles[-6:]:   # last 6 for readability
                body_pct = (c.close - c.open) / c.open * 100 if c.open > 0 else 0
                arrow    = "▲" if c.close >= c.open else "▼"
                candle_rows.append(
                    f"  {c.timestamp.strftime('%H:%M')} {arrow} "
                    f"O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f} "
                    f"({body_pct:+.2f}%) vol={c.volume:.2f}"
                )
            candles_str = "\n".join(candle_rows)
        else:
            candles_str = "  (no history — candles loading)"

        # ── Current (live) candle ─────────────────────────────────────────
        cc = btc_data.current_candle
        if cc and cc.get("open", 0) > 0:
            live_pct = (price - cc["open"]) / cc["open"] * 100
            live_str = (
                f"  LIVE O={cc['open']:.2f} H={cc['high']:.2f} "
                f"L={cc['low']:.2f} C={price:.2f} ({live_pct:+.2f}%) ← current candle"
            )
        else:
            live_str = "  (live candle data pending)"

        return f"""=== {sym}/USD — 15-MINUTE BINARY MARKET ===
Price now:    ${price:,.4f}
Strike (YES threshold): ${strike:,.4f}
Strike note:  {strike_note}
Time left:    {mins_left} min until expiry
Market price: YES={market.yes_price}¢  NO={market.no_price}¢

=== PRICE ACTION ===
15-min change: {chg_15m}
60-min change: {chg_60m}
2-hour change: {chg_2h}
Trend (last 4 candles): {trend}
Candle bias:   {candle_bias}
RSI (approx):  {rsi_str}
Order book:    {ob_signal}
Momentum score: {btc_data.momentum:+.3f} (range -1 to +1)

=== CANDLE HISTORY (15m, oldest → newest) ===
{candles_str}
{live_str}

=== TASK ===
Will {sym} close ABOVE ${strike:,.4f} at {market.close_time.strftime('%H:%M UTC')}?
You have {mins_left} minutes of price movement left.

Analyze trend, RSI, candle bias, momentum, and order book. Make a DECISIVE call.
{_JSON_SCHEMA_HINT}"""

    # ------------------------------------------------------------------
    # JSON parser (shared by all models)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(text: str, model_name: str, latency_ms: float) -> ModelResult:
        # Strip markdown code fences that some models add
        cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.strip("`").strip()

        # Find the first complete JSON object (handles extra prose around it)
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(
                f"{model_name}: no JSON object found in response: {text[:300]!r}"
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{model_name}: JSON parse error — {exc}") from exc

        direction = str(data.get("direction", "")).upper().strip()
        if direction not in ("YES", "NO"):
            raise ValueError(
                f"{model_name}: direction must be YES or NO, got '{direction}'"
            )

        probability = float(data.get("probability", -1))
        confidence  = float(data.get("confidence", -1))

        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{model_name}: probability {probability} out of [0,1]")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{model_name}: confidence {confidence} out of [0,1]")

        # Normalize: probability must always represent P(YES).
        # If a model returned P(direction) instead — e.g. NO with 0.70 — flip it.
        # Inconsistency: NO + prob > 0.5 means model meant "70% chance NO" = P(YES)=0.30
        #                YES + prob < 0.5 means model meant "40% chance YES" — already correct
        # We detect the cross-direction case (NO+high or YES+low) and flip.
        if (direction == "NO" and probability > 0.5) or (direction == "YES" and probability < 0.5):
            probability = 1.0 - probability
        # Re-derive direction from the now-correct P(YES)
        direction = "YES" if probability >= 0.5 else "NO"

        return ModelResult(
            model_name  = model_name,
            direction   = direction,
            probability = probability,
            confidence  = confidence,
            reasoning   = str(data.get("reasoning", ""))[:500],
            latency_ms  = latency_ms,
        )

    # ------------------------------------------------------------------
    # Individual model calls
    # ------------------------------------------------------------------

    async def _call_claude(self, context: str, symbol: str = "BTC") -> ModelResult:
        t0 = time.monotonic()
        msg = await self._anthropic_client.messages.create(  # type: ignore[union-attr]
            model      = settings.CLAUDE_MODEL,
            max_tokens = 300,
            temperature= 0.3,
            system     = _system_prompt(symbol),
            messages   = [{"role": "user", "content": context}],
        )
        text = msg.content[0].text
        return self._parse_result(text, "claude", (time.monotonic() - t0) * 1000)

    async def _call_gpt(self, context: str, symbol: str = "BTC") -> ModelResult:
        t0 = time.monotonic()
        resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model           = settings.GPT_MODEL,
            temperature     = 0.3,
            max_tokens      = 300,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": _system_prompt(symbol)},
                {"role": "user",   "content": context},
            ],
        )
        text = resp.choices[0].message.content or ""
        return self._parse_result(text, "gpt", (time.monotonic() - t0) * 1000)

    # Ordered list of Gemini models to try — first success wins.
    # Google frequently deprecates specific versions; this auto-advances.
    _GEMINI_FALLBACKS = [
        "gemini-2.5-flash",                  # primary
        "gemini-2.5-flash-preview-04-17",    # versioned stable
        "gemini-2.5-pro",                    # pro fallback
        "gemini-2.0-flash-lite-001",         # lite stable versioned
        "gemini-2.0-flash-latest",           # latest alias
    ]

    # Models confirmed dead (404) this session — skip without API call
    _GEMINI_DEAD: set[str] = set()

    async def _call_gemini(self, context: str, symbol: str = "BTC") -> ModelResult:
        t0 = time.monotonic()

        # Build candidate list: configured model first, then fallbacks
        # Skip any model we already know returns 404 this session
        all_candidates = [settings.GEMINI_MODEL] + [
            m for m in self._GEMINI_FALLBACKS if m != settings.GEMINI_MODEL
        ]
        candidates = [m for m in all_candidates if m not in EnsembleEngine._GEMINI_DEAD]
        if not candidates:
            # All known models dead — reset and try everything once more
            EnsembleEngine._GEMINI_DEAD.clear()
            candidates = all_candidates

        last_exc: Exception = RuntimeError("No Gemini models available")
        for model in candidates:
            try:
                response = await self._gemini_client.aio.models.generate_content(  # type: ignore[union-attr]
                    model=model,
                    contents=context,
                    config=types.GenerateContentConfig(
                        system_instruction=_system_prompt(symbol),
                        temperature=0.3,
                    ),
                )
                if model != settings.GEMINI_MODEL and model not in EnsembleEngine._GEMINI_DEAD:
                    log.info("Gemini: fell back to model '%s'", model)
                text = response.text
                return self._parse_result(text, "gemini", (time.monotonic() - t0) * 1000)
            except Exception as exc:
                msg = str(exc)
                # 503 = service temporarily overloaded — try next fallback model
                if "503" in msg or "UNAVAILABLE" in msg or "Service Unavailable" in msg:
                    log.warning("Gemini '%s' overloaded (503) — trying next fallback", model)
                    last_exc = exc
                    continue
                # 404 = model doesn't exist — remember and skip permanently this session
                if "404" in msg or "NOT_FOUND" in msg or "no longer available" in msg or "not found" in msg.lower():
                    EnsembleEngine._GEMINI_DEAD.add(model)
                    log.debug("Gemini model '%s' unavailable (404) — added to skip list", model)
                    last_exc = exc
                    continue
                raise   # unexpected error — don't swallow it

        raise last_exc

    async def _call_deepseek(self, context: str, symbol: str = "BTC") -> ModelResult:
        t0 = time.monotonic()
        resp = await self._deepseek_client.chat.completions.create(  # type: ignore[union-attr]
            model      = settings.DEEPSEEK_MODEL,
            temperature= 0.3,
            max_tokens = 512,
            messages   = [
                {"role": "system", "content": _system_prompt(symbol)},
                {"role": "user",   "content": context},
            ],
        )
        msg  = resp.choices[0].message
        text = msg.content or ""
        # deepseek-reasoner sometimes emits only reasoning_content with empty content
        if not text:
            text = getattr(msg, "reasoning_content", "") or ""
        return self._parse_result(text, "deepseek", (time.monotonic() - t0) * 1000)

    # ------------------------------------------------------------------
    # Parallel execution with individual error isolation
    # ------------------------------------------------------------------

    async def _safe_call(
        self, coro: Any, model_name: str
    ) -> ModelResult | None:
        """
        Run one model call with a timeout. Returns None (never raises) so
        asyncio.gather can still collect the other results.
        """
        try:
            return await asyncio.wait_for(coro, timeout=_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning(
                "%s timed out after %.0fs — excluded from consensus",
                model_name, _CALL_TIMEOUT,
            )
        except Exception as exc:
            log.warning(
                "%s failed (%s: %s) — excluded from consensus",
                model_name, type(exc).__name__, exc,
            )
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def debate(self, btc_data: BtcData, market: Market) -> EnsembleResult:
        """
        Run all 4 models in parallel and aggregate into an EnsembleResult.
        Requires at least 2 successful model responses.
        """
        self._init_clients()
        context = self._build_context(btc_data, market)

        symbol = btc_data.symbol
        log.info(
            "Ensemble debate starting — %s=$%.4f momentum=%.3f market=%s "
            "exp=%s YES=%d¢/NO=%d¢",
            symbol, btc_data.price, btc_data.momentum, market.ticker,
            market.close_time.strftime("%H:%M"),
            market.yes_price, market.no_price,
        )

        # Step 1 — run all 4 models in parallel
        claude_r, gpt_r, gemini_r, deepseek_r = await asyncio.gather(
            self._safe_call(self._call_claude(context, symbol),   "claude"),
            self._safe_call(self._call_gpt(context, symbol),     "gpt"),
            self._safe_call(self._call_gemini(context, symbol),  "gemini"),
            self._safe_call(self._call_deepseek(context, symbol),"deepseek"),
        )

        # Step 2 — require minimum 2 successful models
        valid = [r for r in (claude_r, gpt_r, gemini_r, deepseek_r) if r is not None]
        if len(valid) < 2:
            failed = [
                name for name, r in [("claude", claude_r), ("gpt", gpt_r),
                                      ("gemini", gemini_r), ("deepseek", deepseek_r)]
                if r is None
            ]
            raise RuntimeError(
                f"Only {len(valid)}/4 models responded — need ≥ 2. "
                f"Failed: {', '.join(failed)}. Check API keys."
            )

        # Step 3 — weighted consensus probability (re-normalised to survivors)
        weight_map = {
            "claude":   settings.CLAUDE_WEIGHT,
            "gpt":      settings.GPT_WEIGHT,
            "gemini":   settings.GEMINI_WEIGHT,
            "deepseek": settings.DEEPSEEK_WEIGHT,
        }
        survivor_weights = {r.model_name: weight_map[r.model_name] for r in valid}
        total_w = sum(survivor_weights.values())

        consensus_prob = sum(
            r.probability * survivor_weights[r.model_name] / total_w
            for r in valid
        )

        # Step 4 — model spread
        probs  = [r.probability for r in valid]
        spread = max(probs) - min(probs)

        # Step 5 — confidence (average, penalised by high spread)
        avg_confidence = sum(r.confidence for r in valid) / len(valid)
        confidence = avg_confidence * 0.8 if spread > 0.20 else avg_confidence

        # Step 6 — action gate
        # Count how many models agree on direction (YES vs NO by P(YES) > 0.5)
        yes_votes = sum(1 for r in valid if r.probability > 0.5)
        no_votes  = len(valid) - yes_votes
        majority  = max(yes_votes, no_votes) / len(valid)

        skip_reason: str | None = None
        if spread > settings.MAX_MODEL_SPREAD and majority < 0.75:
            # Only WAIT when spread is high AND models are genuinely split.
            # If ≥3/4 models agree on direction (majority ≥ 0.75), proceed despite spread.
            action = "WAIT"
            skip_reason = (
                f"model spread {spread:.3f} > {settings.MAX_MODEL_SPREAD:.2f} "
                f"with only {int(majority * len(valid))}/{len(valid)} models agreeing on direction"
            )
        elif confidence < settings.MIN_CONFIDENCE:
            action = "SKIP"
            skip_reason = (
                f"confidence {confidence:.3f} below MIN_CONFIDENCE "
                f"{settings.MIN_CONFIDENCE:.2f}"
            )
        else:
            action = "TRADE"

        # Log summary
        latencies = {r.model_name: f"{r.latency_ms:.0f}ms" for r in valid}
        log.info(
            "Ensemble result: prob=%.3f spread=%.3f conf=%.3f action=%s  "
            "votes=YES:%d/NO:%d  "
            "| claude=%s gpt=%s gemini=%s deepseek=%s  latencies=%s",
            consensus_prob, spread, confidence, action,
            yes_votes, no_votes,
            f"{claude_r.probability:.2f}"   if claude_r   else "FAIL",
            f"{gpt_r.probability:.2f}"      if gpt_r      else "FAIL",
            f"{gemini_r.probability:.2f}"   if gemini_r   else "FAIL",
            f"{deepseek_r.probability:.2f}" if deepseek_r else "FAIL",
            latencies,
        )
        if skip_reason:
            log.info("Ensemble skip/wait reason: %s", skip_reason)

        return EnsembleResult(
            consensus_prob = consensus_prob,
            confidence     = confidence,
            spread         = spread,
            action         = action,
            skip_reason    = skip_reason,
            timestamp      = datetime.now(timezone.utc),
            claude         = claude_r,
            gpt            = gpt_r,
            gemini         = gemini_r,
            deepseek       = deepseek_r,
        )

    # ------------------------------------------------------------------
    # Compatibility stub (runner.py calls this after trade close)
    # ------------------------------------------------------------------

    def record_outcome(self, entry_price: float, exit_price: float) -> None:
        """
        No-op. AI models don't require outcome feedback.
        Kept for compatibility with runner.py call site.
        """

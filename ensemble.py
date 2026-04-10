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
# System prompts (exact as specified)
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM = (
    "You are a short-term BTC price analyst. "
    "Analyze momentum and order flow to predict 15-minute price direction. "
    "Respond in JSON only."
)

_GPT_SYSTEM = (
    "You are a bullish BTC analyst. Look for reasons price will move up. "
    "Be objective but lean toward identifying upward catalysts. JSON only."
)

_GEMINI_SYSTEM = (
    "You are a bearish BTC analyst. Look for reasons price will drop. "
    "Be objective but identify downward risks. JSON only."
)

_DEEPSEEK_SYSTEM = (
    "You are a risk manager. Assess the probability of this BTC prediction "
    "market resolving YES or NO. Focus on risk of being wrong. JSON only."
)

_JSON_SCHEMA_HINT = (
    '\n\nRespond with exactly this JSON structure and nothing else:\n'
    '{"direction": "YES" or "NO", "probability": 0.0-1.0, '
    '"confidence": 0.0-1.0, "reasoning": "one sentence"}'
)


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BtcData:
    price:     float          # current BTC/USD price
    momentum:  float          # -1.0 to +1.0 from CoinbaseFeed.get_momentum()
    candles:   list[Candle]   # last 4 completed 15m candles
    imbalance: float          # bid_vol / ask_vol from order book


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
        if btc_data.candles:
            candles_str = "\n".join(
                f"  [{c.timestamp.strftime('%H:%M')}] "
                f"O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} "
                f"C={c.close:.2f} V={c.volume:.4f}"
                for c in btc_data.candles[-4:]
            )
        else:
            candles_str = "  (no completed candles yet)"

        now_utc = datetime.now(timezone.utc)
        minutes_left = max(0, int((market.close_time - now_utc).total_seconds() / 60))

        return (
            f"Current BTC price: ${btc_data.price:,.2f}\n"
            f"15m momentum score: {btc_data.momentum:.3f} (-1 to +1)\n"
            f"Last 4 candles OHLCV:\n{candles_str}\n"
            f"Order book imbalance: {btc_data.imbalance:.3f} (bid volume / ask volume)\n"
            f"Market: Will BTC be above ${market.strike_price:,.2f} "
            f"at {market.close_time.strftime('%H:%M UTC')}?\n"
            f"Current market price: YES at {market.yes_price}¢ / NO at {market.no_price}¢\n"
            f"Time to expiry: {minutes_left} minutes"
            + _JSON_SCHEMA_HINT
        )

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

    async def _call_claude(self, context: str) -> ModelResult:
        t0 = time.monotonic()
        msg = await self._anthropic_client.messages.create(  # type: ignore[union-attr]
            model      = settings.CLAUDE_MODEL,
            max_tokens = 256,
            temperature= 0.1,
            system     = _CLAUDE_SYSTEM,
            messages   = [{"role": "user", "content": context}],
        )
        text = msg.content[0].text
        return self._parse_result(text, "claude", (time.monotonic() - t0) * 1000)

    async def _call_gpt(self, context: str) -> ModelResult:
        t0 = time.monotonic()
        resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model           = settings.GPT_MODEL,
            temperature     = 0.1,
            max_tokens      = 256,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": _GPT_SYSTEM},
                {"role": "user",   "content": context},
            ],
        )
        text = resp.choices[0].message.content or ""
        return self._parse_result(text, "gpt", (time.monotonic() - t0) * 1000)

    async def _call_gemini(self, context: str) -> ModelResult:
        t0 = time.monotonic()
        response = await self._gemini_client.aio.models.generate_content(  # type: ignore[union-attr]
            model=settings.GEMINI_MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=_GEMINI_SYSTEM,
                temperature=0.1,
            ),
        )
        text = response.text
        return self._parse_result(text, "gemini", (time.monotonic() - t0) * 1000)

    async def _call_deepseek(self, context: str) -> ModelResult:
        t0 = time.monotonic()
        resp = await self._deepseek_client.chat.completions.create(  # type: ignore[union-attr]
            model      = settings.DEEPSEEK_MODEL,
            temperature= 0.1,
            max_tokens = 512,
            messages   = [
                {"role": "system", "content": _DEEPSEEK_SYSTEM},
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

        log.info(
            "Ensemble debate starting — BTC=$%.2f momentum=%.3f market=%s "
            "exp=%s YES=%d¢/NO=%d¢",
            btc_data.price, btc_data.momentum, market.ticker,
            market.close_time.strftime("%H:%M"),
            market.yes_price, market.no_price,
        )

        # Step 1 — run all 4 models in parallel
        claude_r, gpt_r, gemini_r, deepseek_r = await asyncio.gather(
            self._safe_call(self._call_claude(context),   "claude"),
            self._safe_call(self._call_gpt(context),     "gpt"),
            self._safe_call(self._call_gemini(context),  "gemini"),
            self._safe_call(self._call_deepseek(context),"deepseek"),
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
        skip_reason: str | None = None
        if spread > settings.MAX_MODEL_SPREAD:
            action = "WAIT"
            skip_reason = (
                f"model spread {spread:.3f} exceeds MAX_MODEL_SPREAD "
                f"{settings.MAX_MODEL_SPREAD:.2f}"
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
            "| claude=%s gpt=%s gemini=%s deepseek=%s  latencies=%s",
            consensus_prob, spread, confidence, action,
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

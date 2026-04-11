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

_CALL_TIMEOUT        = 30.0   # seconds before a model is marked as failed
_STREAK_BEFORE_PAUSE = 3      # consecutive failures before pausing a model
_PAUSE_CYCLES        = 5      # debate() calls to skip before auto-retrying


# ---------------------------------------------------------------------------
# System prompts  (asset symbol is injected at call time)
# ---------------------------------------------------------------------------

def _system_prompt(symbol: str) -> str:
    return (
        f"You are a quantitative trading analyst operating in short-duration binary outcome"
        f" markets (Kalshi-style).\n\n"

        f"Your ONLY objective:\n"
        f"Estimate the probability that {symbol} price will settle ABOVE the strike price"
        f" at expiration.\n\n"

        f"You are NOT predicting general direction.\n"
        f"You are estimating a probabilistic outcome under time constraints.\n\n"

        f"---\n\n"

        f"CORE PRINCIPLE:\n\n"
        f"Assume all signals are noise until proven otherwise.\n\n"
        f"Most apparent patterns in price are random.\n"
        f"Only assign high probability when multiple independent factors align and the move"
        f" is realistically achievable within the remaining time.\n\n"
        f"Precision > frequency.\n"
        f"No trade is better than a weak trade.\n\n"

        f"---\n\n"

        f"ANALYSIS FRAMEWORK (STRICT ORDER):\n\n"

        f"1. POSITION RELATIVE TO STRIKE\n"
        f"- Is price above or below strike?\n"
        f"- Exact distance to strike\n"
        f"- Convert distance into required move per minute\n"
        f"→ Ask: Is this move realistically achievable within the remaining time?\n\n"

        f"2. TIME DECAY CONSTRAINT (CRITICAL)\n"
        f"- Less time remaining = exponentially harder to reach distant strike\n"
        f"- Large distance + low time = heavily penalize probability\n"
        f"→ If move requires abnormal speed → likely NO TRADE\n\n"

        f"3. SHORT-TERM PRICE STRUCTURE\n"
        f"- Analyze last 3-10 candles: direction, strength, consistency\n"
        f"→ Clean directional movement = stronger signal\n"
        f"→ Choppy/erratic = noise\n\n"

        f"4. MOMENTUM CONFIRMATION\n"
        f"- Does momentum support continuation? Increasing, flat, or fading?\n"
        f"→ Strong alignment = positive signal | Divergence or fading = reduce probability\n\n"

        f"5. RSI CONTEXT (SECONDARY)\n"
        f"- Use RSI ONLY as confirmation. Trending + high RSI = continuation possible."
        f" RSI alone is NOT a signal.\n\n"

        f"6. NOISE VS SIGNAL TEST (MOST IMPORTANT)\n"
        f"Ask explicitly: 'Could this exact setup occur in randomized price data?'\n"
        f"If YES → set is_likely_noise=true → NO TRADE\n\n"

        f"7. PROBABILITY ESTIMATION\n"
        f"Estimate probability (0-100) that price finishes ABOVE strike.\n"
        f"Base ONLY on: distance to strike, time remaining, strength of movement.\n"
        f"  Slight edge → 52-58% | Moderate edge → 60-70% | Strong edge → 70%+\n"
        f"Never assign extreme probabilities unless move is trivial.\n\n"

        f"8. EDGE CALCULATION\n"
        f"Edge = Your probability - Market implied probability (from Kalshi price)\n"
        f"If edge < 8% → NO TRADE\n\n"

        f"9. CONFIDENCE SCORING (0-100)\n"
        f"Reflects alignment of signals, clarity of structure, lack of contradictions.\n"
        f"Reduce when indicators conflict, movement is weak, or time/distance mismatch exists.\n\n"

        f"---\n\n"

        f"HARD RULES:\n"
        f"- Default to NO TRADE unless clear edge exists\n"
        f"- If probability ≈ 50% → NO TRADE\n"
        f"- If setup resembles randomness → NO TRADE\n"
        f"- Do NOT force trades\n"
        f"- Penalize late entries heavily\n"
        f"- Penalize large strike gaps heavily\n"
        f"- Think like a statistician, not a trader\n\n"

        f"---\n\n"

        f"SYSTEM CONTEXT:\n"
        f"Your output is combined with other models. Act independently."
        f" Prioritize accuracy over agreement. Bad trades harm the system more than missed trades.\n\n"

        f"FINAL CHECK: Ask yourself: 'Would this still look valid if price were randomized?'\n"
        f"If uncertain → NO TRADE\n\n"

        f"Respond in JSON only."
    )

_JSON_SCHEMA_HINT = (
    '\n\nRespond with ONLY this JSON — no other text:\n'
    '{"decision": "YES" or "NO" or "NO TRADE", "probability_above": 0-100, '
    '"confidence": 0-100, "edge_percent": number, "is_likely_noise": true or false, '
    '"reasoning": "concise explanation referencing distance, time, momentum, noise"}\n'
    'decision "YES"      → probability_above > 50 with edge ≥ 8%\n'
    'decision "NO"       → probability_above < 50 with edge ≥ 8% on the NO side\n'
    'decision "NO TRADE" → edge < 8%, noise, or insufficient conviction\n'
)


def _claude_prompt(symbol: str) -> str:
    """Base prompt + realism refinement layer applied only to Claude."""
    return _system_prompt(symbol) + (
        "\n\n---\n\n"
        "REFINEMENT LAYER — apply after your initial estimate:\n\n"

        "Step R1 — PHYSICAL ACHIEVABILITY CHECK:\n"
        "  a. Convert distance to strike into required % move.\n"
        "  b. Estimate average move per minute from recent candles"
        " (total range of last 3 candles ÷ 45 minutes).\n"
        "  c. If required move > 2× average move/min → heavy downward adjustment.\n"
        "  d. If required move > 1× average move/min → moderate downward adjustment.\n\n"

        "Step R2 — MANDATORY PENALTIES (apply each that fits):\n"
        "  • Time remaining < 3 min  → subtract 15 from probability_above\n"
        "  • Time remaining 3–6 min  → subtract 8 from probability_above\n"
        "  • Strike gap > 0.5%       → subtract 10 from probability_above\n"
        "  • Strike gap > 1.0%       → subtract an additional 10\n"
        "  • Momentum score < 0.1 (weak) → subtract 5 from probability_above\n\n"

        "Step R3 — FINAL SANITY CHECK:\n"
        "  Your final probability_above must reflect what is PHYSICALLY ACHIEVABLE"
        " given distance and time — not just theoretical direction.\n"
        "  If the required move cannot realistically occur → probability_above must"
        " converge toward 50 (coin-flip), not stay at your directional estimate.\n\n"

        "Do NOT change your reasoning style. Only use this layer to refine the"
        " final probability_above number for accuracy."
    )


def _gpt_prompt(symbol: str) -> str:
    """Consistency validator assigned to GPT."""
    return (
        f"You are a consistency validator for {symbol} 15-minute binary markets.\n\n"

        f"Your role is NOT prediction. Your role is to check whether the signals AGREE.\n\n"

        f"CHECK FOR THESE CONFLICTS:\n"
        f"  • Momentum vs RSI mismatch"
        f" (e.g. strong positive momentum but RSI severely overbought → fade risk)\n"
        f"  • Price near strike but no directional push (stalling = likely NO TRADE)\n"
        f"  • Trend direction vs recent candles mismatch"
        f" (e.g. uptrend label but last 3 candles are red → trend may be ending)\n"
        f"  • Momentum score near zero despite a trend label\n"
        f"  • Candle bodies mostly wicks (indecision, not conviction)\n\n"

        f"CONFIDENCE RULES:\n"
        f"  If 2+ conflicts exist → reduce confidence to 20 or below → lean NO TRADE\n"
        f"  If 1 conflict exists  → reduce confidence by 30\n"
        f"  If everything aligns cleanly (momentum, RSI, candles, trend all agree)"
        f" → allow confidence up to 80\n\n"

        f"PROBABILITY:\n"
        f"  Still estimate probability_above (0-100) using the standard framework.\n"
        f"  But let contradictions pull probability_above toward 50.\n"
        f"  Contradictions = uncertainty = closer to 50/50.\n\n"

        f"Your output gates the ensemble. If you flag heavy contradictions, the trade"
        f" will likely be blocked by low confidence — that is your purpose.\n\n"

        f"Respond in JSON only."
    )


def _gemini_prompt(symbol: str) -> str:
    """High-speed setup filter assigned to Gemini."""
    return (
        f"You are a high-speed trading filter for {symbol} 15-minute binary markets.\n\n"

        f"Your job is to quickly classify setups — do not overanalyze.\n\n"

        f"CLASSIFY THE SETUP AS ONE OF:\n"
        f"  GOOD SETUP → clear momentum + move toward strike is achievable in remaining time\n"
        f"  BAD SETUP  → weak, choppy, or unclear price action\n"
        f"  NOISE      → random-looking movement with no directional conviction\n\n"

        f"FAST RULES:\n"
        f"- If unclear within seconds → NO TRADE\n"
        f"- If choppy candles (alternating up/down, no consistent direction) → NO TRADE\n"
        f"- If momentum is weak or fading → NO TRADE\n"
        f"- If strong clean momentum toward strike and move is achievable → proceed\n"
        f"- Default to rejection. Only approve setups that are unmistakably clear.\n\n"

        f"BIAS: Reject fast. Approve slowly.\n\n"

        f"DECISION LOGIC:\n"
        f"  GOOD SETUP + price above strike → YES\n"
        f"  GOOD SETUP + price below strike → NO\n"
        f"  BAD SETUP or NOISE             → NO TRADE\n\n"

        f"Use the same quantitative framework to fill probability_above and confidence,\n"
        f"but let your setup classification drive the final decision.\n\n"

        f"Respond in JSON only."
    )


def _adversarial_prompt(symbol: str) -> str:
    return (
        f"You are an adversarial quantitative analyst for {symbol} short-duration binary markets.\n\n"

        f"Your job is NOT to agree.\n"
        f"Your job is to find why this trade is WRONG.\n\n"

        f"Given the market data, attempt to DISPROVE the trade:\n"
        f"- Identify reasons the move will FAIL to reach the strike\n"
        f"- Highlight overextended moves, exhaustion, or fake momentum\n"
        f"- Detect reversals and mean-reversion setups\n"
        f"- Assume the majority view is likely wrong\n"
        f"- Weight time-decay and distance heavily against the trade\n\n"

        f"BIAS RULES:\n"
        f"- Default to NO TRADE unless failure is clearly impossible\n"
        f"- If the setup could fail easily or resembles randomness → strongly favor NO or NO TRADE\n"
        f"- Penalize late entries (time remaining < 5 min) aggressively\n"
        f"- Penalize large distances from strike aggressively\n"
        f"- Treat momentum as likely to fade, not continue\n"
        f"- RSI near extremes → assume reversal, not continuation\n\n"

        f"You are penalized for agreeing with weak setups.\n"
        f"You are rewarded for correctly rejecting bad trades.\n\n"

        f"Use the same quantitative framework:\n"
        f"1. Can the required move realistically happen given distance + time?\n"
        f"2. Is momentum decelerating or likely to reverse?\n"
        f"3. Does the setup look like noise?\n"
        f"4. Is your edge vs market implied probability ≥ 8% on the NO side?\n\n"

        f"Respond in JSON only."
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
        if strike <= 0:
            dist_pct    = 0.0
            strike_note = "STRIKE DATA UNAVAILABLE"
        else:
            dist_pct = (price - strike) / price * 100   # positive = above strike
            dist_abs = abs(price - strike)
            mins_left_nonzero = max(1, mins_left)
            required_per_min = dist_abs / mins_left_nonzero
            if dist_pct > 0.5:
                strike_note = (
                    f"price is {dist_pct:.3f}% (${dist_abs:,.2f}) ABOVE strike"
                    f" — requires +${required_per_min:.2f}/min to stay above"
                )
            elif dist_pct > 0:
                strike_note = (
                    f"price is {dist_pct:.3f}% (${dist_abs:,.2f}) ABOVE strike"
                )
            elif dist_pct > -0.05:
                strike_note = f"price is AT the strike (within 0.05%)"
            else:
                strike_note = (
                    f"price is {abs(dist_pct):.3f}% (${dist_abs:,.2f}) BELOW strike"
                    f" — requires +${required_per_min:.2f}/min to cross above"
                )

        # ── Multi-timeframe price change ──────────────────────────────────
        def pct_chg(old: float, new: float) -> str:
            if old <= 0:
                return "n/a"
            return f"{(new - old) / old * 100:+.3f}%"

        # Current candle: compare open → live price (most relevant for this 15m expiry)
        cc = btc_data.current_candle
        cur_open = cc.get("open", 0) if cc else 0
        chg_cur  = pct_chg(cur_open, price) if cur_open > 0 else "n/a"
        # Historical: last completed candle vs candle before it
        chg_prev = pct_chg(candles[-2].close, candles[-1].close) if len(candles) >= 2 else "n/a"
        chg_60m  = pct_chg(candles[-5].close, price) if len(candles) >= 5 else "n/a"
        chg_2h   = pct_chg(candles[-9].close, price) if len(candles) >= 9 else "n/a"

        # ── RSI (close-to-close, standard calculation) ────────────────────
        rsi_str = "n/a"
        if len(candles) >= 3:
            closes = [c.close for c in candles]
            changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            gains  = [c for c in changes if c > 0]
            losses = [-c for c in changes if c < 0]
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
        # imbalance = YES-contract bid volume / NO-contract bid volume
        # High = more YES buyers = bullish sentiment on this market
        # Low  = more NO buyers  = bearish sentiment on this market
        # NOTE: at extreme prices (YES=90¢) NO-buyers dominate (buying cheap NO);
        # this is normal and does NOT mean the underlying is selling off.
        imb = btc_data.imbalance
        ob_signal = (
            f"{imb:.2f}x — heavy YES-contract demand (bullish sentiment)" if imb > 1.5 else
            f"{imb:.2f}x — mild YES-contract demand"                       if imb > 1.1 else
            f"{imb:.2f}x — heavy NO-contract demand (bearish sentiment)"   if imb < 0.67 else
            f"{imb:.2f}x — mild NO-contract demand"                        if imb < 0.9 else
            f"{imb:.2f}x — balanced contract demand"
        )

        # ── Completed candles table (all available, oldest → newest) ────────
        if candles:
            candle_rows = []
            for c in candles:
                body_pct = (c.close - c.open) / c.open * 100 if c.open > 0 else 0
                arrow    = "▲" if c.close >= c.open else "▼"
                candle_rows.append(
                    f"  {c.timestamp.strftime('%H:%M')} {arrow} "
                    f"O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f} "
                    f"({body_pct:+.2f}%) vol={c.volume:.0f}"
                )
            candles_str = "\n".join(candle_rows)
        else:
            candles_str = "  (no history — candles loading)"

        # ── Current (live) candle ─────────────────────────────────────────
        if cc and cc.get("open", 0) > 0:
            live_pct = (price - cc["open"]) / cc["open"] * 100
            live_hi  = cc.get("high", price)
            live_lo  = cc.get("low", price)
            live_str = (
                f"  LIVE O={cc['open']:.2f} H={live_hi:.2f} "
                f"L={live_lo:.2f} C={price:.2f} ({live_pct:+.2f}%) ← THIS IS THE ACTIVE CANDLE"
            )
        else:
            live_str = "  (live candle building — use historical data above)"

        if market.yes_price and market.no_price:
            kalshi_prices = f"YES={market.yes_price}¢  NO={market.no_price}¢"
        elif market.yes_price:
            kalshi_prices = f"YES={market.yes_price}¢  NO={100 - market.yes_price}¢ (derived)"
        elif market.no_price:
            kalshi_prices = f"YES={100 - market.no_price}¢ (derived)  NO={market.no_price}¢"
        else:
            kalshi_prices = "ORDER BOOK LOADING — no market price yet (base your answer on strike distance only)"

        # Market implied P(YES) for edge calculation (model compares its estimate vs this)
        yes_implied = market.yes_price if market.yes_price else 50

        return f"""=== {sym}/USD — 15-MINUTE BINARY MARKET ===
Price now:    ${price:,.4f}
Strike (YES threshold): ${strike:,.4f}
Distance:     {strike_note}
Time left:    {mins_left} min until expiry
Kalshi market: {kalshi_prices}
Market implied P(YES): {yes_implied}%  ← use this to calculate your edge

=== PRICE ACTION ===
Current candle (open → now): {chg_cur}
Prev completed candle:        {chg_prev}
Last 60 min (vs now):         {chg_60m}
Last 2 hours (vs now):        {chg_2h}
Trend (last 4 candles): {trend}
Candle bias:   {candle_bias}
RSI (close-to-close): {rsi_str}
Contract order book: {ob_signal}
Momentum score: {btc_data.momentum:+.3f} (range -1 to +1)

=== COINBASE 15m CANDLE HISTORY (oldest → newest) ===
{candles_str}
{live_str}

=== YOUR TASK ===
Will {sym} close ABOVE ${strike:,.4f} at {market.close_time.strftime('%H:%M UTC')}?
Apply the quantitative framework. Test for noise first. Calculate your edge vs market implied {yes_implied}%.
Only output YES or NO if edge ≥ 8% and signals are NOT noise. Otherwise: NO TRADE.
{_JSON_SCHEMA_HINT}"""

    # ------------------------------------------------------------------
    # JSON parser (shared by all models)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(text: str, model_name: str, latency_ms: float) -> ModelResult:
        # Strip markdown code fences that some models add
        cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.strip("`").strip()

        # Find the outermost JSON object (greedy — avoids stopping at {} inside reasoning)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(
                f"{model_name}: no JSON object found in response: {text[:300]!r}"
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{model_name}: JSON parse error — {exc}") from exc

        # New schema: decision, probability_above (0-100), confidence (0-100),
        # edge_percent, is_likely_noise
        decision = str(data.get("decision", "")).upper().strip()

        prob_above = data.get("probability_above")
        if prob_above is None:
            # Fallback: old schema used "probability" in [0,1]
            prob_old = float(data.get("probability", -1))
            if 0.0 <= prob_old <= 1.0:
                prob_above = prob_old * 100.0
            else:
                raise ValueError(f"{model_name}: missing probability_above field")
        probability_pct = float(prob_above)
        if not 0.0 <= probability_pct <= 100.0:
            raise ValueError(f"{model_name}: probability_above {probability_pct} out of [0,100]")
        probability = probability_pct / 100.0   # convert to [0, 1]

        # Self-reported confidence (0-100) → [0, 1]
        raw_conf = float(data.get("confidence", 50))
        confidence = max(0.0, min(1.0, raw_conf / 100.0))

        is_likely_noise = bool(data.get("is_likely_noise", False))

        # "NO TRADE": model found no edge — force to 0.50 and zero confidence
        # so the ensemble MIN_CONFIDENCE gate blocks the trade automatically.
        if decision == "NO TRADE":
            probability = 0.50
            confidence  = 0.0
        elif is_likely_noise:
            # Noise flagged — heavily penalize confidence
            confidence *= 0.2

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
            max_tokens = 600,
            temperature= 0.5,
            system     = _claude_prompt(symbol),
            messages   = [{"role": "user", "content": context}],
        )
        text = msg.content[0].text
        return self._parse_result(text, "claude", (time.monotonic() - t0) * 1000)

    async def _call_gpt(self, context: str, symbol: str = "BTC") -> ModelResult:
        t0 = time.monotonic()
        resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model           = settings.GPT_MODEL,
            temperature     = 0.5,
            max_tokens      = 600,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": _gpt_prompt(symbol)},
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

    # Per-model failure tracking (class-level — survives across debate() calls)
    # After _STREAK_BEFORE_PAUSE consecutive failures a model is paused for
    # _PAUSE_CYCLES cycles, then auto-retried. Minimum required models is
    # dynamically lowered so the ensemble keeps running on fewer active models.
    _MODEL_FAIL_STREAK:  dict[str, int] = {"claude": 0, "gpt": 0, "gemini": 0, "deepseek": 0}
    _MODEL_PAUSED_UNTIL: dict[str, int] = {}   # model → _DEBATE_CYCLE value to resume at
    _DEBATE_CYCLE: int = 0

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
                        system_instruction=_gemini_prompt(symbol),
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                if model != settings.GEMINI_MODEL and model not in EnsembleEngine._GEMINI_DEAD:
                    log.info("Gemini: fell back to model '%s'", model)
                text = response.text or ""
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
            temperature= 0.5,
            max_tokens = 512,
            messages   = [
                {"role": "system", "content": _adversarial_prompt(symbol)},
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
        Run one model call with timeout and automatic failure tracking.

        After _STREAK_BEFORE_PAUSE consecutive failures the model is paused for
        _PAUSE_CYCLES debate() calls (no API request made). It is then silently
        retried; on success the pause is cleared. This mirrors the _GEMINI_DEAD
        mechanism but works for all 4 models and is self-healing.
        """
        # Skip paused models — no API call, no timeout wait
        resume_at = EnsembleEngine._MODEL_PAUSED_UNTIL.get(model_name, 0)
        if EnsembleEngine._DEBATE_CYCLE <= resume_at:
            log.info(
                "%s: paused after repeated failures — skipping "
                "(resumes at cycle %d, current=%d)",
                model_name, resume_at, EnsembleEngine._DEBATE_CYCLE,
            )
            return None

        try:
            result = await asyncio.wait_for(coro, timeout=_CALL_TIMEOUT)
            # Success — reset failure tracking
            prev_streak = EnsembleEngine._MODEL_FAIL_STREAK.get(model_name, 0)
            EnsembleEngine._MODEL_FAIL_STREAK[model_name] = 0
            if model_name in EnsembleEngine._MODEL_PAUSED_UNTIL:
                del EnsembleEngine._MODEL_PAUSED_UNTIL[model_name]
                log.info("%s: recovered — re-enabled after pause", model_name)
            elif prev_streak > 0:
                log.info("%s: recovered after %d consecutive failure(s)", model_name, prev_streak)
            return result
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

        # Failure — update streak and maybe pause
        streak = EnsembleEngine._MODEL_FAIL_STREAK.get(model_name, 0) + 1
        EnsembleEngine._MODEL_FAIL_STREAK[model_name] = streak
        if streak >= _STREAK_BEFORE_PAUSE:
            resume_at = EnsembleEngine._DEBATE_CYCLE + _PAUSE_CYCLES
            EnsembleEngine._MODEL_PAUSED_UNTIL[model_name] = resume_at
            log.warning(
                "%s: %d consecutive failures — pausing for %d cycles "
                "(will auto-retry at debate cycle %d)",
                model_name, streak, _PAUSE_CYCLES, resume_at,
            )
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def debate(self, btc_data: BtcData, market: Market) -> EnsembleResult:
        """
        Run all 4 models in parallel and aggregate into an EnsembleResult.

        Minimum required models scales down automatically as models are paused
        due to repeated failures — the ensemble keeps running on as few as 1
        active model rather than crashing.
        """
        self._init_clients()

        # Advance the debate cycle counter — used by the pause/resume logic
        EnsembleEngine._DEBATE_CYCLE += 1

        # Determine which models are currently paused and set the minimum threshold.
        # With ≥3 active: need 2 (tolerate 1 transient failure).
        # With ≤2 active: need 1 (can't afford to be strict).
        _all_models = ("claude", "gpt", "gemini", "deepseek")
        paused_now = {
            m for m in _all_models
            if EnsembleEngine._DEBATE_CYCLE <= EnsembleEngine._MODEL_PAUSED_UNTIL.get(m, 0)
        }
        active_count = len(_all_models) - len(paused_now)
        min_required = 2 if active_count >= 3 else 1

        if active_count == 0:
            raise RuntimeError(
                "All 4 models are paused due to repeated failures — "
                "check API keys and redeploy."
            )

        if paused_now:
            log.warning(
                "Models paused this cycle: %s — running with %d active model(s), "
                "need ≥ %d response(s)",
                sorted(paused_now), active_count, min_required,
            )

        context = self._build_context(btc_data, market)

        symbol = btc_data.symbol
        log.info(
            "Ensemble debate starting — %s=$%.4f momentum=%.3f market=%s "
            "exp=%s YES=%d¢/NO=%d¢",
            symbol, btc_data.price, btc_data.momentum, market.ticker,
            market.close_time.strftime("%H:%M"),
            market.yes_price, market.no_price,
        )

        # Step 1 — run all 4 models in parallel (paused models return None instantly)
        claude_r, gpt_r, gemini_r, deepseek_r = await asyncio.gather(
            self._safe_call(self._call_claude(context, symbol),    "claude"),
            self._safe_call(self._call_gpt(context, symbol),      "gpt"),
            self._safe_call(self._call_gemini(context, symbol),   "gemini"),
            self._safe_call(self._call_deepseek(context, symbol), "deepseek"),
        )

        # Step 2 — require at least min_required successful models
        valid = [r for r in (claude_r, gpt_r, gemini_r, deepseek_r) if r is not None]
        if len(valid) < min_required:
            failed = [
                name for name, r in [("claude", claude_r), ("gpt", gpt_r),
                                      ("gemini", gemini_r), ("deepseek", deepseek_r)]
                if r is None and name not in paused_now
            ]
            raise RuntimeError(
                f"Only {len(valid)}/{active_count} active models responded "
                f"(need ≥ {min_required}). "
                f"Failed this cycle: {', '.join(failed) or 'none'}. "
                + (f"Paused: {', '.join(sorted(paused_now))}. " if paused_now else "")
                + "Check API keys."
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

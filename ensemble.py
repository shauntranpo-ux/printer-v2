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
_PAUSE_CYCLES        = 5      # debate() calls to skip before auto-retrying (transient errors)
_PAUSE_CYCLES_HARD   = 70     # pause cycles for permanent errors (billing/auth/bad key)
                               # ~70 debate calls ÷ 7 assets = ~10 main cycles ≈ ~10 min

# Error substrings that signal a permanent/billing failure — long pause, no fast-retry
_PERMANENT_ERROR_HINTS = (
    "credit balance",           # Anthropic billing
    "insufficient_quota",       # OpenAI billing
    "quota",                    # generic quota
    "billing",                  # generic billing
    "invalid api key",          # bad key
    "invalid_api_key",
    "authentication",           # auth failure
    "permission_denied",
    "access denied",
)


# ---------------------------------------------------------------------------
# System prompts  (asset symbol is injected at call time)
# ---------------------------------------------------------------------------

def _system_prompt(symbol: str) -> str:
    return (
        f"You are a quantitative probability analyst for {symbol} short-duration binary markets.\n\n"

        f"Your ONLY objective:\n"
        f"Estimate the probability (0-100) that {symbol} price will settle ABOVE the strike"
        f" at expiration, then output whichever direction is more likely.\n\n"

        f"You MUST always output YES or NO — never refuse to give a direction.\n"
        f"Low confidence is fine. The risk system filters weak trades — your job is only"
        f" to give the most accurate probability you can.\n\n"

        f"---\n\n"

        f"ANALYSIS FRAMEWORK:\n\n"

        f"1. POSITION RELATIVE TO STRIKE\n"
        f"- Is price above or below strike? Exact distance.\n"
        f"- Convert distance to required move per minute.\n"
        f"- If move requires abnormal speed → pull probability toward 50.\n\n"

        f"2. TIME DECAY\n"
        f"- Less time = harder to reach distant strikes.\n"
        f"- Large gap + low time → probability closer to 50.\n\n"

        f"3. PRICE STRUCTURE (last 3-10 candles)\n"
        f"- Clean directional movement → adjust probability toward that direction.\n"
        f"- Choppy/erratic → probability closer to 50.\n\n"

        f"4. MOMENTUM\n"
        f"- Strong momentum aligned with direction → increase probability.\n"
        f"- Fading or diverging momentum → pull toward 50.\n\n"

        f"5. RSI (secondary only)\n"
        f"- Use as confirmation. Do not base the decision on RSI alone.\n\n"

        f"6. PROBABILITY OUTPUT\n"
        f"- probability_above > 50 → output YES\n"
        f"- probability_above < 50 → output NO\n"
        f"- probability_above = 50 → pick the direction with any slight edge\n"
        f"- Set confidence low (20-40) when signals are weak or mixed.\n"
        f"- Set confidence high (70-90) only when signals are clear and aligned.\n\n"

        f"---\n\n"

        f"RULES:\n"
        f"- Always give YES or NO. Never skip.\n"
        f"- Accuracy over conviction: a 52% probability is still a valid YES.\n"
        f"- Penalize late entries and large strike gaps by pulling probability toward 50.\n"
        f"- Your output is weighted with 3 other models — be independent and honest.\n\n"

        f"Respond in JSON only."
    )

_JSON_SCHEMA_HINT = (
    '\n\nRespond with ONLY this JSON — no other text:\n'
    '{"decision": "YES" or "NO", "probability_above": 0-100, '
    '"confidence": 0-100, '
    '"reasoning": "1-2 sentences: distance to strike, time left, momentum direction"}\n'
    'decision "YES" → probability_above > 50 (price more likely to finish above strike)\n'
    'decision "NO"  → probability_above ≤ 50 (price more likely to finish below strike)\n'
    'Always give a direction. Set confidence low if signals are weak.\n'
)


def _claude_prompt(symbol: str) -> str:
    """Base prompt + physical achievability refinement layer for Claude."""
    return _system_prompt(symbol) + (
        "\n\n---\n\n"
        "REFINEMENT LAYER — apply after your initial estimate:\n\n"

        "Step R1 — PHYSICAL ACHIEVABILITY:\n"
        "  a. Required % move = distance to strike ÷ current price × 100.\n"
        "  b. Average move/min = total range of last 3 candles ÷ 45 minutes.\n"
        "  c. Required > 2× average → pull probability 15 points toward 50.\n"
        "  d. Required > 1× average → pull probability 8 points toward 50.\n\n"

        "Step R2 — TIME PENALTY (entry window is 0.5-8 min in, so 7-14.5 min remain):\n"
        "  • Time remaining < 8 min  → pull 10 points toward 50\n"
        "  • Time remaining < 10 min → pull 5 points toward 50\n"
        "  • Strike gap > 0.5%       → pull 10 points toward 50\n"
        "  • Strike gap > 1.0%       → pull an additional 10 toward 50\n\n"

        "After applying penalties, output YES if probability_above > 50, NO if ≤ 50.\n"
        "Do NOT refuse to give a direction."
    )


def _gpt_prompt(symbol: str) -> str:
    """Consistency validator + physical achievability check."""
    return (
        f"You are a signal consistency validator for {symbol} 15-minute binary markets.\n\n"

        f"Your job: check whether price structure, momentum, RSI, and candles AGREE"
        f" on direction, then apply physical achievability penalties before outputting.\n\n"

        f"STEP 1 — CONFLICT CHECKS:\n"
        f"  • Momentum vs RSI mismatch (strong momentum + overbought/oversold RSI → fade risk)\n"
        f"  • Trend label vs recent candles mismatch (uptrend but last 3 candles red)\n"
        f"  • Momentum near zero despite trend label\n"
        f"  • Mostly wick candles (indecision — body < 30% of candle range)\n"
        f"  • Volume spike on opposing move (high-vol candle in wrong direction)\n\n"

        f"CONFIDENCE FROM CONFLICT COUNT:\n"
        f"  2+ conflicts  → confidence 15-25 (signals disagree)\n"
        f"  1 conflict    → confidence 30-50 (mild doubt)\n"
        f"  All aligned   → confidence 60-80 (clear setup)\n\n"

        f"STEP 2 — PHYSICAL ACHIEVABILITY (apply after initial estimate):\n"
        f"  R1 — Required move check:\n"
        f"    a. Required move = distance to strike in dollars.\n"
        f"    b. Avg move/min = total range of last 3 candles / 45 minutes.\n"
        f"    c. Required > 2× avg/min → pull probability 15 points toward 50.\n"
        f"    d. Required > 1× avg/min → pull probability 8 points toward 50.\n"
        f"    Note: if price is ABOVE strike, holding flat wins — penalize only downside risk.\n\n"
        f"  R2 — Time penalty (entry window is 0.5-8 min in, so 7-14.5 min remain):\n"
        f"    • Time remaining < 8 min  → pull 10 points toward 50\n"
        f"    • Time remaining < 10 min → pull 5 points toward 50\n"
        f"    • Strike gap > 0.5%       → pull 10 points toward 50\n"
        f"    • Strike gap > 1.0%       → pull an additional 10 toward 50\n\n"

        f"PROBABILITY OUTPUT:\n"
        f"  After all adjustments, output probability_above (0-100).\n"
        f"  Always output YES (if > 50) or NO (if ≤ 50). Never skip.\n\n"

        f"Respond in JSON only."
    )


def _gemini_prompt(symbol: str) -> str:
    """Full quantitative setup classifier with explicit framework."""
    return (
        f"You are a quantitative setup classifier for {symbol} 15-minute binary markets.\n\n"

        f"Your ONLY objective: estimate the probability (0-100) that {symbol} price will"
        f" settle ABOVE the strike at expiration, classify the setup quality, then output"
        f" YES or NO. Always give a direction — never refuse.\n\n"

        f"ANALYSIS FRAMEWORK (evaluate in order):\n\n"

        f"1. POSITION RELATIVE TO STRIKE\n"
        f"   - Is price above or below strike? Calculate exact distance and direction.\n"
        f"   - If price is ABOVE strike: holding flat wins. Only downside pressure matters.\n"
        f"   - If price is BELOW strike: needs positive movement. Calculate required move/min.\n"
        f"   - Large gap + short time → pull probability toward 50.\n\n"

        f"2. TIME DECAY (entry window is 0.5-8 min in, so you always see 7-14.5 min remaining)\n"
        f"   - Time remaining < 8 min  → moderate decay: pull probability 8 points toward 50\n"
        f"   - Time remaining < 10 min → mild decay: pull probability 4 points toward 50\n"
        f"   - Strike gap > 0.5%       → pull 8 points toward 50\n"
        f"   - Strike gap > 1.0%       → pull additional 8 points toward 50\n\n"

        f"3. MOMENTUM\n"
        f"   - Momentum score range: -1.0 (strong bearish) to +1.0 (strong bullish)\n"
        f"   - |momentum| > 0.5 and aligned with direction → increase probability 5-10 pts\n"
        f"   - |momentum| < 0.2 → weak signal, pull toward 50\n"
        f"   - Momentum diverging from price trend → fade risk, pull toward 50\n\n"

        f"4. CANDLE STRUCTURE\n"
        f"   - 3+ bullish candles in last 4 → upward bias\n"
        f"   - Mostly wick candles (body < 30% of range) → indecision, lower confidence\n"
        f"   - Alternating bull/bear candles → choppy, pull toward 50\n"
        f"   - Volume spike: candle marked +20%+ vs avg = conviction; -20%+ = weak move\n\n"

        f"5. RSI (confirmation only — do not base decision on RSI alone)\n"
        f"   - RSI > 70 (overbought) with strong upward momentum → fade risk\n"
        f"   - RSI < 30 (oversold) with downward momentum → bounce risk\n"
        f"   - Otherwise use as mild confirmation of trend direction\n\n"

        f"6. ORDER BOOK IMBALANCE\n"
        f"   - Imbalance > 1.5x (heavy YES demand) → mild bullish signal (+3 pts)\n"
        f"   - Imbalance < 0.67x (heavy NO demand) → mild bearish signal (-3 pts)\n"
        f"   - Near 1.0x → balanced, no adjustment\n\n"

        f"SETUP CLASSIFICATION AND CONFIDENCE:\n"
        f"  CLEAR SETUP: all signals aligned, reachable strike, momentum confirming\n"
        f"    → confidence 65-80, probability skewed 60-75% in that direction\n"
        f"  MODERATE SETUP: most signals aligned, minor conflicts\n"
        f"    → confidence 40-60, probability 54-62% in favored direction\n"
        f"  WEAK SETUP: choppy, mixed, or conflicting signals\n"
        f"    → confidence 20-35, probability 50-54% (still pick a side)\n\n"

        f"RULES:\n"
        f"  - probability_above > 50 → output YES\n"
        f"  - probability_above ≤ 50 → output NO\n"
        f"  - Always give one direction. Never refuse.\n"
        f"  - A 52% probability is a valid YES. Accuracy over conviction.\n\n"

        f"Respond in JSON only."
    )


def _adversarial_prompt(symbol: str) -> str:
    """Adversarial bear — structured failure analysis with quantitative penalty rules."""
    return (
        f"You are an adversarial analyst for {symbol} 15-minute binary markets.\n\n"

        f"Your bias: assume the obvious trade will FAIL. Quantify the failure probability.\n\n"

        f"STEP 1 — APPLY THESE QUANTITATIVE PENALTIES to the naive probability:\n\n"

        f"Physical achievability:\n"
        f"  • Required move > 2× avg candle range/min → pull probability 15 pts toward 50\n"
        f"  • Required move > 1× avg candle range/min → pull probability 8 pts toward 50\n"
        f"  • Note: if price is ABOVE strike, required move is zero (holding flat wins);\n"
        f"    instead penalize only for significant downside momentum\n\n"

        f"Time decay (entry window is 0.5-8 min in, so 7-14.5 min remain at evaluation):\n"
        f"  • Time remaining < 8 min  → pull 10 pts toward 50 (late entry, limited recovery time)\n"
        f"  • Time remaining < 10 min → pull 5 pts toward 50\n\n"

        f"Momentum fade risk:\n"
        f"  • RSI > 70 AND momentum score > +0.5 → overbought fade risk: pull 10 pts toward 50\n"
        f"  • RSI < 30 AND momentum score < -0.5 → oversold bounce risk: pull 10 pts toward 50\n"
        f"  • Momentum score near zero (|momentum| < 0.15) → no directional conviction: pull 8 pts toward 50\n\n"

        f"Noise detection:\n"
        f"  • 2+ alternating bull/bear candles in last 4 → choppy: pull 6 pts toward 50\n"
        f"  • Mostly wick candles (body < 30% of range) in last 3 → indecision: pull 6 pts toward 50\n"
        f"  • Volume -30% below average on directional move → weak conviction: pull 5 pts toward 50\n\n"

        f"STEP 2 — LOOK FOR FAILURE REASONS:\n"
        f"  • Is momentum likely to fade or reverse before expiry?\n"
        f"  • Is the strike too far given average candle velocity?\n"
        f"  • Are candles showing exhaustion (shrinking bodies, rising wicks)?\n"
        f"  • Does order book imbalance contradict the price trend?\n\n"

        f"OUTPUT:\n"
        f"  After applying all penalties, output the direction that is actually more likely.\n"
        f"  If price will fail to cross the strike → NO (or YES if already safely above).\n"
        f"  Set probability_above to your skeptical, penalty-adjusted estimate.\n"
        f"  Always output YES or NO — never skip.\n\n"

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
        secs_left     = max(0, (market.close_time - now_utc).total_seconds())
        mins_left     = int(secs_left // 60)
        secs_rem      = int(secs_left % 60)
        time_left_str = f"{mins_left}m {secs_rem}s"
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
                    f" — holding flat wins; must not drop more than ${required_per_min:.2f}/min"
                )
            elif dist_pct > 0:
                strike_note = (
                    f"price is {dist_pct:.3f}% (${dist_abs:,.2f}) ABOVE strike"
                    f" — holding flat is sufficient to win"
                )
            elif dist_pct > -0.05:
                strike_note = f"price is AT the strike (within 0.05%) — coin flip on direction"
            else:
                strike_note = (
                    f"price is {abs(dist_pct):.3f}% (${dist_abs:,.2f}) BELOW strike"
                    f" — needs to gain ${required_per_min:.2f}/min to cross above by expiry"
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

        # ── RSI (Wilder smoothing, up to 14 periods) ──────────────────────
        rsi_str = "n/a"
        if len(candles) >= 3:
            closes  = [c.close for c in candles]
            changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            period  = min(14, len(changes))
            avg_g   = sum(max(ch, 0)  for ch in changes[:period]) / period
            avg_l   = sum(max(-ch, 0) for ch in changes[:period]) / period
            for ch in changes[period:]:
                avg_g = (avg_g * (period - 1) + max(ch, 0))  / period
                avg_l = (avg_l * (period - 1) + max(-ch, 0)) / period
            rsi       = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 100.0
            rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "neutral"
            rsi_str   = f"{rsi:.0f} ({rsi_label})"

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
            avg_vol = sum(c.volume for c in candles) / len(candles)
            candle_rows = []
            for c in candles:
                body_pct    = (c.close - c.open) / c.open * 100 if c.open > 0 else 0
                arrow       = "▲" if c.close >= c.open else "▼"
                vol_diff_pct = (c.volume - avg_vol) / avg_vol * 100 if avg_vol > 0 else 0
                if vol_diff_pct > 20:
                    vol_note = f"+{vol_diff_pct:.0f}% vs avg"
                elif vol_diff_pct < -20:
                    vol_note = f"{vol_diff_pct:.0f}% vs avg"
                else:
                    vol_note = "avg"
                candle_rows.append(
                    f"  {c.timestamp.strftime('%H:%M')} {arrow} "
                    f"O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f} "
                    f"({body_pct:+.2f}%) vol={c.volume:.0f} [{vol_note}]"
                )
            candles_str = "\n".join(candle_rows)
        else:
            avg_vol     = 0.0
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
        avg_vol_str = f"{avg_vol:.0f}" if avg_vol > 0 else "n/a"

        return f"""=== {sym}/USD — 15-MINUTE BINARY MARKET ===
Price now:    ${price:,.4f}
Strike (YES threshold): ${strike:,.4f}
Distance:     {strike_note}
Time left:    {time_left_str} until expiry
Kalshi market: {kalshi_prices}
Market implied P(YES): {yes_implied}%  ← use this to calculate your edge

=== PRICE ACTION ===
Current candle (open → now): {chg_cur}
Prev completed candle:        {chg_prev}
Last 60 min (vs now):         {chg_60m}
Last 2 hours (vs now):        {chg_2h}
Trend (last 4 candles): {trend}
Candle bias:   {candle_bias}
RSI (Wilder-14): {rsi_str}
Contract order book: {ob_signal}
Momentum score: {btc_data.momentum:+.3f} (range -1 to +1)
Avg candle volume: {avg_vol_str}  (per-candle deviation shown in history below)

=== COINBASE 15m CANDLE HISTORY (oldest → newest) ===
{candles_str}
{live_str}

=== YOUR TASK ===
Will {sym} close ABOVE ${strike:,.4f} at {market.close_time.strftime('%H:%M UTC')}?
Entry window context: you see markets 0.5-8 min into the 15-min window, so time
remaining is always 7-14.5 min. Calibrate all time-decay penalties accordingly.
Apply the quantitative framework. Calculate your edge vs market implied {yes_implied}%.
Always output YES or NO — the EV gate will handle filtering weak signals.
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

        prob_above = data.get("probability_above")
        if prob_above is None:
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

        # Direction always derived from probability — models always give YES or NO.
        # If model sends "NO TRADE" (old schema or hallucination), derive from probability.
        decision = str(data.get("decision", "")).upper().strip()
        if decision == "YES":
            direction = "YES"
        elif decision == "NO":
            direction = "NO"
        else:
            # Fallback: derive from probability
            direction = "YES" if probability > 0.5 else "NO"

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
                        response_mime_type="application/json",
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
            model           = settings.DEEPSEEK_MODEL,
            temperature     = 0.5,
            max_tokens      = 512,
            response_format = {"type": "json_object"},
            messages        = [
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
        self, fn: Any, model_name: str
    ) -> ModelResult | None:
        """
        Run one model call with timeout and automatic failure tracking.

        Accepts a zero-argument callable (lambda) rather than a pre-created
        coroutine so that paused models never create an unawaited coroutine
        (which triggers RuntimeWarning). The coroutine is created only when
        the model is not paused.

        After _STREAK_BEFORE_PAUSE consecutive failures the model is paused for
        _PAUSE_CYCLES debate() calls (no API request made). It is then silently
        retried; on success the pause is cleared. This mirrors the _GEMINI_DEAD
        mechanism but works for all 4 models and is self-healing.
        """
        # Skip paused models — no API call, no coroutine created, no timeout wait
        resume_at = EnsembleEngine._MODEL_PAUSED_UNTIL.get(model_name, 0)
        if EnsembleEngine._DEBATE_CYCLE < resume_at:
            log.info(
                "%s: paused after repeated failures — skipping "
                "(resumes at cycle %d, current=%d)",
                model_name, resume_at, EnsembleEngine._DEBATE_CYCLE,
            )
            return None

        permanent = False
        try:
            result = await asyncio.wait_for(fn(), timeout=_CALL_TIMEOUT)
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
            exc_str = str(exc).lower()
            permanent = any(hint in exc_str for hint in _PERMANENT_ERROR_HINTS)
            log.warning(
                "%s failed (%s: %s) — excluded from consensus",
                model_name, type(exc).__name__, exc,
            )

        # Failure — update streak and maybe pause
        streak = EnsembleEngine._MODEL_FAIL_STREAK.get(model_name, 0) + 1
        EnsembleEngine._MODEL_FAIL_STREAK[model_name] = streak
        if streak >= _STREAK_BEFORE_PAUSE:
            pause_len = _PAUSE_CYCLES_HARD if permanent else _PAUSE_CYCLES
            resume_at = EnsembleEngine._DEBATE_CYCLE + pause_len
            EnsembleEngine._MODEL_PAUSED_UNTIL[model_name] = resume_at
            log.warning(
                "%s: %d consecutive failures — pausing for %d cycles%s "
                "(will auto-retry at debate cycle %d)",
                model_name, streak, pause_len,
                " [PERMANENT ERROR — long pause]" if permanent else "",
                resume_at,
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
            if EnsembleEngine._DEBATE_CYCLE < EnsembleEngine._MODEL_PAUSED_UNTIL.get(m, 0)
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
        # Lambdas defer coroutine creation until inside _safe_call so paused models
        # never create an unawaited coroutine (avoids RuntimeWarning).
        claude_r, gpt_r, gemini_r, deepseek_r = await asyncio.gather(
            self._safe_call(lambda: self._call_claude(context, symbol),    "claude"),
            self._safe_call(lambda: self._call_gpt(context, symbol),      "gpt"),
            self._safe_call(lambda: self._call_gemini(context, symbol),   "gemini"),
            self._safe_call(lambda: self._call_deepseek(context, symbol), "deepseek"),
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

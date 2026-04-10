"""
ensemble.py — 4-model AI ensemble engine

Models:
  1. Trend          — EMA crossover + ADX
  2. MeanReversion  — Bollinger Bands + RSI
  3. Momentum       — MACD + Rate of Change
  4. VolatilityRegime — ATR regime + Z-score
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from coinbase_feed import Candle

log = logging.getLogger(__name__)

MIN_CANDLES = 50    # minimum history before any model produces a signal


# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

class Direction:
    YES  = "yes"
    NO   = "no"
    FLAT = "flat"


@dataclass
class ModelSignal:
    name: str
    prob: float         # P(price up) in [0, 1]
    weight: float
    valid: bool = True  # False if not enough data


@dataclass
class EnsembleSignal:
    direction: str              # "yes" | "no" | "flat"
    confidence: float           # distance from 0.5, scaled to [0,1]
    raw_prob: float             # weighted average P(up)
    models: list[ModelSignal]
    weights: dict[str, float]   # current model weights


# ---------------------------------------------------------------------------
# Rolling accuracy tracker (for dynamic weight adjustment)
# ---------------------------------------------------------------------------

class AccuracyTracker:
    """Tracks rolling binary accuracy for one model."""

    def __init__(self, window: int = 50):
        self._window = window
        self._hits: deque[int] = deque(maxlen=window)

    def record(self, predicted_up: bool, actual_up: bool) -> None:
        self._hits.append(int(predicted_up == actual_up))

    @property
    def accuracy(self) -> float:
        if len(self._hits) < 5:
            return 0.5  # neutral until enough data
        return float(np.mean(self._hits))

    @property
    def n(self) -> int:
        return len(self._hits)


# ---------------------------------------------------------------------------
# Indicator helpers (pure numpy, no external TA lib dependency at runtime)
# ---------------------------------------------------------------------------

def _closes(candles: list[Candle]) -> np.ndarray:
    return np.array([c.close for c in candles], dtype=float)


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(arr)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(arr: np.ndarray, period: int = 14) -> float:
    if len(arr) < period + 1:
        return 50.0
    deltas = np.diff(arr[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, prev_c = candles[i].high, candles[i].low, candles[i - 1].close
        if h == 0 and l == 0:
            # Candles built from ticker feed don't have H/L; fallback to close diff
            trs.append(abs(candles[i].close - prev_c))
        else:
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    return float(np.mean(trs[-period:]))


def _bollinger(arr: np.ndarray, period: int = 20, n_std: float = 2.0):
    """Returns (upper, middle, lower)."""
    if len(arr) < period:
        m = arr[-1] if len(arr) > 0 else 0.0
        return m, m, m
    window = arr[-period:]
    mid = window.mean()
    std = window.std()
    return mid + n_std * std, mid, mid - n_std * std


def _macd(arr: np.ndarray, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line) at the last bar."""
    if len(arr) < slow + signal:
        return 0.0, 0.0
    ema_fast = _ema(arr, fast)
    ema_slow = _ema(arr, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line, signal)
    return float(macd_line[-1]), float(sig_line[-1])


def _adx(candles: list[Candle], period: int = 14) -> float:
    """Simplified ADX using close prices as a proxy when H/L unavailable."""
    closes = _closes(candles)
    if len(closes) < period * 2:
        return 0.0
    # Use absolute EMA slope as ADX proxy
    ema_short = _ema(closes, period)
    slope = abs(ema_short[-1] - ema_short[-period]) / (ema_short[-period] + 1e-9)
    return min(float(slope * 100 * period), 100.0)


# ---------------------------------------------------------------------------
# Individual models
# ---------------------------------------------------------------------------

class TrendModel:
    """EMA crossover (9/21) gated by ADX > 20."""

    name = "trend"

    def predict(self, candles: list[Candle]) -> float:
        closes = _closes(candles)
        if len(closes) < 30:
            return 0.5
        ema9  = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        adx   = _adx(candles, 14)

        cross = ema9[-1] - ema21[-1]
        prev_cross = ema9[-2] - ema21[-2]

        if adx < 15:
            return 0.5  # no trend, stay neutral

        # Crossover in last bar
        if cross > 0 and prev_cross <= 0:
            return 0.72
        if cross < 0 and prev_cross >= 0:
            return 0.28

        # Continuation
        if cross > 0:
            strength = min(abs(cross) / (closes[-1] * 0.001 + 1e-9), 1.0)
            return 0.55 + 0.15 * strength
        else:
            strength = min(abs(cross) / (closes[-1] * 0.001 + 1e-9), 1.0)
            return 0.45 - 0.15 * strength


class MeanReversionModel:
    """Bollinger Band touch + RSI extreme."""

    name = "mean_rev"

    def predict(self, candles: list[Candle]) -> float:
        closes = _closes(candles)
        if len(closes) < 25:
            return 0.5
        upper, mid, lower = _bollinger(closes, 20, 2.0)
        rsi = _rsi(closes, 14)
        price = closes[-1]

        # Oversold: price below lower band, RSI < 35 → expect bounce up
        if price < lower and rsi < 35:
            pct_below = (lower - price) / (lower + 1e-9)
            return min(0.65 + pct_below * 0.5, 0.85)

        # Overbought: price above upper band, RSI > 65 → expect pullback
        if price > upper and rsi > 65:
            pct_above = (price - upper) / (upper + 1e-9)
            return max(0.35 - pct_above * 0.5, 0.15)

        # Near middle band — mild mean pull
        if price > mid:
            return 0.47
        return 0.53


class MomentumModel:
    """MACD crossover + Rate of Change."""

    name = "momentum"

    def predict(self, candles: list[Candle]) -> float:
        closes = _closes(candles)
        if len(closes) < 40:
            return 0.5
        macd, sig = _macd(closes)
        roc = (closes[-1] - closes[-10]) / (closes[-10] + 1e-9)   # 10-bar ROC

        # MACD above signal + positive ROC = bullish momentum
        macd_bull = macd > sig
        roc_bull  = roc > 0

        if macd_bull and roc_bull:
            strength = min(abs(roc) * 50, 1.0)
            return 0.60 + 0.15 * strength

        if not macd_bull and not roc_bull:
            strength = min(abs(roc) * 50, 1.0)
            return 0.40 - 0.15 * strength

        # Mixed signals
        if macd_bull:
            return 0.54
        return 0.46


class VolatilityRegimeModel:
    """ATR regime detection + Z-score of price."""

    name = "vol"

    def predict(self, candles: list[Candle]) -> float:
        closes = _closes(candles)
        if len(closes) < 30:
            return 0.5

        atr = _atr(candles, 14)
        atr_pct = atr / (closes[-1] + 1e-9)

        # Z-score of price relative to 20-bar window
        window = closes[-20:]
        z = (closes[-1] - window.mean()) / (window.std() + 1e-9)

        # High volatility regime: trust mean reversion (z-score)
        if atr_pct > 0.015:
            if z < -1.5:
                return 0.65     # oversold in high vol → snap back
            if z > 1.5:
                return 0.35     # overbought in high vol → fade
            return 0.5

        # Low volatility regime: slight trend bias
        if z > 0.5:
            return 0.54
        if z < -0.5:
            return 0.46
        return 0.5


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------

class EnsembleEngine:
    def __init__(
        self,
        weights: dict[str, float],
        confidence_min: float = 0.60,
        weight_update_every: int = 20,   # recalibrate weights every N predictions
    ):
        self._models = {
            "trend":    TrendModel(),
            "mean_rev": MeanReversionModel(),
            "momentum": MomentumModel(),
            "vol":      VolatilityRegimeModel(),
        }
        self._weights = dict(weights)
        self._confidence_min = confidence_min
        self._weight_update_every = weight_update_every
        self._trackers = {k: AccuracyTracker(window=50) for k in self._models}
        self._prediction_count = 0
        # Ring buffer of past prices to score previous predictions
        self._pending: deque[tuple[dict[str, float], float]] = deque(maxlen=10)

    def predict(self, candles: list[Candle]) -> EnsembleSignal:
        if len(candles) < MIN_CANDLES:
            log.debug("Not enough candles (%d < %d)", len(candles), MIN_CANDLES)
            return EnsembleSignal(
                direction=Direction.FLAT,
                confidence=0.0,
                raw_prob=0.5,
                models=[],
                weights=dict(self._weights),
            )

        # Score previous prediction if price has moved
        self._score_pending(candles[-1].close)

        model_signals: list[ModelSignal] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for key, model in self._models.items():
            prob = model.predict(candles)
            w = self._weights[key]
            model_signals.append(ModelSignal(name=key, prob=prob, weight=w))
            weighted_sum += prob * w
            weight_total += w

        raw_prob = weighted_sum / (weight_total + 1e-9)

        # Convert to direction + confidence
        # confidence = how far raw_prob is from 0.5, normalized to [0,1]
        confidence = abs(raw_prob - 0.5) * 2.0

        if confidence < (self._confidence_min - 0.5) * 2.0 or confidence < 0.05:
            direction = Direction.FLAT
        elif raw_prob > 0.5:
            direction = Direction.YES
        else:
            direction = Direction.NO

        signal = EnsembleSignal(
            direction=direction,
            confidence=confidence,
            raw_prob=raw_prob,
            models=model_signals,
            weights=dict(self._weights),
        )

        # Queue for future scoring
        self._pending.append(({k: s.prob for k, s in zip(self._models, model_signals)}, candles[-1].close))
        self._prediction_count += 1

        if self._prediction_count % self._weight_update_every == 0:
            self._recalibrate_weights()

        log.info(
            "Ensemble → %s (conf=%.2f, prob=%.3f) | trend=%.2f mr=%.2f mom=%.2f vol=%.2f",
            direction, confidence, raw_prob,
            model_signals[0].prob, model_signals[1].prob,
            model_signals[2].prob, model_signals[3].prob,
        )
        return signal

    def record_outcome(self, entry_price: float, exit_price: float) -> None:
        """Call after a trade closes to update model accuracy."""
        actual_up = exit_price > entry_price
        if self._pending:
            probs, _ = self._pending[-1]
            for key, prob in probs.items():
                self._trackers[key].record(prob > 0.5, actual_up)

    def _score_pending(self, current_price: float) -> None:
        """Score predictions from ~1 candle ago if available."""
        if not self._pending:
            return
        probs, past_price = self._pending[0]
        if current_price == past_price:
            return
        actual_up = current_price > past_price
        for key, prob in probs.items():
            self._trackers[key].record(prob > 0.5, actual_up)
        # Only pop if we've had movement (avoid scoring same bar twice)
        if len(self._pending) > 1:
            self._pending.popleft()

    def _recalibrate_weights(self) -> None:
        """Adjust weights proportional to each model's rolling accuracy."""
        accs = {k: self._trackers[k].accuracy for k in self._models}
        total = sum(accs.values())
        if total == 0:
            return
        new_weights = {k: accs[k] / total for k in self._models}
        # Smooth update: 70% old weight, 30% accuracy-based
        for k in self._models:
            self._weights[k] = 0.70 * self._weights[k] + 0.30 * new_weights[k]
        log.info(
            "Weight recalibration: trend=%.3f mr=%.3f mom=%.3f vol=%.3f",
            self._weights["trend"], self._weights["mean_rev"],
            self._weights["momentum"], self._weights["vol"],
        )

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

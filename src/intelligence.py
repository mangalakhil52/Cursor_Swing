"""Swing-trading intelligence: multi-day RS, trend quality, pullbacks, breakouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.constants import DIRECTION_LONG, DIRECTION_SHORT


@dataclass
class MarketContext:
    bias: str
    day_change_pct: float
    week_change_pct: float
    atr_pct: float
    above_ema: bool


@dataclass
class StockIntelligence:
    rs_5d: float
    rs_20d: float
    atr_pct: float
    dist_from_ema_pct: float
    dist_from_high_20d_pct: float
    trend_quality: float          # 0-100
    pullback_quality: float       # 0-100 (higher = cleaner pullback in trend)
    breakout_quality: float       # 0-100
    volume_ratio: float
    is_dead: bool
    dead_reason: str
    ema_fast: float
    ema_slow: float
    recent_high: float
    recent_low: float
    fair_value: float = 0.0
    fair_value_distance_pct: float = 0.0
    fair_value_distance_atr: float = 0.0
    displacement_ratio: float = 0.0
    body_to_range: float = 0.0
    displacement_direction: str = "NONE"
    breaks_structure: bool = False
    setup_grade: str = "B"
    notes: list[str] = field(default_factory=list)


def build_market_context(benchmark_daily: pd.DataFrame, bias: str) -> MarketContext:
    if benchmark_daily.empty or len(benchmark_daily) < 6:
        return MarketContext(bias, 0.0, 0.0, 1.0, False)

    close = benchmark_daily["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    week_ago = float(close.iloc[-6]) if len(close) >= 6 else prev
    day_change = ((last - prev) / prev) * 100
    week_change = ((last - week_ago) / week_ago) * 100
    atr = float((benchmark_daily["high"] - benchmark_daily["low"]).tail(14).mean())
    atr_pct = (atr / last) * 100 if last else 1.0
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    return MarketContext(
        bias=bias,
        day_change_pct=round(day_change, 2),
        week_change_pct=round(week_change, 2),
        atr_pct=round(atr_pct, 2),
        above_ema=last >= ema21,
    )


def _pct_change(series: pd.Series, bars: int) -> float:
    if len(series) <= bars:
        return 0.0
    a = float(series.iloc[-1])
    b = float(series.iloc[-(bars + 1)])
    if b == 0:
        return 0.0
    return ((a - b) / b) * 100


def analyze_swing_stock(
    daily: pd.DataFrame,
    atr_value: float,
    last_price: float,
    volume_ratio: float,
    market: MarketContext,
    nifty_closes: pd.Series | None = None,
) -> StockIntelligence:
    notes: list[str] = []
    close = daily["close"].astype(float)
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)

    stock_5d = _pct_change(close, 5)
    stock_20d = _pct_change(close, 20)
    nifty_5d = _pct_change(nifty_closes, 5) if nifty_closes is not None else market.week_change_pct
    nifty_20d = _pct_change(nifty_closes, 20) if nifty_closes is not None else market.week_change_pct * 2
    rs_5d = stock_5d - nifty_5d
    rs_20d = stock_20d - nifty_20d

    atr_pct = (atr_value / last_price) * 100 if last_price else 0.0
    ema_fast = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema_slow

    dist_ema = ((last_price - ema_slow) / ema_slow) * 100 if ema_slow else 0.0
    recent_high = float(high.tail(20).max())
    recent_low = float(low.tail(20).min())
    dist_high = ((last_price - recent_high) / recent_high) * 100 if recent_high else 0.0

    # Video adaptation for daily swing:
    # - 20-day volume-weighted typical price = rolling "fair price"
    # - displacement = current candle body larger than previous candle body
    # - A+ additionally closes through recent structure
    volume = daily["volume"].astype(float)
    typical = (high + low + close) / 3
    volume_20 = float(volume.tail(20).sum())
    fair_value = (
        float((typical.tail(20) * volume.tail(20)).sum() / volume_20)
        if volume_20 > 0
        else float(close.tail(20).mean())
    )
    fair_distance_pct = (
        ((last_price - fair_value) / fair_value) * 100 if fair_value else 0.0
    )
    fair_distance_atr = (
        (last_price - fair_value) / atr_value if atr_value > 0 else 0.0
    )

    current_open = float(daily["open"].iloc[-1])
    current_body = abs(last_price - current_open)
    previous_body = abs(
        float(daily["close"].iloc[-2]) - float(daily["open"].iloc[-2])
    )
    current_range = float(high.iloc[-1] - low.iloc[-1])
    displacement_ratio = current_body / max(previous_body, last_price * 0.001)
    body_to_range = current_body / current_range if current_range > 0 else 0.0
    if last_price > current_open:
        displacement_direction = DIRECTION_LONG
    elif last_price < current_open:
        displacement_direction = DIRECTION_SHORT
    else:
        displacement_direction = "NONE"

    prior_structure_high = float(high.iloc[:-1].tail(5).max())
    prior_structure_low = float(low.iloc[:-1].tail(5).min())
    breaks_structure = (
        displacement_direction == DIRECTION_LONG
        and last_price > prior_structure_high
    ) or (
        displacement_direction == DIRECTION_SHORT
        and last_price < prior_structure_low
    )

    is_displacement = displacement_ratio > 1.0 and body_to_range >= 0.55
    if is_displacement and breaks_structure and body_to_range >= 0.60:
        setup_grade = "A+"
        notes.append(
            "A+ video setup: displacement candle closed through 5-day structure"
        )
    elif is_displacement:
        setup_grade = "A"
        notes.append(
            f"A video setup: candle body {displacement_ratio:.1f}x previous "
            f"({body_to_range:.0%} of range)"
        )
    else:
        setup_grade = "B"

    # Trend quality: stacked EMAs + higher lows
    trend_quality = 40.0
    if ema_fast > ema_slow > ema_50 and last_price > ema_fast:
        trend_quality = 90.0
        notes.append("Bullish EMA stack (9>21>50) — Stage-2 style trend")
    elif ema_fast > ema_slow and last_price > ema_slow:
        trend_quality = 75.0
        notes.append("Uptrend intact above 21 EMA")
    elif ema_fast < ema_slow < ema_50 and last_price < ema_fast:
        trend_quality = 85.0
        notes.append("Bearish EMA stack — swing short structure")
    elif ema_fast < ema_slow:
        trend_quality = 60.0

    # Pullback quality: price near EMA in trend without breaking structure
    pullback_quality = 20.0
    if ema_fast > ema_slow and -3.0 <= dist_ema <= 1.5:
        pullback_quality = 85.0
        notes.append(f"Constructive pullback to 21 EMA ({dist_ema:+.1f}%)")
    elif ema_fast > ema_slow and -5.0 <= dist_ema < -3.0:
        pullback_quality = 60.0
        notes.append("Deeper pullback — wait for reclaim")
    elif ema_fast < ema_slow and -1.5 <= dist_ema <= 3.0:
        pullback_quality = 80.0
        notes.append("Rally into EMA in downtrend — short pullback zone")

    # Breakout quality: near/through 20d high with volume
    breakout_quality = 20.0
    if dist_high >= -0.5 and volume_ratio >= 1.3 and rs_20d > 0:
        breakout_quality = 90.0
        notes.append("Breaking/holding 20-day high with volume + RS")
    elif dist_high >= -1.5 and volume_ratio >= 1.1:
        breakout_quality = 70.0
        notes.append("Near 20-day high — breakout watch")

    if rs_20d >= 3:
        notes.append(f"20d relative strength leader vs Nifty ({rs_20d:+.1f}%)")
    elif rs_20d <= -3:
        notes.append(f"20d relative weakness vs Nifty ({rs_20d:+.1f}%)")
    if volume_ratio >= 1.5:
        notes.append(f"Volume expansion {volume_ratio:.1f}x — institutions interested")

    is_dead = False
    dead_reason = ""
    if atr_pct < 1.0:
        is_dead = True
        dead_reason = f"ATR {atr_pct:.2f}% too tight for swing risk/reward"
    elif abs(rs_5d) < 0.3 and abs(rs_20d) < 1.0 and volume_ratio < 0.9:
        is_dead = True
        dead_reason = "No relative edge and sleepy volume — skip"
    elif trend_quality < 55 and pullback_quality < 50 and breakout_quality < 50:
        is_dead = True
        dead_reason = "No swing structure (trend/pullback/breakout all weak)"

    return StockIntelligence(
        rs_5d=round(rs_5d, 2),
        rs_20d=round(rs_20d, 2),
        atr_pct=round(atr_pct, 2),
        dist_from_ema_pct=round(dist_ema, 2),
        dist_from_high_20d_pct=round(dist_high, 2),
        trend_quality=round(trend_quality, 1),
        pullback_quality=round(pullback_quality, 1),
        breakout_quality=round(breakout_quality, 1),
        volume_ratio=round(volume_ratio, 2),
        is_dead=is_dead,
        dead_reason=dead_reason,
        ema_fast=round(ema_fast, 2),
        ema_slow=round(ema_slow, 2),
        recent_high=round(recent_high, 2),
        recent_low=round(recent_low, 2),
        fair_value=round(fair_value, 2),
        fair_value_distance_pct=round(fair_distance_pct, 2),
        fair_value_distance_atr=round(fair_distance_atr, 2),
        displacement_ratio=round(displacement_ratio, 2),
        body_to_range=round(body_to_range, 2),
        displacement_direction=displacement_direction,
        breaks_structure=breaks_structure,
        setup_grade=setup_grade,
        notes=notes,
    )


def conviction_grade(score: float, intel: StockIntelligence, confluence: int) -> str:
    from src.constants import CONVICTION_A, CONVICTION_B, CONVICTION_C

    if intel.is_dead:
        return CONVICTION_C
    if score >= 78 and confluence >= 2 and abs(intel.rs_20d) >= 2:
        return CONVICTION_A
    if score >= 65 and confluence >= 1:
        return CONVICTION_B
    return CONVICTION_C


def swing_playbook(
    direction: str,
    entry: float,
    stop: float,
    target_1: float,
    target: float,
    trigger: float,
    intel: StockIntelligence,
    hold_days: str,
) -> list[str]:
    steps = [
        f"SWING TRADE — plan to hold {hold_days} (not intraday)",
    ]
    if direction == DIRECTION_LONG:
        steps.append(f"Enter on hold above INR {trigger:,.2f} (or buy pullback toward 21 EMA ~{intel.ema_slow:,.2f})")
        steps.append(f"Hard stop below structure: INR {stop:,.2f}")
        steps.append(f"Target 1: INR {target_1:,.2f} — book 40-50% and move stop to cost")
        steps.append(f"Target 2: INR {target:,.2f} — trail the balance above rising 9/21 EMA")
    else:
        steps.append(f"Enter on hold below INR {trigger:,.2f}")
        steps.append(f"Hard stop: INR {stop:,.2f}")
        steps.append(f"Target 1: INR {target_1:,.2f} — cover 40-50% and move stop to cost")
        steps.append(f"Target 2: INR {target:,.2f} — trail the balance")
    steps.append("Review only at EOD — avoid noise of 5-min charts")
    steps.append("Exit early if thesis breaks (EMA stack flips or RS collapses vs Nifty)")
    steps.append(f"Time stop: if no progress in {hold_days.split('-')[0]} days, reassess")
    return steps

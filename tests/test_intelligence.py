"""Tests for swing intelligence."""

import pandas as pd

from src.intelligence import MarketContext, StockIntelligence, analyze_swing_stock, conviction_grade


def _daily(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-08-01", periods=n, tz="Asia/Kolkata")
    close = pd.Series([100 + i * 0.8 for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
            "volume": [1_200_000.0] * n,
        },
        index=idx,
    )


def test_leader_not_dead():
    market = MarketContext("BULLISH", 0.2, 1.0, 1.2, True)
    daily = _daily()
    # Flat index → rising stock = positive RS
    nifty = pd.Series([100.0] * len(daily), index=daily.index)
    intel = analyze_swing_stock(
        daily=daily,
        atr_value=3.0,
        last_price=float(daily["close"].iloc[-1]),
        volume_ratio=1.5,
        market=market,
        nifty_closes=nifty,
    )
    assert intel.is_dead is False
    assert intel.rs_20d > 0


def test_tight_atr_is_dead():
    market = MarketContext("NEUTRAL", 0.0, 0.0, 1.0, True)
    daily = _daily()
    intel = analyze_swing_stock(
        daily=daily,
        atr_value=0.4,
        last_price=150.0,
        volume_ratio=0.8,
        market=market,
        nifty_closes=daily["close"],
    )
    assert intel.is_dead is True


def test_conviction_grades():
    intel = StockIntelligence(
        rs_5d=2,
        rs_20d=4,
        atr_pct=2.5,
        dist_from_ema_pct=-1.0,
        dist_from_high_20d_pct=-0.2,
        trend_quality=85,
        pullback_quality=80,
        breakout_quality=70,
        volume_ratio=1.4,
        is_dead=False,
        dead_reason="",
        ema_fast=100,
        ema_slow=98,
        recent_high=105,
        recent_low=90,
    )
    assert conviction_grade(82, intel, 2) == "A"
    assert conviction_grade(66, intel, 1) == "B"


def test_video_a_plus_displacement_breaks_structure():
    daily = _daily()
    # Make the previous candle small and the latest candle body-dominant.
    daily.loc[daily.index[-2], ["open", "high", "low", "close"]] = [
        145.0,
        146.0,
        144.0,
        145.5,
    ]
    daily.loc[daily.index[-1], ["open", "high", "low", "close"]] = [
        145.0,
        152.0,
        144.5,
        151.0,
    ]
    market = MarketContext("BULLISH", 0.2, 1.0, 1.2, True)
    intel = analyze_swing_stock(
        daily=daily,
        atr_value=3.0,
        last_price=151.0,
        volume_ratio=1.6,
        market=market,
        nifty_closes=pd.Series([100.0] * len(daily), index=daily.index),
    )
    assert intel.setup_grade == "A+"
    assert intel.breaks_structure is True
    assert intel.displacement_ratio > 1
    assert intel.body_to_range >= 0.60
    assert intel.fair_value > 0


def test_video_grade_b_when_no_displacement():
    daily = _daily()
    market = MarketContext("BULLISH", 0.2, 1.0, 1.2, True)
    intel = analyze_swing_stock(
        daily=daily,
        atr_value=3.0,
        last_price=float(daily["close"].iloc[-1]),
        volume_ratio=1.0,
        market=market,
        nifty_closes=pd.Series([100.0] * len(daily), index=daily.index),
    )
    assert intel.setup_grade == "B"
    assert intel.breaks_structure is False

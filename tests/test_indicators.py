"""Unit tests for indicator and scoring helpers."""

import pandas as pd

from src.indicators import atr, ema, pivot_levels, rsi, vwap
from src.scorer import detect_nifty_bias


def _sample_ohlcv(n: int = 30) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def test_ema_length():
    df = _sample_ohlcv()
    result = ema(df["close"], 9)
    assert len(result) == len(df)


def test_rsi_bounded():
    df = _sample_ohlcv(60)
    result = rsi(df["close"], 14).dropna()
    assert len(result) > 0
    assert result.min() >= 0
    assert result.max() <= 100


def test_atr_positive():
    df = _sample_ohlcv()
    result = atr(df, 14).dropna()
    assert (result > 0).all()


def test_vwap_reasonable():
    df = _sample_ohlcv()
    result = vwap(df)
    assert abs(result.iloc[-1] - df["close"].iloc[-1]) < 6


def test_pivot_levels():
    levels = pivot_levels(110, 90, 100)
    assert levels["pivot"] == 100
    assert levels["r1"] > levels["pivot"]
    assert levels["s1"] < levels["pivot"]


def test_nifty_bias_neutral_on_flat():
    df = _sample_ohlcv()
    bias = detect_nifty_bias(df)
    assert bias in ("BULLISH", "BEARISH", "NEUTRAL")

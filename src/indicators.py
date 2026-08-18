"""Technical indicators used for intraday setup detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP. Pass a single session — VWAP resets each trading day."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum().replace(0, np.nan)
    return cumulative_tp_vol / cumulative_vol


def last_session(df: pd.DataFrame) -> pd.DataFrame:
    """Slice an intraday frame down to its most recent trading date."""
    if df is None or df.empty:
        return df
    session_date = df.index[-1].date()
    return df[[ts.date() == session_date for ts in df.index]]


def pivot_levels(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    pivot = (prev_high + prev_low + prev_close) / 3
    return {
        "pivot": pivot,
        "r1": 2 * pivot - prev_low,
        "r2": pivot + (prev_high - prev_low),
        "s1": 2 * pivot - prev_high,
        "s2": pivot - (prev_high - prev_low),
    }


def enrich_daily(df: pd.DataFrame, ema_fast: int, ema_slow: int, rsi_period: int, atr_period: int) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out, atr_period)
    out["vol_sma"] = volume_sma(out["volume"])
    return out


def enrich_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a single session's intraday bars (VWAP is session-scoped)."""
    out = last_session(df).copy()
    out["vwap"] = vwap(out)
    out["ema_9"] = ema(out["close"], 9)
    out["rsi"] = rsi(out["close"], 14)
    return out

"""Tests for latest-session reconstruction from intraday bars."""

from datetime import date

import pandas as pd

from src.data_fetcher import DataFetcher
from src.indicators import last_session


def _daily(n: int = 40, end: str = "2026-07-27") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n, tz="Asia/Kolkata")
    base = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 2,
            "low": base - 2,
            "close": base + 1,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )


def _intraday_session(day: str = "2026-07-28") -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:15", f"{day} 15:15", freq="15min", tz="Asia/Kolkata")
    n = len(idx)
    close = pd.Series([200.0 + i for i in range(n)], index=idx)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 3,
            "close": close,
            "volume": [50_000.0] * n,
        },
        index=idx,
    )


def test_session_bar_aggregates_intraday():
    intra = _intraday_session()
    result = DataFetcher.session_bar_from_intraday(intra)
    assert result is not None
    session_date, bar = result
    assert session_date == date(2026, 7, 28)
    assert bar["open"] == intra["open"].iloc[0]
    assert bar["close"] == intra["close"].iloc[-1]
    assert bar["high"] == intra["high"].max()
    assert bar["low"] == intra["low"].min()
    assert bar["volume"] == intra["volume"].sum()


def test_merge_appends_missing_session():
    daily = _daily()
    intra = _intraday_session()
    merged, appended = DataFetcher.merge_latest_session(daily, intra)
    assert appended is True
    assert len(merged) == len(daily) + 1
    assert merged.index[-1].date() == date(2026, 7, 28)


def test_merge_is_noop_when_daily_current():
    daily = _daily(end="2026-07-28")
    intra = _intraday_session()
    merged, appended = DataFetcher.merge_latest_session(daily, intra)
    assert appended is False
    assert len(merged) == len(daily)


def test_snapshot_uses_true_gap():
    fetcher = DataFetcher()
    daily = _daily()
    intra = _intraday_session()
    snap = fetcher.snapshot_from_daily("TEST", daily, intra)

    assert snap is not None
    assert snap.session_date == date(2026, 7, 28)
    assert snap.session_from_intraday is True

    prev_close = float(daily["close"].iloc[-1])
    expected_gap = ((intra["open"].iloc[0] - prev_close) / prev_close) * 100
    assert abs(snap.gap_pct - expected_gap) < 1e-6
    # Gap (open vs prev close) must differ from the full-day move
    assert abs(snap.gap_pct - snap.day_change_pct) > 1e-6


def test_last_session_filters_to_one_day():
    two_days = pd.concat([_intraday_session("2026-07-27"), _intraday_session("2026-07-28")])
    session = last_session(two_days)
    assert {ts.date() for ts in session.index} == {date(2026, 7, 28)}

"""Fetch OHLCV data for NSE symbols via Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.constants import IST


@dataclass
class MarketSnapshot:
    symbol: str
    daily: pd.DataFrame
    intraday: Optional[pd.DataFrame]
    last_price: float
    prev_close: float
    gap_pct: float          # True gap: session open vs previous close
    day_change_pct: float   # Move from previous close to latest price
    session_date: date
    session_from_intraday: bool = False


class DataFetcher:
    """Downloads daily and intraday bars for the watchlist."""

    def __init__(self, exchange_suffix: str = ".NS") -> None:
        self.exchange_suffix = exchange_suffix

    def _ticker(self, symbol: str) -> str:
        return symbol if symbol.startswith("^") else f"{symbol}{self.exchange_suffix}"

    def fetch_daily(self, symbol: str, days: int = 90) -> pd.DataFrame:
        ticker = self._ticker(symbol)
        end = datetime.now(IST)
        start = end - timedelta(days=days)
        df = yf.download(
            ticker,
            start=start.date(),
            end=(end + timedelta(days=1)).date(),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df.empty:
            return df
        return self._normalize_single(df)

    def fetch_daily_batch(
        self,
        symbols: list[str],
        days: int = 90,
        chunk_size: int = 80,
    ) -> dict[str, pd.DataFrame]:
        """Download daily bars for many symbols in chunks (much faster than one-by-one)."""
        end = datetime.now(IST)
        start = end - timedelta(days=days)
        out: dict[str, pd.DataFrame] = {}

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            tickers = [self._ticker(s) for s in chunk]
            raw = yf.download(
                tickers,
                start=start.date(),
                end=(end + timedelta(days=1)).date(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            if raw.empty:
                continue

            if len(chunk) == 1:
                out[chunk[0]] = self._normalize_single(raw)
                continue

            for symbol, ticker in zip(chunk, tickers):
                try:
                    if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                        df = raw[ticker].copy()
                    else:
                        continue
                    df = self._normalize_single(df)
                    if not df.empty:
                        out[symbol] = df
                except Exception:  # noqa: BLE001
                    continue

        return out

    def fetch_intraday(self, symbol: str, period: str = "5d") -> pd.DataFrame:
        ticker = self._ticker(symbol)
        df = yf.download(
            ticker,
            period=period,
            interval="15m",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df.empty:
            return df
        return self._normalize_single(df)

    def fetch_intraday_batch(
        self,
        symbols: list[str],
        period: str = "5d",
        chunk_size: int = 40,
    ) -> dict[str, pd.DataFrame]:
        """Download 15m bars for a shortlist of symbols."""
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            tickers = [self._ticker(s) for s in chunk]
            raw = yf.download(
                tickers,
                period=period,
                interval="15m",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            if raw.empty:
                continue

            if len(chunk) == 1:
                out[chunk[0]] = self._normalize_single(raw)
                continue

            for symbol, ticker in zip(chunk, tickers):
                try:
                    if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                        df = raw[ticker].copy()
                    else:
                        continue
                    df = self._normalize_single(df)
                    if not df.empty:
                        out[symbol] = df
                except Exception:  # noqa: BLE001
                    continue
        return out

    @staticmethod
    def _normalize_single(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            # Single-ticker sometimes still MultiIndex: (OHLCV, ticker)
            if out.columns.nlevels == 2:
                level0 = [str(c).lower() for c in out.columns.get_level_values(0)]
                if set(level0) & {"open", "high", "low", "close", "adj close", "volume"}:
                    out.columns = level0
                else:
                    out.columns = [str(col[0]).lower() for col in out.columns]
            else:
                out.columns = [str(col[0]).lower() for col in out.columns]
        else:
            out.columns = [str(col).lower() for col in out.columns]

        keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
        if len(keep) < 4:
            return pd.DataFrame()
        out = out[keep].dropna(how="all")
        out = out.dropna(subset=["close"])
        if out.empty:
            return out

        if out.index.tzinfo is None:
            out.index = out.index.tz_localize("UTC").tz_convert(IST)
        else:
            out.index = out.index.tz_convert(IST)
        return out

    @staticmethod
    def session_bar_from_intraday(intraday: pd.DataFrame) -> Optional[tuple[date, dict]]:
        """Aggregate the most recent session's intraday bars into one daily OHLCV bar.

        Yahoo publishes the current/just-closed session as an all-NaN daily row for
        hours after the close, so the daily series alone lags by a full session.
        """
        if intraday is None or intraday.empty:
            return None

        session_date = intraday.index[-1].date()
        mask = [ts.date() == session_date for ts in intraday.index]
        day = intraday[mask]
        if day.empty:
            return None

        return session_date, {
            "open": float(day["open"].iloc[0]),
            "high": float(day["high"].max()),
            "low": float(day["low"].min()),
            "close": float(day["close"].iloc[-1]),
            "volume": float(day["volume"].sum()),
        }

    @classmethod
    def merge_latest_session(
        cls,
        daily: pd.DataFrame,
        intraday: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool]:
        """Append the newest session from intraday bars if the daily series lags."""
        if daily.empty:
            return daily, False

        session = cls.session_bar_from_intraday(intraday)
        if session is None:
            return daily, False

        session_date, bar = session
        if session_date <= daily.index[-1].date():
            return daily, False

        stamp = pd.Timestamp(session_date).tz_localize(IST)
        row = pd.DataFrame([bar], index=pd.DatetimeIndex([stamp], name=daily.index.name))
        merged = pd.concat([daily, row[daily.columns]])
        return merged, True

    def snapshot_from_daily(
        self,
        symbol: str,
        daily: pd.DataFrame,
        intraday: Optional[pd.DataFrame] = None,
    ) -> Optional[MarketSnapshot]:
        if daily.empty:
            return None

        intraday = intraday if intraday is not None and not intraday.empty else None
        merged, from_intraday = self.merge_latest_session(daily, intraday)
        if len(merged) < 30:
            return None

        last_price = float(merged["close"].iloc[-1])
        prev_close = float(merged["close"].iloc[-2])
        session_open = float(merged["open"].iloc[-1])

        return MarketSnapshot(
            symbol=symbol,
            daily=merged,
            intraday=intraday,
            last_price=last_price,
            prev_close=prev_close,
            gap_pct=((session_open - prev_close) / prev_close) * 100,
            day_change_pct=((last_price - prev_close) / prev_close) * 100,
            session_date=merged.index[-1].date(),
            session_from_intraday=from_intraday,
        )

    def build_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        daily = self.fetch_daily(symbol)
        if daily.empty:
            return None
        intraday = self.fetch_intraday(symbol)
        return self.snapshot_from_daily(
            symbol,
            daily,
            intraday if not intraday.empty else None,
        )

    def fetch_benchmark(self, symbol: str) -> pd.DataFrame:
        """Benchmark daily series, topped up with the latest session if Yahoo lags."""
        daily = self.fetch_daily(symbol, days=60)
        if daily.empty:
            return daily
        intraday = self.fetch_intraday(symbol)
        merged, _ = self.merge_latest_session(daily, intraday if not intraday.empty else None)
        return merged

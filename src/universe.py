"""Load trade universes (watchlist or index/equity constituent files)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_symbols(config: dict, base_dir: Path | None = None) -> list[str]:
    """
    Resolve the scan universe from config.

    Priority:
    1. universe.list_file (CSV with Symbol/SYMBOL column)
    2. universe.watchlist (inline YAML list)
    """
    base = base_dir or Path.cwd()
    universe = config.get("universe", {})

    list_file = universe.get("list_file")
    if list_file:
        path = Path(list_file)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            raise FileNotFoundError(
                f"Universe file not found: {path}. "
                "Run: python scripts/refresh_universe.py --source nse_equity"
            )
        series = universe.get("series")  # e.g. ["EQ"] or null = all rows in file
        symbols = _symbols_from_csv(path, series_filter=series)
        exclude = {str(s).strip().upper() for s in (universe.get("exclude") or [])}
        if exclude:
            symbols = [s for s in symbols if s not in exclude]
        return symbols

    watchlist = universe.get("watchlist") or []
    if not watchlist:
        raise ValueError("No symbols configured. Set universe.list_file or universe.watchlist.")
    return [str(s).strip().upper() for s in watchlist if str(s).strip()]


def _symbols_from_csv(path: Path, series_filter: list[str] | None = None) -> list[str]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    col = None
    for candidate in ("Symbol", "symbol", "SYMBOL", "Ticker", "ticker"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(f"No Symbol column in {path}. Columns: {list(df.columns)}")

    if series_filter and "Series" in df.columns:
        allowed = {str(s).strip().upper() for s in series_filter}
        series = df["Series"].astype(str).str.strip().str.upper()
        df = df[series.isin(allowed)]
    elif series_filter and "SERIES" in df.columns:
        allowed = {str(s).strip().upper() for s in series_filter}
        series = df["SERIES"].astype(str).str.strip().str.upper()
        df = df[series.isin(allowed)]

    symbols = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return symbols

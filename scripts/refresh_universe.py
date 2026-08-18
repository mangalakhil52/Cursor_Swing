#!/usr/bin/env python3
"""Refresh the scan universe from NSE (full equity / total market / nifty500)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

SOURCES = {
    # Full NSE cash equity list (~2000+ EQ names)
    "nse_equity": {
        "urls": [
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        ],
        "default_out": "data/nse_equity.csv",
        "series_filter": ["EQ"],
    },
    # Broader than Nifty 500 (~750)
    "nifty_total_market": {
        "urls": [
            "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
            "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
        ],
        "default_out": "data/nifty_total_market.csv",
        "series_filter": None,
    },
    "nifty500": {
        "urls": [
            "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        ],
        "default_out": "data/nifty500.csv",
        "series_filter": None,
    },
}


def _download(urls: list[str]) -> bytes:
    last_error: Exception | None = None
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            resp.raise_for_status()
            if len(resp.content) < 500:
                raise ValueError("Downloaded file too small")
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Download failed: {last_error}")


def build_universe(source: str, dest: Path, series_filter: list[str] | None) -> tuple[int, Path]:
    meta = SOURCES[source]
    raw = _download(meta["urls"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download.csv")
    tmp.write_bytes(raw)

    df = pd.read_csv(tmp)
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = None
    for candidate in ("Symbol", "SYMBOL", "symbol"):
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"No Symbol column. Columns: {list(df.columns)}")

    if "SERIES" in df.columns and series_filter:
        series = df["SERIES"].astype(str).str.strip().str.upper()
        df = df[series.isin([s.upper() for s in series_filter])]

    out = pd.DataFrame(
        {
            "Symbol": (
                df[symbol_col]
                .astype(str)
                .str.strip()
                .str.upper()
            ),
        }
    )
    # Keep useful metadata when present
    for src, dst in (
        ("NAME OF COMPANY", "Company Name"),
        ("Company Name", "Company Name"),
        ("ISIN NUMBER", "ISIN Code"),
        ("ISIN Code", "ISIN Code"),
        ("SERIES", "Series"),
        ("Series", "Series"),
        ("Industry", "Industry"),
    ):
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src].astype(str).str.strip().values

    out = out[out["Symbol"].ne("") & out["Symbol"].notna()]
    out = out.drop_duplicates(subset=["Symbol"]).sort_values("Symbol")
    out.to_csv(dest, index=False)
    tmp.unlink(missing_ok=True)
    return len(out), dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh NSE scan universe CSV")
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES.keys()),
        default="nse_equity",
        help="Universe source (default: full NSE equity EQ list)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--include-be",
        action="store_true",
        help="For nse_equity, also include BE series (trade-to-trade)",
    )
    args = parser.parse_args()

    meta = SOURCES[args.source]
    dest = args.output or Path(meta["default_out"])
    series_filter = meta["series_filter"]
    if args.source == "nse_equity":
        series_filter = ["EQ", "BE"] if args.include_be else ["EQ"]

    count, path = build_universe(args.source, dest, series_filter)
    print(f"Saved {count} symbols from '{args.source}' to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

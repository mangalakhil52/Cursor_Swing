"""Tests for universe loading."""

from pathlib import Path

from src.universe import load_symbols


def test_load_nse_equity_universe():
    root = Path(__file__).resolve().parents[1]
    config = {"universe": {"list_file": "data/nse_equity.csv"}}
    symbols = load_symbols(config, base_dir=root)
    assert len(symbols) >= 1500
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert all(s == s.upper() for s in symbols)


def test_load_nifty500_csv_if_present():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "nifty500.csv"
    if not path.exists():
        return
    config = {"universe": {"list_file": "data/nifty500.csv"}}
    symbols = load_symbols(config, base_dir=root)
    assert len(symbols) == 500
    assert "RELIANCE" in symbols


def test_load_inline_watchlist():
    config = {"universe": {"watchlist": ["reliance", "TCS"]}}
    symbols = load_symbols(config)
    assert symbols == ["RELIANCE", "TCS"]

"""F&O eligibility helpers. Overnight shorts are permitted only for symbols in this universe."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_fno_symbols(path: str | Path = 'data/fno_universe.csv') -> set[str]:
    p=Path(path)
    if not p.exists(): return set()
    d=pd.read_csv(p)
    col=next((c for c in ('Symbol','symbol','SYMBOL','Ticker','ticker') if c in d.columns),None)
    if not col: return set()
    return set(d[col].astype(str).str.strip().str.upper().dropna())

def short_allowed(symbol: str, fno_symbols: set[str]) -> bool:
    return str(symbol).strip().upper() in fno_symbols

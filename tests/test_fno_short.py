from pathlib import Path
import pandas as pd
from src.fno_universe import load_fno_symbols, short_allowed

def test_short_fails_closed_when_symbol_is_not_fno(tmp_path):
    p=tmp_path/'fno.csv'; pd.DataFrame({'Symbol':['RELIANCE','HDFCBANK']}).to_csv(p,index=False)
    symbols=load_fno_symbols(p)
    assert short_allowed('RELIANCE',symbols)
    assert not short_allowed('ABCNOTFNO',symbols)

def test_missing_universe_fails_closed(tmp_path):
    symbols=load_fno_symbols(tmp_path/'missing.csv')
    assert symbols==set()
    assert not short_allowed('RELIANCE',symbols)

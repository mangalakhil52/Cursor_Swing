"""Maintain persistent Nifty benchmark history used by point-in-time research."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

COLS=['open','high','low','close','volume']

def norm(df):
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.columns=[str(c).lower().strip() for c in df.columns]
    d=df[[c for c in COLS if c in df.columns]].copy()
    if len(d.columns)<4: raise ValueError('benchmark OHLC data incomplete')
    d.index=pd.to_datetime(d.index,errors='coerce')
    if getattr(d.index,'tz',None) is not None:d.index=d.index.tz_localize(None)
    return d[~d.index.isna()].loc[~d.index.duplicated(keep='last')].sort_index()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--years',type=int,default=5); ap.add_argument('--out',default='data/history/_benchmark.csv'); a=ap.parse_args()
    p=Path(a.out); existing=norm(pd.read_csv(p,index_col=0,parse_dates=True)) if p.exists() else None
    period=f'{a.years}y' if existing is None or existing.empty else '10d'
    fresh=yf.download('^NSEI',period=period,interval='1d',auto_adjust=False,progress=False,threads=False)
    if fresh.empty: raise SystemExit('No benchmark data returned')
    merged=norm(fresh) if existing is None else norm(pd.concat([existing,norm(fresh)]))
    if existing is None or not merged.equals(existing):
        p.parent.mkdir(parents=True,exist_ok=True); merged.to_csv(p)
    print(f'Benchmark history: {len(merged)} rows, {merged.index.min().date()} -> {merged.index.max().date()}')

if __name__=='__main__':main()

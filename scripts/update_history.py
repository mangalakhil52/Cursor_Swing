"""Maintain persistent per-symbol daily OHLCV history in the repository.

First run: downloads the configured lookback once.
Later runs: downloads only a small recent window and merges it into the stored CSV.
The repository therefore becomes the durable historical cache used by research.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    df=df.rename(columns={str(c).lower():str(c).lower() for c in df.columns})
    cols=[c for c in ['open','high','low','close','volume'] if c in df.columns]
    df=df[cols].copy()
    df.index=pd.to_datetime(df.index).tz_localize(None)
    return df[~df.index.duplicated(keep='last')].sort_index()


def update_symbol(symbol: str, out: Path, years: int) -> bool:
    existing=None
    if out.exists():
        existing=pd.read_csv(out,index_col=0,parse_dates=True)
        existing=normalize(existing)
    period=f'{years}y' if existing is None or existing.empty else '10d'
    try:
        fresh=yf.download(symbol+'.NS',period=period,interval='1d',auto_adjust=False,progress=False,threads=False)
        if fresh.empty:return False
        fresh=normalize(fresh)
        merged=fresh if existing is None or existing.empty else pd.concat([existing,fresh])
        merged=normalize(merged)
        merged.to_csv(out)
        return True
    except Exception as exc:
        print(f'ERROR {symbol}: {exc}')
        return False


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--years',type=int,default=5)
    ap.add_argument('--data-dir',default='data/history')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    universe=pd.read_csv('data/nse_equity.csv')
    symbols=universe['Symbol'].astype(str).str.strip().str.upper().drop_duplicates().tolist()
    if args.limit>0:symbols=symbols[:args.limit]
    root=Path(args.data_dir); root.mkdir(parents=True,exist_ok=True)
    ok=0
    for i,symbol in enumerate(symbols,1):
        if update_symbol(symbol,root/f'{symbol}.csv',args.years):ok+=1
        if i%50==0:print(f'Processed {i}/{len(symbols)}; updated {ok}')
    print(f'History update complete: {ok}/{len(symbols)} symbols updated')
    return 0

if __name__=='__main__':raise SystemExit(main())

"""Maintain persistent per-symbol daily OHLCV history in the repository.

First run downloads the configured lookback once. Later runs download only a
small recent window and merge it into the stored CSVs.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

COLUMNS=['open','high','low','close','volume']

def normalize(df:pd.DataFrame)->pd.DataFrame:
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.columns=[str(c).lower().strip() for c in df.columns]
    missing=[c for c in COLUMNS if c not in df.columns]
    if missing: raise ValueError(f'missing OHLCV columns: {missing}')
    d=df[COLUMNS].copy(); idx=pd.to_datetime(d.index,errors='coerce')
    if getattr(idx,'tz',None) is not None: idx=idx.tz_localize(None)
    d.index=idx; d=d[~d.index.isna()]; return d[~d.index.duplicated(keep='last')].sort_index()

def update_symbol(symbol:str,out:Path,years:int)->tuple[bool,str]:
    existing=None
    if out.exists(): existing=normalize(pd.read_csv(out,index_col=0,parse_dates=True))
    period=f'{years}y' if existing is None or existing.empty else '10d'
    try:
        fresh=yf.download(symbol+'.NS',period=period,interval='1d',auto_adjust=False,progress=False,threads=False)
        if fresh.empty:return False,'empty'
        fresh=normalize(fresh); merged=fresh if existing is None or existing.empty else normalize(pd.concat([existing,fresh]))
        if existing is not None and merged.equals(existing): return False,'unchanged'
        out.parent.mkdir(parents=True,exist_ok=True); merged.to_csv(out); return True,'updated'
    except Exception as exc:
        print(f'ERROR {symbol}: {exc}'); return False,'error'

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--years',type=int,default=5); ap.add_argument('--data-dir',default='data/history'); ap.add_argument('--universe',default='data/nifty500.csv'); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args()
    universe=pd.read_csv(args.universe); col=next((c for c in ('Symbol','SYMBOL','symbol') if c in universe.columns),None)
    if col is None: raise SystemExit(f'No symbol column in {args.universe}: {list(universe.columns)}')
    symbols=universe[col].astype(str).str.strip().str.upper().drop_duplicates().tolist()
    if args.limit>0:symbols=symbols[:args.limit]
    root=Path(args.data_dir); root.mkdir(parents=True,exist_ok=True); manifest={}; stats={'symbols':len(symbols),'changed':0,'new_history':0,'unchanged':0,'empty':0,'errors':0}
    for i,symbol in enumerate(symbols,1):
        path=root/f'{symbol}.csv'; existed=path.exists(); changed,status=update_symbol(symbol,path,args.years)
        if changed: stats['changed']+=1; stats['new_history']+=int(not existed)
        elif status=='unchanged':stats['unchanged']+=1
        elif status=='empty':stats['empty']+=1
        else:stats['errors']+=1
        if path.exists():
            try:
                d=normalize(pd.read_csv(path,index_col=0,parse_dates=True)); manifest[symbol]={'file':path.name,'rows':len(d),'start':d.index.min().date().isoformat(),'end':d.index.max().date().isoformat()}
            except Exception: pass
        if i%50==0:print(f'Processed {i}/{len(symbols)}; changed={stats["changed"]}, unchanged={stats["unchanged"]}, errors={stats["errors"]}')
    (root/'_manifest.json').write_text(json.dumps({'updated_at':datetime.now(timezone.utc).isoformat(),'universe':args.universe,'stats':stats,'symbols':manifest},indent=2),encoding='utf-8')
    print(f'History update complete: changed={stats["changed"]}, new={stats["new_history"]}, unchanged={stats["unchanged"]}, errors={stats["errors"]}')
    return 0
if __name__=='__main__':raise SystemExit(main())

"""Persistent signal registry for deduplication and lifecycle tracking."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS=['signal_id','session_date','symbol','direction','score','status']

def load_registry(path='data/signal_registry.csv')->pd.DataFrame:
    p=Path(path)
    if not p.exists(): return pd.DataFrame(columns=COLUMNS)
    d=pd.read_csv(p)
    for c in COLUMNS:
        if c not in d.columns:d[c]=None
    return d[COLUMNS]

def register(signals:pd.DataFrame,path='data/signal_registry.csv',cooldown_days=3)->pd.DataFrame:
    old=load_registry(path); x=signals.copy()
    if x.empty:return old
    x['session_date']=pd.to_datetime(x['session_date']); old['session_date']=pd.to_datetime(old['session_date'],errors='coerce')
    existing={(str(r.symbol).upper(),str(r.direction).upper(),pd.Timestamp(r.session_date)) for _,r in old.iterrows()}
    out=[]
    for _,r in x.sort_values('score',ascending=False).iterrows():
        key=(str(r.symbol).upper(),str(r.direction).upper(),pd.Timestamp(r.session_date))
        if key in existing:continue
        prior=old[(old.symbol.astype(str).str.upper()==key[0])&(old.direction.astype(str).str.upper()==key[1])]
        if not prior.empty and (key[2]-prior.session_date.max()).days<cooldown_days:continue
        out.append({'signal_id':f"{key[2]:%Y%m%d}-{len(old)+len(out)+1}-{key[0]}",'session_date':key[2],'symbol':key[0],'direction':key[1],'score':float(r.score),'status':'NEW'})
    result=pd.concat([old,pd.DataFrame(out,columns=COLUMNS)],ignore_index=True); Path(path).parent.mkdir(parents=True,exist_ok=True); result.to_csv(path,index=False); return result
